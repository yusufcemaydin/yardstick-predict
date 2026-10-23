"""The model's newer inputs: close matches, the first guess, the conversation, and the model without LLM analysis."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_model import ONE_WINDOW, rows_for, synthetic

from yardstick_predict import model as M


def unit_rows(n: int, size: int = 16, seed: int = 0) -> np.ndarray:
    vectors = np.random.default_rng(seed).normal(size=(n, size)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_close_matches_summarize_the_nearest_past_tasks():
    vectors = unit_rows(300)
    results = np.random.default_rng(1).normal(size=(300, 4)).astype(np.float32)
    results[:, 3] = (results[:, 3] > 0).astype(np.float32)  # whether each answered directly
    started = np.arange(300, dtype=float) * 600
    found = M.similar_tasks(
        vectors,
        results,
        vectors,
        np.ones(300, bool),
        earlier_only=True,
        started=started,
        finished=started + 60,
        rich=True,
        half_life_days=30.0,
    )
    assert found.shape == (300, 5 + M.RICH_COLUMNS)
    i = 200
    order = np.argsort(-(vectors[:i] @ vectors[i]))
    np.testing.assert_allclose(found[i, 5:9], results[order[0]], atol=1e-5)  # the closest task's outcomes
    np.testing.assert_allclose(found[i, 10:14], results[order[: M.CLOSE]].mean(axis=0), atol=1e-5)
    top = order[: M.NEIGHBORS]
    np.testing.assert_allclose(found[i, 15:18], results[top, :3].std(axis=0), atol=1e-4)
    work = top[results[top, 3] == 0]
    np.testing.assert_allclose(found[i, 18:21], results[work, :3].mean(axis=0), atol=1e-5)


def test_newer_similar_tasks_count_more_in_the_weighted_average():
    vectors = np.ones((60, 4), dtype=np.float32) / 2  # every task equally similar
    results = np.zeros((60, 4), dtype=np.float32)
    results[30:, :3] = 1.0  # the newer half took more
    started = np.arange(60, dtype=float) * 86400  # a day apart
    found = M.similar_tasks(vectors[:30], results[:30], vectors[:1], np.ones(1, bool), earlier_only=False)
    assert found.shape == (1, 5)
    found = M.similar_tasks(
        vectors,
        results,
        vectors[:1],
        np.ones(1, bool),
        earlier_only=False,
        started=np.array([60 * 86400.0]),
        rich=True,
        pool_started=started,
        half_life_days=7.0,
    )
    plain_average, weighted = found[0, 0], found[0, -3]
    assert plain_average == pytest.approx(0.5) and weighted > 0.9


def test_the_first_guess_reads_the_whole_embedding_and_only_earlier_tasks():
    rng = np.random.default_rng(0)
    inputs = rng.normal(size=(400, 20))
    truth = inputs @ rng.normal(size=20)
    results = np.column_stack([truth + rng.normal(0, 0.1, 400), np.full(400, np.nan)])
    guesses, weights, constants = M._fit_guess(inputs, results)
    first = 400 // M.GUESS_BLOCKS
    assert np.isnan(guesses[:first, 0]).all()  # the oldest tasks have nothing before them to learn from
    assert np.corrcoef(guesses[first:, 0], truth[first:])[0, 1] > 0.95
    assert np.isnan(guesses[:, 1]).all() and np.isnan(constants[1])  # nothing known, nothing guessed
    np.testing.assert_allclose(inputs @ weights[:, 0] + constants[0], truth, atol=0.3)


def test_a_guess_never_learns_from_a_task_that_finished_later():
    rng = np.random.default_rng(1)
    inputs = rng.normal(size=(200, 5))
    results = (inputs @ rng.normal(size=5))[:, None]
    started = np.arange(200, dtype=float)
    finished = started.copy()
    finished[:100] = 1e9  # the first half was still running when every later task started
    guesses, _, _ = M._fit_guess(inputs, results, started, finished)
    changed = results.copy()
    changed[:100] += 50.0  # what the still-running tasks took, which no guess may know
    again, _, _ = M._fit_guess(inputs, changed, started, finished)
    np.testing.assert_array_equal(guesses, again)
    # Until 30 tasks had finished, there was nothing to guess from.
    assert np.isnan(guesses[:140]).all() and not np.isnan(guesses[140:]).any()


def with_context(tasks, seed: int = 2, conversation: bool = True):
    """Rows with a prompt and a context, the project's numbers and, with `conversation`, the conversation's."""
    rng = np.random.default_rng(seed)
    vectors = {
        "prompt": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks},
        "context": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks},
    }
    numbers = {"project_seconds": 4.5, **({"previous_seconds": 4.0} if conversation else {})}
    return [replace(r, numbers=numbers) for r in rows_for(tasks, vectors)]


def test_the_model_reads_the_conversation_only_when_the_rows_come_with_the_conversation_feature():
    tasks = synthetic(200)
    full = M.fit(tasks, with_context(tasks), {})
    plain_rows = with_context(tasks, conversation=False)
    plain = M.fit(tasks, plain_rows, {})
    assert {"previous_seconds", "prompt_context_closeness", "project_seconds"} <= set(full.meta["numbers"])
    assert M.JOINT in full.meta["text_fields"] and full.meta["guess"]["fields"] == ["prompt", "context"]
    assert full.meta["conversation"] and M.JOINT in full.prepare(with_context(tasks[:1]))[0].vectors
    # Without the feature, the context is still read, but not how close the prompt is to it, nor the two together.
    assert not plain.meta["conversation"] and plain.meta["numbers"] == ["project_seconds"]
    assert plain.meta["text_fields"] == ["prompt", "context"] and plain.prepare(plain_rows) is plain_rows


def test_a_task_llm_analysis_couldnt_read_is_forecast_by_the_model_without_it(tmp_path, monkeypatch):
    tasks = synthetic(400)
    rng = np.random.default_rng(3)
    # What LLM analysis said tells Claude's tasks apart, and says nothing about the other model's.
    rows = [
        replace(r, numbers={"llm_minutes": float(np.log(t.seconds) if t.model == "claude-opus-5-5" else rng.normal())})
        for t, r in zip(tasks, rows_for(tasks))
    ]
    tested, real = [], M.test
    monkeypatch.setattr(M, "test", lambda *args, **kwargs: tested.append(args[-1]) or real(*args, **kwargs))
    model = M.train(tasks, rows)
    # One model with LLM analysis and one without it, each with its own test, for every AI model.
    assert isinstance(model, M.WithoutLLM) and tested == [True, False]
    assert "llm_minutes" in model.meta["numbers"] and "llm_minutes" not in model.without.meta["numbers"]
    assert model.meta is model.model.meta and "llm" not in model.meta["report"]["models"]["claude-opus-5-5"]
    for name in model.meta["categories"]["model"]:
        assert model.served_without_llm(name) == model.without.served(name)
    assert model.served_without_llm("brand-new-model") == (False, "unknown_model")
    read = rows[-2]  # a task on Claude
    unread = replace(read, numbers={})
    assert model.forecast([read, unread]) == [model.model.forecast([read])[0], model.without.forecast([unread])[0]]
    model.save(tmp_path / "model")
    again = M.load(tmp_path / "model")
    assert isinstance(again, M.WithoutLLM) and again.without.meta["served"] == model.without.meta["served"]
    after, before = again.forecast([read, unread]), model.forecast([read, unread])
    assert [r["seconds"]["expected"] for r in after] == pytest.approx([r["seconds"]["expected"] for r in before])


def test_training_without_llm_analysis_tests_and_trains_one_model(monkeypatch):
    tasks = synthetic(200)
    tested, real = [], M.test
    monkeypatch.setattr(M, "test", lambda *args, **kwargs: tested.append(args[-1]) or real(*args, **kwargs))
    model = M.train(tasks, with_context(tasks), ONE_WINDOW)
    assert isinstance(model, M.Model) and tested == [True]
    assert "conversation" not in model.meta["report"]["models"]["claude-opus-5-5"]


def test_a_model_made_by_a_version_before_the_previous_release_has_to_be_trained_again(tmp_path):
    tasks = synthetic(200)
    model = M.train(tasks, rows_for(tasks), ONE_WINDOW)
    model.save(tmp_path / "model")
    meta = json.loads((tmp_path / "model" / "model.json").read_text())
    (tmp_path / "model" / "model.json").write_text(json.dumps({**meta, "format": M.PREVIOUS_FORMAT - 1}))
    with pytest.raises(M.OutdatedModel, match="refresh"):
        M.load(tmp_path / "model")
