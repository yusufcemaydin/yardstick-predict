"""The model's newer inputs: close matches, the first guess, and the versions with and without the conversation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_model import rows_for, synthetic

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


def with_context(tasks, seed: int = 2):
    rng = np.random.default_rng(seed)
    vectors = {
        "prompt": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks},
        "context": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks},
    }
    rows = rows_for(tasks, vectors)
    return [replace(r, numbers={"previous_seconds": 4.0, "project_seconds": 4.5}) for r in rows]


def test_each_version_reads_the_conversation_or_leaves_it_out():
    tasks = synthetic(200)
    rows = with_context(tasks)
    full = M.fit(tasks, rows, {}, conversation=True)
    plain = M.fit(tasks, rows, {}, conversation=False)
    assert {"previous_seconds", "prompt_context_closeness", "project_seconds"} <= set(full.meta["numbers"])
    assert M.JOINT in full.meta["text_fields"] and full.meta["guess"]["fields"] == ["prompt", "context"]
    assert "previous_seconds" not in plain.meta["numbers"] and "project_seconds" in plain.meta["numbers"]
    assert M.JOINT not in plain.meta["text_fields"]


def test_the_choice_of_versions_saves_and_forecasts_each_ai_model_with_its_own(tmp_path):
    tasks = synthetic(300)
    rows = with_context(tasks)
    versions = {
        M.VERSIONS[True]: M.fit(tasks, rows, {}, conversation=True),
        M.VERSIONS[False]: M.fit(tasks, rows, {}, conversation=False),
    }
    for version in versions.values():
        version.meta.update(report={"models": {}}, served={}, trained_at=0, tasks=300)
    uses = {"claude-opus-5-5": M.VERSIONS[True], "gpt-6.1-sol": M.VERSIONS[False]}
    choice = M.Choice.made(versions, uses)
    asked = rows[:6]
    found = choice.forecast(asked)
    for row, result in zip(asked, found):
        own = versions[uses[row.model]].forecast([row])[0]
        assert result["seconds"]["expected"] == pytest.approx(own["seconds"]["expected"])
    choice.save(tmp_path / "model")
    again = M.load(tmp_path / "model")
    assert isinstance(again, M.Choice) and sorted(again.versions) == sorted(versions)
    assert [r["seconds"]["expected"] for r in again.forecast(asked)] == pytest.approx(
        [r["seconds"]["expected"] for r in found]
    )


def report(served: bool, score: float) -> dict:
    return {"served": served, "heads": ["seconds"], "seconds": {"score": score}, "tasks_tested": 50}


def test_each_ai_model_gets_the_version_that_did_better_on_the_test():
    reports = {
        (True, False): {
            "models": {"a": report(True, 0.8), "b": report(True, 1.0), "c": report(False, 0.5), "d": report(True, 1.0)}
        },
        (False, False): {
            "models": {"a": report(True, 1.0), "b": report(True, 0.8), "c": report(True, 0.9), "d": report(True, 1.0)}
        },
    }
    plain, full = (False, False), (True, False)
    assert M._choose(reports) == {"a": full, "b": plain, "c": plain, "d": plain}


def test_an_ai_model_uses_llm_analysis_only_when_it_did_better_with_it():
    reports = {
        (True, True): {"models": {"a": report(True, 0.7), "b": report(True, 1.0), "c": report(True, 0.6)}},
        (True, False): {"models": {"a": report(True, 0.8), "b": report(True, 1.0), "c": report(True, 0.9)}},
        (False, True): {"models": {"a": report(True, 0.9), "b": report(True, 1.0), "c": report(False, 0.5)}},
        (False, False): {"models": {"a": report(True, 1.0), "b": report(True, 1.0), "c": report(True, 1.0)}},
    }
    assert M._choose(reports) == {"a": (True, True), "b": (False, False), "c": (True, True)}


def test_a_task_llm_analysis_couldnt_read_is_forecast_by_the_version_without_it(tmp_path):
    tasks = synthetic(400)
    rng = np.random.default_rng(3)
    # What LLM analysis said tells Claude's tasks apart, and says nothing about the other model's.
    rows = [
        replace(r, numbers={"llm_minutes": float(np.log(t.seconds) if t.model == "claude-opus-5-5" else rng.normal())})
        for t, r in zip(tasks, rows_for(tasks))
    ]
    model = M.train(tasks, rows)
    assert isinstance(model, M.Choice) and model.meta["report"]["models"]["claude-opus-5-5"]["llm"]
    fallback = model.meta["without_llm"]["claude-opus-5-5"]
    assert fallback["version"] in model.versions and not fallback["version"].endswith("-llm")
    assert model.served_without_llm("claude-opus-5-5") == (fallback["served"], fallback["reason"])
    for name in set(model.meta["uses"]) - set(model.meta["without_llm"]):
        assert model.served_without_llm(name) == model.served(name)
    read = rows[-2]  # a task on Claude
    unread = replace(read, numbers={})
    uses = model.meta["uses"]["claude-opus-5-5"]
    assert model.forecast([read, unread]) == [
        model.versions[uses].forecast([read])[0],
        model.versions[fallback["version"]].forecast([unread])[0],
    ]
    model.save(tmp_path / "model")
    again = M.load(tmp_path / "model")
    assert again.meta["without_llm"] == model.meta["without_llm"]
    after, before = again.forecast([unread])[0], model.forecast([unread])[0]
    assert after["seconds"]["expected"] == pytest.approx(before["seconds"]["expected"])


def test_a_model_made_by_an_older_version_has_to_be_trained_again(tmp_path):
    tasks = synthetic(200)
    model = M.train(tasks, rows_for(tasks))
    model.save(tmp_path / "model")
    meta = json.loads((tmp_path / "model" / "model.json").read_text())
    (tmp_path / "model" / "model.json").write_text(json.dumps({**meta, "format": 2}))
    with pytest.raises(M.OutdatedModel, match="refresh"):
        M.load(tmp_path / "model")
