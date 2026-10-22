from __future__ import annotations

import math
import random

import numpy as np
import pytest

from yardstick_predict import model as M
from yardstick_predict.archives import Task

START = 1_780_000_000.0


def task(i: int, model: str, prompt: str, seconds: float, tokens: int, direct: bool, usd: float | None = 0.5) -> Task:
    return Task(
        id=f"t{i}",
        archive="test",
        agent="claude-code" if model.startswith("claude") else "codex",
        agent_version="1",
        session=f"s{i // 5}",
        project="demo",
        origin="person",
        started_at=START + i * 600,
        model=model,
        effort="high",
        prompt=prompt,
        context="",
        context_tokens=20_000,
        seconds=seconds,
        input_tokens=tokens // 10,
        cache_write_tokens=0,
        cache_read_tokens=tokens - tokens // 10,
        output_tokens=0,
        usd=usd,
        calls=1 if direct else 5,
        direct=direct,
        stopped=False,
        idle=False,
    )


def synthetic(n: int, models=("claude-opus-5-5", "gpt-6.1-sol"), seed: int = 0) -> list[Task]:
    """Tasks where a longer prompt means a longer, bigger task, plus noise."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        model = models[i % len(models)]
        words = rng.randint(1, 80)
        direct = words < 10 and rng.random() < 0.8
        pace = 1.0 if model == "claude-opus-5-5" else 2.0
        seconds = math.exp(math.log(pace * (5 if direct else 30 + 4 * words)) + rng.gauss(0, 0.4))
        tokens = int(math.exp(math.log(20_000 if direct else 50_000 * (1 + words / 10)) + rng.gauss(0, 0.4)))
        tasks.append(task(i, model, "word " * words, seconds, tokens, direct, usd=tokens / 1e6))
    return tasks


def rows_for(tasks: list[Task], vectors: dict | None = None, numbers: dict | None = None) -> list[M.Features]:
    return [
        M.Features(
            t.model,
            t.effort,
            t.agent,
            t.origin,
            len(t.prompt),
            vectors={name: found[t.id] for name, found in (vectors or {}).items() if t.id in found},
            numbers=(numbers or {}).get(t.model, {}),
        )
        for t in tasks
    ]


def test_curve_math_agrees_with_sampling():
    rng = np.random.default_rng(0)
    w, c1, c2 = 0.3, (1.0, 0.5), (4.0, 1.0)
    draws = np.where(rng.random(400_000) < w, rng.normal(*c1, 400_000), rng.normal(*c2, 400_000))
    assert M.expected(w, c1, c2) == pytest.approx(np.exp(draws).mean(), rel=0.02)
    z = 3.0
    assert M.pit(z, w, c1, c2) == pytest.approx(np.mean(draws <= z), abs=0.005)
    sample_crps = np.mean(np.abs(draws[:20_000] - z)) - 0.5 * np.mean(np.abs(draws[:20_000] - draws[20_000:40_000]))
    assert M.crps(np.array(z), w, c1, c2) == pytest.approx(sample_crps, abs=0.02)


def test_the_weibull_curve_agrees_with_sampling():
    rng = np.random.default_rng(1)
    family = M.FAMILY["weibull"]
    mu, s = 2.0, 0.6
    y = np.exp(mu) * rng.weibull(1 / s, 400_000)  # a Weibull with shape 1/s and scale e^mu
    assert family.mean(mu, s) == pytest.approx(y.mean(), rel=0.01)
    z = np.log(np.median(y))
    assert M.pit(z, 0.0, (0, 1), (mu, s), family) == pytest.approx(0.5, abs=0.01)
    numeric = M.crps(
        np.array([z]), np.array([0.0]), (np.array([0.0]), np.array([1.0])), (np.array([mu]), np.array([s])), family
    )
    sample = np.log(y[:20_000])
    direct = np.mean(np.abs(sample - z)) - 0.5 * np.mean(np.abs(sample - np.log(y[20_000:40_000])))
    assert numeric[0] == pytest.approx(direct, abs=0.02)


def test_a_sharp_right_forecast_scores_better_than_a_wide_one():
    sharp = M.crps(np.array(2.0), 0.0, (0.0, 1.0), (2.0, 0.1))
    wide = M.crps(np.array(2.0), 0.0, (0.0, 1.0), (2.0, 2.0))
    wrong = M.crps(np.array(2.0), 0.0, (0.0, 1.0), (6.0, 0.1))
    assert sharp < wide < wrong


def test_training_learns_and_gives_forecasts_only_where_they_passed():
    tasks = synthetic(900)
    model = M.train(tasks, rows_for(tasks))
    report = model.meta["report"]
    assert report["tested_on"] > 0 and report["windows"] == 3
    opus = report["models"]["claude-opus-5-5"]
    assert opus["served"], opus
    assert opus["seconds"]["better_than_simple_rule"] > 0.05
    assert 0.8 <= opus["seconds"]["in_90_range"] <= 1.0
    assert 0.5 <= opus["seconds"]["expected_over_real_total"] <= 2.0
    assert model.served("claude-opus-5-5") == (True, None)
    assert model.served("some-new-model") == (False, "unknown_model")
    forecast = model.forecast(rows_for(tasks[:3]))
    assert forecast[0]["shape"] == "two_part_lognormal" and forecast[0]["seconds"]["expected"] > 0


def test_a_model_with_few_tasks_gets_no_forecasts():
    tasks = synthetic(600, models=("claude-opus-5-5",)) + [
        task(10_000 + i, "claude-sonnet-5", "word " * 5, 20, 30_000, False) for i in range(10)
    ]
    model = M.train(tasks, rows_for(tasks))
    assert model.served("claude-sonnet-5") == (False, "not_enough_data")


def test_too_few_tasks_to_train():
    tasks = synthetic(10)
    with pytest.raises(M.NotEnoughData):
        M.train(tasks, rows_for(tasks))


def test_stopped_and_waiting_tasks_are_not_learned_from():
    tasks = synthetic(50)
    tasks[0].stopped = True
    tasks[1].idle = True
    assert len(M.usable(tasks)) == 48


def test_saving_and_loading_gives_the_same_forecasts(tmp_path):
    tasks = synthetic(400)
    rng = np.random.default_rng(1)
    vectors = {"prompt": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks}}
    model = M.train(tasks, rows_for(tasks, vectors))
    assert model.meta["text_fields"] == ["prompt"]
    rows = rows_for(tasks[:20], vectors)
    before = model.forecast(rows)
    model.save(tmp_path / "model")
    after = M.load(tmp_path / "model").forecast(rows)
    assert [r["seconds"]["expected"] for r in before] == pytest.approx([r["seconds"]["expected"] for r in after])
    assert not list((tmp_path / "model").glob("*.pkl"))


def test_similar_tasks_never_look_at_the_future():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(300, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    results = rng.normal(size=(300, 4)).astype(np.float32)
    found = M.similar_tasks(vectors, results, vectors, np.ones(300, bool), earlier_only=True)
    assert np.isnan(found[: M.MIN_NEIGHBORS]).all()
    for i in (M.MIN_NEIGHBORS, 50, 299):
        sims = vectors[:i] @ vectors[i]
        top = np.argsort(-sims)[: min(M.NEIGHBORS, i)]
        np.testing.assert_allclose(found[i, :4], results[top].mean(axis=0), atol=1e-5)


def kinds(n: int = 900, seed: int = 3):
    """Tasks whose time depends on what the text says, not on its length."""
    rng = random.Random(seed)
    basis = np.random.default_rng(seed).normal(size=(4, 16)).astype(np.float32)
    tasks, prompt, context = [], {}, {}
    for i in range(n):
        kind = rng.randrange(4)
        seconds = math.exp(math.log([10, 60, 300, 1500][kind]) + rng.gauss(0, 0.3))
        t = task(i, "claude-opus-5-5", "same length prompt", seconds, int(seconds * 1000), False)
        tasks.append(t)
        prompt[t.id] = np.random.default_rng(i).normal(0, 1, 16).astype(np.float32)
        context[t.id] = basis[kind] + np.random.default_rng(i).normal(0, 0.1, 16).astype(np.float32)
    return tasks, prompt, context


def test_an_added_text_feature_helps_when_it_carries_the_signal():
    tasks, prompt, context = kinds()
    plain = M.train(tasks, rows_for(tasks, {"prompt": prompt})).meta["report"]["models"]["claude-opus-5-5"]
    added = M.train(tasks, rows_for(tasks, {"prompt": prompt, "context": context})).meta["report"]
    added = added["models"]["claude-opus-5-5"]
    assert added["seconds"]["better_than_simple_rule"] > plain["seconds"]["better_than_simple_rule"] + 0.2


def test_number_features_such_as_benchmarks_reach_the_trees():
    tasks = synthetic(300)
    numbers = {"claude-opus-5-5": {"benchmark_price_input": 0.7}, "gpt-6.1-sol": {"benchmark_price_input": 0.4}}
    model = M.train(tasks, rows_for(tasks, numbers=numbers))
    assert model.meta["numbers"] == ["benchmark_price_input"]
    assert model.matrix(rows_for(tasks[:2], numbers=numbers)).shape[1] == len(M.CATEGORIES) + 2


@pytest.mark.parametrize(
    "settings",
    [
        {"head": "expected"},
        {"head": "expected", "loss": "tweedie"},
        {"family": "weibull"},
        {"loss": "crps"},
        {"dropout": 0.1},
        {"heads": ["seconds"]},
        {"test_share": 0.2, "test_windows": 2, "validation_share": 0.2, "half_life_days": None},
    ],
)
def test_every_advanced_setting_trains_and_forecasts(settings):
    tasks = synthetic(300)
    model = M.train(tasks, rows_for(tasks), settings)
    result = model.forecast(rows_for(tasks[:5]))[0]
    heads = settings.get("heads", list(M.TARGETS))
    for target in M.TARGETS:
        if target in heads:
            assert result[target]["expected"] > 0
        else:
            assert result[target] is None
    if settings.get("head") == "expected":
        assert result["shape"] is None and result["seconds"]["distribution"] is None
        assert model.meta["report"]["models"]["claude-opus-5-5"]["seconds"]["in_90_range"] is None
    elif settings.get("family") == "weibull":
        assert result["shape"] == "two_part_weibull"


@pytest.mark.parametrize(
    "settings, words",
    [
        ({"heads": ["minutes"]}, "heads"),
        ({"head": "median"}, "head"),
        ({"family": "cauchy"}, "distribution"),
        ({"loss": "gamma"}, "loss"),
        ({"family": "weibull", "loss": "crps"}, "crps"),
        ({"dropout": 2}, "dropout"),
        ({"test_windows": 0}, "test_windows"),
    ],
)
def test_bad_settings_get_a_plain_reason(settings, words):
    with pytest.raises(ValueError) as caught:
        M.settings_for(settings)
    assert words in str(caught.value)
