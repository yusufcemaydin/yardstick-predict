from __future__ import annotations

import json
import math
import os
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb

from yardstick_predict import model as M
from yardstick_predict import pipeline
from yardstick_predict.archives import Task

START = 1_780_000_000.0
ONE_WINDOW = {"test_windows": 1}
"""Test each AI model in one window, not three, so training is quicker. For tests that don't check the test itself."""


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
    assert report["tested_on"] > 0 and report["windows"] == 6  # three for each of the two AI models
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


def test_cost_forecasts_are_only_given_where_cost_was_tested():
    tasks = synthetic(300)
    for t in tasks[int(len(tasks) * 0.65) :]:
        if t.model == "gpt-6.1-sol":
            t.usd = None  # so the test has no costs to check this model's cost forecasts on
    model = M.train(tasks, rows_for(tasks))
    assert model.served("gpt-6.1-sol") == (True, None)
    rows = rows_for(tasks[-2:])
    found = {row.model: result for row, result in zip(rows, model.forecast(rows))}
    assert found["gpt-6.1-sol"]["usd"] is None and found["gpt-6.1-sol"]["seconds"]["expected"] > 0
    assert found["claude-opus-5-5"]["usd"]["expected"] > 0


def test_a_head_with_nothing_to_learn_fails_with_a_plain_reason():
    tasks = synthetic(120)
    for t in tasks:
        t.usd = None
    with pytest.raises(M.NotEnoughData) as caught:
        M.train(tasks, rows_for(tasks), {"heads": ["usd"]})
    assert "usd" in str(caught.value)


def test_stopped_and_waiting_tasks_are_not_learned_from():
    tasks = synthetic(50)
    tasks[0].stopped = True
    tasks[1].idle = True
    assert len(M.usable(tasks)) == 48


def test_saving_and_loading_gives_the_same_forecasts(tmp_path):
    tasks = synthetic(400)
    rng = np.random.default_rng(1)
    vectors = {"prompt": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks}}
    model = M.train(tasks, rows_for(tasks, vectors), ONE_WINDOW)
    assert model.meta["text_fields"] == ["prompt"]
    rows = rows_for(tasks[:20], vectors)
    before = model.forecast(rows)
    model.save(tmp_path / "model")
    after = M.load(tmp_path / "model").forecast(rows)
    assert [r["seconds"]["expected"] for r in before] == pytest.approx([r["seconds"]["expected"] for r in after])
    assert not list((tmp_path / "model").glob("*.pkl"))


def test_a_save_that_fails_keeps_the_old_model(tmp_path, monkeypatch):
    folder = tmp_path / "predictor" / "model"
    old_tasks, new_tasks = synthetic(60), synthetic(60, seed=1)
    rows = rows_for(old_tasks[:3])
    M.train(old_tasks, rows_for(old_tasks)).save(folder)
    before = M.load(folder).forecast(rows)
    new = M.train(new_tasks, rows_for(new_tasks))
    real_rename, failed = os.rename, []

    def rename(source, target, *args, **kwargs):
        if Path(target) == folder and not failed:  # the first move into the model's place fails
            failed.append(source)
            raise OSError("The disk went away.")
        return real_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(OSError):
        new.save(folder)
    monkeypatch.undo()
    assert failed and M.load(folder).forecast(rows) == before
    assert [path.name for path in folder.parent.iterdir()] == ["model"]


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


def test_similar_tasks_for_training_had_finished_before_the_task_started(monkeypatch):
    tasks = [task(i, "claude-opus-5-5", "word", 60.0, 50_000, False) for i in range(60)]
    tasks[10].seconds = 10 * 3600  # still running when every later task here started
    vectors = {"prompt": {t.id: np.ones(8, dtype=np.float32) for t in tasks}}
    found = []
    real = M.similar_tasks

    def similar_tasks(*args, **kwargs):
        result = real(*args, **kwargs)
        if kwargs.get("earlier_only"):
            found.append(result)
        return result

    monkeypatch.setattr(M, "similar_tasks", similar_tasks)
    M.fit(tasks, rows_for(tasks, vectors), {})
    seconds = found[0][:, 0]  # the average log seconds of each task's similar tasks
    assert np.isfinite(seconds[M.MIN_NEIGHBORS + 1 :]).all()
    np.testing.assert_allclose(seconds[np.isfinite(seconds)], math.log(60.0))


def test_a_test_window_learns_only_from_tasks_that_finished_before_it_starts(monkeypatch):
    tasks = synthetic(150)
    for t in tasks[::4]:
        t.seconds = 5 * 3600  # still running when the next 30 tasks start
    learned, scored = [], []
    real_fit, real_score = M.fit, M._score_window

    def fit(train_tasks, train_rows, settings, *rest):
        learned.append(train_tasks)
        return real_fit(train_tasks, train_rows, settings, *rest)

    def score(trial, train_tasks, test_tasks, test_rows):
        scored.append((train_tasks, test_tasks))
        return real_score(trial, train_tasks, test_tasks, test_rows)

    monkeypatch.setattr(M, "fit", fit)
    monkeypatch.setattr(M, "_score_window", score)
    M.test(tasks, rows_for(tasks), {})
    # Three windows for each AI model. Two that open with the same tasks known share one trial.
    assert len(scored) == 6 and len(learned) <= len(scored)
    for simple_rule, window in scored:
        assert len({t.model for t in window}) == 1  # a window forecasts only its own AI model's tasks
        opens = min(t.started_at for t in window)
        finished = [t.id for t in tasks if t.started_at + t.seconds <= opens]
        assert [t.id for t in simple_rule] == finished  # what it learned from, of every AI model
    assert all([t.id for t in fitted] in [[t.id for t in s] for s, _ in scored] for fitted in learned)


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
    model = M.train(tasks, rows_for(tasks, numbers=numbers), ONE_WINDOW)
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
    model = M.train(tasks, rows_for(tasks), {**ONE_WINDOW, **settings})
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
    "loss, seconds",
    [
        ("gamma", {"claude-opus-5-5": 10.0, "gpt-6.1-sol": 10.0}),
        ("tweedie", {"claude-opus-5-5": 10.0, "gpt-6.1-sol": 10.0}),
        ("gamma", {"claude-opus-5-5": 10.0, "gpt-6.1-sol": 20.0}),
    ],
)
def test_a_target_that_never_changes_still_gives_strict_json(loss, seconds):
    tasks = synthetic(300)
    for t in tasks:
        t.seconds = seconds[t.model]
    model = M.train(tasks, rows_for(tasks), {"head": "expected", "loss": loss})
    json.dumps(model.meta, allow_nan=False)  # fails on NaN and infinity, as strict JSON readers do
    for result in model.meta["report"]["models"].values():
        # The simple rule is nearly exact here, so the share can land a hair either side of 0. It must be a number.
        assert math.isfinite(result["seconds"]["better_than_simple_rule"])
    assert all(math.isfinite(r["seconds"]["expected"]) for r in model.forecast(rows_for(tasks[:4])))


def test_a_short_half_life_trains_with_an_ai_model_used_years_ago():
    tasks = synthetic(400)
    for task in tasks[:80:2]:
        task.model, task.started_at = "claude-old", task.started_at - 3 * 365 * 86400
    tasks.sort(key=lambda t: t.started_at)
    model = M.train(tasks, rows_for(tasks), {**ONE_WINDOW, "half_life_days": 0.5})  # a task counts half after 12 hours
    assert model.forecast(rows_for(tasks[-1:]))[0]["seconds"]["expected"] > 0


def test_a_head_is_only_judged_on_enough_known_values():
    tasks = synthetic(300)
    for task in tasks[200:]:
        task.usd = None  # no cost known for the newest tasks, as on a model with no listed price
    model = M.train(tasks, rows_for(tasks))
    opus = model.meta["report"]["models"]["claude-opus-5-5"]
    assert opus["served"] and opus["heads"] == ["seconds", "tokens"]
    forecast = model.forecast(rows_for(tasks[:1]))[0]
    assert forecast["seconds"]["expected"] > 0 and forecast["usd"] is None


def test_an_ai_model_used_only_long_ago_is_still_in_the_report():
    tasks = synthetic(300)
    for task in tasks[:60:2]:
        task.model = "claude-old"
    model = M.train(tasks, rows_for(tasks))
    old = model.meta["report"]["models"]["claude-old"]
    assert old["tasks_tested"] == 0 and not old["served"] and old["reason"] == "not_enough_data"
    assert model.served("claude-old") == (False, "not_enough_data")


@pytest.mark.parametrize(
    "settings, words",
    [
        ({"heads": ["minutes"]}, "heads"),
        ({"head": "median"}, "head"),
        ({"family": "cauchy"}, "distribution"),
        ({"loss": "gamma"}, "loss"),
        ({"family": "weibull", "loss": "crps"}, "crps"),
        ({"head": "expected", "family": "weibull"}, "shape"),
        ({"shape": "weibull"}, "Unknown model setting 'shape'"),
        ({"dropout": 2}, "dropout"),
        ({"test_windows": 0}, "test_windows"),
    ],
)
def test_bad_settings_get_a_plain_reason(settings, words):
    with pytest.raises(ValueError) as caught:
        M.settings_for(settings)
    assert words in str(caught.value)


def test_xgboost_runs_on_one_thread_on_small_data_and_more_threads_change_nothing(monkeypatch):
    tasks = synthetic(200)
    rows = rows_for(tasks)
    one = M.fit(tasks, rows, {})
    assert one.boosters
    for booster in one.boosters.values():
        assert json.loads(booster.save_config())["learner"]["generic_param"]["nthread"] == "1"
    monkeypatch.setattr(M, "THREADS_FROM", 0)  # every training runs on several threads
    many = M.fit(tasks, rows, {})
    alone, together = ([r["seconds"]["expected"] for r in found.forecast(rows)] for found in (one, many))
    if os.name == "nt":
        # On Windows, XGBoost's threads add up their sums in another order, which moves a forecast a little.
        assert alone == pytest.approx(together, rel=0.05)
    else:
        assert alone == together


def test_a_weibull_part_starts_from_the_curve_its_tasks_follow():
    family = M.FAMILY["weibull"]
    z = 4.0 + 0.7 * np.log(np.random.default_rng(2).exponential(size=100_000))  # center 4, width 0.7
    start, _ = M._pooled_curves(["claude-opus-5-5"] * len(z), z, np.ones(len(z)), family)
    center, log_width = start["claude-opus-5-5"]
    assert center == pytest.approx(4.0, abs=0.01)
    assert math.exp(log_width) == pytest.approx(0.7, rel=0.01)
    assert family.mean(center, math.exp(log_width)) == pytest.approx(np.exp(z).mean(), rel=0.02)


def test_the_newest_tasks_judge_trees_against_starting_points_fitted_without_them(monkeypatch):
    tasks = synthetic(200)
    held = max(int(len(tasks) * M.DEFAULTS["validation_share"]), 10)
    weight = M.recency_weights(tasks, M.DEFAULTS["half_life_days"])
    seen = {}
    real_direct, real_part = M._grow_direct, M._grow_part

    def grow_direct(model, *args):
        seen["direct"] = model.meta["direct"]["start"]["*"]
        return real_direct(model, *args)

    def grow_part(model, key, *args):
        seen[key] = model.meta["parts"][key]["start"]["*"]
        return real_part(model, key, *args)

    monkeypatch.setattr(M, "_grow_direct", grow_direct)
    monkeypatch.setattr(M, "_grow_part", grow_part)
    model = M.fit(tasks, rows_for(tasks), {})
    direct = np.array([t.direct for t in tasks])
    z = np.log([t.seconds for t in tasks])
    older = np.arange(len(tasks)) < len(tasks) - held
    assert seen["direct"] == pytest.approx(np.average(direct[older], weights=weight[older]))
    work = older & ~direct
    assert seen["seconds.work"][0] == pytest.approx(np.average(z[work], weights=weight[work]))
    # The final model starts from every task, the newest too.
    assert model.meta["parts"]["seconds.work"]["start"]["*"][0] == pytest.approx(
        np.average(z[~direct], weights=weight[~direct])
    )


def test_a_model_whose_weight_sits_on_two_recent_tasks_counts_as_about_two_tasks():
    rng = np.random.default_rng(0)
    # One model ran 200 tasks 300 days ago and 2 today. Another ran 500 tasks this month.
    age = np.r_[np.full(200, 300.0), [0.0, 0.1], rng.random(500) * 30]
    z = np.r_[rng.normal(5, 1, 200), [6.0, 6.02], rng.normal(4, 1, 500)]
    models = ["claude-old"] * 202 + ["claude-new"] * 500
    weight = 0.5 ** (age / 30)
    mean, variance, n = M._moments(z[:202], weight[:202])
    assert 2 <= n < 3 and M._moments(z[:5], np.ones(5))[2] == pytest.approx(5)
    start, _ = M._pooled_curves(models, z, weight)
    every_mean, every_variance, _ = M._moments(z, weight)
    # As if it had 10 more tasks like all models, so its spread includes the gap between the two centers.
    center = (n * mean + 10 * every_mean) / (n + 10)
    spread = (n * (variance + (mean - center) ** 2) + 10 * (every_variance + (every_mean - center) ** 2)) / (n + 10)
    assert start["claude-old"] == pytest.approx([center, 0.5 * math.log(spread)])
    assert math.exp(start["claude-old"][1]) > 1.0  # not the narrow curve of its two recent tasks


def test_the_crps_width_step_uses_the_curvature_crps_really_has():
    s = 1.3
    data = xgb.DMatrix(np.zeros((1, 1)), label=np.zeros((1, 2)))
    _, hess = M._curve_objective(M.LogNormal(), "crps")(np.array([[0.0, math.log(s)]]), data)

    def expected_crps(log_width: float) -> float:  # a normal curve's CRPS, averaged over results from N(0, s²)
        width = math.exp(log_width)
        return math.sqrt(2 / math.pi) * math.sqrt(width**2 + s**2) - width / math.sqrt(math.pi)

    h, middle = 1e-4, math.log(s)
    curvature = (expected_crps(middle + h) - 2 * expected_crps(middle) + expected_crps(middle - h)) / h**2
    assert hess[0, 1] == pytest.approx(curvature, rel=1e-3)


def test_the_width_factor_is_picked_by_crps_when_the_trees_learn_by_crps():
    rng = np.random.default_rng(3)
    z = np.where(rng.random(400) < 0.9, rng.normal(0, 1, 400), rng.normal(0, 6, 400))  # some results far off
    picked = {}
    for loss in ("log_score", "crps"):
        curve = {"start": {"*": [0.0, 0.0]}}
        meta = {
            "settings": M.settings_for({"loss": loss}),
            "direct": {"start": {"*": 0.0}},
            "parts": {"seconds.direct": dict(curve), "seconds.work": dict(curve)},
            "width_scale": {},
        }
        model = M.Model(meta, {})
        n = len(z)
        M._fit_width_scale(
            model, "seconds", np.zeros((n, 1)), ["claude-opus-5-5"] * n, z, np.ones(n, bool), np.arange(n)
        )
        picked[loss] = model.meta["width_scale"]["seconds"]["claude-opus-5-5"]
    grid = np.geomspace(0.5, 3.0, 46)
    by_crps = grid[np.argmin([M.crps(z, 0.5, (0.0, g), (0.0, g)).mean() for g in grid])]
    assert picked["crps"] == pytest.approx(by_crps)
    assert picked["log_score"] > picked["crps"]  # the log score widens more for the few far-off results


def test_the_simple_rule_and_the_starting_curves_are_never_narrower_than_a_forecast():
    tasks = [task(i, "claude-opus-5-5", "same", 60.0, 50_000, False) for i in range(40)]
    assert M.simple_rule(tasks, "seconds")["claude-opus-5-5"]["work"][1] == M.WIDTH_RANGE[0]
    start, _ = M._pooled_curves([t.model for t in tasks], np.full(40, math.log(60.0)), np.ones(40))
    assert math.exp(start["claude-opus-5-5"][1]) == pytest.approx(M.WIDTH_RANGE[0])


def test_an_ai_model_needs_60_tasks_of_its_own_to_be_tested_whatever_the_share():
    assert M.min_tasks_to_test({}) == M.min_tasks_to_test({"test_share": 0.5}) == 60
    tasks = synthetic(60, models=("claude-opus-5-5",))
    report = M.test(tasks, rows_for(tasks), {})
    # 30% of 60 is 18, but a model is tested on at least 30 of its tasks, with 30 before them.
    assert report["min_tasks_to_test"] == 60 and report["tested_on"] == 30
    assert report["models"]["claude-opus-5-5"]["tasks"] == 60
    report = M.test(tasks[:59], rows_for(tasks[:59]), {})
    assert (report["min_tasks_to_test"], report["windows"], report["tested_on"]) == (60, 0, 0)


def test_each_ai_model_is_tested_on_its_own_newest_tasks_however_many_came_after_on_others():
    # GPT ran 100 tasks first. Then Claude ran 400, more than the newest 30% of all the tasks.
    tasks = synthetic(100, models=("gpt-6.1-sol",)) + synthetic(400, models=("claude-opus-5-5",), seed=1)
    for i, t in enumerate(tasks):
        t.id, t.started_at = f"t{i}", START + i * 600
    report = M.test(tasks, rows_for(tasks), {})
    gpt, claude = report["models"]["gpt-6.1-sol"], report["models"]["claude-opus-5-5"]
    assert gpt["tasks_tested"] == 30 and claude["tasks_tested"] == 120  # 30% of each model's own tasks
    assert report["windows"] == 6 and report["tested_on"] == 150
    assert gpt["reason"] != "not_enough_data"


def test_tested_on_counts_only_the_windows_that_ran():
    tasks = synthetic(100, models=("claude-opus-5-5",))  # windows open at tasks 70, 80 and 90
    for i, t in enumerate(tasks[:70]):
        t.seconds = 45_000 - 600 * i  # each finishes after task 70 started, before task 80 did
    report = M.test(tasks, rows_for(tasks), {})
    assert (report["windows"], report["trained_on"], report["tested_on"]) == (2, 80, 20)


def test_a_test_window_reuses_the_similar_task_numbers_it_would_work_out_itself(monkeypatch):
    tasks = synthetic(120)
    for t in tasks[::5]:
        t.seconds = 3 * 3600  # still running while the next tasks start
    rng = np.random.default_rng(4)
    rows = rows_for(tasks, {"prompt": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks}})
    every = M._similar_features(tasks, rows)["prompt"]
    finished = M.finish_times(tasks)
    known = [i for i in range(96) if finished[i] <= tasks[96].started_at]  # what the window opening at 96 learns from
    alone = M._similar_features([tasks[i] for i in known], [rows[i] for i in known])["prompt"]
    np.testing.assert_allclose(every[known], alone, rtol=1e-5, atol=1e-6)  # the same tasks, up to rounding
    calls, real = [], M._similar_features
    monkeypatch.setattr(M, "_similar_features", lambda *args: calls.append(args) or real(*args))
    assert M.test(tasks, rows, {})["windows"] == 6  # three for each of the two AI models
    assert len(calls) == 1  # worked out once for all six windows


def test_the_weibull_crps_comes_out_the_same_worked_out_in_blocks():
    rng = np.random.default_rng(5)
    n = 2500  # more than one block
    z, w = rng.normal(3, 1, n), rng.random(n)
    c1, c2 = (rng.normal(1, 0.3, n), rng.uniform(0.1, 1, n)), (rng.normal(4, 0.5, n), rng.uniform(0.3, 2, n))
    family = M.FAMILY["weibull"]
    together = M.crps(z, w, c1, c2, family)
    picks = range(0, n, 97)
    alone = [M.crps(z[i : i + 1], w[i : i + 1], (c1[0][i], c1[1][i]), (c2[0][i], c2[1][i]), family)[0] for i in picks]
    np.testing.assert_allclose(together[list(picks)], alone)


def test_a_model_saved_in_a_newer_format_gives_a_plain_error(tmp_path):
    M.Model({"format": M.FORMAT + 1, "text_fields": []}, {}).save(tmp_path / "model")
    with pytest.raises(M.NewerFormat) as caught:
        M.load(tmp_path / "model")
    assert "newer version" in str(caught.value) and "uv tool upgrade yardstick-predict" in str(caught.value)


def test_a_model_saved_by_version_0_2_still_loads_and_forecasts(tmp_path):
    tasks = synthetic(200)
    model = M.fit(tasks, rows_for(tasks), {})
    assert "tree_settings" not in model.meta and "direct_objective" not in model.meta
    model.save(tmp_path / "model")
    meta = json.loads((tmp_path / "model" / "model.json").read_text())
    for spec in [meta["direct"], *meta["parts"].values()]:
        del spec["effort_start"]  # 0.2 kept one starting point per AI model
    del meta["effort_counts"]
    meta["served"] = {"claude-opus-5-5": {"served": True, "reason": None, "heads": list(M.TARGETS)}}
    (tmp_path / "model" / "model.json").write_text(json.dumps(meta))
    old = M.load(tmp_path / "model")
    assert all(r["seconds"]["expected"] > 0 for r in old.forecast(rows_for(tasks[-3:])))
    assert old.served("claude-opus-5-5", "max") == (True, None)  # 0.2 didn't count effort levels


def test_each_effort_level_starts_from_its_own_curve_pulled_toward_its_model():
    rng = np.random.default_rng(6)
    levels = ["high"] * 160 + ["max"] * 80 + ["low"] * 3
    rng.shuffle(levels)
    took = {"high": 60.0, "max": 600.0, "low": 6.0}
    tasks = []
    for i, level in enumerate(levels):
        tasks.append(task(i, "claude-opus-5-5", "same", took[level] * math.exp(rng.normal(0, 0.3)), 50_000, False))
        tasks[-1].effort = level
    model = M.fit(tasks, rows_for(tasks), {"heads": ["seconds"]})
    spec = model.meta["parts"]["seconds.work"]
    own, by_level = spec["start"]["claude-opus-5-5"][0], spec["effort_start"]["claude-opus-5-5"]
    assert by_level["max"][0] == pytest.approx(math.log(600), abs=0.3)
    assert by_level["high"][0] == pytest.approx(math.log(60), abs=0.3)
    # Three tasks count for little, so their level starts nearer the model's own curve than their own.
    assert abs(by_level["low"][0] - own) < abs(math.log(6) - own) / 2
    high, top = model.forecast(
        [M.Features("claude-opus-5-5", level, "claude-code", "person", 4) for level in ("high", "max")]
    )
    assert top["seconds"]["expected"] > 5 * high["seconds"]["expected"]


def test_the_default_effort_level_is_the_one_used_lately():
    tasks = [task(i, "claude-opus-5-5", "word", 60.0, 50_000, False) for i in range(100)]  # all at high
    for i in range(20):
        tasks.append(task(200 + i, "claude-opus-5-5", "word", 60.0, 50_000, False))
        tasks[-1].effort, tasks[-1].started_at = "max", tasks[-1].started_at + 120 * 86400
    assert M._defaults(tasks, M.recency_weights(tasks, 30.0))["claude-opus-5-5"]["effort"] == "max"
    assert M._defaults(tasks)["claude-opus-5-5"]["effort"] == "high"  # every task counted alike


def test_effort_counts_list_each_models_levels_most_used_first():
    tasks = synthetic(10)  # the two models take turns, all at high
    for t in tasks[:3]:
        t.effort = "max"
    assert M._effort_counts(tasks) == {
        "claude-opus-5-5": [["high", 3], ["max", 2]],
        "gpt-6.1-sol": [["high", 4], ["max", 1]],
    }


def test_an_effort_level_the_model_never_ran_at_gets_no_forecast():
    meta = {
        "categories": {"model": ["claude-opus-5-5"]},
        "served": {"claude-opus-5-5": {"served": True, "reason": None, "heads": ["seconds"]}},
        "effort_counts": {"claude-opus-5-5": [["high", 80], ["max", 30]]},
    }
    model = M.Model(meta, {})
    assert model.served("claude-opus-5-5") == model.served("claude-opus-5-5", "max") == (True, None)
    assert model.served("claude-opus-5-5", "low") == (False, "not_enough_data")


def test_dropout_really_drops_trees(monkeypatch):
    monkeypatch.setattr(M, "MAX_DROPOUT_ROUNDS", 60)
    tasks = synthetic(200)
    rows = rows_for(tasks)
    plain, dropped = M.fit(tasks, rows, {}), M.fit(tasks, rows, {"dropout": 0.3})
    assert dropped.boosters
    for key, booster in dropped.boosters.items():
        config = json.loads(booster.save_config())["learner"]["gradient_booster"]
        assert float(config["dart_train_param"]["rate_drop"]) == pytest.approx(0.3)
        spec = dropped.meta["direct"] if key == "direct" else dropped.meta["parts"][key]
        assert booster.num_boosted_rounds() == spec["rounds"] <= M.MAX_DROPOUT_ROUNDS  # a forecast uses every tree
    assert [r["seconds"]["expected"] for r in plain.forecast(rows)] != [
        r["seconds"]["expected"] for r in dropped.forecast(rows)
    ]


def test_with_dropout_the_trees_kept_are_the_model_early_stopping_picked():
    rng = np.random.default_rng(7)
    n = 400
    matrix = np.column_stack([np.zeros((n, len(M.CATEGORIES))), rng.normal(size=(n, 2))])
    label = matrix[:, -2] + rng.normal(0, 3, n)  # a weak signal in much noise, so trees soon stop helping
    model, margin, grow, check = M.Model({}, {}), np.zeros(n), np.arange(300), np.arange(300, n)

    def objective(predt, data):
        return M._weighted(predt - data.get_label(), np.ones_like(predt), data)

    def metric(predt, data):
        return "score", float(np.mean((predt - data.get_label()) ** 2))

    params = M._params({"dropout": 0.3})
    start_score = float(np.mean(label[check] ** 2))
    booster, rounds = M._grow(
        model, "x", params, matrix, margin, label, np.ones(n), grow, check, objective, metric, start_score
    )
    assert 0 < rounds < M.MAX_DROPOUT_ROUNDS and booster.num_boosted_rounds() == rounds
    data = model.dmatrix(matrix[grow], margin[grow], label=label[grow], weight=np.ones(len(grow)))
    again = xgb.train({**params, "disable_default_eval_metric": 1, "nthread": 1}, data, rounds, obj=objective)
    np.testing.assert_array_equal(booster.predict(data, output_margin=True), again.predict(data, output_margin=True))


def test_a_forecast_compares_each_different_vector_with_past_tasks_once(monkeypatch):
    tasks = synthetic(200)
    rng = np.random.default_rng(8)
    model = M.fit(tasks, rows_for(tasks, {"prompt": {t.id: rng.normal(size=8).astype(np.float32) for t in tasks}}), {})
    a, b = rng.normal(size=8).astype(np.float32), rng.normal(size=8).astype(np.float32)
    rows = [
        M.Features(name, "high", "claude-code", "person", 4, vectors={"prompt": vector})
        for vector in (a, b, a)
        for name in ("claude-opus-5-5", "gpt-6.1-sol")
    ]
    rows.append(M.Features("claude-opus-5-5", "high", "claude-code", "person", 4))  # no prompt vector
    asked, real = [], M.similar_tasks

    def similar_tasks(pool, results, queries, *args, **kwargs):
        asked.append(len(queries))
        return real(pool, results, queries, *args, **kwargs)

    monkeypatch.setattr(M, "similar_tasks", similar_tasks)
    found = model.matrix(rows)
    assert asked == [3]  # a, b and the row without one
    pool, results, _ = model.pools["prompt"]
    queries, has = M._unit([r.vectors.get("prompt") for r in rows], 8)
    # The prompt's block comes after the categories, the prompt's length and the number features.
    start = len(M.CATEGORIES) + 1 + len(model.meta["numbers"])
    expected = real(pool, results, queries, has, earlier_only=False)
    np.testing.assert_allclose(found[:, start : start + 5], expected, rtol=1e-6)


def test_a_free_ai_model_keeps_its_time_and_token_forecasts():
    paid = task(0, "claude-opus-5-5", "word", 60.0, 50_000, False, usd=0.25)
    free = task(1, "gpt-6.1-sol", "word", 60.0, 50_000, False, usd=0.0)
    values = M._known_values([paid, free], "usd")
    assert values[0] == 0.25 and np.isnan(values[1])  # a $0 cost can't go on a log scale, so it isn't learned
    tasks = synthetic(300)
    for t in tasks:
        if t.model == "gpt-6.1-sol":
            t.usd = 0.0  # as on a free AI model
    model = M.train(tasks, rows_for(tasks))
    assert model.served("gpt-6.1-sol") == (True, None)
    assert model.meta["report"]["models"]["gpt-6.1-sol"]["heads"] == ["seconds", "tokens"]
    found = model.forecast(rows_for(tasks[-2:]))
    assert all(f["seconds"]["expected"] > 0 for f in found)
    assert [f["usd"] is None for f in found] == [t.model == "gpt-6.1-sol" for t in tasks[-2:]]


def test_a_load_that_read_the_old_settings_gets_the_old_trees_while_a_save_waits(tmp_path, monkeypatch):
    def tiny(generation: str, label: float) -> M.Model:
        booster = xgb.train(
            {"nthread": 1, "base_score": 0.0}, xgb.DMatrix(np.zeros((4, 1)), label=np.full(4, label)), 1
        )
        return M.Model({"format": M.FORMAT, "generation": generation, "text_fields": []}, {"seconds": booster})

    folder = tmp_path / "model"
    predictor = SimpleNamespace(dir=tmp_path, model_dir=folder, config={})
    old, new = tiny("old", 1.0), tiny("new", 50.0)
    old.save(folder)
    real_read, saving = Path.read_text, []

    def save_as_a_refresh_does():
        with pipeline._model_lock(predictor):
            new.save(folder)

    def read_text(self, *args, **kwargs):
        text = real_read(self, *args, **kwargs)
        if self == folder / "model.json" and not saving:
            saving.append(threading.Thread(target=save_as_a_refresh_does))
            saving[0].start()
            saving[0].join(timeout=0.5)  # time enough for a save that doesn't wait to swap the new model in
            assert saving[0].is_alive(), "the save didn't wait for the load to finish"
        return text

    monkeypatch.setattr(Path, "read_text", read_text)
    loaded = pipeline.load_model(predictor)
    monkeypatch.undo()
    saving[0].join(timeout=10)
    point = xgb.DMatrix(np.zeros((1, 1)))
    assert loaded.meta["generation"] == "old"
    assert loaded.boosters["seconds"].predict(point) == pytest.approx(old.boosters["seconds"].predict(point))
    assert not saving[0].is_alive() and pipeline.load_model(predictor).meta["generation"] == "new"
