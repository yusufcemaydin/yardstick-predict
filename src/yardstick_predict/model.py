"""The forecasting model: XGBoost trees that learn a curve for each task.

For a task on an AI model, it forecasts seconds, tokens and dollars: the output heads. By default each
head is a curve with two parts, because an agent either answers directly, without tools, or works with
tools:

1. the chance of a direct answer,
2. a log-normal curve for a direct answer,
3. a log-normal curve for work with tools.

A log-normal curve is a bell curve on a log scale. For each part, the trees learn two numbers per task,
the curve's center and its width, by maximum likelihood: the curves under which what really happened
is most likely.

The features are the AI model, the thinking level, the agent, who started the task and the prompt's
length, plus the chosen features. Each embedded feature, such as the prompt, becomes what the most
similar past tasks really took. That works better than feeding the trees the embedding itself, which
is many weak numbers the trees overfit.

The trees start from the simple rule: each AI model's own curve, the spread of its past tasks. A model
with few tasks has its curve pulled toward the curve of all models. Trees are added only while they beat
that start on the newest tasks, so with little data the model stays close to the simple rule, and with a
lot of data it gets sharper.

Training holds back the newest tasks as a test. Forecasts for an AI model are only given when they did
at least as well as the simple rule on that test, their ranges held and their expected values added up.

Advanced settings change the heads, the curve's shape, the loss, dropout and how the data is split.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
except Exception as error:  # On a Mac without Homebrew's OpenMP, XGBoost can't load in a running program.
    if "libomp" not in str(error):
        raise
    raise ImportError(
        "XGBoost can't find the OpenMP library on this Mac. The yardstick-predict command handles this by "
        "itself. From your own Python code, start Python with DYLD_FALLBACK_LIBRARY_PATH set to the .dylibs "
        "folder inside scikit-learn, or install OpenMP with: brew install libomp"
    ) from None

from .archives import Task

TARGETS = ("seconds", "tokens", "usd")
PARTS = ("direct", "work")
CATEGORIES = ("model", "effort", "agent", "origin")

DEFAULTS = {
    "type": "xgboost",
    "heads": list(TARGETS),
    "head": "distribution",
    "family": "lognormal",
    "loss": "log_score",
    "dropout": 0.0,
    "test_share": 0.3,
    "test_windows": 3,
    "validation_share": 0.15,
    "half_life_days": 30.0,
}
"""The settings a predictor uses unless it's told otherwise. Everything here is an advanced setting."""
LOSSES = {"distribution": ("log_score", "crps"), "expected": ("gamma", "tweedie")}
FAMILIES = ("lognormal", "weibull")

NEIGHBORS = 100
"""How many of the most similar past tasks the similar-task features average over."""
MIN_NEIGHBORS = 10
NEIGHBOR_POOL = 20_000
"""The newest tasks a forecast compares its embedded features with."""
MIN_TASKS = 30
"""Fewer finished tasks than this, and there is nothing to train on."""
MIN_PART_TASKS = 30
"""A part with fewer tasks than this keeps its starting curves and grows no trees."""
MIN_TEST_TASKS = 30
"""A model needs this many earlier tasks and this many test tasks before its forecasts are given."""
POOLING = 10
"""A model's starting curve is pulled toward the curve of all models as if it had this many tasks."""
TOLERANCE = 0.02
"""Forecasts may score up to 2% worse than the simple rule on the test and still be given."""
MIN_RANGE_HELD = 0.75
"""The 90% range must hold for at least 75% of test tasks."""
TOTALS_RANGE = (0.5, 2.0)
"""The expected values must add up to between half and twice the real total."""
WIDTH_RANGE = (0.05, 3.0)
MAX_ROUNDS = 2000
MAX_DROPOUT_ROUNDS = 300
"""With dropout, trees learn slowly and rarely stop early, so dropout gets fewer rounds to stay quick."""
PATIENCE = 50
TWEEDIE_POWER = 1.5
TREES = {
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20.0,
    "lambda": 1.0,
    "seed": 0,
}


class NotEnoughData(Exception):
    pass


def settings_for(given: dict | None) -> dict:
    """Fill in the defaults and check every setting. Raises ValueError with a plain reason."""
    given = dict(given or {})
    settings = {**DEFAULTS, **given}
    if settings["head"] == "expected" and "loss" not in given:
        settings["loss"] = "gamma"
    heads = settings["heads"]
    if isinstance(heads, str):
        heads = [h.strip() for h in heads.split(",") if h.strip()]
    if not heads or any(h not in TARGETS for h in heads):
        raise ValueError(f"Choose heads from: {', '.join(TARGETS)}.")
    settings["heads"] = [t for t in TARGETS if t in heads]
    if settings["head"] not in LOSSES:
        raise ValueError(f"The head is one of: {', '.join(LOSSES)}.")
    if settings["family"] not in FAMILIES:
        raise ValueError(f"The distribution is one of: {', '.join(FAMILIES)}.")
    if settings["loss"] not in LOSSES[settings["head"]]:
        raise ValueError(f"With {settings['head']} heads, the loss is one of: {', '.join(LOSSES[settings['head']])}.")
    if settings["loss"] == "crps" and settings["family"] != "lognormal":
        raise ValueError("The crps loss works with the lognormal distribution only.")
    checks = (
        ("dropout", 0.0, 0.9, "between 0 and 0.9"),
        ("test_share", 0.05, 0.5, "between 0.05 and 0.5"),
        ("validation_share", 0.05, 0.5, "between 0.05 and 0.5"),
    )
    for name, low, high, words in checks:
        if not isinstance(settings[name], (int, float)) or not low <= settings[name] <= high:
            raise ValueError(f"{name} must be {words}.")
    if not isinstance(settings["test_windows"], int) or not 1 <= settings["test_windows"] <= 10:
        raise ValueError("test_windows must be a whole number from 1 to 10.")
    half_life = settings["half_life_days"]
    if half_life is not None and (not isinstance(half_life, (int, float)) or half_life <= 0):
        raise ValueError("half_life_days must be a positive number, or null to weigh every task the same.")
    if settings["type"] != "xgboost":
        raise ValueError("This module trains the xgboost model type.")
    return settings


@dataclass
class Features:
    """What the model knows about a task before it runs."""

    model: str
    effort: str
    agent: str
    origin: str
    prompt_chars: int
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    """One embedding for each embedded feature the task has, such as its prompt."""
    numbers: dict[str, float] = field(default_factory=dict)
    """The task's number features, such as its model's benchmarks."""


def usable(tasks: list[Task]) -> list[Task]:
    """Tasks the model can learn from, oldest first. Stopped tasks and tasks left waiting are left out."""
    keep = [t for t in tasks if not t.stopped and not t.idle and t.seconds > 0 and t.tokens > 0 and t.model]
    return sorted(keep, key=lambda t: (t.started_at, t.id))


def target_value(task: Task, target: str) -> float | None:
    value = {"seconds": task.seconds, "tokens": float(task.tokens), "usd": task.usd}[target]
    return value if value is not None and value > 0 else None


def finish_times(tasks: list[Task]) -> np.ndarray:
    """When each task finished: its start plus its seconds. A task's outcome is known only from then on.

    Seconds leave out time spent waiting for a person, so for a task that waited this is a lower bound:
    it really finished later.
    """
    return np.array([t.started_at + t.seconds for t in tasks], dtype=float)


def recency_weights(tasks: list[Task], half_life_days: float | None) -> np.ndarray:
    if half_life_days is None or not tasks:
        return np.ones(len(tasks))
    newest = max(t.started_at for t in tasks)
    age_days = np.array([(newest - t.started_at) / 86400 for t in tasks])
    return 0.5 ** (age_days / half_life_days)


# Curve math -----------------------------------------------------------------------------------------

_erf = np.frompyfunc(math.erf, 1, 1)
_lgamma = np.frompyfunc(math.lgamma, 1, 1)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
_LOG_WIDTHS = (math.log(WIDTH_RANGE[0]), math.log(WIDTH_RANGE[1]))


def _norm_cdf(x) -> np.ndarray:
    return 0.5 * (1.0 + np.asarray(_erf(np.asarray(x, dtype=float) / math.sqrt(2.0)), dtype=float))


def _norm_pdf(x) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class LogNormal:
    """A bell curve on the log scale."""

    name = "lognormal"

    def logpdf(self, w):
        return -0.5 * w * w - 0.5 * math.log(2 * math.pi)

    def cdf(self, w):
        return _norm_cdf(w)

    def mean(self, mu, s):
        return np.exp(mu + s * s / 2)

    def log_score_steps(self, w, s):
        """Gradients and Fisher information of the log score for the center and the log width."""
        return -w / s, 1.0 - w * w, 1.0 / (s * s), np.full_like(w, 2.0)


class Weibull:
    """The Weibull curve, common for durations. On the log scale it's a Gumbel curve."""

    name = "weibull"

    def logpdf(self, w):
        return w - np.exp(np.minimum(w, 50))

    def cdf(self, w):
        return 1.0 - np.exp(-np.exp(np.minimum(w, 50)))

    def mean(self, mu, s):
        return np.exp(mu + np.asarray(_lgamma(1.0 + np.asarray(s, dtype=float)), dtype=float))

    def log_score_steps(self, w, s):
        e = np.exp(np.minimum(w, 50))
        return (1.0 - e) / s, 1.0 + w * (1.0 - e), 1.0 / (s * s), np.full_like(w, 1.8237)


FAMILY = {"lognormal": LogNormal(), "weibull": Weibull()}


def log_score(z, w, c1, c2, family=None) -> np.ndarray:
    """The negative log likelihood of a two-part curve, on the log scale. Lower is better."""
    family = family or FAMILY["lognormal"]
    (mu1, s1), (mu2, s2) = c1, c2
    l1 = np.log(np.clip(w, 1e-9, 1.0)) - np.log(s1) + family.logpdf((z - mu1) / s1)
    l2 = np.log(np.clip(1 - w, 1e-9, 1.0)) - np.log(s2) + family.logpdf((z - mu2) / s2)
    top = np.maximum(l1, l2)
    return -(top + np.log(np.exp(l1 - top) + np.exp(l2 - top)))


def pit(z, w, c1, c2, family=None) -> np.ndarray:
    """The chance the forecast gave to a result at or below the real one."""
    family = family or FAMILY["lognormal"]
    (mu1, s1), (mu2, s2) = c1, c2
    return w * family.cdf((z - mu1) / s1) + (1 - w) * family.cdf((z - mu2) / s2)


def expected(w, c1, c2, family=None):
    """The expected value: the average, not the middle. Rare long runs pull it above the middle."""
    family = family or FAMILY["lognormal"]
    return w * family.mean(*c1) + (1 - w) * family.mean(*c2)


def _a(m, variance) -> np.ndarray:
    s = np.sqrt(variance)
    u = m / s
    return 2.0 * s * _norm_pdf(u) + m * (2.0 * _norm_cdf(u) - 1.0)


def crps(z, w, c1, c2, family=None) -> np.ndarray:
    """The continuous ranked probability score of a two-part curve, on the log scale. Lower is better."""
    family = family or FAMILY["lognormal"]
    (mu1, s1), (mu2, s2) = c1, c2
    if family.name == "lognormal":
        v1, v2 = s1 * s1, s2 * s2
        near = w * _a(z - mu1, v1) + (1 - w) * _a(z - mu2, v2)
        spread = (
            w * w * _a(0.0 * mu1, 2 * v1)
            + (1 - w) ** 2 * _a(0.0 * mu2, 2 * v2)
            + 2 * w * (1 - w) * _a(mu1 - mu2, v1 + v2)
        )
        return near - 0.5 * spread
    z, w = np.atleast_1d(z), np.atleast_1d(w) * np.ones_like(np.atleast_1d(z))
    mu1, s1, mu2, s2 = (np.atleast_1d(v) * np.ones_like(z) for v in (mu1, s1, mu2, s2))
    reach = 12 * np.maximum(s1, s2)
    low = np.minimum(np.minimum(mu1, mu2) - reach, z) - 1
    high = np.maximum(np.maximum(mu1, mu2) + reach, z) + 1
    x = low[:, None] + (high - low)[:, None] * np.linspace(0, 1, 600)[None, :]
    curve = pit(x, w[:, None], (mu1[:, None], s1[:, None]), (mu2[:, None], s2[:, None]), family)
    return _trapezoid((curve - (x >= z[:, None])) ** 2, x, axis=1)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# The trees' scores --------------------------------------------------------------------------------------


def _label(data: xgb.DMatrix) -> np.ndarray:
    return data.get_label().reshape(data.num_row(), -1)[:, 0]


def _weighted(grad, hess, data: xgb.DMatrix):
    weight = data.get_weight()
    if len(weight):
        shape = (-1, 1) if grad.ndim == 2 else (-1,)
        grad, hess = grad * weight.reshape(shape), hess * weight.reshape(shape)
    return grad, hess


def _curve_objective(family, loss):
    """Gradients for each task's center and log width. Widths stay inside WIDTH_RANGE."""

    def objective(predt: np.ndarray, data: xgb.DMatrix):
        z = _label(data)
        mu, log_width = predt[:, 0], np.clip(predt[:, 1], *_LOG_WIDTHS)
        s = np.exp(log_width)
        w = (z - mu) / s
        if loss == "crps":
            g_mu = -(2 * _norm_cdf(w) - 1)
            g_ls = s * (2 * _norm_pdf(w) - 1 / math.sqrt(math.pi))
            h_mu = np.maximum(2 * _norm_pdf(w) / s, 1e-3 / s)
            h_ls = s / math.sqrt(math.pi)
        else:
            g_mu, g_ls, h_mu, h_ls = family.log_score_steps(w, s)
        return _weighted(np.column_stack([g_mu, g_ls]), np.column_stack([h_mu, h_ls]), data)

    return objective


def _curve_score(family, loss, z, mu, log_width) -> float:
    s = np.exp(np.clip(log_width, *_LOG_WIDTHS))
    if loss == "crps":
        w = (z - mu) / s
        return float(np.mean(s * (w * (2 * _norm_cdf(w) - 1) + 2 * _norm_pdf(w) - 1 / math.sqrt(math.pi))))
    return float(np.mean(np.log(s) - family.logpdf((z - mu) / s)))


def _curve_metric(family, loss):
    def metric(predt: np.ndarray, data: xgb.DMatrix):
        return "score", _curve_score(family, loss, _label(data), predt[:, 0], predt[:, 1])

    return metric


def _mean_objective(loss):
    """Gradients for an expected value on a log link, for the gamma or Tweedie loss."""

    def objective(predt: np.ndarray, data: xgb.DMatrix):
        y = data.get_label()
        mu = np.exp(np.clip(predt, -50, 50))
        if loss == "gamma":
            grad, hess = 1 - y / mu, y / mu
        else:
            p = TWEEDIE_POWER
            grad = -y * mu ** (1 - p) + mu ** (2 - p)
            hess = -(1 - p) * y * mu ** (1 - p) + (2 - p) * mu ** (2 - p)
        return _weighted(grad, np.maximum(hess, 1e-6), data)

    return objective


def mean_score(loss, y, mu) -> np.ndarray:
    """Half the unit deviance of an expected value: 0 when it's right, larger the further off it is."""
    mu = np.maximum(mu, 1e-12)
    if loss == "gamma":
        return y / mu - np.log(y / mu) - 1
    p = TWEEDIE_POWER
    return y ** (2 - p) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p) + mu ** (2 - p) / (2 - p)


def _mean_metric(loss):
    def metric(predt: np.ndarray, data: xgb.DMatrix):
        return "score", float(np.mean(mean_score(loss, data.get_label(), np.exp(np.clip(predt, -50, 50)))))

    return metric


# Similar tasks ------------------------------------------------------------------------------------------


def _unit(rows: list[np.ndarray | None], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack vectors at length 1. The second value marks which rows had a vector."""
    has = np.array([v is not None for v in rows], dtype=bool)
    out = np.zeros((len(rows), size), dtype=np.float32)
    for i, v in enumerate(rows):
        if v is not None:
            v = np.asarray(v, dtype=np.float32)
            out[i] = v / max(float(np.linalg.norm(v)), 1e-9)
    return out, has


def outcomes(tasks: list[Task]) -> np.ndarray:
    """What each task really took: log seconds, log tokens, log dollars and whether it answered directly."""
    rows = []
    for t in tasks:
        usd = target_value(t, "usd")
        rows.append([math.log(t.seconds), math.log(t.tokens), math.log(usd) if usd else np.nan, float(t.direct)])
    return np.array(rows, dtype=np.float32).reshape(len(tasks), 4)


def similar_tasks(
    pool: np.ndarray,
    results: np.ndarray,
    queries: np.ndarray,
    has: np.ndarray,
    earlier_only: bool,
    started: np.ndarray | None = None,
    finished: np.ndarray | None = None,
):
    """For each query, the average of what the most similar past tasks took, and how similar they were.

    With `earlier_only`, the queries are the pool itself in time order. Each one only looks at the
    NEIGHBOR_POOL tasks just before it, the same reach a forecast has, so training never peeks at the future.
    Given each task's start and finish times, it also skips tasks that hadn't finished when the query started.
    """
    out = np.full((len(queries), 5), np.nan, dtype=np.float64)
    if len(pool) == 0:
        return out
    empty = ~np.any(pool, axis=1)
    for start in range(0, len(queries), 512):
        stop = min(start + 512, len(queries))
        low, high = (max(0, start - NEIGHBOR_POOL), stop) if earlier_only else (0, len(pool))
        block = queries[start:stop] @ pool[low:high].T
        block[:, empty[low:high]] = -np.inf
        if earlier_only:
            position = np.arange(low, high)[None, :]
            row = np.arange(start, stop)[:, None]
            unknown = (position >= row) | (position < row - NEIGHBOR_POOL)
            if finished is not None:
                unknown |= finished[None, low:high] > started[start:stop, None]
            block[unknown] = -np.inf
        k = min(NEIGHBORS, high - low)
        top = np.argpartition(-block, k - 1, axis=1)[:, :k]
        similarity = np.take_along_axis(block, top, axis=1)
        valid = np.isfinite(similarity)
        picked = results[low:high][top]
        picked[~valid] = np.nan
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(picked, axis=1)
            closeness = np.nanmean(np.where(valid, similarity, np.nan), axis=1)
        chunk = np.column_stack([means, closeness])
        chunk[valid.sum(axis=1) < MIN_NEIGHBORS] = np.nan
        out[start:stop] = chunk
    out[~has] = np.nan
    return out


# Starting points: the simple rule, pooled -------------------------------------------------------------


def _pooled_curves(models: list[str], z: np.ndarray, weight: np.ndarray) -> dict:
    """Each model's center and log width, pulled toward the curve of all models when it has few tasks."""
    mu_all = float(np.average(z, weights=weight))
    var_all = float(np.average((z - mu_all) ** 2, weights=weight))
    start = {"*": [mu_all, 0.5 * math.log(max(var_all, 0.01))]}
    names = np.array(models)
    for name in set(models):
        rows = names == name
        n, w = int(rows.sum()), weight[rows]
        mu_m = float(np.average(z[rows], weights=w))
        var_m = float(np.average((z[rows] - mu_m) ** 2, weights=w))
        mu = (n * mu_m + POOLING * mu_all) / (n + POOLING)
        var = (n * var_m + POOLING * var_all) / (n + POOLING)
        start[name] = [mu, 0.5 * math.log(max(var, 0.01))]
    return start


def _pooled_values(models: list[str], values: np.ndarray, weight: np.ndarray, log: bool) -> dict:
    """Each model's average, pulled toward the average of all models when it has few tasks."""
    every = float(np.average(values, weights=weight))
    start = {"*": every}
    names = np.array(models)
    for name in set(models):
        rows = names == name
        n = int(rows.sum())
        start[name] = (n * float(np.average(values[rows], weights=weight[rows])) + POOLING * every) / (n + POOLING)
    if log:
        start = {name: math.log(max(value, 1e-12)) for name, value in start.items()}
    return start


# The model ----------------------------------------------------------------------------------------------


class Model:
    def __init__(self, meta: dict, boosters: dict[str, xgb.Booster], pools: dict | None = None) -> None:
        self.meta = meta
        self.boosters = boosters
        self.pools: dict[str, tuple[np.ndarray, np.ndarray]] = pools or {}
        """For each embedded feature: the newest past tasks' embeddings, at length 1, and what they took."""

    @property
    def settings(self) -> dict:
        return self.meta["settings"]

    @property
    def family(self):
        return FAMILY[self.settings["family"]]

    # Features into numbers

    def matrix(self, rows: list[Features], similar: dict[str, np.ndarray] | None = None) -> np.ndarray:
        categories = self.meta["categories"]
        columns = []
        for name in CATEGORIES:
            codes = {value: i for i, value in enumerate(categories[name])}
            columns.append([codes.get(getattr(r, name), np.nan) for r in rows])
        columns.append([math.log1p(max(r.prompt_chars, 0)) for r in rows])
        for name in self.meta["numbers"]:
            columns.append([r.numbers.get(name, np.nan) for r in rows])
        matrix = np.array(columns, dtype=float).T.reshape(len(rows), len(columns))
        for name in self.meta["text_fields"]:
            if similar is not None:
                block = similar[name]
            else:
                pool, results = self.pools[name]
                queries, has = _unit([r.vectors.get(name) for r in rows], pool.shape[1])
                block = similar_tasks(pool, results, queries, has, earlier_only=False)
            matrix = np.hstack([matrix, block])
        return matrix

    def dmatrix(self, matrix: np.ndarray, margin: np.ndarray, label=None, weight=None) -> xgb.DMatrix:
        types = ["c"] * len(CATEGORIES) + ["q"] * (matrix.shape[1] - len(CATEGORIES))
        data = xgb.DMatrix(matrix, label=label, weight=weight, feature_types=types, enable_categorical=True)
        data.set_base_margin(margin)
        return data

    def start(self, spec: dict, models: list[str], columns: int) -> np.ndarray:
        """Each row's starting point: its model's pooled value, or the value of all models."""
        start = spec["start"]
        values = np.array([start.get(m, start["*"]) for m in models], dtype=float)
        return values.reshape(len(models), columns) if columns > 1 else values

    def _predict(self, key: str, spec: dict, matrix: np.ndarray, margin: np.ndarray) -> np.ndarray:
        if not spec.get("rounds"):
            return margin
        data = self.dmatrix(matrix, margin)
        return self.boosters[key].predict(data, output_margin=True, iteration_range=(0, spec["rounds"]))

    def raw_curves(self, matrix: np.ndarray, models: list[str], target: str):
        """The chance of a direct answer and both parts' curves, before the width correction."""
        spec = self.meta["direct"]
        p = 1 / (1 + np.exp(-self._predict("direct", spec, matrix, _logit(self.start(spec, models, 1)))))
        parts = []
        for part in PARTS:
            key = f"{target}.{part}"
            spec = self.meta["parts"][key]
            raw = self._predict(key, spec, matrix, self.start(spec, models, 2)).reshape(len(models), 2)
            parts.append((raw[:, 0], np.exp(np.clip(raw[:, 1], *_LOG_WIDTHS))))
        return np.clip(p, 0.0, 1.0), parts[0], parts[1]

    def curves(self, rows: list[Features]) -> dict:
        """The two-part curve of each row, for every head. Widths include the final correction."""
        matrix = self.matrix(rows)
        models = [r.model for r in rows]
        out: dict = {"p_direct": None}
        for target in TARGETS:
            if f"{target}.work" not in self.meta.get("parts", {}):
                out[target] = None
                continue
            p, (mu1, s1), (mu2, s2) = self.raw_curves(matrix, models, target)
            out["p_direct"] = p
            scales = self.meta["width_scale"].get(target, {})
            scale = np.array([scales.get(m, scales.get("*", 1.0)) for m in models])
            out[target] = {
                "direct": (mu1, np.clip(s1 * scale, *WIDTH_RANGE)),
                "work": (mu2, np.clip(s2 * scale, *WIDTH_RANGE)),
            }
        return out

    def expected_values(self, rows: list[Features]) -> dict[str, np.ndarray | None]:
        """For expected-value heads: each row's expected value, for every head."""
        matrix = self.matrix(rows)
        models = [r.model for r in rows]
        out: dict = {}
        for target in TARGETS:
            spec = self.meta.get("expected", {}).get(target)
            if spec is None:
                out[target] = None
                continue
            out[target] = np.exp(np.clip(self._predict(target, spec, matrix, self.start(spec, models, 1)), -50, 50))
        return out

    def forecast(self, rows: list[Features]) -> list[dict]:
        """Each row's forecast: for every head, the expected value and, when there is one, the curve.

        This is what every model type provides. A head the model doesn't have is None.
        """
        if not rows:
            return []
        if self.settings["head"] == "expected":
            values = self.expected_values(rows)
            return [
                {
                    "shape": None,
                    "direct_probability": None,
                    **{
                        t: None if values[t] is None else {"expected": float(values[t][i]), "distribution": None}
                        for t in TARGETS
                    },
                }
                for i in range(len(rows))
            ]
        curves = self.curves(rows)
        family = self.family
        found = []
        for i in range(len(rows)):
            row: dict = {"shape": f"two_part_{family.name}", "direct_probability": float(curves["p_direct"][i])}
            for target in TARGETS:
                curve = curves[target]
                if curve is None:
                    row[target] = None
                    continue
                (mu1, s1), (mu2, s2) = curve["direct"], curve["work"]
                value = expected(curves["p_direct"][i], (mu1[i], s1[i]), (mu2[i], s2[i]), family)
                row[target] = {
                    "expected": float(value),
                    "distribution": {
                        "direct": {"mu": float(mu1[i]), "sigma": float(s1[i])},
                        "work": {"mu": float(mu2[i]), "sigma": float(s2[i])},
                    },
                }
            found.append(row)
        return found

    def served(self, model: str) -> tuple[bool, str | None]:
        """Whether forecasts for this AI model passed the test, and why not when they didn't."""
        if model not in self.meta["categories"]["model"]:
            return False, "unknown_model"
        verdict = self.meta.get("served", {}).get(model)
        if verdict is None:
            return False, "not_enough_data"
        return verdict["served"], verdict["reason"]

    def save(self, folder: Path) -> None:
        folder.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".model-", dir=folder.parent))
        for name, booster in self.boosters.items():
            booster.save_model(str(staging / f"{name}.json"))
        (staging / "model.json").write_text(json.dumps(self.meta, indent=1) + "\n")
        for name, (vectors, results) in self.pools.items():
            np.save(staging / f"similar-{name}-vectors.npy", vectors.astype(np.float32), allow_pickle=False)
            np.save(staging / f"similar-{name}-results.npy", results.astype(np.float32), allow_pickle=False)
        if folder.exists():
            shutil.rmtree(folder)
        staging.rename(folder)


def load(folder: Path) -> Model:
    """Load a saved model. Every file is plain JSON or plain numbers, so loading runs no code."""
    meta = json.loads((folder / "model.json").read_text())
    boosters = {}
    for path in folder.glob("*.json"):
        if path.name != "model.json":
            booster = xgb.Booster()
            booster.load_model(str(path))
            boosters[path.stem] = booster
    pools = {}
    for name in meta.get("text_fields", []):
        pools[name] = (
            np.load(folder / f"similar-{name}-vectors.npy", allow_pickle=False),
            np.load(folder / f"similar-{name}-results.npy", allow_pickle=False),
        )
    return Model(meta, boosters, pools)


# Training -----------------------------------------------------------------------------------------------


def train(tasks: list[Task], rows: list[Features], settings: dict | None = None) -> Model:
    """Test on the newest tasks, then train the final model on all of them. `rows` match `tasks` one to one."""
    settings = settings_for(settings)
    learnable = {t.id for t in usable(tasks)}
    pairs = sorted(((t, r) for t, r in zip(tasks, rows) if t.id in learnable), key=lambda p: (p[0].started_at, p[0].id))
    if any(r.vectors.get("prompt") is not None for _, r in pairs):
        pairs = [(t, r) for t, r in pairs if r.vectors.get("prompt") is not None]
    if len(pairs) < MIN_TASKS:
        raise NotEnoughData(f"Training needs at least {MIN_TASKS} finished tasks. This dataset has {len(pairs)}.")
    eligible, eligible_rows = [t for t, _ in pairs], [r for _, r in pairs]
    report = test(eligible, eligible_rows, settings)
    final = fit(eligible, eligible_rows, settings)
    final.meta.update(
        report=report,
        served={m: {"served": r["served"], "reason": r["reason"]} for m, r in report["models"].items()},
        trained_at=time.time(),
        tasks=len(eligible),
    )
    return final


def _defaults(tasks: list[Task]) -> dict:
    """What to assume for a forecast request that leaves a feature out, per AI model."""
    by_model: dict[str, list[Task]] = {}
    for t in tasks:
        by_model.setdefault(t.model, []).append(t)
    return {
        model: {
            "effort": Counter(t.effort for t in group).most_common(1)[0][0],
            "agent": Counter(t.agent for t in group).most_common(1)[0][0],
        }
        for model, group in by_model.items()
    }


def _params(settings: dict) -> dict:
    params = dict(TREES)
    if settings["dropout"] > 0:
        params.update(rate_drop=float(settings["dropout"]), skip_drop=0.0)
    return params


def fit(tasks: list[Task], rows: list[Features], settings: dict) -> Model:
    """Fit every head on the tasks, oldest first.

    The newest tasks pick each part's number of rounds and the width correction. Then every part is
    refit on all the tasks with that number of rounds.
    """
    settings = settings_for(settings)
    text_fields = [name for name in dict.fromkeys(n for r in rows for n in r.vectors) if name]
    numbers = sorted({n for r in rows for n in r.numbers})
    meta: dict = {
        "format": 2,
        "settings": settings,
        "categories": {
            name: [v for v, _ in Counter(getattr(t, name) for t in tasks).most_common()] for name in CATEGORIES
        },
        "numbers": numbers,
        "text_fields": text_fields,
        "parts": {},
        "expected": {},
        "width_scale": {},
        "defaults": _defaults(tasks),
        "tree_settings": TREES,
    }
    model = Model(meta, {})
    results = outcomes(tasks)
    started, finished = np.array([t.started_at for t in tasks], dtype=float), finish_times(tasks)
    similar: dict[str, np.ndarray] = {}
    for name in text_fields:
        size = len(next(r.vectors[name] for r in rows if r.vectors.get(name) is not None))
        queries, has = _unit([r.vectors.get(name) for r in rows], size)
        similar[name] = similar_tasks(
            queries, results, queries, has, earlier_only=True, started=started, finished=finished
        )
        newest = np.flatnonzero(has)[-NEIGHBOR_POOL:]
        model.pools[name] = (queries[newest], results[newest])
    matrix = model.matrix(rows, similar)
    models = [t.model for t in tasks]
    held = max(int(len(tasks) * settings["validation_share"]), 10)
    fit_rows = np.arange(len(tasks) - held)
    valid_rows = np.arange(len(tasks) - held, len(tasks))
    weight = recency_weights(tasks, settings["half_life_days"])
    if settings["head"] == "expected":
        for target in settings["heads"]:
            values = np.array([target_value(t, target) or np.nan for t in tasks])
            known = ~np.isnan(values)
            if known.sum() >= MIN_TASKS:
                _grow_mean(model, target, matrix, models, values, known, weight, fit_rows, valid_rows)
    else:
        direct = np.array([t.direct for t in tasks], dtype=float)
        meta["direct"] = {"start": _pooled_shares(models, direct, weight)}
        _grow_direct(model, matrix, models, direct, weight, fit_rows, valid_rows)
        for target in settings["heads"]:
            values = np.array([target_value(t, target) or np.nan for t in tasks])
            known = ~np.isnan(values)
            if known.sum() < MIN_TASKS:
                continue
            z = np.log(np.where(known, values, 1.0))
            for part, flag in (("direct", 1.0), ("work", 0.0)):
                mask = known & (direct == flag)
                pool = np.flatnonzero(mask) if mask.sum() >= 2 else np.flatnonzero(known)
                key = f"{target}.{part}"
                meta["parts"][key] = {"start": _pooled_curves([models[i] for i in pool], z[pool], weight[pool])}
                _grow_part(model, key, matrix, models, z, weight, mask, fit_rows, valid_rows)
            _fit_width_scale(model, target, matrix, models, z, known, valid_rows)
    _refit(model, tasks, matrix, models, weight)
    return model


def _pooled_shares(models: list[str], direct: np.ndarray, weight: np.ndarray) -> dict:
    return _pooled_values(models, direct, weight, log=False)


def _useful_rounds(booster: xgb.Booster, start_score: float) -> int:
    """How many rounds to keep: zero when no number of trees beat the starting point on the newest tasks."""
    return booster.best_iteration + 1 if booster.best_score < start_score else 0


def _grow(model, key, params, matrix, margin, label, weight, grow, check, objective, metric, start_score):
    def data(idx, weighted):
        return model.dmatrix(matrix[idx], margin[idx], label=label[idx], weight=weight[idx] if weighted else None)

    booster = xgb.train(
        {**params, "disable_default_eval_metric": 1},
        data(grow, True),
        MAX_DROPOUT_ROUNDS if params.get("rate_drop") else MAX_ROUNDS,
        obj=objective,
        custom_metric=metric,
        evals=[(data(check, False), "valid")],
        early_stopping_rounds=PATIENCE,
        verbose_eval=False,
    )
    rounds = _useful_rounds(booster, start_score)
    return booster, rounds


def _grow_direct(model, matrix, models, direct, weight, fit_rows, valid_rows) -> None:
    spec = model.meta["direct"]
    if min(direct[fit_rows].sum(), len(fit_rows) - direct[fit_rows].sum()) < 10:
        return
    margin = _logit(model.start(spec, models, 1))

    def objective(predt, data):
        p = 1 / (1 + np.exp(-predt))
        return _weighted(p - data.get_label(), np.maximum(p * (1 - p), 1e-6), data)

    def logloss(y, margin_values):
        p = np.clip(1 / (1 + np.exp(-margin_values)), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def metric(predt, data):
        return "score", logloss(data.get_label(), predt)

    params = _params(model.settings)
    booster, rounds = _grow(
        model,
        "direct",
        params,
        matrix,
        margin,
        direct,
        weight,
        fit_rows,
        valid_rows,
        objective,
        metric,
        logloss(direct[valid_rows], margin[valid_rows]),
    )
    if rounds:
        spec.update(rounds=rounds, params=params)
        model.boosters["direct"] = booster
        model.meta["direct_objective"] = "logistic"


def _grow_part(model, key, matrix, models, z, weight, mask, fit_rows, valid_rows) -> None:
    spec = model.meta["parts"][key]
    grow, check = fit_rows[mask[fit_rows]], valid_rows[mask[valid_rows]]
    if len(grow) < MIN_PART_TASKS or len(check) < 5:
        return
    family, loss = model.family, model.settings["loss"]
    margin = model.start(spec, models, 2)
    params = {**_params(model.settings), "multi_strategy": "one_output_per_tree"}
    label = np.column_stack([z, z])
    start_score = _curve_score(family, loss, z[check], margin[check, 0], margin[check, 1])
    booster, rounds = _grow(
        model,
        key,
        params,
        matrix,
        margin,
        label,
        weight,
        grow,
        check,
        _curve_objective(family, loss),
        _curve_metric(family, loss),
        start_score,
    )
    if rounds:
        spec.update(rounds=rounds, params=params)
        model.boosters[key] = booster


def _grow_mean(model, target, matrix, models, values, known, weight, fit_rows, valid_rows) -> None:
    loss = model.settings["loss"]
    rows = np.flatnonzero(known)
    spec = model.meta["expected"][target] = {
        "start": _pooled_values([models[i] for i in rows], values[rows], weight[rows], log=True)
    }
    grow, check = fit_rows[known[fit_rows]], valid_rows[known[valid_rows]]
    if len(grow) < MIN_PART_TASKS or len(check) < 5:
        return
    margin = model.start(spec, models, 1)
    label = np.where(known, values, 1.0)
    start_score = float(np.mean(mean_score(loss, label[check], np.exp(margin[check]))))
    params = _params(model.settings)
    booster, rounds = _grow(
        model,
        target,
        params,
        matrix,
        margin,
        label,
        weight,
        grow,
        check,
        _mean_objective(loss),
        _mean_metric(loss),
        start_score,
    )
    if rounds:
        spec.update(rounds=rounds, params=params)
        model.boosters[target] = booster


def _fit_width_scale(model, target, matrix, models, z, known, valid_rows) -> None:
    """A factor that widens or narrows a target's curves, picked on the newest tasks.

    Each AI model with enough of the newest tasks gets its own factor. The rest share one. This runs
    before the refit, so the trees haven't seen these tasks.
    """
    scales = model.meta["width_scale"][target] = {"*": 1.0}
    rows = valid_rows[known[valid_rows]]
    if len(rows) < 10:
        return
    names = np.array([models[i] for i in rows])
    w, (mu1, s1), (mu2, s2) = model.raw_curves(matrix[rows], names.tolist(), target)
    grid = np.geomspace(0.5, 3.0, 46)
    family = model.family

    def best(mask: np.ndarray) -> float:
        scores = [
            float(
                np.mean(
                    log_score(
                        z[rows][mask],
                        w[mask],
                        (mu1[mask], np.clip(s1[mask] * g, *WIDTH_RANGE)),
                        (mu2[mask], np.clip(s2[mask] * g, *WIDTH_RANGE)),
                        family,
                    )
                )
            )
            for g in grid
        ]
        return float(grid[int(np.argmin(scores))])

    scales["*"] = best(np.ones(len(rows), dtype=bool))
    for name in set(names.tolist()):
        mask = names == name
        if mask.sum() >= MIN_TEST_TASKS:
            scales[name] = best(mask)


def _refit(model: Model, tasks: list[Task], matrix, models, weight) -> None:
    """Refit every grown part on all the tasks, with the number of rounds the newest tasks picked."""
    everything = np.arange(len(tasks))
    loss, family = model.settings["loss"], model.family
    direct = np.array([t.direct for t in tasks], dtype=float)
    if model.meta.get("direct", {}).get("rounds"):
        spec = model.meta["direct"]
        margin = _logit(model.start(spec, models, 1))

        def objective(predt, data):
            p = 1 / (1 + np.exp(-predt))
            return _weighted(p - data.get_label(), np.maximum(p * (1 - p), 1e-6), data)

        data = model.dmatrix(matrix, margin, label=direct, weight=weight)
        model.boosters["direct"] = xgb.train(spec["params"], data, spec["rounds"], obj=objective)
    for key, spec in model.meta["parts"].items():
        if not spec.get("rounds"):
            continue
        target, part = key.split(".")
        values = np.array([target_value(t, target) or np.nan for t in tasks])
        rows = everything[~np.isnan(values) & (direct == (1.0 if part == "direct" else 0.0))]
        z = np.log(values[rows])
        margin = model.start(spec, [models[i] for i in rows], 2)
        data = model.dmatrix(matrix[rows], margin, label=np.column_stack([z, z]), weight=weight[rows])
        model.boosters[key] = xgb.train(spec["params"], data, spec["rounds"], obj=_curve_objective(family, loss))
    for target, spec in model.meta["expected"].items():
        if not spec.get("rounds"):
            continue
        values = np.array([target_value(t, target) or np.nan for t in tasks])
        rows = everything[~np.isnan(values)]
        margin = model.start(spec, [models[i] for i in rows], 1)
        data = model.dmatrix(matrix[rows], margin, label=values[rows], weight=weight[rows])
        model.boosters[target] = xgb.train(spec["params"], data, spec["rounds"], obj=_mean_objective(loss))


# Testing ------------------------------------------------------------------------------------------------


def simple_rule(train_tasks: list[Task], target: str) -> dict:
    """The same two-part curve for every task on an AI model: the spread of its past tasks."""

    def curve(group: list[Task]) -> dict | None:
        pairs = [(t.direct, math.log(v), v) for t in group if (v := target_value(t, target))]
        if len(pairs) < 2:
            return None
        flags = np.array([d for d, _, _ in pairs], dtype=float)
        z = np.array([v for _, v, _ in pairs])
        whole = (float(z.mean()), float(max(z.std(), 0.1)))
        parts = {}
        for part, flag in (("direct", 1.0), ("work", 0.0)):
            zz = z[flags == flag]
            parts[part] = (float(zz.mean()), float(max(zz.std(), 0.1))) if len(zz) >= 2 else whole
        return {"p": float(flags.mean()), "mean": float(np.mean([v for _, _, v in pairs])), **parts}

    rules = {"*": curve(train_tasks)}
    for model in {t.model for t in train_tasks}:
        rules[model] = curve([t for t in train_tasks if t.model == model]) or rules["*"]
    return rules


def test(tasks: list[Task], rows: list[Features], settings: dict) -> dict:
    """Forecast the newest tasks window by window, each from the tasks before it, and judge each AI model.

    A window learns only from tasks that had finished when it starts. A task that started earlier but
    was still running then hadn't shown its outcome yet.
    """
    settings = settings_for(settings)
    if len(tasks) < 100:
        return {"trained_on": len(tasks), "tested_on": 0, "windows": 0, "models": {}, "pit": {}}
    windows = settings["test_windows"]
    size = max(int(len(tasks) * settings["test_share"] / windows), 1)
    first = len(tasks) - size * windows
    finished = finish_times(tasks)
    scored = []
    for k in range(windows):
        cut = first + k * size
        end = cut + size if k < windows - 1 else len(tasks)
        known = [i for i in range(cut) if finished[i] <= tasks[cut].started_at]
        if len(known) < MIN_TASKS:
            continue
        train_tasks, train_rows = [tasks[i] for i in known], [rows[i] for i in known]
        trial = fit(train_tasks, train_rows, settings)
        scored.append(_score_window(trial, train_tasks, tasks[cut:end], rows[cut:end]))
    return _verdicts(scored, trained_on=first, tested_on=len(tasks) - first, settings=settings)


def _score_window(trial: Model, train_tasks: list[Task], test_tasks: list[Task], test_rows: list[Features]) -> dict:
    """Per-task scores for one window, for the trial model and the simple rule.

    Tasks on a model the trial saw fewer than MIN_TEST_TASKS times aren't scored: the test asks how
    well a model with enough history forecasts, not how well it guesses at a model it barely knows.
    """
    trained = Counter(t.model for t in train_tasks)
    keep = [i for i, t in enumerate(test_tasks) if trained[t.model] >= MIN_TEST_TASKS]
    test_tasks, test_rows = [test_tasks[i] for i in keep], [test_rows[i] for i in keep]
    out: dict = {"models": np.array([t.model for t in test_tasks]), "targets": {}}
    if not test_tasks:
        return out
    settings = trial.settings
    if settings["head"] == "expected":
        values_by_target = trial.expected_values(test_rows)
    else:
        curves = trial.curves(test_rows)
    family = trial.family
    for target in settings["heads"]:
        values = np.array([target_value(t, target) or np.nan for t in test_tasks])
        rules = simple_rule(train_tasks, target)
        base = [rules.get(m) or rules["*"] for m in out["models"]]
        if settings["head"] == "expected":
            ours_mean = values_by_target.get(target)
            if ours_mean is None:
                continue
            simple_mean = np.array([b["mean"] for b in base])
            known = ~np.isnan(values)
            y = np.where(known, values, 1.0)
            out["targets"][target] = {
                "real": values,
                "ours": mean_score(settings["loss"], y, ours_mean),
                "simple": mean_score(settings["loss"], y, simple_mean),
                "pit": None,
                "expected": ours_mean,
            }
            continue
        if curves[target] is None:
            continue
        z = np.log(np.where(np.isnan(values), 1.0, values))
        w = curves["p_direct"]
        c1, c2 = curves[target]["direct"], curves[target]["work"]
        bw = np.array([b["p"] for b in base])
        b1 = (np.array([b["direct"][0] for b in base]), np.array([b["direct"][1] for b in base]))
        b2 = (np.array([b["work"][0] for b in base]), np.array([b["work"][1] for b in base]))
        out["targets"][target] = {
            "real": values,
            "ours": crps(z, w, c1, c2, family),
            "simple": crps(z, bw, b1, b2, LogNormal()),
            "pit": pit(z, w, c1, c2, family),
            "expected": expected(w, c1, c2, family),
        }
    return out


def _verdicts(scored: list[dict], trained_on: int, tested_on: int, settings: dict) -> dict:
    """Pool the windows and judge each AI model: give its forecasts, or say why not."""
    models = np.concatenate([w["models"] for w in scored]) if scored else np.array([])
    report: dict = {
        "trained_on": trained_on,
        "tested_on": tested_on,
        "windows": len(scored),
        "models": {},
        "pit": {},
    }
    pooled: dict[str, dict] = {}
    for target in settings["heads"]:
        parts = [w["targets"].get(target) for w in scored]
        if not any(parts):
            continue
        pooled[target] = {}
        for name in ("real", "ours", "simple", "pit", "expected"):
            pieces = []
            for p, w in zip(parts, scored):
                value = p.get(name) if p else None
                pieces.append(np.full(len(w["models"]), np.nan) if value is None else np.asarray(value, dtype=float))
            pooled[target][name] = np.concatenate(pieces) if pieces else np.array([])
        known = ~np.isnan(pooled[target]["real"]) & ~np.isnan(pooled[target]["pit"])
        if known.any():
            report["pit"][target] = np.histogram(pooled[target]["pit"][known], bins=10, range=(0, 1))[0].tolist()
    for model in sorted(set(models.tolist())):
        result: dict = {"tasks_tested": int((models == model).sum())}
        for target, data in pooled.items():
            rows = (models == model) & ~np.isnan(data["real"])
            if not rows.any():
                continue
            chance = data["pit"][rows]
            has_ranges = not np.isnan(chance).all()
            result[target] = {
                "score": round(float(data["ours"][rows].mean()), 4),
                "simple_rule_score": round(float(data["simple"][rows].mean()), 4),
                "better_than_simple_rule": round(float(1 - data["ours"][rows].mean() / data["simple"][rows].mean()), 4),
                "in_50_range": round(float(np.mean((chance > 0.25) & (chance < 0.75))), 4) if has_ranges else None,
                "in_90_range": round(float(np.mean((chance > 0.05) & (chance < 0.95))), 4) if has_ranges else None,
                "expected_over_real_total": round(float(data["expected"][rows].sum() / data["real"][rows].sum()), 4),
            }
        failed = []
        for target in TARGETS:
            if target not in result:
                continue
            r = result[target]
            if r["score"] > r["simple_rule_score"] * (1 + TOLERANCE):
                failed.append(f"{target}.simple_rule")
            if r["in_90_range"] is not None and r["in_90_range"] < MIN_RANGE_HELD:
                failed.append(f"{target}.range_held")
            if not TOTALS_RANGE[0] <= r["expected_over_real_total"] <= TOTALS_RANGE[1]:
                failed.append(f"{target}.totals")
        if result["tasks_tested"] < MIN_TEST_TASKS:
            served, reason = False, "not_enough_data"
        elif failed:
            served, reason = False, "not_accurate_enough"
        else:
            served, reason = True, None
        result["failed_checks"] = failed if reason == "not_accurate_enough" else []
        report["models"][model] = {"served": served, "reason": reason, **result}
    return report
