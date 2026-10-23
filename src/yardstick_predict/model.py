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
length, plus the numbers the pipeline adds, such as the prompt's shape and the project's recent tasks,
and the chosen features. Each embedded feature, such as the prompt, becomes what the most similar past
tasks really took. For the prompt that's the closest task, the closest ten and the closest hundred, how
much they varied, and an average where closer, newer tasks count more. That works better than feeding
the trees the embedding itself, which is many weak numbers the trees overfit. The whole embedding still
gets one say: a weighted sum of all its numbers, with the weights kept small, makes a first guess of
each outcome, and the trees get that guess.

Training reads exactly the features it's given. With the conversation feature, a task with a prompt and a
context also gets how close their meanings are, and the two side by side as one more embedding. With LLM
analysis, training also makes the same model without it, tested on its own, to forecast the tasks LLM
analysis couldn't read.

The trees start from the simple rule: each AI model's own curve, the spread of its past tasks, at each
effort level it ran at. A model with few tasks has its curve pulled toward the curve of all models, and an
effort level with few tasks toward its model's. Trees are added only while they beat that start on the
newest tasks, so with little data the model stays close to the simple rule, and with a lot of data it
gets sharper.

Training holds back the newest tasks as a test. Forecasts for an AI model are only given when they did
at least as well as the simple rule on that test, their ranges held and their expected values added up.

Advanced settings change the heads, the curve's shape, the loss, dropout and how the data is split.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

from . import benchmarks, features
from .archives import Task
from .model_types import BUILT_IN_FORMAT, PREVIOUS_FORMAT

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
CLOSE = 10
"""How many of the closest past tasks the close-match features average over."""
CLOSE_POWER = 4
"""In the weighted average of similar tasks, each counts by its closeness to this power, times how recent it is."""
RICH = ("prompt",)
"""The embedded features whose similar tasks get every summary. The others get the five averages."""
RICH_COLUMNS = 22
JOINT = "prompt_and_context"
"""The prompt and its context side by side, to find past tasks similar in both."""
CONVERSATION = (
    "session_tasks_before",
    "previous_seconds",
    "previous_tokens",
    "previous_usd",
    "previous_direct",
    "minutes_since_previous",
    "conversation_tokens",
    "conversation_tokens_program",
)
"""The number features the conversation feature brings, about the conversation a task is in. Rows with them
also get how close each prompt is to its context, and the two side by side."""
CLOSENESS = "prompt_context_closeness"
"""How close a prompt's meaning is to its context's, for rows with the conversation feature."""
GUESSES = ("guess_seconds", "guess_tokens", "guess_usd", "guess_direct")
GUESS_ALPHAS = (0.3, 3.0, 30.0, 300.0)
"""How hard the first guess's weights are pulled toward 0. Training picks the one that guesses best."""
GUESS_BLOCKS = 10
"""Training's first guesses are made stretch by stretch of time, each from the ones before it."""
FORMAT = BUILT_IN_FORMAT
"""The saved model's format. A model the previous release saved, in PREVIOUS_FORMAT, still forecasts until it's
trained again. One in an older format has to be trained again, and one in a newer format needs a newer version
of yardstick-predict."""
LLM = "llm"
"""The embedded feature LLM analysis writes. Its number features' names start with llm_."""
WITHOUT_LLM = "without-llm"
"""The folder, inside a model's own, that holds the same model without LLM analysis."""
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
THREADS_FROM = 50_000
"""XGBoost runs on one thread below this many rows. On less data, its threads cost more time than they save."""
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


class OutdatedModel(Exception):
    """The saved model was made by a version before the previous release, which read different features. It has
    to be trained again."""


class NewerFormat(Exception):
    """The saved model was written by a newer version of yardstick-predict."""


def settings_for(given: dict | None) -> dict:
    """Fill in the defaults and check every setting. Raises ValueError with a plain reason."""
    given = dict(given or {})
    unknown = [name for name in given if name not in DEFAULTS]
    if unknown:
        raise ValueError(f"Unknown model setting {unknown[0]!r}. The settings are: {', '.join(DEFAULTS)}.")
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
    if settings["head"] == "expected" and settings["family"] != "lognormal":
        raise ValueError(
            "The distribution's shape is for curves. Expected heads have no curve, so leave the shape out."
        )
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
    started_at: float | None = None
    """When the task started, for how recent each similar past task is. A forecast leaves it out: it means now."""


def usable(tasks: list[Task]) -> list[Task]:
    """Tasks the model can learn from, oldest first. Stopped tasks and tasks left waiting are left out."""
    keep = [t for t in tasks if not t.stopped and not t.idle and t.seconds > 0 and t.tokens > 0 and t.model]
    return sorted(keep, key=lambda t: (t.started_at, t.id))


def target_value(task: Task, target: str) -> float | None:
    """What a task really took for one head, or None when the model can't learn it.

    Curves live on a log scale, so only a value above 0 can be learned. A cost of $0, as on a free AI
    model, would sit far below every real cost and pull every curve toward it, so it's left out like an
    unknown cost.
    """
    value = {"seconds": task.seconds, "tokens": float(task.tokens), "usd": task.usd}[target]
    return float(value) if value is not None and value > 0 else None


def _known_values(tasks: list[Task], target: str) -> np.ndarray:
    """What each task really took for one head, with NaN where target_value has none."""
    return np.array([np.nan if (value := target_value(t, target)) is None else value for t in tasks], dtype=float)


def finish_times(tasks: list[Task]) -> np.ndarray:
    """When each task really finished, waits for the person included. A task's outcome is known only from then on.

    A task stored by an older version doesn't know when it finished, so its start plus its seconds stands in.
    Seconds leave out time spent waiting for a person, so for such a task that waited, it really finished later.
    """
    return np.array([t.started_at + t.seconds if t.finished_at is None else t.finished_at for t in tasks], dtype=float)


def recency_weights(tasks: list[Task], half_life_days: float | None) -> np.ndarray:
    if half_life_days is None or not tasks:
        return np.ones(len(tasks))
    newest = max(t.started_at for t in tasks)
    age_days = np.array([(newest - t.started_at) / 86400 for t in tasks])
    # With a short half-life, a task years old would weigh exactly 0, and so could every task of an AI model
    # used only long ago. A tiny least weight keeps its curve defined without giving it any real say.
    return np.maximum(0.5 ** (age_days / half_life_days), 1e-12)


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

    def from_moments(self, mean: float, spread: float) -> tuple[float, float]:
        """The center and width of the curve whose log values have this mean and standard deviation."""
        return mean, spread

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

    def from_moments(self, mean: float, spread: float) -> tuple[float, float]:
        """The center and width of the curve whose log values have this mean and standard deviation.

        On the log scale, the curve's spread is pi / sqrt(6) widths, and its mean sits Euler's constant
        widths below its center.
        """
        s = spread * math.sqrt(6.0) / math.pi
        return mean + np.euler_gamma * s, s

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
    out = np.empty(len(z))
    for start in range(0, len(z), 1024):  # a block of tasks at a time, so the grid of points stays small
        r = slice(start, start + 1024)
        x = low[r, None] + (high - low)[r, None] * np.linspace(0, 1, 600)[None, :]
        curve = pit(x, w[r, None], (mu1[r, None], s1[r, None]), (mu2[r, None], s2[r, None]), family)
        out[r] = _trapezoid((curve - (x >= z[r, None])) ** 2, x, axis=1)
    return out


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
            h_ls = s / (2 * math.sqrt(math.pi))  # its expected value when the width is right
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
        rows.append(
            [math.log(t.seconds), math.log(t.tokens), np.nan if usd is None else math.log(usd), float(t.direct)]
        )
    return np.array(rows, dtype=np.float32).reshape(len(tasks), 4)


def similar_tasks(
    pool: np.ndarray,
    results: np.ndarray,
    queries: np.ndarray,
    has: np.ndarray,
    earlier_only: bool,
    started: np.ndarray | None = None,
    finished: np.ndarray | None = None,
    rich: bool = False,
    pool_started: np.ndarray | None = None,
    half_life_days: float | None = None,
):
    """For each query, what the most similar past tasks took, and how similar they were.

    The first five columns are averages over the NEIGHBORS most similar: log seconds, log tokens, log
    dollars, the share that answered directly, and their closeness. With `rich`, RICH_COLUMNS more follow:
    the closest task's outcomes and closeness, the same averages over the CLOSE closest, how much the
    outcomes varied, the typical outcome of direct answers and of work kept apart, and an average where
    closer and newer tasks count more.

    With `earlier_only`, the queries are the pool itself in time order. Each one only looks at the
    NEIGHBOR_POOL tasks just before it, the same reach a forecast has, so training never peeks at the future.
    Given each task's start and finish times, it also skips tasks that hadn't finished when the query started.
    Otherwise `started` holds each query's start and `pool_started` each past task's, for how recent they are.
    """
    out = np.full((len(queries), 5 + (RICH_COLUMNS if rich else 0)), np.nan, dtype=np.float64)
    if len(pool) == 0:
        return out
    pool_started = started if earlier_only else pool_started
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
            columns = [np.nanmean(picked, axis=1), np.nanmean(np.where(valid, similarity, np.nan), axis=1)]
            if rich:
                age = None
                if started is not None and pool_started is not None:
                    age = (started[start:stop, None] - pool_started[low:high][top]) / 86400
                columns += _close_matches(picked, similarity, valid, age, half_life_days)
        chunk = np.column_stack([c.reshape(stop - start, -1) for c in columns])
        chunk[valid.sum(axis=1) < MIN_NEIGHBORS] = np.nan
        out[start:stop] = chunk
    out[~has] = np.nan
    return out


def _close_matches(picked, similarity, valid, age_days, half_life_days) -> list[np.ndarray]:
    """The rich summaries of each query's similar tasks, as similar_tasks describes them."""
    order = np.argsort(-np.where(valid, similarity, -np.inf), axis=1)
    picked = np.take_along_axis(picked, order[:, :, None], axis=1)
    similarity, valid = np.take_along_axis(similarity, order, axis=1), np.take_along_axis(valid, order, axis=1)
    closeness = np.where(valid, similarity, np.nan)
    outcome, direct = picked[:, :, :3], picked[:, :, 3]
    weight = np.where(valid, np.clip(similarity, 0.0, None) ** CLOSE_POWER, 0.0)
    if age_days is not None and half_life_days:
        weight = weight * 0.5 ** (np.maximum(np.take_along_axis(age_days, order, axis=1), 0.0) / half_life_days)
    weight = weight[:, :, None] * ~np.isnan(outcome)
    total = weight.sum(axis=1)
    weighted = np.nansum(np.nan_to_num(outcome) * weight, axis=1) / np.where(total > 0, total, np.nan)
    return [
        picked[:, 0, :],
        closeness[:, 0],
        np.nanmean(picked[:, :CLOSE, :], axis=1),
        np.nanmean(closeness[:, :CLOSE], axis=1),
        np.nanstd(outcome, axis=1),
        np.nanmean(np.where((direct == 0.0)[:, :, None], outcome, np.nan), axis=1),
        np.nanmean(np.where((direct == 1.0)[:, :, None], outcome, np.nan), axis=1),
        weighted,
    ]


def _length_one(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _reads_conversation(rows: list[Features]) -> bool:
    """Whether the rows come with the conversation feature: its numbers about the conversation so far."""
    return any(name in r.numbers for r in rows for name in CONVERSATION)


def _with_conversation(rows: list[Features], use: bool, joint: bool) -> list[Features]:
    """The rows as a model reads them: with the rest of the conversation's inputs, when it reads the conversation.

    A row with both a prompt and a context also gets how close their meanings are, and with `joint`, the two
    side by side as one more embedding to find similar past tasks by. Without the conversation, the rows stay
    as they are.
    """
    if not use:
        return rows
    found = []
    for r in rows:
        numbers, vectors = dict(r.numbers), dict(r.vectors)
        prompt, context = r.vectors.get("prompt"), r.vectors.get("context")
        if prompt is not None and context is not None:
            numbers[CLOSENESS] = float(_length_one(prompt) @ _length_one(context))
        if joint and prompt is not None:
            other = _length_one(context) if context is not None else np.zeros(len(prompt), dtype=np.float32)
            vectors[JOINT] = np.concatenate([_length_one(prompt), other])
        found.append(replace(r, numbers=numbers, vectors=vectors))
    return found


def _prepared(rows: list[Features]) -> list[Features]:
    """The rows as training reads them. With the conversation feature, the prompt and its context go side by
    side when any row has a context."""
    use = _reads_conversation(rows)
    return _with_conversation(rows, use, use and _has_context(rows))


def _read_by_llm(row: Features) -> bool:
    """Whether the row has what LLM analysis said about its task."""
    return has_feature(row, LLM)


def _without_llm(rows: list[Features]) -> list[Features]:
    """The rows as the model without LLM analysis sees them."""
    return without(rows, LLM)


# Leaving one feature out --------------------------------------------------------------------------------


def _parts(name: str) -> tuple[frozenset[str], frozenset[str], str | None]:
    """The names of a feature's numbers and vectors, and the start every one of its numbers' names has, if any.

    The conversation brings its numbers, and the model adds how close each prompt is to its context and the two
    side by side. Benchmarks and time bring their numbers, and LLM analysis numbers whose names start with llm_,
    with its description's vector. The context, the skills and your own features are a vector or a number of
    their own name.
    """
    if name == "conversation":
        return frozenset({*CONVERSATION, CLOSENESS}), frozenset({JOINT}), None
    if name == LLM:
        return frozenset(), frozenset({LLM}), "llm_"
    if name == "benchmarks":
        return frozenset(benchmarks.FACTS), frozenset(), None
    if name == "time":
        return frozenset(features.TIME_NUMBERS), frozenset(), None
    return frozenset({name}), frozenset({name}), None


def without(rows: list[Features], name: str) -> list[Features]:
    """The rows without one feature: exactly the numbers and vectors it brings, as if it had never been chosen.

    The rows are as the pipeline builds them. The model works out the conversation's closeness number and the
    joint vector of the prompt and its context by itself, only from the conversation's numbers and both vectors,
    so rows without the conversation or the context never get them. Rows the model already read may hold them:
    without the conversation, they go too. The feature check and the model without LLM analysis both leave
    features out here.
    """
    numbers, vectors, start = _parts(name)
    return [
        replace(
            r,
            numbers={k: v for k, v in r.numbers.items() if k not in numbers and not (start and k.startswith(start))},
            vectors={k: v for k, v in r.vectors.items() if k not in vectors},
        )
        for r in rows
    ]


def has_feature(row: Features, name: str) -> bool:
    """Whether a row has a value for a feature: any of its numbers, or its vector."""
    numbers, vectors, start = _parts(name)
    return any(row.vectors.get(v) is not None for v in vectors) or any(
        k in numbers or bool(start and k.startswith(start)) for k in row.numbers
    )


# The first guess: a weighted sum of the whole embedding --------------------------------------------------


def _guess_inputs(rows: list[Features], fields: list[str], sizes: list[int]) -> np.ndarray:
    """Each row's embeddings side by side at length 1, each followed by a 1 when the row has it and a 0 when not."""
    parts = []
    for name, size in zip(fields, sizes):
        vectors, has = _unit([r.vectors.get(name) for r in rows], size)
        parts += [vectors, has[:, None].astype(np.float32)]
    return np.hstack(parts).astype(float)


class _Ridge:
    """Least squares with the weights pulled toward 0, solved once for any pull strength."""

    def __init__(self, x: np.ndarray) -> None:
        self.center = x.mean(axis=0)
        self.x = x - self.center
        # The smaller of the two square matrices is the quicker to split up: with more numbers per row than
        # rows, the rows' own one.
        self.by_column = self.x.shape[1] <= self.x.shape[0]
        gram = self.x.T @ self.x if self.by_column else self.x @ self.x.T
        self.values, self.vectors = np.linalg.eigh(gram)

    def fit(self, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
        """The weights and the constant: a row's guess is its numbers times the weights, plus the constant."""
        mean = float(y.mean())
        if self.by_column:
            weights = self.vectors @ ((self.vectors.T @ (self.x.T @ (y - mean))) / (self.values + alpha))
        else:
            weights = self.x.T @ (self.vectors @ ((self.vectors.T @ (y - mean)) / (self.values + alpha)))
        return weights, mean - float(self.center @ weights)


def _fit_guess(
    inputs: np.ndarray, results: np.ndarray, started: np.ndarray | None = None, finished: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A first guess of each outcome from the whole embedding.

    Training splits the tasks, oldest first, into GUESS_BLOCKS stretches of time. Each stretch's guesses come
    from weights fit only on tasks that had finished before it began, as a forecast's guess knows only earlier
    tasks. The first stretch has nothing before it, so it gets no guess. Returns those guesses, and the weights
    and constants fit on every task, for forecasts. An outcome with too few known values gets no guess.
    """
    n, outputs = len(inputs), results.shape[1]
    started = np.arange(n, dtype=float) if started is None else started
    finished = started if finished is None else finished
    edges = np.linspace(0, n, GUESS_BLOCKS + 1).astype(int)
    guesses = np.full((n, outputs), np.nan)
    weights, constants = np.zeros((inputs.shape[1], outputs)), np.full(outputs, np.nan)
    solved: dict[bytes, _Ridge] = {}

    def solver(rows: np.ndarray) -> _Ridge:
        key = rows.tobytes()
        if key not in solved:
            solved[key] = _Ridge(inputs[rows])
        return solved[key]

    for c in range(outputs):
        y = results[:, c].astype(float)
        known = ~np.isnan(y)
        if known.sum() < MIN_TASKS:
            continue
        stretches = []
        for low, high in zip(edges[1:-1], edges[2:]):
            earlier = known & (np.arange(n) < low) & (finished <= started[low])
            if high > low and earlier.sum() >= MIN_TASKS:
                stretches.append((earlier, slice(low, high)))
        best = None
        for alpha in GUESS_ALPHAS:
            guess = np.full(n, np.nan)
            for earlier, stretch in stretches:
                w, b = solver(earlier).fit(y[earlier], alpha)
                guess[stretch] = inputs[stretch] @ w + b
            scored = known & ~np.isnan(guess)
            error = float(np.mean((guess[scored] - y[scored]) ** 2)) if scored.any() else math.inf
            if best is None or error < best[0]:
                best = (error, alpha, guess)
        guesses[:, c] = best[2]
        weights[:, c], constants[c] = solver(known).fit(y[known], best[1])
    return guesses, weights, constants


def _has_context(rows: list[Features]) -> bool:
    return any(r.vectors.get("context") is not None for r in rows)


def _similar_features(
    tasks: list[Task], rows: list[Features], half_life_days: float | None = DEFAULTS["half_life_days"]
) -> dict[str, np.ndarray]:
    """For each embedded feature, what each task's most similar earlier tasks took, as similar_tasks gives.

    Only tasks that had finished when a task started count. So a test window can reuse these numbers for
    the tasks it learns from: they finished before the window opened, and their similar tasks before them.
    Up to NEIGHBOR_POOL tasks, the similar tasks are the same ones the window's tasks alone would find.
    The rows are as the model reads them, with the prompt and its context side by side when there's one.
    """
    results = outcomes(tasks)
    started, finished = np.array([t.started_at for t in tasks], dtype=float), finish_times(tasks)
    found: dict[str, np.ndarray] = {}
    for name in [name for name in dict.fromkeys(n for r in rows for n in r.vectors) if name]:
        size = len(next(r.vectors[name] for r in rows if r.vectors.get(name) is not None))
        queries, has = _unit([r.vectors.get(name) for r in rows], size)
        found[name] = similar_tasks(
            queries,
            results,
            queries,
            has,
            earlier_only=True,
            started=started,
            finished=finished,
            rich=name in RICH,
            half_life_days=half_life_days,
        )
    return found


# Starting points: the simple rule, pooled -------------------------------------------------------------


def _moments(values: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    """The weighted mean and variance of some values, and how many tasks they're worth.

    Recency weights make older tasks count less, so tasks count by their effective number: the square of
    their weights' sum over the sum of their squared weights. Tasks that weigh the same all count. Of many
    tasks where one carries nearly all the weight, about one counts.
    """
    mean = float(np.average(values, weights=weight))
    variance = float(np.average((values - mean) ** 2, weights=weight))
    return mean, variance, float(weight.sum() ** 2 / np.sum(weight * weight))


def _pulled(own: tuple[float, float, float], toward: tuple[float, float, float]) -> tuple[float, float, float]:
    """Moments pulled toward others, as if POOLING more tasks looked like the others.

    The variance is that of the real and the pretend tasks together, so it includes the gap between the means.
    """
    (mean, variance, n), (other_mean, other_variance, _) = own, toward
    pooled = (n * mean + POOLING * other_mean) / (n + POOLING)
    spread = n * (variance + (mean - pooled) ** 2) + POOLING * (other_variance + (other_mean - pooled) ** 2)
    return pooled, spread / (n + POOLING), n


def _pooled(models: list[str], efforts: list[str] | None, values: np.ndarray, weight: np.ndarray, kept) -> tuple:
    """Starting points for all models, for each model and for each model at each effort level it ran at.

    A model is pulled toward all models, and a model at one effort level toward that model.
    `kept` turns moments into what is kept.
    """
    every = _moments(values, weight)
    start, by_effort = {"*": kept(every)}, {}
    names, levels = np.array(models), None if efforts is None else np.array(efforts)
    for name in sorted(set(models)):
        rows = names == name
        own = _pulled(_moments(values[rows], weight[rows]), every)
        start[name] = kept(own)
        for level in [] if levels is None else sorted(set(levels[rows].tolist())):
            same = rows & (levels == level)
            by_effort.setdefault(name, {})[level] = kept(_pulled(_moments(values[same], weight[same]), own))
    return start, by_effort


def _pooled_curves(models: list[str], z: np.ndarray, weight: np.ndarray, family=None, efforts=None) -> tuple:
    """Each model's center and log width, pulled toward the curve of all models when it has few tasks.

    Also each model's at each effort level, pulled toward the model's own. No width is narrower than a
    forecast's can be.
    """
    family = family or FAMILY["lognormal"]

    def curve(moments) -> list[float]:
        center, width = family.from_moments(moments[0], math.sqrt(moments[1]))
        return [float(center), math.log(max(width, WIDTH_RANGE[0]))]

    return _pooled(models, efforts, z, weight, curve)


def _pooled_values(models: list[str], values: np.ndarray, weight: np.ndarray, log: bool, efforts=None) -> tuple:
    """Each model's average, pulled toward the average of all models when it has few tasks.

    Also each model's at each effort level, pulled toward the model's own.
    """
    return _pooled(models, efforts, values, weight, lambda m: math.log(max(m[0], 1e-12)) if log else m[0])


# The model ----------------------------------------------------------------------------------------------


class Model:
    def __init__(
        self,
        meta: dict,
        boosters: dict[str, xgb.Booster],
        pools: dict | None = None,
        guess: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.meta = meta
        self.boosters = boosters
        self.pools: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = pools or {}
        """For each embedded feature: the newest past tasks' embeddings at length 1, what they took and
        when they started."""
        self.guess = guess
        """The first guess's weights, one column per outcome, and its constants."""

    @property
    def settings(self) -> dict:
        return self.meta["settings"]

    @property
    def family(self):
        return FAMILY[self.settings["family"]]

    # Features into numbers

    def prepare(self, rows: list[Features]) -> list[Features]:
        """The rows as the model reads them: with the rest of the conversation's inputs, when it reads the
        conversation."""
        return _with_conversation(rows, self.meta.get("conversation", False), JOINT in self.meta["text_fields"])

    def matrix(
        self, rows: list[Features], similar: dict[str, np.ndarray] | None = None, guesses: np.ndarray | None = None
    ) -> np.ndarray:
        """The rows as numbers, one column per feature. Training passes what it found for each row; a forecast
        works it out from what the model saved. Rows come from prepare()."""
        categories = self.meta["categories"]
        columns = []
        for name in CATEGORIES:
            codes = {value: i for i, value in enumerate(categories[name])}
            columns.append([codes.get(getattr(r, name), np.nan) for r in rows])
        columns.append([math.log1p(max(r.prompt_chars, 0)) for r in rows])
        for name in self.meta["numbers"]:
            columns.append([r.numbers.get(name, np.nan) for r in rows])
        matrix = np.array(columns, dtype=float).T.reshape(len(rows), len(columns))
        now = time.time()
        started = np.array([now if r.started_at is None else r.started_at for r in rows], dtype=float)
        for name in self.meta["text_fields"]:
            if similar is not None:
                block = similar[name]
            else:
                pool, results, pool_started = self.pools[name]
                queries, has = _unit([r.vectors.get(name) for r in rows], pool.shape[1])
                rich = name in self.meta["rich_fields"]
                if rich and len(set(started.tolist())) > 1:
                    # Rows that start at different times, as in a test window, count each similar task's age
                    # from their own start.
                    block = similar_tasks(
                        pool,
                        results,
                        queries,
                        has,
                        earlier_only=False,
                        started=started,
                        rich=True,
                        pool_started=pool_started,
                        half_life_days=self.settings["half_life_days"],
                    )
                else:
                    # One prompt forecast on several AI models or effort levels brings the same vector many
                    # times. Each different vector is compared with the past tasks once.
                    unique, where = np.unique(queries, axis=0, return_inverse=True)
                    block = similar_tasks(
                        pool,
                        results,
                        unique,
                        np.ones(len(unique), dtype=bool),
                        earlier_only=False,
                        started=np.full(len(unique), started[0]) if len(started) else None,
                        rich=rich,
                        pool_started=pool_started,
                        half_life_days=self.settings["half_life_days"],
                    )
                    block = block[where.reshape(-1)]
                    block[~has] = np.nan
            matrix = np.hstack([matrix, block])
        spec = self.meta["guess"]
        if spec:
            if guesses is None:
                weights, constants = self.guess
                guesses = _guess_inputs(rows, spec["fields"], spec["sizes"]) @ weights + constants
            matrix = np.hstack([matrix, guesses])
        return matrix

    def dmatrix(self, matrix: np.ndarray, margin: np.ndarray, label=None, weight=None) -> xgb.DMatrix:
        types = ["c"] * len(CATEGORIES) + ["q"] * (matrix.shape[1] - len(CATEGORIES))
        data = xgb.DMatrix(matrix, label=label, weight=weight, feature_types=types, enable_categorical=True)
        data.set_base_margin(margin)
        return data

    def start(self, spec: dict, models: list[str], columns: int, efforts: list[str] | None = None) -> np.ndarray:
        """Each row's starting point: its model's at its effort level, its model's, or the value of all models."""
        start, by_effort = spec["start"], spec.get("effort_start") or {}
        found = []
        for m, effort in zip(models, efforts if efforts is not None else [None] * len(models)):
            value = by_effort.get(m, {}).get(effort)
            found.append(start.get(m, start["*"]) if value is None else value)
        values = np.array(found, dtype=float)
        return values.reshape(len(models), columns) if columns > 1 else values

    def _predict(self, key: str, spec: dict, matrix: np.ndarray, margin: np.ndarray) -> np.ndarray:
        if not spec.get("rounds"):
            return margin
        data = self.dmatrix(matrix, margin)
        return self.boosters[key].predict(data, output_margin=True, iteration_range=(0, spec["rounds"]))

    def raw_curves(self, matrix: np.ndarray, models: list[str], target: str, efforts: list[str] | None = None):
        """The chance of a direct answer and both parts' curves, before the width correction."""
        spec = self.meta["direct"]
        p = 1 / (1 + np.exp(-self._predict("direct", spec, matrix, _logit(self.start(spec, models, 1, efforts)))))
        parts = []
        for part in PARTS:
            key = f"{target}.{part}"
            spec = self.meta["parts"][key]
            raw = self._predict(key, spec, matrix, self.start(spec, models, 2, efforts)).reshape(len(models), 2)
            parts.append((raw[:, 0], np.exp(np.clip(raw[:, 1], *_LOG_WIDTHS))))
        return np.clip(p, 0.0, 1.0), parts[0], parts[1]

    def curves(self, rows: list[Features]) -> dict:
        """The two-part curve of each row, for every head. Widths include the final correction."""
        matrix = self.matrix(self.prepare(rows))
        models = [r.model for r in rows]
        out: dict = {"p_direct": None}
        for target in TARGETS:
            if f"{target}.work" not in self.meta.get("parts", {}):
                out[target] = None
                continue
            p, (mu1, s1), (mu2, s2) = self.raw_curves(matrix, models, target, [r.effort for r in rows])
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
        matrix = self.matrix(self.prepare(rows))
        models, efforts = [r.model for r in rows], [r.effort for r in rows]
        out: dict = {}
        for target in TARGETS:
            spec = self.meta.get("expected", {}).get(target)
            if spec is None:
                out[target] = None
                continue
            margin = self.start(spec, models, 1, efforts)
            out[target] = np.exp(np.clip(self._predict(target, spec, matrix, margin), -50, 50))
        return out

    def forecast(self, rows: list[Features]) -> list[dict]:
        """Each row's forecast: for every head, the expected value and, when there is one, the curve.

        This is what every model type provides. A head the model doesn't have is None, and so is a head
        the test never checked for the row's AI model.
        """
        if not rows:
            return []
        if self.settings["head"] == "expected":
            values = self.expected_values(rows)
            found = [
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
            return self._tested_only(rows, found)
        curves = self.curves(rows)
        family = self.family
        found = []
        for i in range(len(rows)):
            p = curves["p_direct"]
            row: dict = {"shape": f"two_part_{family.name}", "direct_probability": None if p is None else float(p[i])}
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
        return self._tested_only(rows, found)

    def _tested_only(self, rows: list[Features], found: list[dict]) -> list[dict]:
        """Set each head the test never checked for the row's AI model to None."""
        verdicts = self.meta.get("served", {})
        for row, result in zip(rows, found):
            tested = verdicts.get(row.model, {}).get("heads")
            if tested is None:  # no verdict for this AI model, or a model saved before verdicts named heads
                continue
            for target in TARGETS:
                if target not in tested:
                    result[target] = None
        return found

    def served(self, model: str, effort: str | None = None) -> tuple[bool, str | None]:
        """Whether forecasts for this AI model passed the test, and why not when they didn't.

        With an effort level, forecasts at a level the model never ran at aren't given: nothing was learned
        about it.
        """
        if model not in self.meta["categories"]["model"]:
            return False, "unknown_model"
        verdict = self.meta.get("served", {}).get(model)
        if verdict is None:
            return False, "not_enough_data"
        levels = self.meta.get("effort_counts", {}).get(model)  # a model saved by 0.2 doesn't count them
        if verdict["served"] and effort is not None and levels is not None and effort not in {e for e, _ in levels}:
            return False, "not_enough_data"
        return verdict["served"], verdict["reason"]

    def save(self, folder: Path) -> None:
        """Write the whole model to a new folder, then swap it in."""
        _swap_in(folder, self.write)

    def write(self, folder: Path) -> None:
        """Write the model's files into an empty folder."""
        for name, booster in self.boosters.items():
            booster.save_model(str(folder / f"{name}.json"))
        (folder / "model.json").write_text(json.dumps(self.meta, indent=1) + "\n")
        for name, (vectors, results, started) in self.pools.items():
            np.save(folder / f"similar-{name}-vectors.npy", vectors.astype(np.float32), allow_pickle=False)
            np.save(folder / f"similar-{name}-results.npy", results.astype(np.float32), allow_pickle=False)
            np.save(folder / f"similar-{name}-started.npy", started.astype(np.float64), allow_pickle=False)
        if self.guess is not None:
            np.save(folder / "guess-weights.npy", self.guess[0].astype(np.float64), allow_pickle=False)
            np.save(folder / "guess-constants.npy", self.guess[1].astype(np.float64), allow_pickle=False)


class WithoutLLM:
    """A model that reads LLM analysis, with the same model without it, for tasks LLM analysis couldn't read.

    It reads like one model: forecast(), served(), save() and a meta dict, as model_types describes. Its meta is
    the model's own. The model without LLM analysis has a test of its own, and a folder of its own inside the
    model's: WITHOUT_LLM.
    """

    def __init__(self, model: Model, without: Model) -> None:
        self.model = model
        self.without = without
        self.meta = model.meta

    @property
    def pools(self) -> dict:
        """Both models' similar-task pools. A forecast embeds a text only when one of them compares it."""
        return {**self.without.pools, **self.model.pools}

    def forecast(self, rows: list[Features]) -> list[dict]:
        """Each row's forecast: by the model, or by the one without LLM analysis when LLM analysis couldn't read it."""
        found: list = [None] * len(rows)
        read = [_read_by_llm(r) for r in rows]
        for model, reads in ((self.model, True), (self.without, False)):
            mine = [i for i, was in enumerate(read) if was == reads]
            if mine:
                for i, result in zip(mine, model.forecast([rows[i] for i in mine])):
                    found[i] = result
        return found

    def served(self, model: str, effort: str | None = None) -> tuple[bool, str | None]:
        return self.model.served(model, effort)

    def served_without_llm(self, model: str, effort: str | None = None) -> tuple[bool, str | None]:
        """Whether a task LLM analysis couldn't read is forecast on this AI model, and why not.

        The model without LLM analysis forecasts it, so its own test decides.
        """
        served, reason = self.served(model, effort)
        if not served and reason != "not_accurate_enough":
            return served, reason  # an unknown AI model, one with too few tasks, or a level it never ran at
        return self.without.served(model, effort)

    def save(self, folder: Path) -> None:
        def write(staging: Path) -> None:
            self.model.write(staging)
            (staging / WITHOUT_LLM).mkdir()
            self.without.write(staging / WITHOUT_LLM)

        _swap_in(folder, write)


def _swap_in(folder: Path, write) -> None:
    """Write a model into a new folder with `write`, then swap it in for the old one.

    The old model is only moved aside until the new one is in its place, so a failed save keeps it.
    """
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-", dir=folder.parent))
    old = staging.with_name(staging.name + "-old")
    try:
        write(staging)
        if folder.exists():
            folder.rename(old)
        staging.rename(folder)
    finally:
        if old.exists() and not folder.exists():
            old.rename(folder)  # the new model never arrived, so the old one goes back
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)


def load(folder: Path):
    """Load a saved model: a Model, or a WithoutLLM. Every file is plain JSON or plain numbers, so loading runs
    no code.

    A model the previous release saved loads as that release loaded it, so it forecasts as it did then.
    """
    meta = json.loads((folder / "model.json").read_text())
    saved = meta.get("format")
    if isinstance(saved, int) and saved > FORMAT:
        raise NewerFormat(
            "This predictor's model was saved by a newer version of yardstick-predict. "
            "Upgrade with: uv tool upgrade yardstick-predict"
        )
    if isinstance(saved, int) and saved == PREVIOUS_FORMAT:
        from . import model_0_3  # a model saved by 0.3. This goes when model_0_3 does.

        return model_0_3.load(folder, meta)
    if not isinstance(saved, int) or saved < FORMAT:
        raise OutdatedModel(
            "This predictor's model was trained by an older version of yardstick-predict. "
            "Train it again with: yardstick-predict refresh"
        )
    if (folder / WITHOUT_LLM).is_dir():
        return WithoutLLM(_load_one(folder), _load_one(folder / WITHOUT_LLM))
    return _load_one(folder)


def _load_one(folder: Path) -> Model:
    meta = json.loads((folder / "model.json").read_text())
    boosters = {}
    for path in folder.glob("*.json"):
        if path.name != "model.json":
            booster = xgb.Booster()
            booster.load_model(str(path))
            booster.set_param({"nthread": 1})  # a forecast has few rows, and threads cost more than they save
            boosters[path.stem] = booster
    pools = {}
    for name in meta.get("text_fields", []):
        pools[name] = (
            np.load(folder / f"similar-{name}-vectors.npy", allow_pickle=False),
            np.load(folder / f"similar-{name}-results.npy", allow_pickle=False),
            np.load(folder / f"similar-{name}-started.npy", allow_pickle=False),
        )
    guess = None
    if meta.get("guess"):
        guess = (
            np.load(folder / "guess-weights.npy", allow_pickle=False),
            np.load(folder / "guess-constants.npy", allow_pickle=False),
        )
    return Model(meta, boosters, pools, guess)


# Training -----------------------------------------------------------------------------------------------


def train(
    tasks: list[Task],
    rows: list[Features],
    settings: dict | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Model | WithoutLLM:
    """Test on the newest tasks, then train the final model on all of them. `rows` match `tasks` one to one.

    The model reads every feature the rows have. When LLM analysis read some of the tasks, the same model is
    also tested and trained without it, for a task LLM analysis couldn't read, and the two come as a WithoutLLM.
    `on_progress` is told how many of the models training fits are done, and how many there are: a trial
    for each test window and the final model, as fit_sizes counts them.
    """
    settings = settings_for(settings)
    eligible, eligible_rows = _eligible(tasks, rows, settings)
    read_by_llm = any(_read_by_llm(r) for r in eligible_rows)
    done, total = 0, len(fit_sizes(eligible, settings)) * (2 if read_by_llm else 1)

    def fitted() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(min(done, total), total)

    if on_progress is not None:
        on_progress(0, total)
    # The test windows and the final models share the similar-task numbers. With the conversation feature,
    # they're worked out with the prompt and its context side by side too.
    similar = _similar_features(eligible, _prepared(eligible_rows), settings["half_life_days"])
    model = _tested(eligible, eligible_rows, settings, similar, on_fit=fitted)
    if not read_by_llm:
        return model
    # A task that LLM analysis couldn't read, as when its AI model couldn't be reached, is forecast by the same
    # model without it. That one is tested too, so it gives forecasts only where it passed.
    return WithoutLLM(model, _tested(eligible, eligible_rows, settings, similar, llm=False, on_fit=fitted))


def _eligible(tasks: list[Task], rows: list[Features], settings: dict) -> tuple[list[Task], list[Features]]:
    """The tasks training learns from and tests on, oldest first, with their rows. Raises NotEnoughData when too few.

    Stopped tasks and tasks left waiting are left out. When any task has an embedded prompt, so are the ones
    without one. `settings` are filled in already, as settings_for gives them.
    """
    learnable = {t.id for t in usable(tasks)}
    pairs = sorted(((t, r) for t, r in zip(tasks, rows) if t.id in learnable), key=lambda p: (p[0].started_at, p[0].id))
    if any(r.vectors.get("prompt") is not None for _, r in pairs):
        pairs = [(t, r) for t, r in pairs if r.vectors.get("prompt") is not None]
    if len(pairs) < MIN_TASKS:
        raise NotEnoughData(f"Training needs at least {MIN_TASKS} finished tasks. This dataset has {len(pairs)}.")
    eligible = [t for t, _ in pairs]
    # A head with too few known values is left out. With no head left, there is nothing to learn.
    known = max(sum(target_value(t, head) is not None for t in eligible) for head in settings["heads"])
    if known < MIN_TASKS:
        raise NotEnoughData(
            f"Training needs at least {MIN_TASKS} tasks with a known value for the "
            f"{' or '.join(settings['heads'])} head. This dataset has {known}."
        )
    return eligible, [r for _, r in pairs]


def _tested(
    tasks: list[Task],
    rows: list[Features],
    settings: dict,
    similar: dict[str, np.ndarray],
    llm: bool = True,
    on_fit: Callable[[], None] | None = None,
) -> Model:
    """The final model, trained on all the tasks, with the verdicts of its test on the newest ones.

    `on_fit` is called each time a model is fitted: each trial of the test, then the final model.
    """
    report = test(tasks, rows, settings, similar, llm, on_trial=on_fit)
    model = fit(tasks, rows, settings, similar, llm)
    if on_fit is not None:
        on_fit()
    served = {
        m: {"served": r["served"], "reason": r["reason"], "heads": r["heads"]} for m, r in report["models"].items()
    }
    model.meta.update(report=report, served=served, trained_at=time.time(), tasks=len(tasks))
    return model


def _defaults(tasks: list[Task], weight: np.ndarray | None = None) -> dict:
    """What to assume for a forecast request that leaves a feature out, per AI model.

    Each task counts by its recency weight, so it's what the model ran with lately, not years ago.
    """
    weight = np.ones(len(tasks)) if weight is None else weight
    by_model: dict[str, dict[str, Counter]] = {}
    for t, w in zip(tasks, weight):
        counts = by_model.setdefault(t.model, {"effort": Counter(), "agent": Counter()})
        counts["effort"][t.effort] += float(w)
        counts["agent"][t.agent] += float(w)
    return {
        model: {name: found.most_common(1)[0][0] for name, found in counts.items()}
        for model, counts in by_model.items()
    }


def _effort_counts(tasks: list[Task]) -> dict[str, list[list]]:
    """How many tasks each AI model ran at each effort level, the most used first."""
    found: dict[str, list[list]] = {}
    for (model, effort), count in sorted(
        Counter((t.model, t.effort) for t in tasks).items(), key=lambda kv: (-kv[1], kv[0])
    ):
        found.setdefault(model, []).append([effort, count])
    return found


def _params(settings: dict) -> dict:
    params = dict(TREES)
    if settings["dropout"] > 0:
        # XGBoost 3.4 drops trees, as DART does, on its tree booster itself. Its booster="dart" is only an
        # old name for the same thing, kept with a warning.
        params.update(rate_drop=float(settings["dropout"]), skip_drop=0.0)
    return params


def _threads(rows: int) -> int:
    """How many threads XGBoost uses on this many rows. The results are the same either way."""
    return 1 if rows < THREADS_FROM else min(4, os.cpu_count() or 1)


def fit(
    tasks: list[Task],
    rows: list[Features],
    settings: dict,
    similar: dict[str, np.ndarray] | None = None,
    llm: bool = True,
) -> Model:
    """Fit every head on the tasks, oldest first.

    The newest tasks pick each part's number of rounds and the width correction. They judge the trees
    against starting points fitted without them, since the trees haven't seen them either. Then every
    starting point is fitted again and every part refit, on all the tasks, with that number of rounds.

    `similar` is what _similar_features gives for these tasks, when it was worked out before. The model
    reads the conversation's inputs when the rows come with the conversation feature. Without `llm`, it
    leaves out what LLM analysis said.
    """
    settings = settings_for(settings)
    conversation = _reads_conversation(rows)
    rows = _prepared(rows)
    if not llm:
        rows = _without_llm(rows)
    text_fields = [name for name in dict.fromkeys(n for r in rows for n in r.vectors) if name]
    numbers = sorted({n for r in rows for n in r.numbers})
    # The first guess reads every embedding the tasks really have. The joint one only repeats two of them.
    guess_fields = [name for name in text_fields if name != JOINT]
    sizes = [len(next(r.vectors[name] for r in rows if r.vectors.get(name) is not None)) for name in guess_fields]
    weight = recency_weights(tasks, settings["half_life_days"])
    meta: dict = {
        "format": FORMAT,
        "settings": settings,
        "categories": {
            name: [v for v, _ in Counter(getattr(t, name) for t in tasks).most_common()] for name in CATEGORIES
        },
        "numbers": numbers,
        "text_fields": text_fields,
        "rich_fields": [name for name in text_fields if name in RICH],
        "conversation": conversation,
        "llm": any(_read_by_llm(r) for r in rows),
        "guess": {"fields": guess_fields, "sizes": sizes, "outputs": list(GUESSES)} if guess_fields else None,
        "parts": {},
        "expected": {},
        "width_scale": {},
        "defaults": _defaults(tasks, weight),
        "effort_counts": _effort_counts(tasks),
    }
    model = Model(meta, {})
    if similar is None:
        similar = _similar_features(tasks, rows, settings["half_life_days"])
    results = outcomes(tasks)
    started, finished = np.array([t.started_at for t in tasks], dtype=float), finish_times(tasks)
    for name in text_fields:
        size = len(next(r.vectors[name] for r in rows if r.vectors.get(name) is not None))
        queries, has = _unit([r.vectors.get(name) for r in rows], size)
        newest = np.flatnonzero(has)[-NEIGHBOR_POOL:]
        model.pools[name] = (queries[newest], results[newest], started[newest])
    guesses = None
    if guess_fields:
        guesses, weights, constants = _fit_guess(_guess_inputs(rows, guess_fields, sizes), results, started, finished)
        model.guess = (weights, constants)
    matrix = model.matrix(rows, similar, guesses)
    models, efforts = [t.model for t in tasks], [t.effort for t in tasks]
    held = max(int(len(tasks) * settings["validation_share"]), 10)
    fit_rows = np.arange(len(tasks) - held)
    valid_rows = np.arange(len(tasks) - held, len(tasks))
    values = {target: _known_values(tasks, target) for target in settings["heads"]}
    heads = [target for target in settings["heads"] if (~np.isnan(values[target])).sum() >= MIN_TASKS]
    if settings["head"] == "expected":
        meta["expected"] = {target: {} for target in heads}
    else:
        meta["direct"] = {}
        meta["parts"] = {f"{target}.{part}": {} for target in heads for part in PARTS}
    _set_starts(model, tasks, weight, fit_rows)
    if settings["head"] == "expected":
        for target in heads:
            known = ~np.isnan(values[target])
            _grow_mean(model, target, matrix, models, values[target], known, weight, fit_rows, valid_rows, efforts)
    else:
        direct = np.array([t.direct for t in tasks], dtype=float)
        _grow_direct(model, matrix, models, direct, weight, fit_rows, valid_rows, efforts)
        for target in heads:
            known = ~np.isnan(values[target])
            z = np.log(np.where(known, values[target], 1.0))
            for part, flag in (("direct", 1.0), ("work", 0.0)):
                mask = known & (direct == flag)
                _grow_part(model, f"{target}.{part}", matrix, models, z, weight, mask, fit_rows, valid_rows, efforts)
            _fit_width_scale(model, target, matrix, models, z, known, valid_rows, efforts)
    _refit(model, tasks, matrix, models, weight)
    return model


def _pooled_shares(models: list[str], direct: np.ndarray, weight: np.ndarray, efforts=None) -> tuple:
    return _pooled_values(models, direct, weight, log=False, efforts=efforts)


def _set_starts(model: Model, tasks: list[Task], weight: np.ndarray, rows: np.ndarray) -> None:
    """Fit every starting point on `rows`: the older tasks while trees grow, then every task for the refit.

    A part with fewer than 2 of these rows starts from the curve of its whole head. Only when that has
    fewer than 2 too does it use the other rows.
    """
    meta, n = model.meta, len(tasks)
    models, efforts = np.array([t.model for t in tasks]), np.array([t.effort for t in tasks])
    direct = np.array([t.direct for t in tasks], dtype=float)
    chosen = np.zeros(n, dtype=bool)
    chosen[rows] = True

    def pick(*masks: np.ndarray) -> np.ndarray:
        for mask in [m & chosen for m in masks] + list(masks):
            if mask.sum() >= 2:
                return np.flatnonzero(mask)
        return np.flatnonzero(masks[-1])

    if "direct" in meta:
        use = pick(np.ones(n, dtype=bool))
        start, by_effort = _pooled_shares(models[use].tolist(), direct[use], weight[use], efforts[use].tolist())
        meta["direct"].update(start=start, effort_start=by_effort)
    for key, spec in meta["parts"].items():
        target, part = key.split(".")
        values = _known_values(tasks, target)
        known = ~np.isnan(values)
        use = pick(known & (direct == (1.0 if part == "direct" else 0.0)), known)
        z = np.log(values[use])
        start, by_effort = _pooled_curves(models[use].tolist(), z, weight[use], model.family, efforts[use].tolist())
        spec.update(start=start, effort_start=by_effort)
    for target, spec in meta["expected"].items():
        values = _known_values(tasks, target)
        use = pick(~np.isnan(values))
        start, by_effort = _pooled_values(models[use].tolist(), values[use], weight[use], True, efforts[use].tolist())
        spec.update(start=start, effort_start=by_effort)


def _useful_rounds(booster: xgb.Booster, start_score: float) -> int:
    """How many rounds to keep: zero when no number of trees beat the starting point on the newest tasks."""
    return booster.best_iteration + 1 if booster.best_score < start_score else 0


def _grow(model, key, params, matrix, margin, label, weight, grow, check, objective, metric, start_score):
    def data(idx, weighted):
        return model.dmatrix(matrix[idx], margin[idx], label=label[idx], weight=weight[idx] if weighted else None)

    params = {**params, "disable_default_eval_metric": 1, "nthread": _threads(len(grow))}
    booster = xgb.train(
        params,
        data(grow, True),
        MAX_DROPOUT_ROUNDS if params.get("rate_drop") else MAX_ROUNDS,
        obj=objective,
        custom_metric=metric,
        evals=[(data(check, False), "valid")],
        early_stopping_rounds=PATIENCE,
        verbose_eval=False,
    )
    rounds = _useful_rounds(booster, start_score)
    if rounds and params.get("rate_drop") and booster.num_boosted_rounds() > rounds:
        # With dropout, each round reweighs trees grown before it. So the first trees of a longer run aren't
        # the model early stopping picked. Growing exactly that many rounds again gives that model back.
        booster = xgb.train(params, data(grow, True), rounds, obj=objective)
    return booster, rounds


def _grow_direct(model, matrix, models, direct, weight, fit_rows, valid_rows, efforts=None) -> None:
    spec = model.meta["direct"]
    if min(direct[fit_rows].sum(), len(fit_rows) - direct[fit_rows].sum()) < 10:
        return
    margin = _logit(model.start(spec, models, 1, efforts))

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


def _grow_part(model, key, matrix, models, z, weight, mask, fit_rows, valid_rows, efforts=None) -> None:
    spec = model.meta["parts"][key]
    grow, check = fit_rows[mask[fit_rows]], valid_rows[mask[valid_rows]]
    if len(grow) < MIN_PART_TASKS or len(check) < 5:
        return
    family, loss = model.family, model.settings["loss"]
    margin = model.start(spec, models, 2, efforts)
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


def _grow_mean(model, target, matrix, models, values, known, weight, fit_rows, valid_rows, efforts=None) -> None:
    loss = model.settings["loss"]
    spec = model.meta["expected"][target]
    grow, check = fit_rows[known[fit_rows]], valid_rows[known[valid_rows]]
    if len(grow) < MIN_PART_TASKS or len(check) < 5:
        return
    margin = model.start(spec, models, 1, efforts)
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


def _fit_width_scale(model, target, matrix, models, z, known, valid_rows, efforts=None) -> None:
    """A factor that widens or narrows a target's curves, picked on the newest tasks.

    Each AI model with enough of the newest tasks gets its own factor. The rest share one. This runs
    before the refit, so neither the trees nor their starting points have seen these tasks. The factor
    scores best by the loss the trees learn by: the log score, or CRPS.
    """
    scales = model.meta["width_scale"][target] = {"*": 1.0}
    rows = valid_rows[known[valid_rows]]
    if len(rows) < 10:
        return
    names = np.array([models[i] for i in rows])
    levels = None if efforts is None else [efforts[i] for i in rows]
    w, (mu1, s1), (mu2, s2) = model.raw_curves(matrix[rows], names.tolist(), target, levels)
    grid = np.geomspace(0.5, 3.0, 46)
    family = model.family
    score = crps if model.settings["loss"] == "crps" else log_score

    def best(mask: np.ndarray) -> float:
        scores = [
            float(
                np.mean(
                    score(
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
    """Fit every starting point again on all the tasks, then refit every grown part on top of them.

    Each part gets the number of rounds the newest tasks picked.
    """
    everything = np.arange(len(tasks))
    _set_starts(model, tasks, weight, everything)
    efforts = [t.effort for t in tasks]
    loss, family = model.settings["loss"], model.family
    direct = np.array([t.direct for t in tasks], dtype=float)
    if model.meta.get("direct", {}).get("rounds"):
        spec = model.meta["direct"]
        margin = _logit(model.start(spec, models, 1, efforts))

        def objective(predt, data):
            p = 1 / (1 + np.exp(-predt))
            return _weighted(p - data.get_label(), np.maximum(p * (1 - p), 1e-6), data)

        data = model.dmatrix(matrix, margin, label=direct, weight=weight)
        params = {**spec["params"], "nthread": _threads(len(tasks))}
        model.boosters["direct"] = xgb.train(params, data, spec["rounds"], obj=objective)
    for key, spec in model.meta["parts"].items():
        if not spec.get("rounds"):
            continue
        target, part = key.split(".")
        values = _known_values(tasks, target)
        rows = everything[~np.isnan(values) & (direct == (1.0 if part == "direct" else 0.0))]
        z = np.log(values[rows])
        margin = model.start(spec, [models[i] for i in rows], 2, [efforts[i] for i in rows])
        data = model.dmatrix(matrix[rows], margin, label=np.column_stack([z, z]), weight=weight[rows])
        params = {**spec["params"], "nthread": _threads(len(rows))}
        model.boosters[key] = xgb.train(params, data, spec["rounds"], obj=_curve_objective(family, loss))
    for target, spec in model.meta["expected"].items():
        if not spec.get("rounds"):
            continue
        values = _known_values(tasks, target)
        rows = everything[~np.isnan(values)]
        margin = model.start(spec, [models[i] for i in rows], 1, [efforts[i] for i in rows])
        data = model.dmatrix(matrix[rows], margin, label=values[rows], weight=weight[rows])
        params = {**spec["params"], "nthread": _threads(len(rows))}
        model.boosters[target] = xgb.train(params, data, spec["rounds"], obj=_mean_objective(loss))


# Testing ------------------------------------------------------------------------------------------------


def simple_rule(train_tasks: list[Task], target: str) -> dict:
    """The same two-part curve for every task on an AI model: the spread of its past tasks."""

    def curve(group: list[Task]) -> dict | None:
        pairs = [(t.direct, math.log(v), v) for t in group if (v := target_value(t, target)) is not None]
        if len(pairs) < 2:
            return None
        flags = np.array([d for d, _, _ in pairs], dtype=float)
        z = np.array([v for _, v, _ in pairs])
        # Its widths stay inside the same range as the model's, so neither wins on a width the other can't have.
        whole = (float(z.mean()), float(np.clip(z.std(), *WIDTH_RANGE)))
        parts = {}
        for part, flag in (("direct", 1.0), ("work", 0.0)):
            zz = z[flags == flag]
            parts[part] = (float(zz.mean()), float(np.clip(zz.std(), *WIDTH_RANGE))) if len(zz) >= 2 else whole
        return {"p": float(flags.mean()), "mean": float(np.mean([v for _, _, v in pairs])), **parts}

    rules = {"*": curve(train_tasks)}
    for model in {t.model for t in train_tasks}:
        rules[model] = curve([t for t in train_tasks if t.model == model]) or rules["*"]
    return rules


def min_tasks_to_test(settings: dict | None = None) -> int:
    """The fewest tasks an AI model needs before the test can judge it.

    Each AI model is tested on at least MIN_TEST_TASKS of its own newest tasks, with at least as many of its
    tasks before them to learn from, whatever share the settings test.
    """
    settings_for(settings)
    return 2 * MIN_TEST_TASKS


def _test_windows(tasks: list[Task], settings: dict) -> list[tuple[str, list[int]]]:
    """Each AI model's test windows, as the positions of its tasks, the windows opening earliest first.

    An AI model is tested on its own newest tasks: the test share of them, and at least MIN_TEST_TASKS, as
    long as at least MIN_TEST_TASKS of its tasks stay before them. So a model used mostly long ago is still
    tested on its newest tasks, however many tasks on other models came after. They're split into windows,
    oldest first, and the last window also takes any left over.
    """
    by_model: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        by_model.setdefault(t.model, []).append(i)
    found = []
    for model, positions in by_model.items():
        size = min(max(int(len(positions) * settings["test_share"]), MIN_TEST_TASKS), len(positions) - MIN_TEST_TASKS)
        if size < MIN_TEST_TASKS:
            continue
        newest, count = positions[-size:], settings["test_windows"]
        each = max(size // count, 1)
        for k in range(count):
            part = newest[k * each : (k + 1) * each] if k < count - 1 else newest[k * each :]
            if part:
                found.append((model, part))
    return sorted(found, key=lambda window: (window[1][0], window[0]))


def _runs(tasks: list[Task], windows: list[tuple[str, list[int]]]) -> list[tuple[list[int], list[int]]]:
    """Each test window that runs, with the positions of the tasks known when it opens: the ones that had finished.

    A task that started earlier but was still running then hadn't shown its outcome yet. A window with fewer
    than MIN_TASKS known doesn't run.
    """
    finished = finish_times(tasks)
    found = []
    for _, part in windows:
        opens = part[0]
        known = [i for i in range(opens) if finished[i] <= tasks[opens].started_at]
        if len(known) >= MIN_TASKS:
            found.append((part, known))
    return found


def fit_sizes(tasks: list[Task], settings: dict | None = None) -> list[int]:
    """How many tasks each model that training fits learns from, in order: a trial for each test window that
    runs, then the final model, on every task. Windows that open with the same tasks known share one trial.

    With LLM analysis, training fits each of them twice, the second time without it. The progress bar counts
    them, and the estimate before setup and refresh adds them up.
    """
    settings = settings_for(settings)
    tasks = usable(tasks)
    sizes: list[int] = []
    for _, known in _runs(tasks, _test_windows(tasks, settings)):
        if not sizes or sizes[-1] != len(known):
            sizes.append(len(known))
    return [*sizes, len(tasks)]


def test(
    tasks: list[Task],
    rows: list[Features],
    settings: dict,
    similar: dict[str, np.ndarray] | None = None,
    llm: bool = True,
    on_trial: Callable[[], None] | None = None,
) -> dict:
    """Forecast each AI model's newest tasks window by window, each from the tasks before it, and judge each model.

    A window learns from the tasks on every AI model that had finished when it opens. Only the window's own
    model's tasks are forecast in it. `similar` is what _similar_features gives for all the tasks, when it was
    worked out before. Without `llm`, the model tested leaves out what LLM analysis said, as fit does.
    `on_trial` is called each time a trial model is fitted.
    """
    settings = settings_for(settings)
    least = min_tasks_to_test(settings)
    windows = _test_windows(tasks, settings)
    if not windows:
        return {
            "trained_on": len(tasks),
            "tested_on": 0,
            "windows": 0,
            "models": {},
            "pit": {},
            "min_tasks_to_test": least,
        }
    if similar is None:
        similar = _similar_features(tasks, _prepared(rows), settings["half_life_days"])
    scored, learned_from, tested_on = [], [], 0
    last: tuple[int, Model, list[Task]] | None = None
    for part, known in _runs(tasks, windows):
        # Windows of two AI models that open with the same tasks known share one trial. The tasks known only
        # grow as windows open later, so the same number of them means the same tasks.
        if last is None or last[0] != len(known):
            train_tasks, train_rows = [tasks[i] for i in known], [rows[i] for i in known]
            # A task's similar tasks had all finished before it started, so before this window opened too.
            sliced = {name: found[known] for name, found in similar.items()}
            last = (len(known), fit(train_tasks, train_rows, settings, sliced, llm), train_tasks)
            if on_trial is not None:
                on_trial()
        _, trial, train_tasks = last
        scored.append(_score_window(trial, train_tasks, [tasks[i] for i in part], [rows[i] for i in part]))
        learned_from.append(len(known))
        tested_on += len(part)
    seen = sorted({t.model for t in tasks})
    # A window with too few finished tasks before it never ran, so its tasks weren't tested.
    trained_on = min(learned_from) if learned_from else len(tasks)
    report = _verdicts(scored, trained_on=trained_on, tested_on=tested_on, settings=settings, seen=seen)
    counts = Counter(t.model for t in tasks)
    for model, result in report["models"].items():
        result["tasks"] = counts[model]  # so a model with too few tasks can say how many it has
    report["min_tasks_to_test"] = least
    report["thresholds"] = {
        "min_test_tasks": MIN_TEST_TASKS,
        "tolerance": TOLERANCE,
        "min_range_held": MIN_RANGE_HELD,
        "totals_range": list(TOTALS_RANGE),
    }
    return report


def compare(
    tasks: list[Task],
    rows: list[Features],
    settings: dict | None = None,
    names: list[str] | tuple[str, ...] = (),
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, dict], dict[str, dict[str, float]]]:
    """The test training runs, with every feature the rows have, then again without each feature in `names`.

    It only tests: no final model is fitted. Returns each test's report by the feature it left out, with "" for
    the test with every feature. Also, for each feature in `names`, the share of each AI model's test tasks that
    have it. The similar-task numbers are worked out once for every test, since leaving a feature out only drops
    its own. A feature no task has would test the same as every feature, so it isn't tested again.
    `on_progress` is told how many trial models are fitted, and how many there are in all.
    """
    settings = settings_for(settings)
    tasks, rows = _eligible(tasks, rows, settings)
    present = [name for name in names if any(has_feature(r, name) for r in rows)]
    total = (len(fit_sizes(tasks, settings)) - 1) * (1 + len(present))
    done = 0

    def tried() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(min(done, total), total)

    if on_progress is not None and total:
        on_progress(0, total)  # the bar starts with the step, so it can tell how fast it goes
    similar = _similar_features(tasks, _prepared(rows), settings["half_life_days"])
    reports = {"": test(tasks, rows, settings, similar, on_trial=tried)}
    for name in present:
        reports[name] = test(tasks, without(rows, name), settings, similar, on_trial=tried)
    for name in names:
        reports.setdefault(name, reports[""])  # no task has it, so the test without it is the same
    tested: dict[str, list[int]] = {}
    for part, _ in _runs(tasks, _test_windows(tasks, settings)):
        for i in part:
            tested.setdefault(tasks[i].model, []).append(i)
    shares = {
        name: {model: float(np.mean([has_feature(rows[i], name) for i in found])) for model, found in tested.items()}
        for name in names
    }
    return reports, shares


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
        values = _known_values(test_tasks, target)
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


def _verdicts(
    scored: list[dict], trained_on: int, tested_on: int, settings: dict, seen: list[str] | None = None
) -> dict:
    """Pool the windows and judge each AI model: give its forecasts, or say why not.

    An AI model in `seen` that the windows never tested, such as one used only long ago, gets no forecasts.
    """
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
            if rows.sum() < MIN_TEST_TASKS:
                continue  # too few known values to judge this head, such as cost on a model with no listed price
            chance = data["pit"][rows]
            has_ranges = not np.isnan(chance).all()
            ours, simple = float(data["ours"][rows].mean()), float(data["simple"][rows].mean())
            result[target] = {
                "score": round(ours, 4),
                "simple_rule_score": round(simple, 4),
                # On a target that never changes, the simple rule's expected value is exactly right, so its
                # score is 0. Dividing by 0 gives NaN, so its score counts as at least 0.0001, the smallest shown.
                "better_than_simple_rule": round((simple - ours) / max(simple, 1e-4), 4),
                "in_50_range": round(float(np.mean((chance > 0.25) & (chance < 0.75))), 4) if has_ranges else None,
                "in_90_range": round(float(np.mean((chance > 0.05) & (chance < 0.95))), 4) if has_ranges else None,
                "expected_over_real_total": round(float(data["expected"][rows].sum() / data["real"][rows].sum()), 4),
                "tasks_tested": int(rows.sum()),
            }
        # Only heads with test results were checked. A head that wasn't tested is never given.
        tested = [target for target in TARGETS if target in result]
        failed = []
        for target in tested:
            r = result[target]
            if r["score"] > r["simple_rule_score"] * (1 + TOLERANCE):
                failed.append(f"{target}.simple_rule")
            if r["in_90_range"] is not None and r["in_90_range"] < MIN_RANGE_HELD:
                failed.append(f"{target}.range_held")
            if not TOTALS_RANGE[0] <= r["expected_over_real_total"] <= TOTALS_RANGE[1]:
                failed.append(f"{target}.totals")
        if result["tasks_tested"] < MIN_TEST_TASKS or not tested:
            served, reason = False, "not_enough_data"
        elif failed:
            served, reason = False, "not_accurate_enough"
        else:
            served, reason = True, None
        result["failed_checks"] = failed if reason == "not_accurate_enough" else []
        report["models"][model] = {"served": served, "reason": reason, "heads": tested, **result}
    for model in seen or []:
        report["models"].setdefault(
            model, {"served": False, "reason": "not_enough_data", "heads": [], "tasks_tested": 0, "failed_checks": []}
        )
    return report
