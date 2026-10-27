"""How the screens name your models, say when each was last updated and how accurate each one is.

Every screen calls a predictor a model, as people do. The models a forecast is about, such as
claude-opus-5-5, are AI models. The default model has (default) after its name in every list.
"""

from __future__ import annotations

import math

from ..store import BrokenPredictor, Predictor, default_name, predictor_names

LEARNS_FROM = {
    "prompt": "prompt text",
    "conversation": "conversation",
    "context": "context",
    "skills": "skills",
    "benchmarks": "model facts",
    "time": "time",
    "llm": "LLM analysis",
}
"""What each feature is called on a model's screen."""

TITLES = {"claude-code": "Claude Code", "codex": "Codex"}
"""What the usual archives of records are called on screen. Another folder of records goes by its own name."""


def title(archive: dict) -> str:
    """What an archive of records is called on screen."""
    return TITLES.get(archive["name"], archive["name"]) if archive["name"] == archive["format"] else archive["name"]


def label(name: str, default: str | None = None) -> str:
    """A model's name as lists show it, with (default) after the default model's."""
    default = default_name() if default is None else default
    return f"{name} (default)" if name == default else name


def updated(name: str) -> str:
    """When the model last learned from your work, in a few words."""
    from ..cli import ago

    try:
        moment = Predictor(name).config.get("refreshed_at")
    except BrokenPredictor:
        return "can't be used"
    return f"updated {ago(moment)}" if moment else "not trained yet"


def models() -> list[tuple[str, str]]:
    """Each model's name as lists show it, and when it was last updated, in name order."""
    default = default_name()
    return [(label(name, default), updated(name)) for name in predictor_names()]


ABOUT_AS_ACCURATE = 0.02
"""Forecasts that come within 2% of the default's are about as accurate as its."""
TARGETS = ("seconds", "tokens", "usd")


def accuracy(names: list[str], default: str | None) -> tuple[dict[str, str], bool]:
    """How accurate each model's forecasts are next to the default's, in a few words, by name, such as 12% more
    accurate than main. A model with no forecasts yet says why. Nothing is said of the default itself, of a model
    that isn't trained or can't be read, or of any model when the default gives no forecasts. Also whether any
    model was compared with the default.

    Accuracy is how close the forecasts came on each AI model's newest tasks, as if those hadn't happened yet, as
    training found it. Only the AI models both models forecast count."""
    from ..cli import trained_facts
    from ..pipeline import model_meta, outdated

    said: dict[str, str] = {}
    found: dict[str, tuple[dict, list[str]]] = {}
    for name in names:
        try:
            predictor = Predictor(name)
        except BrokenPredictor:
            continue
        meta = model_meta(predictor)
        if meta is None:
            continue  # not trained yet, as the time it was updated says
        served = trained_facts(predictor, meta)["forecasts_for"]
        if served:
            found[name] = (meta.get("report") or {}, served)
        else:
            said[name] = "needs training" if outdated(predictor, meta) else "not enough data yet"
    if default not in found:
        return said, False
    base, theirs = found[default]
    any_compared = False
    for name, (report, served) in found.items():
        if name != default:
            ratio = error_ratio(report, base, [model for model in served if model in theirs])
            said[name] = f"no AI model in common with {default}" if ratio is None else compared(ratio, default)
            any_compared = any_compared or ratio is not None
    return said, any_compared


def error_ratio(report: dict, base: dict, models: list[str]) -> float | None:
    """How far off one model's forecasts came next to another's, on these AI models: under 1 when they came closer.
    None when there's no score to compare.

    Each score is first taken next to the simple rule's on the same tasks. So two models scored on different tasks,
    as when one learned from more of them, still compare fairly. On the same tasks, it's the ratio of their scores.
    Time, tokens and cost each count as much as the tasks they were scored on."""
    total = weight = 0.0
    for model in models:
        for target in TARGETS:
            mine, theirs = _next_to_simple_rule(report, model, target), _next_to_simple_rule(base, model, target)
            if mine is not None and theirs is not None:
                count = min(mine[1], theirs[1])
                total += count * math.log(mine[0] / theirs[0])
                weight += count
    return math.exp(total / weight) if weight else None


def _next_to_simple_rule(report: dict, model: str, target: str) -> tuple[float, int] | None:
    """An AI model's score for a target over the simple rule's, with how many tasks were scored. Lower is closer.
    None when the report has no such score."""
    found = ((report.get("models") or {}).get(model) or {}).get(target)
    if not isinstance(found, dict):
        return None
    numbers = score, simple, count = tuple(found.get(key) for key in ("score", "simple_rule_score", "tasks_tested"))
    if not all(isinstance(value, int | float) and not isinstance(value, bool) and value > 0 for value in numbers):
        return None
    return score / simple, int(count)


def compared(ratio: float, default: str) -> str:
    """How accurate a model is next to the default, in a few words, from how far off its forecasts came next to
    the default's."""
    if ratio >= 2:
        return f"far less accurate than {default}"
    if ratio > 1 + ABOUT_AS_ACCURATE:
        return f"{ratio - 1:.0%} less accurate than {default}"
    if ratio < 1 - ABOUT_AS_ACCURATE:
        return f"{1 - ratio:.0%} more accurate than {default}"
    return f"about as accurate as {default}"


def learns_from(config: dict) -> str:
    """The features a model learns from, in plain words, the prompt first."""
    names = [name for name in config.get("features") or ["prompt"] if isinstance(name, str)]
    ordered = ["prompt", *(name for name in names if name != "prompt")]
    words = [LEARNS_FROM.get(name, name) for name in dict.fromkeys(ordered)]
    return ", ".join(words)
