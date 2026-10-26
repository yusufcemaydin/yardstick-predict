"""How the screens name your models and say when each was last updated.

Every screen calls a predictor a model, as people do. The models a forecast is about, such as
claude-opus-5-5, are AI models. The default model has (default) after its name in every list.
"""

from __future__ import annotations

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


def learns_from(config: dict) -> str:
    """The features a model learns from, in plain words, the prompt first."""
    names = [name for name in config.get("features") or ["prompt"] if isinstance(name, str)]
    ordered = ["prompt", *(name for name in names if name != "prompt")]
    words = [LEARNS_FROM.get(name, name) for name in dict.fromkeys(ordered)]
    return ", ".join(words)
