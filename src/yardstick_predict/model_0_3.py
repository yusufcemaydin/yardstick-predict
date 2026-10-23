"""Models saved by 0.3: how this version loads them and forecasts with them, exactly as 0.3 did.

Each version keeps forecasting with the models the previous release saved, until a refresh trains them again.
0.3 saved its models in format 3, model_types.PREVIOUS_FORMAT. Remove this module, and the few lines that
use it, when the previous release no longer saves format 3.

0.3 trained versions of the model: with the conversation's inputs or without them, and with LLM analysis
or without it. They're called conversation and plain, and conversation-llm and plain-llm. Each AI model
used the version that did best on its test. A task LLM analysis couldn't read was forecast by the same
version without LLM analysis, which had its own verdict. When every AI model used one version, 0.3 saved
just that one, as a single model.

Each version is a Model as this version knows it, and reads a row as 0.3 did:

- It reads only the numbers and embeddings it names in its own model.json. Their names haven't changed
  since 0.3, and this version's rows carry them all. A predictor 0.3 saved reads the conversation, because
  its settings get the conversation feature when they're read.
- A version with the conversation adds how close the prompt is to its context, and the two side by side,
  as 0.3 did.
- 0.3 also took the conversation's numbers out of the rows a version without the conversation read. That
  version never names them, so leaving them in changes nothing.
- 0.3 gave every model the conversation's numbers, even with a plugin that named a feature of its own
  conversation. Today that plugin's feature turns the built-in one off, so a forecast gives a model 0.3 saved
  the conversation's numbers whatever plugin is loaded: see saved_by_0_3.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model import Features, Model, _load_one, _read_by_llm, _swap_in
from .model_types import PREVIOUS_FORMAT


class Choice:
    """The versions of a model 0.3 saved, and which one each AI model uses.

    It reads like one model: forecast(), served(), save() and a meta dict, as model_types describes. The meta
    is what the folder's own model.json says: the test report and the effort levels, which every version
    shares, with "versions", "uses" and "without_llm". Each version has a folder of its own.
    """

    def __init__(self, versions: dict[str, Model], meta: dict) -> None:
        self.versions = versions
        self.meta = meta

    @property
    def pools(self) -> dict:
        """Every version's similar-task pools. A forecast embeds a text only when some version compares it."""
        found: dict = {}
        for version in self.versions.values():
            found.update(version.pools)
        return found

    def _version_of(self, row: Features) -> str:
        """The version that forecasts a row: its AI model's, or the one without LLM analysis when it has none."""
        without = (self.meta.get("without_llm") or {}).get(row.model)
        if without is not None and not _read_by_llm(row):
            return without["version"]
        return self.meta["uses"].get(row.model, next(iter(self.versions)))

    def forecast(self, rows: list[Features]) -> list[dict]:
        """Each row's forecast, by its version."""
        found: list = [None] * len(rows)
        picked = [self._version_of(r) for r in rows]
        for name, version in self.versions.items():
            mine = [i for i, chosen in enumerate(picked) if chosen == name]
            if mine:
                for i, result in zip(mine, version.forecast([rows[i] for i in mine])):
                    found[i] = result
        return found

    def served(self, model: str, effort: str | None = None) -> tuple[bool, str | None]:
        """Whether forecasts for this AI model passed the test. Every version holds the same verdicts."""
        return next(iter(self.versions.values())).served(model, effort)

    def served_without_llm(self, model: str, effort: str | None = None) -> tuple[bool, str | None]:
        """Whether a task LLM analysis couldn't read is forecast on this AI model, and why not.

        The version without LLM analysis forecasts it, so that version's test decides.
        """
        served, reason = self.served(model, effort)
        without = (self.meta.get("without_llm") or {}).get(model)
        if without is None or (not served and reason != "not_accurate_enough"):
            return served, reason  # an AI model that never uses it, an unknown one, or a level it never ran at
        return without["served"], without["reason"]

    def save(self, folder: Path) -> None:
        """Write the model as 0.3 wrote it: its own model.json, and a folder for each version."""

        def write(staging: Path) -> None:
            (staging / "model.json").write_text(json.dumps(self.meta, indent=1) + "\n")
            for name, version in self.versions.items():
                (staging / name).mkdir()
                version.write(staging / name)

        _swap_in(folder, write)


def load(folder: Path, meta: dict) -> Model | Choice:
    """Load a model 0.3 saved, as 0.3 did. `meta` is its folder's own model.json."""
    if meta.get("versions"):
        return Choice({name: _load_one(folder / name) for name in meta["versions"]}, meta)
    return _load_one(folder)


def saved_by_0_3(model) -> bool:
    """Whether a loaded model is one 0.3 saved: its versions, or the one version 0.3 saved alone. Such a model gets
    the conversation's numbers whatever plugin is loaded, as 0.3 gave them."""
    return isinstance(model, (Choice, Model)) and model.meta.get("format") == PREVIOUS_FORMAT
