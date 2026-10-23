"""The feature check: which of a predictor's optional features really help its forecasts.

It runs the test that training runs, on each AI model's newest tasks, once with every feature the predictor has,
then once without each optional feature in turn. A feature helped an AI model when that model's test scored
more than 1% worse without it. It didn't help when the test scored about the same, or better, without it. When
the AI model had too few tasks to test, or too few of its test tasks had the feature, there isn't enough data to
tell.

It's advice only. Setup offers it, then suggests the features that helped. The check command runs it on a
predictor you already have, and changes nothing about it. Each verdict is remembered in a file of its own in the
feature store's folder, so the Features screen can show the latest ones next to each feature.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import files, locks
from .archives import Task

HELPED = "helped"
NO_HELP = "no_help"
NOT_ENOUGH = "not_enough_data"
WORDS = {HELPED: "helped", NO_HELP: "didn't help", NOT_ENOUGH: "not enough data"}
"""Each verdict in plain words."""
WORSE_BY = 0.01
"""A feature helped an AI model when its test scored more than 1% worse without it. A lower score is better."""
LEAST_SHARE = 0.2
"""When under 20% of an AI model's test tasks have a feature, its test can't show what the feature does."""
SAMPLE_SHARE = 0.3
SAMPLE_MOST = 500
"""Before the check, LLM analysis reads only each AI model's newest 30% of tasks, and at most 500 in all. Each AI
model is tested on its own newest 30% or so, so those are the tasks the test needs. The rest are read only when
the person keeps LLM analysis after the check, and agrees to what it costs."""
FORMAT = 1
"""The format of the file of remembered verdicts."""


@dataclass
class Inputs:
    """What the check reads from a predictor, as training reads it."""

    tasks: list[Task]
    rows: list
    """Each task's features, as the model reads them."""
    settings: dict
    """The model's settings, such as how many windows its test has."""
    features: list[str]
    """The optional features to check, in the predictor's order."""


@dataclass
class Result:
    """What a check found."""

    features: list[str]
    """The optional features it checked."""
    report: dict
    """The test with every feature, as model.test reports it."""
    verdicts: dict[str, dict[str, dict]]
    """By feature, then by AI model: the verdict, how much worse the test scored without the feature, and the share
    of the AI model's test tasks that have it."""
    checked_at: float = field(default_factory=time.time)

    @property
    def models(self) -> list[str]:
        """Every AI model the test knows about, the most tested first."""
        found = self.report.get("models") or {}
        return sorted(found, key=lambda name: -(found[name].get("tasks_tested") or 0))

    def tasks_tested(self, model: str) -> int:
        """How many of the AI model's tasks the test forecast."""
        return int(((self.report.get("models") or {}).get(model) or {}).get("tasks_tested") or 0)

    def helped(self) -> list[str]:
        """The features that helped at least one AI model."""
        return [
            name
            for name in self.features
            if any(found["verdict"] == HELPED for found in self.verdicts.get(name, {}).values())
        ]


def inputs(predictor) -> Inputs:
    """What the check needs from a predictor: the tasks training learns from and their features.

    The predictor's plugins are loaded first. Raises model_types.PluginError when one can't be.
    """
    from . import features, pipeline

    pipeline.load_plugins(predictor)
    tasks = predictor.tasks()
    if predictor.config.get("max_tasks"):
        # The same newest tasks that training learns from, oldest first.
        tasks = pipeline.learnable(tasks, predictor.config["max_tasks"])[::-1]
    optional = [name for name in pipeline.chosen_features(predictor) if name != features.REQUIRED]
    return Inputs(tasks, pipeline.feature_rows(predictor, tasks), pipeline.settings(predictor), optional)


def run(given: Inputs, on_progress: Callable[[int, int], None] | None = None) -> Result:
    """Test the model with every feature, then without each optional one, and judge each feature for each AI model.

    `on_progress` is told how many of the trial models are fitted, and how many there are. Raises
    model.NotEnoughData when there are too few tasks to train on.
    """
    from . import model

    reports, shares = model.compare(given.tasks, given.rows, given.settings, given.features, on_progress)
    return judge(reports, shares, given.features)


def judge(reports: dict[str, dict], shares: dict[str, dict[str, float]], names: list[str]) -> Result:
    """Each feature's verdict for each AI model, from the tests model.compare ran.

    `reports` holds each test's report by the feature it left out, with "" for the test with every feature.
    `shares` holds, by feature, the share of each AI model's test tasks that have it.
    """
    full = reports.get("") or {}
    verdicts = {}
    for name in names:
        without = (reports.get(name) or {}).get("models") or {}
        verdicts[name] = {
            model: verdict(found, without.get(model) or {}, (shares.get(name) or {}).get(model))
            for model, found in (full.get("models") or {}).items()
        }
    return Result(list(names), full, verdicts)


def verdict(with_all: dict, without: dict, share: float | None) -> dict:
    """One feature's verdict for one AI model.

    `with_all` and `without` are the AI model's results in the test with every feature and in the test without
    this one, as the reports give them. `share` is the share of its test tasks that have the feature. Each head
    both tests scored, such as time or cost, counts alike: the change is the average of how much worse each
    head's score was without the feature, as a share of its score with it. Below 0, it scored better without.
    """
    heads = [head for head in with_all.get("heads") or [] if head in (without.get("heads") or [])]
    change = None
    if heads:
        worse = []
        for head in heads:
            score = with_all[head]["score"]
            worse.append((without[head]["score"] - score) / max(score, 1e-4))  # as the test's report divides
        change = round(sum(worse) / len(worse), 4)
    if change is None or share is None or share < LEAST_SHARE:
        found = NOT_ENOUGH
    else:
        found = HELPED if change > WORSE_BY else NO_HELP
    return {"verdict": found, "worse_without": change, "share": None if share is None else round(share, 4)}


def sample(asks: list, model_of: dict[str, str] | None = None) -> list:
    """The tasks LLM analysis reads before the check: each AI model's newest 30%, and at most 500 in all.

    The test forecasts each AI model's own newest tasks, so an AI model used mostly long ago gets its share too.
    When 30% of each would pass 500 in all, each gets the same smaller share. `asks` are the tasks it would read,
    newest first, as pipeline.analysis_plan gives them, and `model_of` says each task's AI model, by its id.
    Without it, the tasks count as one AI model's.
    """
    model_of = model_of or {}
    by_model: dict[str, list[int]] = {}
    for at, ask in enumerate(asks):
        by_model.setdefault(model_of.get(getattr(ask, "id", ask), ""), []).append(at)
    wanted = sum(math.ceil(len(found) * SAMPLE_SHARE) for found in by_model.values())
    share = SAMPLE_SHARE * min(1.0, SAMPLE_MOST / wanted) if wanted else 0.0
    keep = sorted(at for found in by_model.values() for at in found[: math.ceil(len(found) * share)])
    return [asks[at] for at in keep][:SAMPLE_MOST]


def sampled_too_little(result: Result) -> list[str]:
    """The AI models whose test had LLM answers for some of its tasks, but too few to judge LLM analysis."""
    found = (result.verdicts.get("llm") or {}).items()
    return [model for model, said in found if said["verdict"] == NOT_ENOUGH and (said.get("share") or 0) > 0]


def as_json(result: Result, predictor: str) -> dict:
    """What a check found, for programs."""
    models = result.report.get("models") or {}
    return {
        "version": 1,
        "predictor": predictor,
        "checked_at": result.checked_at,
        "tested_on": int(result.report.get("tested_on") or 0),
        "features": result.features,
        "models": {
            name: {"tasks": int(models[name].get("tasks") or 0), "tasks_tested": result.tasks_tested(name)}
            for name in result.models
        },
        "verdicts": result.verdicts,
    }


def title(name: str) -> str:
    """A feature's name as the Features screen shows it, such as Model facts. Your own features keep theirs."""
    from . import features

    group = next((g for g in features.MENU if g.key == name), None)
    return group.title if group else name


def summary(by_model: dict[str, dict]) -> str:
    """The latest verdicts on one feature in a few words, such as "Last check: helped claude-opus-5-5." Empty
    when there are none."""
    said = []
    for kind, words in ((HELPED, "helped"), (NO_HELP, "didn't help"), (NOT_ENOUGH, "not enough data for")):
        models = [model for model, found in by_model.items() if found.get("verdict") == kind]
        if models:
            said.append(f"{words} {', '.join(models)}")
    return f"Last check: {'; '.join(said)}." if said else ""


# Remembering the verdicts ------------------------------------------------------------------------------


def path() -> Path:
    """The file that remembers each feature's latest verdicts, in the feature store's folder."""
    from . import store

    return store.home() / "features" / "checks.json"


def _saved() -> dict | None:
    """The file's contents, or None when it's missing or broken."""
    try:
        found = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def latest() -> dict[str, dict[str, dict]]:
    """The latest verdict on each feature for each AI model, with when it was checked and on which predictor.

    Empty when nothing was checked, and when the file is missing or broken. An entry in a shape this version
    doesn't know is skipped.
    """
    return _entries(_saved() or {})


def _entries(saved: dict) -> dict[str, dict[str, dict]]:
    found: dict[str, dict[str, dict]] = {}
    by_feature = saved.get("features")
    for name, by_model in (by_feature if isinstance(by_feature, dict) else {}).items():
        for model, entry in (by_model if isinstance(by_model, dict) else {}).items():
            if isinstance(entry, dict) and entry.get("verdict") in WORDS:
                found.setdefault(name, {})[model] = entry
    return found


def remember(result: Result, predictor: str) -> bool:
    """Save each verdict with its date and the predictor's name, for the Features screen and the feature store.

    The file is saved in one step. A verdict of not enough data never replaces one that says whether the feature
    helped: it tells nothing new. A file a newer version saved, in a format this version doesn't know, is left as
    it is. Returns whether the verdicts were saved.
    """
    from . import store

    store.private_home()
    folder = store.home() / "features"
    folder.mkdir(mode=0o700, exist_ok=True)
    if os.name == "posix":
        os.chmod(folder, 0o700)  # it holds what you chose for your predictors, so it's yours alone
    with locks.held(folder / "checks.lock"):
        saved = _saved() or {}
        known = saved.get("format")
        if isinstance(known, int) and not isinstance(known, bool) and known > FORMAT:
            return False
        kept = _entries(saved)
        for name in result.features:
            for model, found in result.verdicts.get(name, {}).items():
                before = kept.get(name, {}).get(model, {}).get("verdict")
                if found["verdict"] == NOT_ENOUGH and before in (HELPED, NO_HELP):
                    continue
                entry = {"verdict": found["verdict"], "checked_at": result.checked_at, "predictor": predictor}
                kept.setdefault(name, {})[model] = entry
        files.write_text(folder / "checks.json", json.dumps({"format": FORMAT, "features": kept}, indent=2) + "\n")
    return True
