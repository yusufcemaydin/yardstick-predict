"""Update: learn from your newest work.

First it says where the new work is and how much there is to read, from file sizes alone. Then it reads
with a labelled bar, shows every cost and time for each model, and asks once. Cancel or Esc spends nothing
and saves nothing. What it read is kept for the rest of the session, so a second Update reads none of it
again. `yardstick-predict refresh` in a terminal uses the same summary and question.
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from .. import cli, embeddings, locks, ui
from ..store import BrokenPredictor, Predictor, predictor_names
from . import keys, listing, summary

_readings = None
"""What this session has read, so a second Update reads none of it again."""


def readings():
    global _readings
    from ..pipeline import Readings

    if _readings is None:
        _readings = Readings()
    return _readings


@dataclass
class Work:
    """One model's update, worked out before anything is spent: what it reads and what it does after."""

    predictor: Predictor
    reads: list
    planned: object = None
    """What pipeline.plan_refresh found it does, once its records are read."""
    before: tuple[str, ...] = ()
    """The AI models it gave forecasts for before."""


def run(names: list[str] | None = None) -> int:
    """Update the named models, or with none, ask which first. Returns the command's exit code."""
    if names is None:
        names = pick()
        if not names:
            return 0
    from .. import feature_store

    with ExitStack() as held:
        ready = take(names, held)
        if not ready:
            return 1
        # Each update reads the feature store to plan what to make, and saves to it. No features delete runs meanwhile.
        held.enter_context(feature_store.writing())
        return update(ready)


def pick() -> list[str]:
    """Which models to update: all of them, or one. With only one model, it isn't asked."""
    names = predictor_names()
    if len(names) <= 1:
        if not names:
            ui.note("There's no model yet. Choose Add New to make one.")
        return names
    ui.out()
    ui.line(ui.bold("Update"))
    ui.out()
    choices = [keys.Choice("Update all", cli.in_words(names))]
    choices += [keys.Choice(name, when) for name, when in listing.models()]
    choices.append(keys.Choice("Back"))
    picked = keys.choose(choices)
    if picked is None or picked == len(choices) - 1:
        return []
    return names if picked == 0 else [names[picked - 1]]


def take(names: list[str], held: ExitStack) -> list[Work]:
    """The models that can be updated now, each with its lock held until the update is done. Any other is said."""
    from .. import model_types
    from ..pipeline import load_plugins, to_read

    found = []
    for name in names:
        try:
            predictor = Predictor(name)
        except BrokenPredictor as error:
            ui.error(str(error))
            continue
        if predictor.newer:
            ui.note(f"{name} was saved by a newer version of yardstick-predict. This version leaves it as it is.")
            continue
        try:
            load_plugins(predictor)
        except model_types.PluginError as error:
            ui.error(str(error))
            continue
        # Two updates at once could each train, and the slower one could save an older model over a newer one.
        if not held.enter_context(locks.held(predictor.dir / "refresh.lock", wait=False)):
            ui.note(f"{name} is being set up or updated by another command. Update it once that's done.")
            continue
        if cli.gone(predictor):
            ui.note(f"There's no model called {name} anymore: it was deleted.")
            continue
        held.callback(predictor.close)
        cli.unknown_features_note(predictor)
        found.append(Work(predictor, to_read(predictor, count_chats=True)))
    return found


def update(work: list[Work]) -> int:
    """Say where the new work is, read it, show what each model does and what it costs, ask once, then do it."""
    from ..pipeline import plan_refresh, scan

    since = when(work)
    # With nothing changed in any archive, nothing is read or listed. The models may still have work to do.
    if any(found.files for job in work for found in job.reads):
        unread = where(work, since)
        try:
            if unread:
                read(unread)
        except keys.Stopped:
            ui.out()
            ui.note("Stopped. Nothing was spent or saved.")
            return 0
    key, _ = embeddings.find_key()
    for job in work:
        found = scan(job.predictor, reads=job.reads, readings=readings())
        job.planned = plan_refresh(job.predictor, found, key)
        job.before = tuple(cli.trained_facts(job.predictor, job.planned.model)["forecasts_for"])
    todo = [job for job in work if not job.planned.up_to_date]
    if not todo:
        for job in work:
            settle(job)
        ui.out()
        ui.line(f"Nothing new {since}.")
        return 0
    if not summary.ask(work, key):
        ui.out()
        ui.note("Nothing was spent or saved.")
        return 0
    if any(job.planned.embed.items or job.planned.asks for job in todo):
        key = key or cli.get_key(todo[0].predictor.embedding_model, True)
        if not key:
            return 1
    ui.out()
    code = 0
    for job in work:
        code = max(code, apply(job, key))
    return code


def when(work: list[Work]) -> str:
    """Since when the models last learned, as in: Nothing new since 12 hours ago."""
    moments = [job.predictor.config.get("refreshed_at") for job in work]
    if not moments or not all(moments):
        return "since they were made" if len(work) > 1 else "since it was made"
    then = cli.ago(min(moments))
    return "since a moment ago" if then == "just now" else f"since {then}"


# Where the new work is -----------------------------------------------------------------------------------


def short_path(archive: dict) -> str:
    """The archive's folder, with your home folder written as ~."""
    path = str(archive["path"])
    home = str(Path.home())
    return "~" + path[len(home) :] if path.startswith(home + os.sep) else path


def where(work: list[Work], since: str) -> list[tuple[dict, dict]]:
    """Show each archive's folder, how many chats changed and how much that is to read. Returns what there is
    to read, by archive: every file any model needs from it, read once for all of them."""
    groups: dict[tuple, dict] = {}
    for job in work:
        for found in job.reads:
            archive = found.archive
            which = (archive["name"], archive["format"], str(archive["path"]))
            group = groups.setdefault(which, {"archive": archive, "files": {}, "chats": set()})
            group["files"].update(found.files)
            group["chats"] |= found.chats
    if not groups:
        return []
    ui.out()
    ui.line(f"New work {since}")
    rows, unread = [], []
    for group in groups.values():
        archive, files, chats = group["archive"], group["files"], group["chats"]
        if not files:
            facts = "nothing new"
        else:
            count = len(chats) or len(files)
            changed = f"{count:,} chat{'' if count == 1 else 's'} changed"
            if readings().get(archive, files) is not None:
                facts = ui.join([changed, "read already"])
            else:
                facts = ui.join([changed, f"{ui.size(sum(size for size, _ in files.values()))} to read"])
                unread.append((archive, files))
        rows.append((listing.title(archive), ui.dim(short_path(archive))))
        rows.append(("", facts))
    ui.table(rows, indent=4)
    return unread


def read(unread: list[tuple[dict, dict]]) -> None:
    """Read the new work with a labelled bar, and keep what was read for this session. Esc stops it."""
    from ..archives import read_records

    total = sum(size for _, files in unread for size, _ in files.values())
    ui.out()
    ui.note("Free. Your chats never leave this computer. Esc stops it.")
    now = time.time()
    done = 0
    with keys.watching() as watch:

        def told(size: int) -> None:
            nonlocal done
            done = min(done + size, total)
            ui.progress(done, total, "bytes")
            watch.stop_if_asked()

        ui.working("Reading")
        ui.progress(0, total, "bytes")
        for archive, files in unread:
            start = done
            reading = read_records(archive, files, now, told)
            done = start + sum(size for size, _ in files.values())  # the files the reader passed over count too
            readings().put(archive, files, reading)
    ui.ok(f"Read {ui.size(total)}.")


# Doing it ----------------------------------------------------------------------------------------------------


def settle(job: Work) -> None:
    """For a model with nothing new: remember the files that were read, so they aren't read again."""
    found = job.planned.found
    job.predictor.remember_files(found.files)
    if found.reader is not None:
        job.predictor.config["reader"] = found.reader
        job.predictor.save_config()


def apply(job: Work, key: str | None) -> int:
    """Save what was found, analyze, embed and train, each step with a labelled bar. Then one line on how it
    went. Returns 1 when a step failed, else 0."""
    from ..model import NotEnoughData
    from ..pipeline import plan_embeddings, refresh_benchmarks, run_analysis, save_scan, train

    predictor, planned = job.predictor, job.planned
    name = predictor.name
    if planned.up_to_date:
        settle(job)
        ui.ok(f"{name} is up to date")
        return 0
    save_scan(predictor, planned.found)
    warning = refresh_benchmarks(predictor)
    if warning:
        ui.note(warning)
    if planned.asks:
        ui.working(f"{name}  Analyzing with {summary.analyst(predictor.config['llm'])}")
        _, problem = run_analysis(predictor, planned.asks, lambda done, total: ui.progress(done, total, "tasks"))
        if problem:
            ui.note(f"{name}: {problem} It trains without LLM analysis for now. The next update tries again.")
    if not embed(predictor, planned.embed, key):
        return 1
    # The descriptions LLM analysis just wrote are embedded too. The summary priced them already.
    if not embed(predictor, plan_embeddings(predictor, key), key):
        return 1
    ui.working(f"{name}  Training")
    try:
        model = train(predictor, lambda done, total: ui.progress(done, total, "steps"))
    except NotEnoughData as error:
        ui.fail(f"{name} wasn't trained: {error}")
        return 1
    cli.default_when_ready(predictor)
    report = model.meta.get("report") or {}
    served = cli.served_models(report)
    words = f"forecasts on for {cli.in_words(served)}" if served else "no forecasts yet"
    ui.ok(ui.join([f"{name} updated", words]))
    for model_name in served:
        if model_name not in job.before:
            ui.note(f"  {model_name} forecasts are on now: it passed the test.")
    for model_name in job.before:
        if model_name not in served:
            ui.note(f"  {model_name} forecasts are off now: {off_because(report, model_name)}")
    return 0


def embed(predictor: Predictor, plan, key: str | None) -> bool:
    """Embed what the plan holds with a labelled bar. False when Voyage stopped it."""
    from ..pipeline import run_embeddings

    if not plan.items:
        return True
    ui.working(f"{predictor.name}  Embedding with Voyage")
    try:
        run_embeddings(predictor, plan, key, lambda done, total: ui.progress(done, total, "texts"), ui.note)
    except embeddings.VoyageError as error:
        ui.fail(f"{predictor.name} stopped: {cli.explain(error)}")
        ui.note("What was embedded so far is saved. Update again to finish.")
        return False
    return True


def off_because(report: dict, model_name: str) -> str:
    """Why an AI model's forecasts turned off, in a few plain words."""
    result = (report.get("models") or {}).get(model_name)
    if not isinstance(result, dict):
        return "none of its tasks are left to learn from."
    if result.get("reason") == "not_enough_data":
        return "too few of its tasks are left to test it."
    failed = result.get("failed_checks") or []
    if failed:
        names = {"seconds": "time", "tokens": "tokens", "usd": "cost"}
        try:
            return cli.failed_check(failed[0], result, names, cli.thresholds(report))
        except (KeyError, TypeError, ValueError):
            pass
    return "it didn't pass the test."
