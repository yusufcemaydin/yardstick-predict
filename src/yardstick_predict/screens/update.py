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
    # A key is needed only for the texts that go to Voyage: LLM analysis needs none of it. It's asked for before the
    # summary, so the summary counts its free tokens, and nothing is asked after the yes.
    voyage = [job for job in todo if job.planned.embed.texts or job.planned.described]
    if voyage and not key:
        key = cli.get_key(voyage[0].predictor.embedding_model, True)
        if not key:
            return 1
    if key:
        cli.damaged_usage_note(key)
    from .. import spending

    costs = {
        id(job): spending.cost_of(
            job.predictor.config,
            job.predictor.embedding_model,
            job.planned.embed,
            job.planned.asks,
            job.planned.described,
        )
        for job in todo
    }
    cli.keep_to_limit(list(costs.values()), "update")  # with --max-cost, before anything is asked or spent
    if not summary.ask(work, key):
        ui.out()
        ui.note("Nothing was spent or saved.")
        return 0
    ui.out()
    code = 0
    global _unreachable
    _unreachable = False
    for number, job in enumerate(work):
        # What the models after this one may cost is kept out of what this one may spend, under a limit.
        later = sum(costs[id(other)].most_usd or 0.0 for other in work[number + 1 :] if id(other) in costs)
        code = max(code, apply(job, key, later))
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
        rows.append((listing.title(archive), short_path(archive)))
        rows.append(("", facts))
    # A long folder is cut short at its start, where it's least telling, so each line fits the terminal.
    widest = max(ui.width(title) for title, _ in rows)
    room = ui.columns() - 1 - 4 - widest - 3
    fitted = []
    for title, facts in rows:
        if title and ui.width(facts) > room:
            facts = ui.mark("wait") + facts[-max(room - 1, 1) :]
        fitted.append((title, ui.dim(facts) if title else facts))
    ui.table(fitted, indent=4)
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


def apply(job: Work, key: str | None, later: float = 0.0) -> int:
    """Save what was found, analyze, embed and train, each step with a labelled bar. Then one line on how it
    went. Returns 1 when a step failed, else 0. Under a --max-cost limit, `later` is the most the models updated
    after this one may cost, kept out of what this one may spend."""
    from .. import llm, spending
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
        budget = None
        if cli._limit is not None and llm.PROVIDERS[predictor.config["llm"]["provider"]].billed:
            reserve = spending.embedding_most_usd(planned.embed, predictor.embedding_model, planned.described)
            budget = llm.Budget(cli._limit.left() - reserve - later)
        ui.working(f"{name}  Analyzing with {summary.analyst(predictor.config['llm'])}")
        try:
            _, problem = run_analysis(
                predictor, planned.asks, lambda done, total: ui.progress(done, total, "tasks"), budget
            )
        finally:
            if budget is not None:
                cli._limit.spent += budget.spent
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
            ui.note(f"  {model_name} forecasts are on now: there's enough data.")
    for model_name in job.before:
        if model_name not in served:
            ui.note(f"  {model_name} forecasts are off now: not enough data.")
    return 0


_unreachable = False
"""Whether Voyage couldn't be reached during this update. The models after that one don't try: each would wait
a few minutes for an answer that doesn't come."""


def embed(predictor: Predictor, plan, key: str | None) -> bool:
    """Embed what the plan holds with a labelled bar. False when Voyage stopped it."""
    global _unreachable
    from .. import spending
    from ..pipeline import run_embeddings

    if not plan.items:
        return True
    if _unreachable:
        ui.fail(f"{predictor.name} wasn't embedded: Voyage couldn't be reached a moment ago.")
        ui.note("What was found is saved. Update again once Voyage can be reached.")
        return False
    most = spending.embedding_most_usd(plan, predictor.embedding_model)
    if cli._limit is not None and not cli._limit.fits(most):
        ui.fail(f"{predictor.name} stopped at your spending limit: embedding may cost up to {spending.dollars(most)}.")
        ui.note("What was done so far is saved. With a higher --max-cost, the next update goes on from there.")
        return False
    ui.working(f"{predictor.name}  Embedding with Voyage")
    try:
        run_embeddings(predictor, plan, key, lambda done, total: ui.progress(done, total, "texts"), ui.note)
    except embeddings.VoyageError as error:
        _unreachable = _unreachable or error.status == 0
        ui.fail(f"{predictor.name} stopped: {cli.explain(error)}")
        ui.note("What was embedded so far is saved. Update again to finish.")
        return False
    if cli._limit is not None:
        cli._limit.spent += most
    return True
