"""Estimate: forecast a task before you run it, one step at a time.

Describe the task. With more than one model, choose which of your models forecasts it. Then pick the AI
models: the ones with enough data behind them, or any you've used, from a list. Either way a list shows them 6
at a time. The forecasts show 6 at a time too, each with the ranges the real result lands in half the time and
9 times out of 10. After them you choose what's next: another page, another task, other AI models for the same
task, another of your models for it or home.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .. import cli, ui
from ..store import default_name, predictor_names
from . import checklist, keys, listing

PAGE = 6
"""How many forecasts a page shows."""


@dataclass
class Ready:
    """One of your models, loaded to forecast, with the AI models it knows and how much data each has."""

    predictor: object
    model: object
    pairs: list
    runs: dict
    too_few: list
    verdicts: dict


def run() -> None:
    """The Estimate screen, until you go back home."""
    from ..forecast import RequestError

    several = len(predictor_names()) > 1
    task, name, loaded = None, None, None
    while True:
        if task is None:
            task = describe()
            if not task:
                return
        if loaded is None:
            name = which(name) if several else default_name() or predictor_names()[0]
            if name is None:
                task = None  # back to describing the task
                continue
            loaded = ready(name)
            if loaded is not None and not loaded.pairs:
                too_few(name, loaded.too_few)
                loaded = None
            if loaded is None:  # it can't forecast, and the reason was said
                if not several:
                    return
                continue  # choose another of your models, for the same task
        chosen = pick(loaded.pairs, loaded.runs, loaded.verdicts, loaded.too_few)
        if chosen is None:
            task = None  # back to describing the task
            continue
        asked = argparse.Namespace(context=None, skill=[], distribution=False)
        try:
            result = cli.forecast_pairs(
                loaded.predictor, loaded.model, task, chosen, asked, rough=True, runs=loaded.runs
            )
        except RequestError as error:
            ui.error(error.message)
            result = None
        step = show(task, result, loaded.predictor, several)
        if step == "home":
            return
        if step == "task":
            task = None
        elif step == "model":
            loaded = None


def which(last: str | None) -> str | None:
    """Ask which of your models forecasts the task. It starts on the one used last, else on the default. None goes
    back to the task."""
    names = predictor_names()
    ui.out()
    ui.line("Which of your models should forecast it?")
    ui.out()
    choices = [keys.Choice(label, when) for label, when in listing.models()] + [keys.Choice("Back")]
    first = last if last in names else default_name()
    picked = keys.choose(choices, start=names.index(first) if first in names else 0)
    return None if picked is None or picked == len(names) else names[picked]


NEEDS_TRAINING = "{name} needs training before it can forecast. Choose Update on the home screen to train it."


def ready(name: str) -> Ready | None:
    """The model, loaded to forecast, or None when it can't forecast, with the reason said."""
    from ..forecast import RequestError, load

    try:
        predictor = cli.request_predictor(name)
        model = load(predictor)
    except RequestError as error:
        ui.error(NEEDS_TRAINING.format(name=name) if error.code == "not_trained" else error.message)
        return None
    pairs, runs, too_few = cli.every_pair(predictor, None, model.meta)
    verdicts = (model.meta.get("report") or {}).get("models") or {}
    return Ready(predictor, model, pairs, runs, too_few, verdicts)


def describe() -> str:
    """Ask what the task is. An empty answer goes back home."""
    ui.out()
    title, says = "Estimate", "Forecast a task before you run it"
    if len(title) + 2 + len(says) <= ui.columns() - 3:
        ui.line(f"{ui.bold(title)}  {ui.dim(says)}")
    else:  # what it does goes under its name
        ui.line(ui.bold(title))
        ui.note(says)
    ui.out()
    ui.line("Describe the task, as you'd ask a coding agent.")
    ui.note("For example: add a dark mode toggle to the settings page")
    ui.note("Press Enter on its own to go back.")
    return input(f"  {ui.accent(ui.mark('pick'))} ").strip()


def status(model: str, effort: str | None, runs: dict, verdicts: dict) -> str:
    """A warning when there's not enough data for this AI model at this effort level. Nothing when there is."""
    warning = cli.rough_warning({"reason": _reason(verdicts.get(model)), "runs": runs.get((model, effort))})
    return ui.yellow(f"{ui.mark('warn')} {warning}") if warning else ""


def _reason(verdict: dict | None) -> str | None:
    if not verdict:
        return "not_enough_data"
    return None if verdict.get("served") else verdict.get("reason") or "not_enough_data"


def pick(pairs: list, runs: dict, verdicts: dict, too_few: list) -> list | None:
    """Which AI models to forecast, at which effort levels. None goes back to the task."""
    # Enough data means the AI model's forecasts held on your past tasks, at a level it ran at often.
    tested = [pair for pair in pairs if not status(*pair, runs, verdicts)]
    names = list(dict.fromkeys(name for name, _ in tested))
    while True:
        ui.out()
        ui.line("Which AI models should it forecast?")
        ui.out()
        named = (cli.in_words(names) if len(names) <= 3 else f"{len(names)} AI models") if names else "none yet"
        choices = [
            keys.Choice("AI models with enough data", named),
            keys.Choice("Choose from a list", f"every one you've used, {len(pairs)} in all"),
            keys.Choice("Back"),
        ]
        picked = keys.choose(choices)
        if picked is None or picked == 2:
            return None
        if picked == 0 and not tested:
            ui.note("No AI model has enough data yet. Choose from the list, where each one says how sure it is.")
            continue
        shown = tested if picked == 0 else pairs
        rows = [keys.Choice(_label(pair, shown), status(*pair, runs, verdicts)) for pair in shown]
        notes = []
        if picked == 1 and too_few:
            listed = cli.in_words([f"{name} ({count:,} run{'' if count == 1 else 's'})" for name, count in too_few])
            notes.append(f"Not enough data to forecast {listed}.")
        ui.out()
        ticked = checklist.tick(rows, notes=notes)
        if ticked is not None:
            return [shown[number] for number in ticked]


def too_few(name: str, found: list) -> None:
    """Say that no AI model ran enough tasks for the model to forecast it yet, with how many each ran."""
    listed = cli.in_words([f"{model} ({count:,} run{'' if count == 1 else 's'})" for model, count in found])
    text = f"Not enough data to forecast with {name} yet. Each AI model needs {cli.SHOWN_FROM} runs."
    text += f" These ran fewer: {listed}." if listed else ""
    for part in ui.wrapped(f"{text} Update it once there are more."):
        ui.note(part)


def _label(pair: tuple, every: list) -> str:
    """An AI model and its effort level, lined up with the others in the list. The default level isn't named."""
    pad = max(len(name) for name, _ in every)
    name, effort = pair
    return f"{name.ljust(pad)}  {cli.shown_effort(effort)}".rstrip()


def show(task: str, result: dict | None, predictor, several: bool = False) -> str:
    """The forecasts, a page at a time, then what's next. Returns task, models, model or home. With `several` of
    your models, another of them can forecast the same task."""
    forecasts = (result or {}).get("forecasts") or []
    page, count = 0, max(1, -(-len(forecasts) // PAGE))
    while True:
        if forecasts:
            ui.out()
            cli.show_title(task, f"page {page + 1} of {count}" if count > 1 else "")
            shown = {**result, "forecasts": forecasts[page * PAGE : (page + 1) * PAGE], "too_few": []}
            cli.show_forecasts(task, shown, predictor, screen=True)
        ui.out()
        steps = []
        if page + 1 < count:
            steps.append(("next", keys.Choice("Next page", f"forecasts {(page + 1) * PAGE + 1} on")))
        if page > 0:
            steps.append(("back", keys.Choice("Previous page")))
        steps += [
            ("task", keys.Choice("Estimate another task")),
            ("models", keys.Choice("Choose other AI models", "for the same task")),
        ]
        if several:
            steps.append(("model", keys.Choice("Use another of your models", "for the same task")))
        steps.append(("home", keys.Choice("Back to home")))
        picked = keys.choose([choice for _, choice in steps], footer=[keys.hint("esc home")])
        step = "home" if picked is None else steps[picked][0]
        if step == "next":
            page += 1
        elif step == "back":
            page -= 1
        else:
            return step
