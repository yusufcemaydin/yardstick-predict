"""Estimate: forecast a task before you run it, one step at a time.

Describe the task. Then pick the AI models: the ones that passed the test, or any you've used, from a list.
Either way a checklist shows them 6 at a time. The forecasts show 6 at a time too, with the range each lands
in 9 times out of 10. After them you choose what's next: another page, another task, other AI models for the
same task, or home.
"""

from __future__ import annotations

import argparse

from .. import cli, ui
from . import checklist, keys

PAGE = 6
"""How many forecasts a page shows."""


def run() -> None:
    """The Estimate screen, until you go back home."""
    from ..forecast import RequestError, load

    try:
        predictor = cli.request_predictor(None)
        model = load(predictor)
    except RequestError as error:
        ui.error(error.message)
        return
    pairs, runs, too_few = cli.every_pair(predictor, None, model.meta)
    verdicts = (model.meta.get("report") or {}).get("models") or {}
    task = None
    while True:
        if task is None:
            task = describe()
            if not task:
                return
        chosen = pick(pairs, runs, verdicts, too_few)
        if chosen is None:
            task = None  # back to describing the task
            continue
        asked = argparse.Namespace(context=None, skill=[], distribution=False)
        try:
            result = cli.forecast_pairs(predictor, model, task, chosen, asked, rough=True, runs=runs)
        except RequestError as error:
            ui.error(error.message)
            result = None
        step = show(task, result, predictor)
        if step == "home":
            return
        if step == "task":
            task = None


def describe() -> str:
    """Ask what the task is. An empty answer goes back home."""
    ui.out()
    ui.line(f"{ui.bold('Estimate')}  {ui.dim('Forecast a task before you run it')}")
    ui.out()
    ui.line("Describe the task, as you'd ask a coding agent.")
    ui.note("For example: add a dark mode toggle to the settings page")
    ui.note("Press Enter on its own to go back.")
    return input(f"  {ui.accent(ui.mark('pick'))} ").strip()


def status(model: str, effort: str | None, runs: dict, verdicts: dict) -> str:
    """How sure a forecast for this AI model at this effort level can be, in a few words."""
    warning = cli.rough_warning({"reason": _reason(verdicts.get(model)), "runs": runs.get((model, effort))})
    return ui.yellow(f"{ui.mark('warn')} {warning}") if warning else ui.green("passed the test")


def _reason(verdict: dict | None) -> str | None:
    if not verdict:
        return "not_enough_data"
    return None if verdict.get("served") else verdict.get("reason") or "not_enough_data"


def pick(pairs: list, runs: dict, verdicts: dict, too_few: list) -> list | None:
    """Which AI models to forecast, at which effort levels. None goes back to the task."""
    tested = [pair for pair in pairs if (verdicts.get(pair[0]) or {}).get("served")]
    names = list(dict.fromkeys(name for name, _ in tested))
    while True:
        ui.out()
        ui.line("Which AI models should it forecast?")
        ui.out()
        named = cli.in_words(names) if len(names) <= 3 else f"{len(names)} AI models"
        choices = [
            keys.Choice("Tested AI models", f"{named}: passed the test" if names else "none passed yet"),
            keys.Choice("Choose from a list", f"every one you've used, {len(pairs)} in all"),
            keys.Choice("Back"),
        ]
        picked = keys.choose(choices)
        if picked is None or picked == 2:
            return None
        if picked == 0 and not tested:
            ui.note("No AI model passed the test yet. Choose from the list, where each one says how sure it is.")
            continue
        shown = tested if picked == 0 else pairs
        rows = [keys.Choice(_label(pair, shown), status(*pair, runs, verdicts)) for pair in shown]
        notes = []
        if picked == 1 and too_few:
            listed = cli.in_words([f"{name} ({count:,})" for name, count in too_few])
            notes.append(f"Too few of your tasks to forecast: {listed}.")
        ui.out()
        ticked = checklist.tick(rows, notes=notes)
        if ticked is not None:
            return [shown[number] for number in ticked]


def _label(pair: tuple, every: list) -> str:
    """An AI model and its effort level, lined up with the others in the list."""
    pad = max(len(name) for name, _ in every)
    name, effort = pair
    return f"{name.ljust(pad)}  {effort or ''}".rstrip()


def show(task: str, result: dict | None, predictor) -> str:
    """The forecasts, a page at a time, then what's next. Returns task, models or home."""
    forecasts = (result or {}).get("forecasts") or []
    page, count = 0, max(1, -(-len(forecasts) // PAGE))
    while True:
        if forecasts:
            ui.out()
            ui.line(f"{ui.bold(task)}" + (f"  {ui.dim(f'page {page + 1} of {count}')}" if count > 1 else ""))
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
            ("home", keys.Choice("Back to home")),
        ]
        picked = keys.choose([choice for _, choice in steps], footer=[keys.hint("esc home")])
        step = "home" if picked is None else steps[picked][0]
        if step == "next":
            page += 1
        elif step == "back":
            page -= 1
        else:
            return step
