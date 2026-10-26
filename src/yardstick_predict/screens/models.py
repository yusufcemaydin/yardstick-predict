"""Models: your models, and for each one Update, Make default and Rename."""

from __future__ import annotations

from .. import cli, ui
from ..store import BrokenPredictor, Busy, Predictor, default_name, name_problem, predictor_names, set_default
from . import keys, listing


def run() -> None:
    """List the models. Choosing one opens its screen. Back or Esc goes home."""
    while True:
        names = predictor_names()
        if not names:
            ui.note("There's no model yet. Choose Add New to make one.")
            return
        ui.out()
        ui.line(ui.bold("Models"))
        ui.out()
        choices = [keys.Choice(name, when) for name, when in listing.models()] + [keys.Choice("Back")]
        picked = keys.choose(choices)
        if picked is None or picked == len(names):
            return
        name: str | None = names[picked]
        while name is not None:
            name = model(name)


def model(name: str) -> str | None:
    """One model's screen: what it knows and learns from, then what you can do with it. Returns the name of the
    model to show again, or None to go back to the list."""
    from ..pipeline import model_meta
    from . import update

    try:
        predictor = Predictor(name)
    except BrokenPredictor as error:
        ui.error(str(error))
        return None
    facts = cli.trained_facts(predictor, model_meta(predictor))
    tasks = predictor.count_tasks()
    config = predictor.config
    predictor.close()
    ui.out()
    ui.line(ui.bold(name))
    ui.line(ui.join([f"{tasks:,} task{'' if tasks == 1 else 's'}", listing.updated(name)]))
    served = facts["forecasts_for"]
    ui.line(f"Forecasts on for {cli.in_words(served)}" if served else "No forecasts yet")
    ui.line(f"Learns from: {listing.learns_from(config)}")
    ui.out()
    actions = ["update", "default", "rename", "back"]
    if default_name() == name:
        actions.remove("default")
    picked = keys.choose([CHOICES[action] for action in actions])
    action = "back" if picked is None else actions[picked]
    if action == "update":
        update.run([name])
        return name
    if action == "default":
        set_default(name)
        ui.ok(f"{name} is now the default. Estimate and Claude Code's status line use it.")
        return name
    if action == "rename":
        return rename(name)
    return None


CHOICES = {
    "update": keys.Choice("Update", "Learn from your newest work"),
    "default": keys.Choice("Make default", "Use it when you don't pick one"),
    "rename": keys.Choice("Rename", "Give it another name"),
    "back": keys.Choice("Back"),
}


def rename(old: str) -> str:
    """Ask for a new name and give it. Returns the model's name afterwards, new or not."""
    from ..store import rename_predictor

    ui.note("Use 1 to 40 lowercase letters, digits, dashes or underscores. Press Enter to keep the name.")
    while True:
        new = input(ui.prompt(f"New name for {old}:")).strip()
        if not new or new == old:
            ui.note(f"It's still called {old}.")
            return old
        problem = name_problem(new)
        if problem:
            ui.note(problem)
            continue
        if new in predictor_names():
            ui.note(f"There's already a model called {new}. Pick another name.")
            continue
        was_default = default_name() == old
        try:
            rename_predictor(old, new)
        except (ValueError, Busy) as error:
            ui.error(str(error))
            return old
        except OSError as error:
            ui.error(f"Couldn't rename {old}: {error.strerror or error}. Nothing changed.")
            return old
        cli.say_renamed(old, new, was_default)
        return new
