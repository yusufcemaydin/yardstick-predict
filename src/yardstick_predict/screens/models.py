"""Models: your models, and for each one Update, Make default, Rename and Delete."""

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
    ui.line(ui.bold(ui.cut(name, ui.columns() - 3)))
    ui.line(ui.join([f"{tasks:,} task{'' if tasks == 1 else 's'}", listing.updated(name)]))
    served = facts["forecasts_for"]
    for text in (
        f"Forecasts on for {cli.in_words(served)}" if served else "No forecasts yet",
        f"Learns from: {listing.learns_from(config)}",
    ):
        for part in ui.wrapped(text):  # long lists of AI models or features are broken to fit
            ui.line(part)
    ui.out()
    actions = ["update", "default", "rename", "delete", "back"]
    if default_name() == name:
        actions.remove("default")
    picked = keys.choose([CHOICES[action] for action in actions])
    action = "back" if picked is None else actions[picked]
    if action == "update":
        update.run([name])
        return name
    if action == "default":
        set_default(name)
        done(f"{name} is now the default.")
        note("Estimate and Claude Code's status line use it.")
        return name
    if action == "rename":
        return rename(name)
    if action == "delete":
        return delete(name)
    return None


CHOICES = {
    "update": keys.Choice("Update", "Learn from your newest work"),
    "default": keys.Choice("Make default", "Use it when you don't pick one"),
    "rename": keys.Choice("Rename", "Give it another name"),
    "delete": keys.Choice("Delete", "Remove it for good"),
    "back": keys.Choice("Back"),
}


def rename(old: str) -> str:
    """Ask for a new name and give it. Returns the model's name afterwards, new or not."""
    from ..store import rename_predictor

    note("Use 1 to 40 lowercase letters, digits, dashes or underscores. Press Enter to keep the name.")
    while True:
        new = input(ui.prompt(f"New name for {old}:")).strip()
        if not new or new == old:
            note(f"It's still called {old}.")
            return old
        problem = name_problem(new)
        if problem:
            note(problem)
            continue
        if new in predictor_names():
            note(f"There's already a model called {new}. Pick another name.")
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
        done(f"Renamed {old} to {new}.")
        if was_default:
            note(f"{new} is still the default.")
        return new


def delete(name: str) -> str | None:
    """Say exactly what deleting the model removes and what stays, then ask, with keeping it first. Returns the
    model's name when it's kept, or None once it's deleted, to go back to the list."""
    from ..store import being_refreshed, contents, delete_predictor

    # A setup or update of it would make it again, so it's refused before anything is asked.
    if being_refreshed(name):
        note(f"{name} is being set up or updated by another command. Delete it once that's done.")
        return name
    was_default = default_name() == name
    ui.out()
    ui.line(ui.bold(ui.cut(f"Delete {name}", ui.columns() - 3)))
    for quiet, text in cli.what_goes(name, contents(name), was_default, called="model"):
        indent = len(text) - len(text.lstrip())
        for part in ui.wrapped(text.strip(), 2 + indent):
            ui.line(" " * indent + (ui.dim(part) if quiet else part))
    ui.out()
    choices = [keys.Choice("Keep it"), keys.Choice("Delete it for good")]
    if keys.choose(choices, footer=[keys.hint("esc keep it")]) != 1:
        note("Nothing was deleted.")
        return name
    try:
        delete_predictor(name)
    except Busy as error:
        ui.error(str(error))
        return name
    except OSError as error:
        ui.error(
            f"Couldn't delete all of {name}: {error.strerror or error}. It's still listed. Try again when no other "
            "program is using its files."
        )
        return name
    done(f"Deleted {name}.")
    left, now = predictor_names(), default_name()
    if was_default and now is not None:
        note(f"{now} is your only model now. It's the default." if len(left) == 1 else f"{now} is the default now.")
    elif was_default and left:
        note(
            "It was your default. There's no default now. Until you choose one, Claude Code's status line shows no "
            "forecasts. Choose Make default on the model you want."
        )
    return None


def done(text: str) -> None:
    """What worked, with its mark, broken to fit the terminal."""
    for number, part in enumerate(ui.wrapped(text, 4)):
        if number == 0:
            ui.ok(part)
        else:
            ui.line(f"  {part}")


def note(text: str) -> None:
    """A note broken to fit the terminal, so a long model name never runs past its edge."""
    for part in ui.wrapped(text):
        ui.note(part)
