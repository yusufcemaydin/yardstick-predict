"""The home screen: a greeting, your models and what you can do next.

It shows only when a person runs yardstick-predict in a terminal with nothing after it. It reads only the
models' settings, so it opens at once. When a choice finishes, the home screen comes back.
"""

from __future__ import annotations

import shutil
import textwrap

from .. import __version__, cli, ui
from ..store import BrokenPredictor, default_name, predictor_names
from . import keys, listing, version

DESCRIPTION = "Forecasts a coding-agent task's time and cost before it runs"
NO_BOX_UNDER = 50
"""The box is left out in a terminal narrower than this many columns."""
BOX_MODELS = 5
"""How many models the box names, the default first. Models lists them all."""


def run() -> int:
    """Show the home screen until Quit or Esc. Returns the command's exit code."""
    while True:
        names = predictor_names()
        greet(names)
        actions = ["add", "estimate", "update", "models"] if not names else ["estimate", "update", "models", "add"]
        choices = [CHOICES[action](bool(names)) for action in actions] + [keys.Choice("Quit")]
        ui.out()
        ui.line("What would you like to do?")
        ui.out()
        picked = keys.choose(choices, footer=[keys.hint("esc quit"), "Every command: yardstick-predict --help"])
        if picked is None or picked == len(actions):
            return 0
        act(actions[picked], names)


CHOICES = {
    "estimate": lambda any_model: keys.Choice("Estimate", "Forecast a task's time and cost"),
    "update": lambda any_model: keys.Choice("Update", "Learn from your newest work"),
    "models": lambda any_model: keys.Choice("Models", "Update, rename, delete or pick the default"),
    "add": lambda any_model: keys.Choice("Add New", "Make another model" if any_model else "Make your first model"),
}


def greet(names: list[str]) -> None:
    """The box at the top: the tool's name and version, what it does and your models."""
    columns = shutil.get_terminal_size((80, 24)).columns
    # A screen reader would read out each line of the box, so numbered menus get none.
    boxed = columns >= NO_BOX_UNDER and not ui.numbered()
    # A line in the box has its edges and a space on each side. Without the box, two spaces before it.
    room = columns - 1 - (4 if boxed else 2)
    lines = [f"{ui.accent(ui.mark('logo'))} {ui.bold('yardstick-predict')} {ui.dim(__version__)}"]
    lines += [f"  {part}" for part in textwrap.wrap(DESCRIPTION, 42)]
    lines.append("")
    default = default_name()
    first = sorted(names, key=lambda name: name != default)[:BOX_MODELS]  # sorted keeps the rest in name order
    found = [(listing.label(name, default), listing.updated(name)) for name in first]
    if found:
        widest = max(ui.width(when) for _, when in found)
        pad = min(max(ui.width(name) for name, _ in found), max(room - 2 - 2 - widest, 8))
        lines.append(f"  {ui.bold('Your models')}")
        for name, when in found:
            name = ui.cut(name, pad)  # a name too long for the terminal is cut short
            lines.append(ui.cut(f"  {name}{' ' * (pad - ui.width(name))}  {ui.dim(when)}", room))
        if len(names) > len(found):
            lines.append(f"  {ui.dim(f'and {len(names) - len(found)} more in Models')}")
    else:
        lines.append("  No models yet")
    ui.out()
    if not boxed:
        for text in lines:
            ui.out(f"  {text}" if text else "")
    else:
        box(lines)
    newer = version.notice()
    if newer:
        ui.out()
        for part in ui.wrapped(newer):
            ui.line(ui.accent(part))
    if not names:
        ui.out()
        for part in ui.wrapped("Add New reads your Claude Code and Codex chats. It needs a Voyage API key."):
            ui.line(part)


def box(lines: list[str]) -> None:
    """Lines in a box with round corners, or plain ones where the terminal can't show those."""
    top_left, top_right, bottom_left, bottom_right, across, down = ui.mark("box")
    inner = max(ui.width(text) for text in lines) + 2
    edge = ui.dim
    ui.out(edge(top_left + across * inner + top_right))
    for text in lines:
        ui.out(f"{edge(down)} {text}{' ' * (inner - 2 - ui.width(text))} {edge(down)}")
    ui.out(edge(bottom_left + across * inner + bottom_right))


def act(action: str, names: list[str]) -> None:
    """Do what was chosen. A problem is said in plain words, and the home screen comes back."""
    from . import models, update

    try:
        if action == "add":
            cli.cmd_create(cli.make_parser().parse_args(["create"]))
        elif not names:
            ui.out()
            ui.note("There's no model yet. Choose Add New to make one.")
        elif action == "estimate":
            from . import estimate

            estimate.run()
        elif action == "update":
            update.run()
        else:
            models.run()
    except BrokenPredictor as error:
        ui.error(str(error))
    except SystemExit as stop:
        if not isinstance(stop.code, str):
            raise
        ui.error(stop.code)
