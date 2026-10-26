"""A list to pick from, 6 to a page, where several can be ticked.

Enter picks the line you're on. To pick several, tick them with space first: Enter then picks the ticked ones.
A choice's number picks it at once, as in every menu, or ticks it once some are ticked. The up and down arrows
move, on to the next page at the end of one. The left and right arrows turn the page. Esc goes back. A
terminal that can't pass single key presses gets each page numbered, and the numbers typed in.
"""

from __future__ import annotations

from .. import ui
from . import keys

PAGE = 6
"""How many choices a page shows."""


def tick(
    choices: list[keys.Choice], ticked: set[int] | None = None, notes: list[str] | None = None
) -> list[int] | None:
    """Show the choices with a box before each and return the numbers of the ones picked, from 0, in order: the
    ticked ones, or the one Enter was pressed on when none is ticked. None when Esc goes back. `ticked` are ticked
    at the start. `notes` go under the page, such as what a mark means."""
    ticked = set(ticked or ())
    read = keys.reader()
    if read is None:
        return _typed(choices, notes or [])
    at, drawn = 0, 0
    while True:
        drawn = _draw(choices, ticked, at, notes or [], drawn)
        key = read()
        first = at - at % PAGE
        if key == keys.ENTER:
            return sorted(ticked) if ticked else [at]
        if key == keys.ESC:
            return None
        elif key == keys.UP:
            at = (at - 1) % len(choices)
        elif key == keys.DOWN:
            at = (at + 1) % len(choices)
        elif key == keys.RIGHT:
            at = first + PAGE if first + PAGE < len(choices) else at
        elif key == keys.LEFT:
            at = max(first - PAGE, 0)
        elif key == keys.SPACE:
            ticked ^= {at}
        elif key.isdigit() and 1 <= int(key) <= min(PAGE, len(choices) - first):
            at = first + int(key) - 1
            if not ticked:
                _draw(choices, ticked, at, notes or [], drawn)
                return [at]
            ticked ^= {at}


def pages(choices: list) -> int:
    return max(1, -(-len(choices) // PAGE))


def _page(choices: list[keys.Choice], ticked: set[int], first: int, at: int | None) -> list[str]:
    """The page that starts at `first`: each choice numbered from 1 on the page, with its box, lined up."""
    shown = choices[first : first + PAGE]
    pad = max(ui.width(choice.label) for choice in choices)
    found = []
    for number, choice in enumerate(shown, start=1):
        place = first + number - 1
        box = ui.accent(ui.mark("ticked")) if place in ticked else ui.dim(ui.mark("unticked"))
        name = f"{number}. {box} {ui.bold(choice.label) if place == at else choice.label}"
        lead = f"  {ui.accent(ui.mark('pick'))} " if place == at else "    "
        found.append(keys.row(lead, name, pad - ui.width(choice.label), choice.hint))
    return found


def _status(choices: list, ticked: set[int], first: int) -> str:
    """Which page this is, when there are several, and how many are ticked, when any are."""
    parts = [f"Page {first // PAGE + 1} of {pages(choices)}"] if pages(choices) > 1 else []
    if ticked:
        parts.append(f"{len(ticked)} ticked")
    return ui.join(parts)


def _hint(choices: list, ticked: set[int]) -> str:
    """Which keys do what, with what Enter picks now."""
    enter = f"enter forecast the {len(ticked)} ticked" if ticked else "enter forecast this one"
    turn = [f"{ui.mark('pages')} page"] if pages(choices) > 1 else []
    space = "space tick or untick" if ticked else "space tick several"
    return ui.join([f"{ui.mark('keys')} move", enter, space, *turn, "esc go back"])


def _draw(choices, ticked: set[int], at: int, notes: list[str], drawn: int) -> int:
    """Draw the page holding the picked choice, over the one drawn before. Every page takes as many lines, so a
    shorter last page leaves nothing of a longer one behind. Returns how many lines it drew."""
    first = at - at % PAGE
    lines = _page(choices, ticked, first, at)
    if pages(choices) > 1:
        lines += [""] * (PAGE - len(lines))
    lines += ["", *keys.notes([_status(choices, ticked, first), *notes, _hint(choices, ticked)])]
    return keys.redraw(lines, drawn)


def _typed(choices: list[keys.Choice], notes: list[str]) -> list[int]:
    """The list for a terminal that can't pass single key presses: a page at a time, the numbers to pick typed in,
    such as 2, or 1 3 for two. n and p turn the page, also between numbers, as in 1 n 2."""
    first = 0
    while True:
        for text in _page(choices, set(), first, None):
            ui.out(text)
        ui.out()
        if pages(choices) > 1:
            ui.line(ui.dim(_status(choices, set(), first)))
        for note in notes:
            ui.note(note)
        more = " n or p turns the page." if pages(choices) > 1 else ""
        reply = input(ui.prompt(f"Type the numbers to forecast, such as 2 or 1 3.{more}")).strip().lower()
        picked: set[int] = set()
        for word in reply.replace(",", " ").split():
            if word == "n" and first + PAGE < len(choices):
                first += PAGE
            elif word == "p":
                first = max(first - PAGE, 0)
            elif word.isdigit() and 1 <= int(word) <= min(PAGE, len(choices) - first):
                picked.add(first + int(word) - 1)
        if picked:
            return sorted(picked)
        if not reply:
            ui.note("Type at least one number.")
