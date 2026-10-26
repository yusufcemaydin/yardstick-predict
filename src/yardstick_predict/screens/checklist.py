"""A checklist shown a page at a time: each choice with a box to tick, 6 to a page.

Space or a choice's number ticks it or unticks it. The up and down arrows move, and move on to the next page
at its end. The left and right arrows turn the page. Enter goes on once at least one is ticked, and Esc goes
back. A terminal that can't pass single key presses gets each page numbered, and numbers typed in.
"""

from __future__ import annotations

from .. import ui
from . import keys

PAGE = 6
"""How many choices a page shows."""


def tick(
    choices: list[keys.Choice], ticked: set[int] | None = None, notes: list[str] | None = None
) -> list[int] | None:
    """Show the choices with a box before each and return the numbers of the ticked ones, from 0, in order. None
    when Esc goes back. `ticked` are ticked at the start. `notes` go under the page, such as what a mark means."""
    ticked = set(ticked or ())
    read = keys.reader()
    if read is None:
        return _typed(choices, ticked, notes or [])
    at, drawn, problem = 0, 0, ""
    while True:
        drawn = _draw(choices, ticked, at, notes or [], problem, drawn)
        problem = ""
        key = read()
        first = at - at % PAGE
        if key == keys.ENTER:
            if ticked:
                return sorted(ticked)
            problem = "Tick at least one first."
        elif key == keys.ESC:
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
        name = f"{number}. {box} {choice.label}"
        tail = " " * (pad - ui.width(choice.label) + 2) + ui.dim(choice.hint) if choice.hint else ""
        if place == at:
            found.append(f"  {ui.accent(ui.mark('pick'))} {ui.bold(name)}{tail}")
        else:
            found.append(f"    {name}{tail}")
    return found


def _status(choices: list, ticked: set[int], first: int) -> str:
    count = len(ticked)
    parts = [f"{count} ticked"]
    if pages(choices) > 1:
        parts.insert(0, f"Page {first // PAGE + 1} of {pages(choices)}")
    return ui.join(parts)


def _draw(choices, ticked: set[int], at: int, notes: list[str], problem: str, drawn: int) -> int:
    """Draw the page holding the picked choice, over the one drawn before. Every page takes as many lines, so a
    shorter last page leaves nothing of a longer one behind. Returns how many lines it drew."""
    first = at - at % PAGE
    lines = _page(choices, ticked, first, at)
    if pages(choices) > 1:
        lines += [""] * (PAGE - len(lines))
    turn = [f"{ui.mark('pages')} page"] if pages(choices) > 1 else []
    hint = ui.join([f"{ui.mark('keys')} move", "space tick", *turn, "enter go on"])
    lines += [
        "",
        f"  {ui.dim(_status(choices, ticked, first))}",
        f"  {ui.yellow(problem)}" if problem else "",
        *(f"  {ui.dim(note)}" for note in notes),
        f"  {ui.dim(ui.join([hint, 'esc go back']))}",
    ]
    if drawn:
        print(f"\033[{drawn}F", end="")  # back to the first line drawn before, to draw again in place
        for text in lines:
            print(f"\033[2K{text}", flush=True)
    else:
        for text in lines:
            ui.out(text)
    return len(lines)


def _typed(choices: list[keys.Choice], ticked: set[int], notes: list[str]) -> list[int]:
    """The checklist for a terminal that can't pass single key presses: a page at a time, numbers typed in."""
    first = 0
    while True:
        for text in _page(choices, ticked, first, None):
            ui.out(text)
        ui.out()
        ui.line(ui.dim(_status(choices, ticked, first)))
        for note in notes:
            ui.note(note)
        more = " n or p for another page," if pages(choices) > 1 else ""
        reply = input(ui.prompt(f"Numbers to tick or untick,{more} Enter to go on:")).strip().lower()
        if not reply:
            if ticked:
                return sorted(ticked)
            ui.note("Tick at least one first.")
            continue
        for word in reply.replace(",", " ").split():
            if word == "n" and first + PAGE < len(choices):
                first += PAGE
            elif word == "p":
                first = max(first - PAGE, 0)
            elif word.isdigit() and 1 <= int(word) <= min(PAGE, len(choices) - first):
                ticked ^= {first + int(word) - 1}
