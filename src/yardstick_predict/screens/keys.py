"""Single key presses and the menus they move through.

A menu is moved with the arrow keys and chosen with Enter. A number key picks its choice at once, and Esc
goes back. It uses the standard library only: termios and tty on macOS and Linux, msvcrt on Windows. A
terminal that can't pass single key presses gets the choices numbered, and a number typed in.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from .. import ui

UP, DOWN, LEFT, RIGHT, ENTER, ESC, SPACE = "up", "down", "left", "right", "enter", "esc", " "
"""The keys a menu knows by name. A number key comes as its digit."""

_ARROWS = {"[A": UP, "OA": UP, "[B": DOWN, "OB": DOWN, "[C": RIGHT, "OC": RIGHT, "[D": LEFT, "OD": LEFT}
_WINDOWS_ARROWS = {"H": UP, "P": DOWN, "M": RIGHT, "K": LEFT}


@dataclass(frozen=True)
class Choice:
    """One line of a menu: what it's called and, after it, what it does."""

    label: str
    hint: str = ""


def reader() -> Callable[[], str] | None:
    """A way to read one key press at a time from this terminal, or None when it can't pass them."""
    found = _terminal()
    if found is None:
        return None
    return _windows_key if found == "windows" else _posix_key


def _terminal() -> str | None:
    """How this terminal passes single key presses: windows or posix. None when it can't."""
    try:
        if not sys.stdin.isatty():
            return None
        if os.name == "nt":
            import msvcrt  # noqa: F401

            return "windows"
        import termios

        termios.tcgetattr(sys.stdin.fileno())
    except (ImportError, OSError, ValueError, AttributeError):
        return None
    return "posix"


def _posix_key() -> str:
    """The next key pressed, with the terminal set to pass each one at once and show none of them."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd, termios.TCSANOW)  # Ctrl+C still stops it, and keys pressed ahead aren't thrown away
        first = os.read(fd, 1)
        if first != b"\x1b":
            return _named(first.decode(errors="replace"))
        # An arrow key sends more bytes at once. Esc on its own sends none.
        if not select.select([fd], [], [], 0.05)[0]:
            return ESC
        rest = os.read(fd, 8).decode(errors="replace")
        return _ARROWS.get(rest[:2], "")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _windows_key() -> str:
    import msvcrt

    first = msvcrt.getwch()
    if first in ("\x00", "\xe0"):  # an arrow key comes in two parts
        return _WINDOWS_ARROWS.get(msvcrt.getwch(), "")
    if first == "\x1b":
        return ESC
    if first == "\x03":
        raise KeyboardInterrupt
    return _named(first)


def _named(key: str) -> str:
    if key in ("\r", "\n"):
        return ENTER
    if key == "\x04":  # Ctrl+D
        return ESC
    return {"k": UP, "j": DOWN, "q": ESC}.get(key, key)


def choose(choices: list[Choice], footer: list[str] | None = None, start: int = 0) -> int | None:
    """Show a menu and return the number of the choice made, from 0. None when Esc goes back.

    `footer` lines go under the choices, such as which keys do what.
    """
    keys = reader()
    if keys is None:
        return _typed(choices)
    footer = footer if footer is not None else [hint("esc go back")]
    at, drawn = start, 0
    while True:
        drawn = _draw(choices, at, footer, drawn)
        key = keys()
        if key == ENTER:
            return at
        if key == ESC:
            return None
        if key == UP:
            at = (at - 1) % len(choices)
        elif key == DOWN:
            at = (at + 1) % len(choices)
        elif key.isdigit() and 1 <= int(key) <= len(choices):
            _draw(choices, int(key) - 1, footer, drawn)
            return int(key) - 1


def hint(back: str) -> str:
    """The line under a menu that says which keys do what. `back` is what Esc does there, such as esc quit."""
    return ui.join([f"{ui.mark('keys')} move", "enter choose", back])


def _draw(choices: list[Choice], at: int, footer: list[str], drawn: int) -> int:
    """Draw the menu with one choice picked, over the one drawn before when there was one. Returns its lines."""
    return redraw([*_lines(choices, at), "", *notes(footer)], drawn)


def columns() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def notes(texts: list[str]) -> list[str]:
    """Plain lines under a menu, dimmed and broken to fit the terminal, so none of them wraps on its own."""
    room = max(columns() - 4, 20)
    return [f"  {ui.dim(part)}" for text in texts for part in (textwrap.wrap(text, room) or [""])]


def redraw(lines: list[str], drawn: int) -> int:
    """Draw lines over the ones drawn before, when there were some, and return how many there are now.

    Each line fits the terminal, so the count of lines is the count of rows. The rows drawn before are cleared to
    the end of the screen first, so a shorter drawing leaves nothing of a longer one behind.
    """
    if drawn:
        print(f"\033[{drawn}F\033[J", end="")  # back to the first row drawn before, and clear from there down
        for text in lines:
            print(text, flush=True)
    else:
        for text in lines:
            ui.out(text)
    return len(lines)


def row(lead: str, name: str, pad: int, hint: str) -> str:
    """A menu's line: what comes before the name, the name, and its hint lined up after it. A hint too long for the
    terminal is cut short, so the line never wraps."""
    room = columns() - 1 - ui.width(lead) - ui.width(name) - 2 - pad
    if not hint or room < 4:
        return f"{lead}{name}"
    shown = hint if ui.width(hint) <= room else hint[: room - 1].rstrip() + ui.mark("wait")
    return f"{lead}{name}{' ' * (pad + 2)}{ui.dim(shown)}"


def _lines(choices: list[Choice], at: int | None) -> list[str]:
    """Each choice on its own line, numbered, with what it does lined up after it and the one picked marked."""
    pad = max(ui.width(choice.label) for choice in choices)
    found = []
    for number, choice in enumerate(choices, start=1):
        name = f"{number}. {choice.label}"
        lead = f"  {ui.accent(ui.mark('pick'))} " if number - 1 == at else "    "
        shown = ui.bold(name) if number - 1 == at else name
        found.append(row(lead, shown, pad - ui.width(choice.label), choice.hint))
    return found


def _typed(choices: list[Choice]) -> int:
    """The menu for a terminal that can't pass single key presses: every choice numbered, and a number typed in."""
    for text in _lines(choices, None):
        ui.out(text)
    ui.out()
    while True:
        reply = input(ui.prompt("Type a number:", "[1]")).strip() or "1"
        if reply.isdigit() and 1 <= int(reply) <= len(choices):
            return int(reply) - 1
        ui.note(f"Type a number from 1 to {len(choices)}.")


class Stopped(Exception):
    """Esc was pressed while a long step ran."""


@dataclass
class _Watch:
    check: Callable[[], bool]

    def stop_if_asked(self) -> None:
        """Raise Stopped when Esc was pressed since the last look."""
        if self.check():
            raise Stopped


@contextmanager
def watching() -> Iterator[_Watch]:
    """While the block runs, an Esc pressed makes stop_if_asked raise Stopped. Keys pressed meanwhile aren't shown.

    In a terminal that can't pass single key presses, Esc does nothing and Ctrl+C still stops.
    """
    found = _terminal()
    if found is None:
        yield _Watch(lambda: False)
        return
    if found == "windows":
        import msvcrt

        def pressed() -> bool:
            found = False
            while msvcrt.kbhit():
                found = msvcrt.getwch() == "\x1b" or found
            return found

        yield _Watch(pressed)
        return
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)

    def waiting() -> bool:
        found = False
        while select.select([fd], [], [], 0)[0]:
            got = os.read(fd, 16)
            found = got == b"\x1b" or found  # an arrow key starts with Esc too, but sends more with it
        return found

    tty.setcbreak(fd, termios.TCSANOW)  # an Esc pressed a moment before counts too
    try:
        yield _Watch(waiting)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
