"""How the command looks in a terminal.

A few quiet styles: bold headings, a cyan mark for questions, a green mark for what worked, a red one
for what didn't, and grey for notes. When the output goes to a file or another program, or NO_COLOR is
set, it is plain text without marks. FORCE_COLOR turns the styles on anyway.
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

_MARKS = {"ok": ("✓", "+"), "fail": ("✗", "x"), "wait": ("…", "..."), "bar": ("━", "#"), "rest": ("─", "-")}
_ANSI = re.compile(r"\033\[[0-9;]*m")
_transient = False
"""Whether the last thing printed was a line the next output replaces, like a progress bar."""


def styled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb" or not sys.stdout.isatty():
        return False
    return sys.platform != "win32" or _windows_colors()


@lru_cache(maxsize=1)
def _windows_colors() -> bool:
    """Turn on color codes in a Windows console. Windows Terminal has them on already."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def mark(name: str) -> str:
    fancy, plain = _MARKS[name]
    try:
        fancy.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return plain
    return fancy


def _style(start: str, end: str, text: str) -> str:
    return f"\033[{start}m{text}\033[{end}m" if styled() else text


def bold(text: str) -> str:
    return _style("1", "22", text)


def dim(text: str) -> str:
    return _style("2", "22", text)


def accent(text: str) -> str:
    return _style("36", "39", text)


def green(text: str) -> str:
    return _style("32", "39", text)


def red(text: str) -> str:
    return _style("31", "39", text)


def width(text: str) -> int:
    """How many columns a text takes on screen, leaving out color codes."""
    return len(_ANSI.sub("", text))


def join(parts: list[str]) -> str:
    """Short facts on one line, split by a quiet dot, or by commas in plain text."""
    return dim(" · ").join(parts) if styled() else ", ".join(parts)


def _clear() -> None:
    global _transient
    if _transient:
        print("\r\033[2K", end="")
        _transient = False


def out(text: str = "") -> None:
    _clear()
    print(text, flush=True)


def heading(title: str) -> None:
    out()
    out(bold(title))


def line(text: str = "") -> None:
    """A line under a heading."""
    out(f"  {text}" if text else "")


def note(text: str) -> None:
    line(dim(text))


def ok(text: str) -> None:
    line(f"{green(mark('ok'))} {text}" if styled() else text)


def fail(text: str) -> None:
    for number, part in enumerate(text.splitlines()):
        if number == 0:
            line(f"{red(mark('fail'))} {part}" if styled() else part)
        else:
            line(f"  {part}" if styled() else part)


def working(text: str) -> None:
    """Say what's happening. In a terminal, the next line printed replaces it."""
    global _transient
    if styled():
        _clear()
        print(f"  {dim(mark('wait') + ' ' + text)}", end="", flush=True)
        _transient = True
    else:
        line(f"{text}...")


def progress(done: int, total: int, unit: str) -> None:
    global _transient
    if not styled():
        return
    size = 28
    filled = round(size * done / total) if total else size
    bar = accent(mark("bar") * filled) + dim(mark("rest") * (size - filled))
    print(f"\r\033[2K  {bar}  {done:,} / {total:,} {unit}", end="", flush=True)
    _transient = True


def prompt(question: str, hint: str = "") -> str:
    """The text in front of an answer: a question and, dimmed, the answer you get by pressing Enter."""
    _clear()
    tail = f" {dim(hint)}" if hint else ""
    return f"  {accent('?')} {question}{tail} " if styled() else f"  {question}{tail} "


def ask(question: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        reply = input(prompt(question, hint)).strip().lower()
        if not reply:
            return default
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        note("Please answer y or n.")


def table(rows: list[tuple[str, ...]], indent: int = 2, align: str = "") -> None:
    """Rows in lined-up columns. align has an l or r for each column; the default is l."""
    if not rows:
        return
    columns = max(len(row) for row in rows)
    widths = [max((width(row[i]) for row in rows if i < len(row)), default=0) for i in range(columns)]
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            pad = " " * (widths[i] - width(cell))
            right = i < len(align) and align[i] == "r"
            cells.append(pad + cell if right else (cell + pad if i < len(row) - 1 else cell))
        out(" " * indent + "   ".join(cells).rstrip())


def error(text: str) -> None:
    """A problem that stops the command, on the error stream."""
    _clear()
    loud = styled() and sys.stderr.isatty()
    first, *rest = text.splitlines() or [""]
    print(f"{_style('31', '39', mark('fail')) if loud else 'Error:'} {first}", file=sys.stderr)
    for part in rest:
        print(f"  {part}", file=sys.stderr)


def stopped() -> None:
    """Say the command stopped, as when you press Ctrl+C, or Ctrl+D at a question. On the error stream.

    So a program reading a JSON answer on standard output never finds these words in it.
    """
    _clear()
    quiet = _style("2", "22", "Stopped.") if sys.stderr.isatty() else "Stopped."
    print(f"\n  {quiet}", file=sys.stderr, flush=True)
