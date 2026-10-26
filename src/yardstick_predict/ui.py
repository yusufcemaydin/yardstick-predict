"""How the command looks in a terminal.

A few quiet styles: bold headings, a cyan mark for questions, a green mark for what worked, a red one
for what didn't, and grey for notes. When the output goes to a file or another program, or NO_COLOR is
set, it is plain text without marks. FORCE_COLOR turns the styles on anyway.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from functools import lru_cache

_MARKS = {
    "ok": ("✓", "+"),
    "fail": ("✗", "x"),
    "wait": ("…", "..."),
    "bar": ("━", "#"),
    "rest": ("─", "-"),
    "dot": ("·", "-"),
    "pick": ("›", ">"),
    "warn": ("⚠", "!"),
    "pages": ("←→", "left and right"),
    "ticked": ("●", "[x]"),
    "unticked": ("○", "[ ]"),
    "logo": ("◆", "*"),
    "keys": ("↑↓", "up and down"),
    "box": ("╭╮╰╯─│", "++++-|"),
}
_ANSI = re.compile(r"\033\[[0-9;]*m")
_transient = False
"""Whether the last thing printed was a line the next output replaces, like a progress bar."""
_label: str | None = None
"""What the step that's working says it does, so its progress bar keeps saying it."""

LEFT_AFTER_SECONDS = 5.0
"""A progress bar says how long is left only once it has run this long..."""
LEFT_AFTER_SHARE = 0.03
"""...and done this share of the work, so its first guess isn't a wild one."""
SMOOTHING_SECONDS = 60.0
"""How fast a bar moves counts its last minute or so the most, and older work less and less. So one slow or
quick batch doesn't make the time left jump, and a step that slows down, such as one waiting on a limit, still
shows it."""


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


def yellow(text: str) -> str:
    return _style("33", "39", text)


def width(text: str) -> int:
    """How many columns a text takes on screen, leaving out color codes."""
    return len(_ANSI.sub("", text))


def columns() -> int:
    """How wide the terminal is, 80 when it doesn't say."""
    return shutil.get_terminal_size((80, 24)).columns


def cut(text: str, room: int) -> str:
    """Text cut short to fit `room` columns, with a mark where it was cut. Text that fits keeps its colors. Text
    that's cut loses them, so no color code is cut in two."""
    if width(text) <= room:
        return text
    plain = _ANSI.sub("", text)
    return plain[: max(room - 1, 0)].rstrip() + mark("wait") if room > 1 else plain[:room]


def wrapped(text: str, lead: int = 2) -> list[str]:
    """Plain text broken into lines that fit the terminal after `lead` columns, so none of them wraps on its own."""
    return textwrap.wrap(text, max(columns() - lead - 1, 20)) or [""]


def join(parts: list[str]) -> str:
    """Short facts on one line, split by a quiet dot, or by commas in plain text."""
    return dim(f" {mark('dot')} ").join(parts) if styled() else ", ".join(parts)


def _clear() -> None:
    global _transient
    if _transient:
        print("\r\033[2K", end="")
        _transient = False


def out(text: str = "") -> None:
    global _label
    _clear()
    _label = None  # a step that printed a line of its own is done
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
    """Say what's happening. In a terminal, the next line printed replaces it, and a progress bar keeps saying it."""
    global _transient, _label
    if styled():
        _clear()
        print(f"  {dim(mark('wait') + ' ' + text)}", end="", flush=True)
        _transient = True
        _label = text
    else:
        line(f"{text}...")


def size(count: float) -> str:
    """A number of bytes read at a glance: 800 KB, 3.4 MB, 120 MB, 1.2 GB. Never 1000 KB: that's 1.0 MB."""
    if count >= 999.5e6:
        return f"{count / 1e9:.1f} GB"
    if count >= 9.95e6:
        return f"{count / 1e6:.0f} MB"
    if count >= 999.5e3:
        return f"{count / 1e6:.1f} MB"
    return f"{max(count, 0) / 1e3:.0f} KB"


def hours(seconds: float) -> str:
    """Hours to a tenth, without a needless 0: 1.5, 2."""
    return f"{seconds / 3600:.1f}".removesuffix(".0")


def time_left(seconds: float) -> str:
    """How long is left, in words calm enough to watch: under a minute, about 4 min, about 1.5 h."""
    if seconds < 60:
        return "under a minute left"
    if seconds < 90 * 60:
        return f"about {max(round(seconds / 60), 1)} min left"
    return f"about {hours(seconds)} h left"


def _now() -> float:
    """Seconds on a clock that only goes forward. Tests replace it to pass minutes at once."""
    return time.monotonic()


@dataclass
class _Bar:
    """The progress bar being drawn: how far it has got, and how fast it has gone lately."""

    total: int
    unit: str
    started: float
    first: int
    """How much was done when the bar started."""
    done: int
    at: float
    """When the bar last moved."""
    work: float = 0.0
    """The work done so far, the older the less it counts."""
    took: float = 0.0
    """The time it took, counted the same way. The work over this is how fast the bar moves lately."""

    def move(self, done: int, now: float) -> None:
        fade = math.exp(-max(now - self.at, 0.0) / SMOOTHING_SECONDS)
        self.work = self.work * fade + max(done - self.done, 0)
        self.took = self.took * fade + max(now - self.at, 0.0)
        self.done, self.at = done, now

    def left(self, now: float) -> float | None:
        """Seconds left at the pace so far. None until there's enough to go on, and once it's done."""
        enough = now - self.started >= LEFT_AFTER_SECONDS and self.done - self.first >= LEFT_AFTER_SHARE * self.total
        if not enough or self.done >= self.total or self.work <= 0 or self.took <= 0:
            return None
        return (self.total - self.done) * self.took / self.work


_bar: _Bar | None = None


def progress(done: int, total: int, unit: str) -> None:
    """A bar for a long step, which replaces itself as it moves. It only shows in a terminal.

    Once the step has run a few seconds and done a few percent, it also says about how long is left. A bar
    starts when one with another total or unit comes, or when it goes back, so a step starts its own with
    progress(0, total, unit). A unit of "bytes" shows sizes, such as 120 MB.
    """
    global _transient, _bar
    if not styled():
        return
    now = _now()
    bar = _bar
    if bar is None or (bar.total, bar.unit) != (total, unit) or done < bar.done:
        bar = _bar = _Bar(total, unit, started=now, first=done, done=done, at=now)
    else:
        bar.move(done, now)
    length = 28
    filled = min(round(length * done / total), length) if total else length
    track = accent(mark("bar") * filled) + dim(mark("rest") * (length - filled))
    counts = f"{size(done)} / {size(total)}" if unit == "bytes" else f"{done:,} / {total:,} {unit}"
    left = bar.left(now)
    tail = dim(f" {mark('dot')} {time_left(left)}") if left is not None else ""
    shown = f"{track}  {counts}{tail}"
    if _label:
        # The step's own words stay in front of its bar, cut short so the line never wraps.
        room = shutil.get_terminal_size((80, 24)).columns - 1 - 2 - width(shown) - 2
        label = _label if len(_label) <= room else _label[: max(room - 1, 0)].rstrip() + mark("wait")
        shown = f"{dim(label)}  {shown}" if room >= 8 else shown
    print(f"\r\033[2K  {shown}", end="", flush=True)
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


def table(rows: list[tuple[str, ...]], indent: int = 2, align: str = "", gap: int = 3) -> None:
    """Rows in lined-up columns, `gap` spaces apart. align has an l or r for each column; the default is l."""
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
        out(" " * indent + (" " * gap).join(cells).rstrip())


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
