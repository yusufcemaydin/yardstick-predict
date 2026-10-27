"""How the command looks in a terminal.

A few quiet styles: bold headings, a cyan mark for questions, a green mark for what worked, a red one
for what didn't, and grey for notes. When the output goes to a file or another program, or NO_COLOR is
set, it is plain text without marks. FORCE_COLOR turns the styles on anyway. With numbered menus asked for, as a
screen reader needs, nothing redraws itself in place: no progress bar, and every menu is typed into.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import textwrap
import time
import unicodedata
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
MOST_BAR = 28
"""How many columns a progress bar takes where the terminal has room..."""
LEAST_BAR = 10
"""...and how few it keeps in a narrow one, where the time left and the step's words make way first."""
LEAST_LABEL = 8
"""A step's words in front of its bar are left out, rather than cut shorter than this."""


def styled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if os.environ.get("TERM") == "dumb" or not sys.stdout.isatty():
        return False
    return sys.platform != "win32" or _windows_colors()


def codes_work() -> bool:
    """Whether the terminal acts on the codes that move the cursor and clear lines, which menus redraw with. A
    Windows console needs them turned on, which this does. A dumb terminal has none."""
    if os.environ.get("TERM") == "dumb":
        return False
    return sys.platform != "win32" or _windows_colors()


NUMBERED = "YARDSTICK_NUMBERED_MENUS"
"""Set to 1 for menus with numbers to type, and nothing on screen that redraws itself, as a screen reader needs. The
setting numbered_menus, true in settings.json, does the same."""


@lru_cache(maxsize=1)
def numbered() -> bool:
    """Whether numbered menus were asked for: every menu typed into, and nothing that redraws itself in place. It's
    looked up once a command."""
    asked = os.environ.get(NUMBERED, "").strip().lower()
    if asked:
        return asked not in ("0", "false", "no")
    from .store import settings

    return settings().get("numbered_menus") is True


def redraws() -> bool:
    """Whether a line may be drawn over, as a progress bar draws over itself: in a styled terminal, unless numbered
    menus were asked for."""
    return styled() and not numbered()


@lru_cache(maxsize=1)
def _windows_colors() -> bool:
    """Turn on the codes for color and for moving the cursor in a Windows console. Windows Terminal has them on
    already."""
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


def number_typed(text: str) -> bool:
    """Whether a key or an answer is a whole number in the digits 0 to 9. Python's own test also takes ² and other
    marks that look like digits, which int() can't read, as a key on some keyboards types."""
    return text.isascii() and text.isdigit()


def width(text: str) -> int:
    """How many columns a text takes on screen, leaving out color codes. A wide character, as Chinese and Japanese
    use, takes two. A mark that sits on the character before it takes none."""
    shown = _ANSI.sub("", text)
    return len(shown) if shown.isascii() else sum(_columns_of(char) for char in shown)


def _columns_of(char: str) -> int:
    if unicodedata.combining(char) or char in "​‌‍️":
        return 0
    return 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1


def columns() -> int:
    """How wide the terminal is, 80 when it doesn't say."""
    return shutil.get_terminal_size((80, 24)).columns


def plain(text: str) -> str:
    """Text without its color codes."""
    return _ANSI.sub("", text)


_ESCAPE = re.compile(
    r"\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"  # a code that colors text or moves the cursor
    r"|[\]PX^_][^\x07\x1b]*(?:\x07|\x1b\\)?"  # a string for the terminal itself, such as a window title or a link
    r"|[ -~]"  # a code of one more character
    r")?"  # an Esc on its own goes too
)
_HIDDEN = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f؜‎‏‪-‮⁦-⁩]")
"""Characters a terminal acts on instead of showing, other than new lines and tabs: a carriage return, which writes
over a line, and the marks that turn text the other way round."""


def clean(text: str) -> str:
    """Text from outside this program, such as another service's error, made safe to show: without the codes and
    hidden characters that could make a terminal show something else, change its title or write over a line."""
    return _HIDDEN.sub("", _ESCAPE.sub("", str(text)))


def cut(text: str, room: int) -> str:
    """Text cut short to fit `room` columns, with a mark where it was cut. Text that fits keeps its colors. Text
    that's cut loses them, so no color code is cut in two."""
    if width(text) <= room:
        return text
    plain = _ANSI.sub("", text)
    end = mark("wait")
    if room <= width(end):
        return _start(plain, room)
    return _start(plain, room - width(end)).rstrip() + end


def _start(text: str, room: int) -> str:
    """As much of the start of a text as fits `room` columns."""
    used = 0
    for place, char in enumerate(text):
        used += _columns_of(char)
        if used > room:
            return text[:place]
    return text


def wrapped(text: str, lead: int = 2) -> list[str]:
    """Plain text broken into lines that fit the terminal after `lead` columns, so none of them wraps on its own.

    It breaks between words, never at a hyphen inside one, and never inside a command, so a command can be copied
    whole. A word too long for a line, such as a long path, is broken where the line ends. A command too long for
    one keeps a line of its own, and the terminal wraps it."""
    return _words(text, max(columns() - lead - 1, 20))


def _words(text: str, room: int) -> list[str]:
    glued = _COMMAND.sub(lambda found: found.group().replace(" ", _GLUE), text)
    parts = []
    for part in textwrap.wrap(glued, room, break_long_words=False, break_on_hyphens=False) or [""]:
        if _GLUE in part or len(part) <= room:
            parts.append(part.replace(_GLUE, " "))
        else:  # a word too long for a line, such as a path, is broken where the line ends
            parts += [part[at : at + room] for at in range(0, len(part), room)]
    return parts


_COMMAND = re.compile(
    r"\b(?:uv tool upgrade yardstick-predict|yardstick-predict)"
    r"(?: (?:(?:create|delete|check|default|rename|--predictor)(?: (?!--)[^\s\"]+)?"
    r"|refresh|predict|status|list|features|claude-code|install|remove|--[a-z-]+|[A-Z]{2,}\b|\"[^\"]*\"))*"
)
"""A command in a message, with its subcommands, flags, names and quoted text. A line never breaks inside one, so
it can be copied whole: a command broken over two lines would run as two."""
_GLUE = "\ue000"


_ONE_STYLE = re.compile(r"(\033\[[0-9;]*m)([^\033]*)(\033\[[0-9;]*m)")


def _broken(text: str, room: int) -> list[str]:
    """Text broken into lines of `room` columns, 20 at least. Text in one style keeps it on every line. Text in
    more loses them, so no color code is broken in two."""
    if width(text) <= room:
        return [text]
    whole = _ONE_STYLE.fullmatch(text)
    if whole is None:
        return _words(plain(text), max(room, 20))
    start, words, end = whole.groups()
    return [f"{start}{part}{end}" for part in _words(words, max(room, 20))]


def fit(text: str, lead: int = 2) -> list[str]:
    """A line as it fits the terminal after `lead` columns: as it is when it fits, else broken into lines that start
    as far in as it does."""
    if width(text) <= columns() - lead - 1:
        return [text]
    body = text.lstrip(" ")
    indent = len(text) - len(body)
    return [" " * indent + part for part in _broken(body, columns() - lead - indent - 1)]


def join(parts: list[str]) -> str:
    """Short facts on one line, split by a quiet dot, or by commas in plain text."""
    return dim(f" {mark('dot')} ").join(parts) if styled() else ", ".join(parts)


def _clear() -> None:
    global _transient
    if _transient:
        print("\r\033[2K", end="")
        _transient = False


def out(text: str = "") -> None:
    """A line, or a few, as they are. One too wide for the terminal is broken to fit, so none wraps on its own."""
    global _label
    _clear()
    _label = None  # a step that printed a line of its own is done
    for given in text.split("\n"):
        for part in fit(given, 0):
            print(part, flush=True)


def heading(title: str) -> None:
    out()
    out(bold(title))


def line(text: str = "") -> None:
    """A line under a heading. One too wide for the terminal is broken to fit."""
    out(f"  {text}" if text else "")


def note(text: str) -> None:
    line(dim(text))


def ok(text: str) -> None:
    _marked(green(mark("ok")), text)


def fail(text: str) -> None:
    """What didn't work, with its mark. It's cleaned first, as it may hold another service's words."""
    _marked(red(mark("fail")), clean(text))


def _marked(sign: str, text: str) -> None:
    """Lines with a mark in front, in a styled terminal, broken to fit. The lines after the first line up after it."""
    lead = width(sign) + 1 if styled() else 0
    for number, part in enumerate(row for given in text.splitlines() for row in fit(given, 2 + lead)):
        line(f"{sign if number == 0 else ' ' * (lead - 1)} {part}" if lead else part)


def working(text: str) -> None:
    """Say what's happening. In a terminal, the next line printed replaces it, and a progress bar keeps saying it."""
    global _transient, _label
    if redraws():
        _clear()
        # Cut short to fit, as a line that wraps can't be drawn over: only its last row would go.
        shown = cut(text, max(columns() - 4 - width(mark("wait")), 1))
        print(f"  {dim(mark('wait') + ' ' + shown)}", end="", flush=True)
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
    """A bar for a long step, which replaces itself as it moves. It only shows in a terminal, and not with numbered
    menus, where nothing redraws itself.

    Once the step has run a few seconds and done a few percent, it also says about how long is left. A bar
    starts when one with another total or unit comes, or when it goes back, so a step starts its own with
    progress(0, total, unit). A unit of "bytes" shows sizes, such as 120 MB.
    """
    global _transient, _bar
    if not redraws():
        return
    now = _now()
    bar = _bar
    if bar is None or (bar.total, bar.unit) != (total, unit) or done < bar.done:
        bar = _bar = _Bar(total, unit, started=now, first=done, done=done, at=now)
    else:
        bar.move(done, now)
    counts = f"{size(done)} / {size(total)}" if unit == "bytes" else f"{done:,} / {total:,} {unit}"
    left = bar.left(now)
    tail = f" {mark('dot')} {time_left(left)}" if left is not None else ""
    # The whole line fits the terminal, as a line that wraps can't be drawn over. The counts always show. Then a
    # bar of at least LEAST_BAR, the time left, and the step's own words in front, cut short when they must be.
    # The bar grows into what's left, up to MOST_BAR.
    room = columns() - 3
    if LEAST_BAR + 2 + width(counts) + width(tail) > room:
        tail = ""
    after = width(counts) + width(tail)
    label = ""
    if _label and room - after - LEAST_BAR - 4 >= LEAST_LABEL:
        label = cut(_label, room - after - LEAST_BAR - 4)
    length = min(MOST_BAR, room - after - 2 - (width(label) + 2 if label else 0))
    if length < 1:
        shown = cut(counts, max(room, 1))
    else:
        filled = min(round(length * done / total), length) if total else length
        track = accent(mark("bar") * filled) + dim(mark("rest") * (length - filled))
        shown = f"{dim(label)}  " if label else ""
        shown += f"{track}  {counts}{dim(tail)}"
    print(f"\r\033[2K  {shown}", end="", flush=True)
    _transient = True


def prompt(question: str, hint: str = "") -> str:
    """The text in front of an answer: a question and, dimmed, the answer you get by pressing Enter. A question too
    wide for the terminal is broken into lines, with the answer typed after the last."""
    _clear()
    lead = 4 if styled() else 2
    parts = [question]
    if lead + width(question) + width(hint) + 2 > columns() - 1:
        parts = wrapped(plain(question), lead)
        if hint and lead + width(parts[-1]) + width(hint) + 2 > columns() - 1:
            parts.append("")  # the hint goes on a line of its own
    if hint:
        parts[-1] = f"{parts[-1]} {dim(hint)}".lstrip(" ")
    first = f"  {accent('?')} " if styled() else "  "
    return first + f"\n{' ' * lead}".join(parts) + " "


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
    """Rows in lined-up columns, `gap` spaces apart. align has an l or r for each column; the default is l.

    A row too wide for the terminal breaks its last cell: the words go on under it, or, with under 20 columns left
    there, on lines of their own under the row."""
    if not rows:
        return
    count = max(len(row) for row in rows)
    widths = [max((width(row[i]) for row in rows if i < len(row)), default=0) for i in range(count)]
    start = indent + sum(widths[:-1]) + gap * (count - 1)
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            pad = " " * (widths[i] - width(cell))
            right = i < len(align) and align[i] == "r"
            cells.append(pad + cell if right else (cell + pad if i < len(row) - 1 else cell))
        shown = " " * indent + (" " * gap).join(cells).rstrip()
        if width(shown) <= columns() - 1 or len(row) < count or count == 1:
            out(shown)
            continue
        head = " " * indent + (" " * gap).join(cells[:-1])
        if columns() - 1 - start >= 20:
            first, *rest = _broken(row[-1], columns() - 1 - start)
            out(f"{head}{' ' * gap}{first}")
        else:
            out(head.rstrip())
            rest = _broken(row[-1], columns() - 1 - indent - 4)
        for part in rest:
            out(" " * (start if columns() - 1 - start >= 20 else indent + 4) + part)


def error(text: str) -> None:
    """A problem that stops the command, on the error stream. It's cleaned first, as it may hold another service's
    words."""
    _clear()
    loud = styled() and sys.stderr.isatty()
    sign = _style("31", "39", mark("fail")) if loud else "Error:"
    parts = clean(text).splitlines() or [""]
    if sys.stderr.isatty():  # broken to fit the terminal it's shown in
        parts = [part for given in parts for part in fit(given, width(sign) + 1)]
    first, *rest = parts
    print(f"{sign} {first}", file=sys.stderr)
    for part in rest:
        print(f"  {part}", file=sys.stderr)


def stopped() -> None:
    """Say the command stopped, as when you press Ctrl+C, or Ctrl+D at a question. On the error stream.

    So a program reading a JSON answer on standard output never finds these words in it.
    """
    _clear()
    quiet = _style("2", "22", "Stopped.") if sys.stderr.isatty() else "Stopped."
    print(f"\n  {quiet}", file=sys.stderr, flush=True)
