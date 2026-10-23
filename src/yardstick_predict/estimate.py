"""How long setup and refresh take, and the room they need, said before they start.

Each step's time comes from what's known by then: how many tokens go to Voyage, how many tasks LLM analysis
reads and how many tasks training learns from. A few rough constants turn those into seconds, bytes on disk
and bytes of memory. Nothing is timed while the command runs. Training's and the dataset's constants were
measured once, on synthetic tasks on a recent laptop. Voyage's and the AI models' speeds can't be measured
without paying for them, so theirs are worked out from how the requests are sent. The time is said as a
range, because the real one varies with the computer, the network and the tasks.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from . import ui
from .archives import Task

EMBED_TOKENS_A_SECOND = 40_000
"""About how many tokens Voyage embeds a second. A rough figure, not measured: that needs a paid key. Requests
go one after another, each with up to 1,000 texts, and one takes a few seconds. That's for an account with a
payment method. An account without one may get only 10,000 tokens a minute, and then embedding waits for
Voyage's limits and takes far longer."""

LLM_SECONDS_PER_TASK = {"anthropic": 1.0, "openai": 1.0, "claude": 2.0, "codex": 1.2}
"""About how long LLM analysis takes for each task, by the way it's paid. The ChatGPT subscription's figure was
measured once, through Codex, over a few thousand tasks. The others are rough figures: an answer runs to about
200 tokens a task, written at about 50 tokens a second. An API key sends 8 tasks in a request and 4 requests at
once. A subscription sends 20 in a request, 2 at once, through the Claude Code or Codex command, which takes a
few seconds to start each time."""

TRAIN_SECONDS_PER_MODEL = 1.5
TRAIN_SECONDS_PER_TASK = 0.00035
"""Training fits a model for each test window, and then the final one, as model.fit_sizes counts them. Each
takes about 1.5 seconds, plus 0.35 milliseconds for each task it learns from. Measured on synthetic tasks
with the default features, from 1,000 to 30,000 tasks: 15 seconds for 1,000 tasks on two AI models, and
about a minute for 12,000 on three or for 30,000 on two. Working out similar tasks is counted in."""

RANGE = (0.7, 1.5)
"""The time said runs from this share of the estimate to this share of it. Memory uses the top of it."""

DISK_PER_TASK = 500
"""Bytes the dataset takes for a task besides its texts, with the record file it came from. Measured: about
360 for a task and 175 for a record file."""
DISK_PER_TEXT_BYTE = 1.25
"""Bytes the dataset takes for each byte of a task's texts. Measured: 1.04 to 1.34 for prompts and
conversations of real lengths, as the database leaves room between rows."""
DISK_PER_VECTOR = 1_450
"""Bytes the dataset takes for one embedding: 256 numbers of 4 bytes, with the hash of its text. Measured."""
DISK_PER_ANSWER = 600
"""Bytes the dataset takes for what LLM analysis said about a task. Measured: about 400, more with a
description."""

MEMORY_BASE = 300_000_000
MEMORY_PER_TASK = 100_000
"""The memory training needs: Python and its libraries, then this much for each task it learns from.
Measured as the most memory training on synthetic tasks used: 290 MB for 1,000 tasks, 1.4 GB for 12,000,
1.9 GB for 12,000 with LLM analysis and 2.6 GB for 30,000."""
MEMORY_PER_TEXT_BYTE = 3
"""The tasks' texts are in memory too, in more than one copy while training runs."""

STEPS = ("LLM analysis", "embedding", "feature check", "training")
"""The steps after the question, in the order they run."""


@dataclass
class Estimate:
    """What the steps still to come take, each at the middle of its range."""

    steps: dict[str, float]
    """Each step's seconds, by its name in STEPS. A step with nothing to do is left out."""
    disk: float
    """Bytes the dataset takes once it's done."""
    memory: float
    """The most memory training uses, in bytes."""

    @property
    def seconds(self) -> float:
        return sum(self.steps.values())


def text_bytes(tasks: list[Task]) -> int:
    """How many bytes the tasks' texts take, all told. A character outside English takes two to four."""
    return sum(len(f"{t.prompt}{t.context}{t.skills}{t.prior_skills}".encode(errors="replace")) for t in tasks)


def of(
    tasks: list[Task],
    config: dict,
    tokens: int,
    vectors: int,
    asks: int,
    new_tasks: list[Task] | None = None,
    dataset: int = 0,
    checked: int = 0,
) -> Estimate:
    """What the rest of setup or a refresh takes.

    `tasks` are all the predictor's tasks once the new ones are in, and `config` its settings: its features
    and their options, how LLM analysis runs, its model's settings and how many tasks it learns from. `tokens`
    go to Voyage and make `vectors` embeddings. LLM analysis reads `asks` tasks, and when it describes them, the
    descriptions are embedded too. `new_tasks` are the ones the dataset doesn't have yet, all of `tasks` by
    default, and `dataset` is how many bytes it takes now. With the feature check, `checked` is how many features
    it leaves out, one at a time: it fits each test window's trial once with every feature and once without
    each, and no final model.
    """
    from . import features, llm, model
    from .pipeline import learnable

    new_tasks = tasks if new_tasks is None else new_tasks
    how = config.get("llm") or {}
    analyzed = "llm" in (config.get("features") or []) and how.get("provider") in llm.PROVIDERS
    if asks and "description" in features.options(config, "llm"):
        tokens += asks * llm.DESCRIPTION_TOKENS
        vectors += asks
    settings = dict(config.get("model") or {})
    # A model type from a plugin is counted as if it trained like the built-in one.
    built_in = settings if settings.get("type", "xgboost") == "xgboost" else {}
    learn_from = learnable(tasks, config.get("max_tasks"))
    sizes = model.fit_sizes(learn_from, built_in)

    def fitting(models: list[int]) -> float:
        """How long fitting models on these many tasks each takes."""
        return sum(TRAIN_SECONDS_PER_MODEL + TRAIN_SECONDS_PER_TASK * size for size in models)

    steps = {
        "LLM analysis": asks * LLM_SECONDS_PER_TASK.get(how.get("provider"), max(LLM_SECONDS_PER_TASK.values())),
        "embedding": tokens / EMBED_TOKENS_A_SECOND,
        "feature check": (1 + checked) * fitting(sizes[:-1]) if checked else 0.0,
        "training": fitting(sizes * (2 if analyzed else 1)),
    }
    disk = (
        dataset
        + DISK_PER_TASK * len(new_tasks)
        + DISK_PER_TEXT_BYTE * text_bytes(new_tasks)
        + DISK_PER_VECTOR * vectors
        + DISK_PER_ANSWER * asks
    )
    memory = MEMORY_BASE + MEMORY_PER_TASK * len(learn_from) + MEMORY_PER_TEXT_BYTE * text_bytes(tasks)
    return Estimate({name: seconds for name, seconds in steps.items() if seconds > 0}, disk, memory)


def span(low: float, high: float) -> str:
    """A stretch of time in words: under a minute, about a minute, about 4 to 8 minutes, about 1.5 to 3 hours."""
    if high < 60:
        return "under a minute"
    if high >= 90 * 60:
        start, end = ui.hours(low), ui.hours(high)
        return f"about {end} hours" if start == end else f"about {start} to {end} hours"
    start, end = max(round(low / 60), 1), max(round(high / 60), 1)
    if start == end:
        return "about a minute" if end == 1 else f"about {end} minutes"
    return f"about {start} to {end} minutes"


def rough(seconds: float) -> str:
    """One step's time in a few words: under a minute, 3 min, 1.5 h."""
    if seconds < 60:
        return "under a minute"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min"
    return f"{ui.hours(seconds)} h"


def line(found: Estimate) -> str:
    """How long the rest takes and what it's made of, then the room the dataset takes, as one line.

    Each step's own time is said only when the whole takes a minute or more: "It takes about 4 to 8 minutes:
    embedding 3 min, training 2 min." Otherwise the steps are only named: "Embedding and training take under
    a minute."
    """
    low, high = found.seconds * RANGE[0], found.seconds * RANGE[1]
    when = span(low, high)
    names = [name for name in STEPS if name in found.steps] or ["it"]
    if len(names) > 1 and high >= 60:
        took = f"It takes {when}: {', '.join(f'{name} {rough(found.steps[name])}' for name in names)}."
    elif len(names) > 1:
        took = f"{', '.join(names[:-1])} and {names[-1]} take {when}."
    else:
        took = f"{names[0]} takes {when}."
    return f"{took[0].upper()}{took[1:]} The dataset takes about {ui.size(found.disk)} on disk."


def memory_note(found: Estimate, total: int | None) -> str | None:
    """A word on memory, only when training may use more than a quarter of the computer's. None otherwise, and
    when the computer doesn't say how much it has."""
    most = found.memory * RANGE[1]
    if not total or most <= total / 4:
        return None
    return f"Training may use up to about {ui.size(most)} of memory, more than a quarter of this computer's."


def total_memory() -> int | None:
    """How much memory this computer has, in bytes, as the operating system says. None when it doesn't say."""
    try:
        if sys.platform == "win32":
            found = _windows_memory()
        else:
            found = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:  # the operating system has no answer, or none Python can ask for
        return None
    return found if found > 0 else None


def _windows_memory() -> int:
    """The memory Windows says this computer has, in bytes, or 0 when it doesn't say."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("free_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("free_virtual", ctypes.c_ulonglong),
            ("free_extended_virtual", ctypes.c_ulonglong),
        ]

    status = Status()
    status.size = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return 0
    return int(status.total)
