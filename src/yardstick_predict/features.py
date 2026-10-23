"""Features: what the model knows about a task before it runs.

Always used, at no cost: the prompt's meaning and layout, the AI model, the effort level, the agent, who
started the task, its similar past tasks and the project's recent tasks. These can be added:

- conversation: the conversation so far: what its earlier tasks took and how big it is. It costs nothing.
- context: the end of the conversation before the prompt, embedded.
- skills: the skills known before the task runs, embedded by their descriptions: the ones its prompt
  invokes, and the ones used earlier in the conversation.
- benchmarks: public facts about each AI model, such as its prices and context size.
- llm: an AI model reads each task before it starts and describes the work.

Developers can add their own features with register_feature(). A text feature is embedded like the
prompt. A number feature goes to the model as it is.
"""

from __future__ import annotations

import json
import math
import numbers as _numbers
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import benchmarks
from .archives import TEXT_CHARS, Task, prompt_skills

REQUIRED = "prompt"
BUILT_IN = {
    "conversation": "the conversation so far: what its earlier tasks took and how big it is",
    "context": "the end of the conversation before the prompt, embedded",
    "skills": "the skills the prompt invokes or the conversation used before it, embedded by their descriptions",
    "benchmarks": "public facts about each AI model: prices, context size, longest answer and release date",
    "llm": "an AI model reads each task before it starts and describes the work",
}
DEFAULT = ("prompt", "conversation", "context")
TEXT = {"prompt", "context", "skills", "llm"}
CONTEXT_CHARS = 1_500
"""How much of the conversation, from its end, the context feature embeds."""
SKILLS_CHARS = 4_000
LEVELS = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5, "ultra": 5}
"""Effort levels in order. As one scale, a level with few tasks can learn from the levels next to it."""

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_CUSTOM: dict[str, tuple[str, Callable[[TaskInfo], object]]] = {}
_LIST_ITEM = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]")
_FILE_NAME = re.compile(
    r"[\w./-]+\.(?:py|ts|tsx|js|jsx|mjs|md|json|go|rs|swift|kt|java|css|scss|html|sql|ya?ml|toml|sh|rb|php|c|h|cpp)\b"
)
_LINK = re.compile(r"https?://")


@dataclass
class TaskInfo:
    """A task as features see it, whether it comes from past records or from a forecast request."""

    prompt: str
    model: str
    effort: str = "default"
    agent: str = ""
    origin: str = "program"
    context: str = ""
    skills: list[str] = field(default_factory=list)
    """The skills known before the task starts. The ones the agent goes on to use aren't known yet."""
    project: str = ""
    extra: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    """What an AI model said about the task before it ran, when the predictor uses LLM analysis."""

    @classmethod
    def of(cls, task: Task, skill_options: list[str] | None = None, llm: dict | None = None) -> TaskInfo:
        """A past task as a forecast would have seen it before it ran.

        Its skills are the ones known then: those its prompt invokes and those used earlier in its
        conversation, or only the ones `skill_options` keeps. The skills it went on to use are known only
        after it ran, so a forecast can't have them. `llm` is what LLM analysis said about the task.
        """
        return cls(
            prompt=task.prompt,
            model=task.model,
            effort=task.effort,
            agent=task.agent,
            origin=task.origin,
            context=task.context,
            skills=known_skills([s for s in (task.prior_skills or "").split("\n") if s], task.prompt, skill_options),
            project=task.project,
            llm=llm or {},
        )


@dataclass(frozen=True)
class Option:
    key: str
    says: str


@dataclass(frozen=True)
class Group:
    """One feature as the setup shows it, with the options under it."""

    key: str
    title: str
    says: str
    options: tuple[Option, ...] = ()
    always: bool = False


MENU = (
    Group(
        "task",
        "Task",
        "what the prompt means and how it's laid out, the AI model, the effort level and who started it",
        always=True,
    ),
    Group(
        "similar",
        "Similar past tasks",
        "the closest matches, how much they varied, and a first guess from the prompt's meaning",
        always=True,
    ),
    Group("history", "Project history", "the project's recent tasks, from any conversation", always=True),
    Group(
        "conversation",
        "Conversation",
        "the conversation so far, before each prompt",
        (Option("so_far", "what its earlier tasks took and how big it is"),),
    ),
    Group(
        "context",
        "Context",
        "the end of the conversation before each prompt",
        (Option("meaning", "its meaning, read by Voyage"),),
    ),
    Group(
        "skills",
        "Skills",
        "the skills known before a task starts, read by Voyage with their descriptions",
        (
            Option("named", "skills the prompt invokes, like /review"),
            Option("earlier", "skills used earlier in the conversation"),
        ),
    ),
    Group(
        "benchmarks",
        "Model facts",
        "public facts about each AI model",
        (Option("facts", "prices, context size, longest answer and release date"),),
    ),
    Group(
        "llm",
        "LLM analysis",
        "an AI model reads each task before it starts and describes the work",
        (
            Option("kind", "the kind of task and its expected size"),
            Option("description", "a short description of the work, read by Voyage"),
        ),
    ),
)
"""The features as the setup lists them. The first three are always used."""


def options(config: dict, name: str) -> list[str]:
    """The options a predictor uses for one feature. A feature saved before it had options uses them all."""
    group = next((g for g in MENU if g.key == name), None)
    every = [o.key for o in group.options] if group else []
    chosen = (config.get("options") or {}).get(name)
    return [o for o in every if o in chosen] if isinstance(chosen, list) else every


def known_skills(known: list[str], prompt: str, chosen: list[str] | None = None) -> list[str]:
    """The skills known before a task starts that the skills options keep.

    "named" keeps the ones the prompt invokes, and "earlier" the ones the conversation used before it. A
    skill the prompt invokes counts as named, even when the conversation used it before. Without options,
    both are kept.
    """
    chosen = ["named", "earlier"] if chosen is None else chosen
    named = prompt_skills(prompt)
    found = set(named) if "named" in chosen else set()
    if "earlier" in chosen:
        found |= set(known) - named
    return sorted(found)


def register_feature(name: str, kind: str, function: Callable[[TaskInfo], object]) -> None:
    """Add a feature of your own. `kind` is "text" to embed it, or "number" to use it as it is.

    The function gets a TaskInfo and returns a string or a number, or None when it has nothing. A number
    may be any real, finite number, NumPy's too. A bool is not a number here. For forecast requests, a
    task's "extra" object reaches the function as `info.extra`.
    """
    if not _NAME.match(name) or name in TEXT or name in BUILT_IN:
        raise ValueError(f"Pick a new feature name of lowercase letters, digits and underscores, not {name!r}.")
    if name.startswith("llm_"):
        raise ValueError(f"Names that start with llm_ are kept for LLM analysis. Pick another one than {name!r}.")
    if kind not in ("text", "number"):
        raise ValueError('A feature\'s kind is "text" or "number".')
    _CUSTOM[name] = (kind, function)


def available() -> dict[str, str]:
    """Every feature that can be added to the prompt, with what it is."""
    return {**BUILT_IN, **{name: f"your own {kind} feature" for name, (kind, _) in _CUSTOM.items()}}


def check(chosen: list[str] | tuple[str, ...]) -> list[str]:
    """The prompt first, then each chosen feature once. Raises ValueError for a feature that doesn't exist."""
    unknown = [name for name in chosen if name != REQUIRED and name not in available()]
    if unknown:
        raise ValueError(f"Unknown feature {unknown[0]!r}. Choose from: {', '.join(available())}.")
    return [REQUIRED] + [name for name in dict.fromkeys(chosen) if name != REQUIRED]


def is_text(name: str) -> bool:
    return name in TEXT or _CUSTOM.get(name, ("",))[0] == "text"


def text(name: str, info: TaskInfo) -> str:
    """The text a text feature embeds. Empty when the task has none.

    The prompt is cut the way the records are read, without spaces at its ends. So a prompt sent for a
    forecast is the same text as the task it becomes in training, and the cache knows its vector.
    """
    if name == "prompt":
        return info.prompt.strip()[:TEXT_CHARS].strip()
    if name == "context":
        return info.context[-CONTEXT_CHARS:]
    if name == "skills":
        return skills_text(info.skills)
    if name == "llm":
        return str(info.llm.get("description") or "")[:TEXT_CHARS]
    value = _CUSTOM[name][1](info)
    return str(value)[:TEXT_CHARS] if value else ""


def number(value: object) -> float | None:
    """A real, finite number as a float, NumPy's included. None for anything else, and for a bool."""
    if isinstance(value, bool) or not isinstance(value, _numbers.Real):
        return None
    try:
        found = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return found if math.isfinite(found) else None


def numbers(chosen: list[str], info: TaskInfo, table: dict | None) -> dict[str, float]:
    """The number features of a task: the ones always used, then the chosen ones."""
    found = shape(text("prompt", info))  # the prompt as training reads it, however long the one sent
    if info.effort in LEVELS:
        found["effort_level"] = float(LEVELS[info.effort])
    for name in chosen:
        if name == "benchmarks":
            found.update(benchmarks.facts(table or {}, info.model))
        elif name == "llm":
            from . import llm

            found.update(llm.numbers(info.llm))
        elif name in _CUSTOM and _CUSTOM[name][0] == "number":
            value = number(_CUSTOM[name][1](info))
            if value is not None:
                found[name] = value
    return found


def _file_names(prompt: str) -> int:
    """How many words of the prompt are or hold a file name. Each word is cut short, so a long run of text
    without spaces stays quick to search."""
    return sum(1 for word in prompt.split() if _FILE_NAME.search(word[:300]))


def shape(prompt: str) -> dict[str, float]:
    """How the prompt is laid out: its list items, file names, links and lines, and whether it ends in a question.

    A prompt that lists ten steps or names twelve files usually means more work than one that doesn't.
    """
    return {
        "prompt_list_items": math.log1p(len(_LIST_ITEM.findall(prompt))),
        "prompt_file_names": math.log1p(_file_names(prompt)),
        "prompt_links": math.log1p(len(_LINK.findall(prompt))),
        "prompt_lines": math.log1p(prompt.count("\n") + 1),
        "prompt_asks": float(prompt.rstrip().endswith("?")),
    }


# Skills -------------------------------------------------------------------------------------------------


SKIPPED_FOLDERS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages"})
"""Folders that hold code a skill uses, never a skill of its own. Looking through them would take long."""
SKILLS_SAVED_FOR = 86_400.0
"""How long the descriptions found are kept, at most. Any change to the skill folders is noticed sooner."""
FRONTMATTER_CHARS = 16_000


def _skill_folders() -> list[Path]:
    home = Path(os.path.expanduser("~"))
    return [home / ".claude" / "skills", home / ".claude" / "plugins", home / ".codex" / "skills"]


def _changed_at(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _skill_files(roots: list[Path]) -> tuple[list[str], dict[str, int | None]]:
    """Every SKILL.md in the skill folders, and the time each folder and file looked at last changed.

    It follows links to folders, as when a skill is installed as a link. A folder reached a second time,
    through another link or a link that loops back, is not looked through again.
    """
    found: list[str] = []
    watched: dict[str, int | None] = {}
    seen: set = set()
    for root in roots:
        watched[str(root)] = _changed_at(str(root))
        if not root.is_dir():
            continue
        for folder, subfolders, names in os.walk(root, followlinks=True):
            try:
                info = os.stat(folder)
            except OSError:
                subfolders[:] = []
                continue
            watched[folder] = info.st_mtime_ns
            if "SKILL.md" in names:
                path = os.path.join(folder, "SKILL.md")
                found.append(path)
                watched[path] = _changed_at(path)
            node = (info.st_dev, info.st_ino) if info.st_ino else os.path.realpath(folder)
            if node in seen:
                subfolders[:] = []
                continue
            seen.add(node)
            subfolders[:] = sorted(name for name in subfolders if name not in SKIPPED_FOLDERS)
    return found, watched


def _saved_path() -> Path:
    from . import store

    return store.home() / "cache" / "skills.json"


def _saved_descriptions(roots: list[Path]) -> dict[str, str] | None:
    """The descriptions found by an earlier look, when no folder or file it looked at changed since."""
    try:
        saved = json.loads(_saved_path().read_text())
        if saved["version"] != 1 or saved["roots"] != [str(root) for root in roots]:
            return None
        if not 0 <= time.time() - saved["made_at"] < SKILLS_SAVED_FOR:
            return None
        if any(_changed_at(path) != changed for path, changed in saved["watched"].items()):
            return None
        descriptions = saved["descriptions"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return descriptions if isinstance(descriptions, dict) else None


def _save_descriptions(roots: list[Path], watched: dict, descriptions: dict[str, str]) -> None:
    from . import embeddings, files

    saved = {
        "version": 1,
        "roots": [str(root) for root in roots],
        "made_at": time.time(),
        "watched": watched,
        "descriptions": descriptions,
    }
    try:
        embeddings.cache_folder()
        files.write_text(_saved_path(), json.dumps(saved) + "\n")
    except OSError:
        pass  # the next process looks again


@lru_cache(maxsize=1)
def skill_descriptions() -> dict[str, str]:
    """Each installed skill's description, by name, read from the frontmatter of the skills' own SKILL.md files.

    Looking through the skill folders can take a second, so what it finds is kept in the yardstick folder.
    Another process uses that until a folder or SKILL.md it looked at changes.
    """
    roots = _skill_folders()
    saved = _saved_descriptions(roots)
    if saved is not None:
        return saved
    paths, watched = _skill_files(roots)
    found: dict[str, str] = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as file:
                head = file.read(FRONTMATTER_CHARS)
        except OSError:
            continue
        lines = frontmatter(head)
        description = yaml_text(lines, "description") if lines is not None else None
        if not description:
            continue
        name = yaml_text(lines, "name")
        for key in dict.fromkeys([Path(path).parent.name, name or Path(path).parent.name]):
            found.setdefault(key, description)
    _save_descriptions(roots, watched, found)
    return found


def skills_text(skills: list[str]) -> str:
    """The skills as one text: each skill's name and description. A plugin's skill is found by its own name."""
    names = sorted({skill.strip() for skill in skills if skill and skill.strip()})
    if not names:
        return ""
    known = skill_descriptions()
    lines = []
    for skill in names:
        if ": " in skill:
            lines.append(skill)
            continue
        description = known.get(skill) or known.get(skill.split(":")[-1], "")
        lines.append(f"{skill}: {description}" if description else skill)
    return "\n".join(lines)[:SKILLS_CHARS]


# Reading a SKILL.md's frontmatter -----------------------------------------------------------------------


def frontmatter(text: str) -> list[str] | None:
    """The lines of a file's frontmatter: between a first line of --- and the next --- or ... line. None without."""
    lines = re.split(r"\r\n|\r|\n", text.lstrip("\ufeff"))
    if not lines or lines[0].rstrip() != "---":
        return None
    for end in range(1, len(lines)):
        if lines[end].rstrip() in ("---", "..."):
            return lines[1:end]
    return lines[1:]


def yaml_text(lines: list[str], key: str) -> str | None:
    """The text of a top-level key in YAML frontmatter, without spaces at its ends. None when it has none.

    It reads the forms a text takes in YAML: plain, in 'single' or "double" quotes, each over one line or
    several, and the | and > blocks, with or without - or + and a number for the indentation.
    """
    pattern = re.compile(rf"{re.escape(key)}[ \t]*:(?:[ \t]+(.*))?$")
    for at, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            found = _scalar((match.group(1) or "").strip(), lines[at + 1 :]).strip()
            return found or None
    return None


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _scalar(first: str, rest: list[str]) -> str:
    """A YAML text that begins with `first`, on the key's own line, and may go on into the lines after it."""
    block = re.match(r"([|>])([1-9][-+]?|[-+][1-9]?)?(?:[ \t]+#.*)?$", first)
    if block:
        indent = re.search(r"[1-9]", block.group(2) or "")
        return _block(rest, block.group(1), int(indent.group()) if indent else None)
    if first.startswith('"'):
        return _quoted(first[1:], rest, '"')
    if first.startswith("'"):
        return _quoted(first[1:], rest, "'")
    return _plain(first, rest)


def _continuation(rest: list[str]) -> list[str]:
    """The lines after a key that still belong to its text: blank ones, and ones indented more than the key."""
    taken = []
    for line in rest:
        if line.strip() and not line[0].isspace():
            break
        taken.append(line)
    return taken


def _plain(first: str, rest: list[str]) -> str:
    """A text without quotes. Its lines join with spaces, a blank line becomes a line break, and # starts a note."""
    words = [re.split(r"[ \t]+#", first, maxsplit=1)[0].strip()]
    if first.startswith("#"):
        words = [""]
    for line in _continuation(rest):
        content = line.strip()
        if content.startswith("#"):
            break
        cut = re.split(r"[ \t]+#", content, maxsplit=1)
        words.append(cut[0].strip())
        if len(cut) > 1:
            break
    return _join(words)


def _join(lines: list[str]) -> str:
    """Join a text's lines as YAML folds them: with a space, and a line break for each blank line between."""
    out = ""
    after_text = False
    for line in lines:
        if not line:
            out += "\n"
            after_text = False
        else:
            out += (" " if after_text else "") + line
            after_text = True
    return out.strip("\n ")


_ESCAPES = {
    "0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n", "v": "\v", "f": "\f", "r": "\r",
    "e": "\x1b", " ": " ", '"': '"', "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0", "L": "\u2028", "P": "\u2029",
}  # fmt: skip


def _unescape(text: str) -> str:
    def replace(match: re.Match) -> str:
        code = match.group(1)
        if code[0] in "xuU":
            try:
                return chr(int(code[1:], 16))
            except ValueError:
                return match.group(0)
        return _ESCAPES.get(code, match.group(0))

    return re.sub(r"\\(x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)", replace, text)


def _closing(line: str, quote: str) -> int:
    """Where an unescaped closing quote is in a line, or -1. In single quotes, '' stands for one quote."""
    at = 0
    while at < len(line):
        if quote == '"' and line[at] == "\\":
            at += 2
            continue
        if line[at] == quote:
            if quote == "'" and line[at + 1 : at + 2] == "'":
                at += 2
                continue
            return at
        at += 1
    return -1


def _quoted(first: str, rest: list[str], quote: str) -> str:
    """A text in quotes, over one line or several. Its lines join as YAML folds them.

    In "double" quotes a backslash starts an escape such as \\n, and one at the end of a line joins the
    next line on without a space. In 'single' quotes, '' stands for one quote.
    """
    lines = []
    for line in [first, *rest]:
        end = _closing(line, quote)
        lines.append(line if end < 0 else line[:end])
        if end >= 0:
            break
    last = len(lines) - 1
    joined = ""
    for n, line in enumerate(lines):
        part = line if n == 0 else line.lstrip(" \t")
        if n < last:
            part = part.rstrip(" \t")
        if n == 0:
            joined = part
        elif quote == '"' and (len(joined) - len(joined.rstrip("\\"))) % 2 == 1:
            joined = joined[:-1] + part
        elif not part:
            joined += "\n"
        else:
            joined += ("" if not joined or joined.endswith("\n") else " ") + part
    return _unescape(joined) if quote == '"' else joined.replace("''", "'")


def _block(lines: list[str], style: str, indent: int | None = None) -> str:
    """The lines of a YAML | or > block, as one text. | keeps the line breaks, > folds lines into one.

    The block's lines are the ones indented at least as far as its first line, or as far as the number
    after | or > says.
    """
    taken: list[str] = []
    for line in lines:
        if not line.strip():
            taken.append("")
            continue
        if indent is None:
            indent = _indent(line)
            if indent == 0:
                break
        if _indent(line) < indent:
            break
        taken.append(line[indent:])
    while taken and not taken[-1]:
        taken.pop()
    if style == "|":
        return "\n".join(taken).strip()
    out = ""
    previous = None
    for line in taken:
        if not line:
            out += "\n"
            previous = "blank"
        elif line[0] in " \t":  # a more indented line keeps its line breaks
            out += ("\n" if previous in ("text", "more") else "") + line
            previous = "more"
        else:
            out += (" " if previous == "text" else "\n" if previous == "more" else "") + line
            previous = "text"
    return out.strip()
