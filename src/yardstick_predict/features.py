"""Features: what the model knows about a task before it runs.

Always used, at no cost: the prompt's meaning and layout, the AI model, the effort level, the agent, who
started the task, its similar past tasks, the project's recent tasks and the conversation so far. These
can be added:

- context: the end of the conversation before the prompt, embedded.
- skills: the skills known before the task starts, embedded by their descriptions: those named in the
  prompt, and those used earlier in the conversation.
- benchmarks: public facts about each AI model, such as its prices and context size.
- llm: an AI model reads each task before it starts and describes the work.

Developers can add their own features with register_feature(). A text feature is embedded like the
prompt. A number feature goes to the model as it is.
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import benchmarks
from .archives import TEXT_CHARS, Task

REQUIRED = "prompt"
BUILT_IN = {
    "context": "the end of the conversation before the prompt, embedded",
    "skills": "the skills named in the prompt or used earlier in the conversation, embedded by their descriptions",
    "benchmarks": "public facts about each AI model: prices, context size, longest answer and release date",
    "llm": "an AI model reads each task before it starts and describes the work",
}
DEFAULT = ("prompt", "context")
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
    def of(cls, task: Task, skills: list[str] | None = None, llm: dict | None = None) -> TaskInfo:
        """A recorded task. `skills` are the ones known before it started; by default, those its prompt names."""
        return cls(
            prompt=task.prompt,
            model=task.model,
            effort=task.effort,
            agent=task.agent,
            origin=task.origin,
            context=task.context,
            skills=named_skills(task.prompt) if skills is None else skills,
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
    Group("history", "History", "the project's recent tasks, the conversation so far and its size", always=True),
    Group(
        "context",
        "Conversation",
        "the end of the conversation before each prompt",
        (Option("meaning", "its meaning, read by Voyage"),),
    ),
    Group(
        "skills",
        "Skills",
        "the skills known before a task starts, read by Voyage with their descriptions",
        (Option("named", "skills named in the prompt"), Option("earlier", "skills used earlier in the conversation")),
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


def named_skills(prompt: str) -> list[str]:
    """The installed skills a prompt names as a slash command, like /review. A path like /src/app isn't one."""
    known = skill_descriptions()
    names = set(_COMMAND.findall(prompt)) | set(_SLASH.findall(prompt))
    return sorted(name for name in names if name in known or name.split(":")[-1] in known)


_COMMAND = re.compile(r"<command-name>/?([\w:.-]+)</command-name>")
_SLASH = re.compile(r"(?:^|(?<=[\s(]))/([a-z][\w:-]*)(?![\w/-])", re.MULTILINE)


def register_feature(name: str, kind: str, function: Callable[[TaskInfo], object]) -> None:
    """Add a feature of your own. `kind` is "text" to embed it, or "number" to use it as it is.

    The function gets a TaskInfo and returns a string or a number, or None when it has nothing. For
    forecast requests, a task's "extra" object reaches the function as `info.extra`.
    """
    if not _NAME.match(name) or name in TEXT or name in BUILT_IN:
        raise ValueError(f"Pick a new feature name of lowercase letters, digits and underscores, not {name!r}.")
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
    """The text a text feature embeds. Empty when the task has none."""
    if name == "prompt":
        return info.prompt[:TEXT_CHARS]
    if name == "context":
        return info.context[-CONTEXT_CHARS:]
    if name == "skills":
        return skills_text(info.skills)
    if name == "llm":
        return str(info.llm.get("description") or "")[:TEXT_CHARS]
    value = _CUSTOM[name][1](info)
    return str(value)[:TEXT_CHARS] if value else ""


def numbers(chosen: list[str], info: TaskInfo, table: dict | None) -> dict[str, float]:
    """The number features of a task: the ones always used, then the chosen ones."""
    found = shape(info.prompt)
    if info.effort in LEVELS:
        found["effort_level"] = float(LEVELS[info.effort])
    for name in chosen:
        if name == "benchmarks":
            found.update(benchmarks.facts(table or {}, info.model))
        elif name == "llm":
            from . import llm

            found.update(llm.numbers(info.llm))
        elif name in _CUSTOM and _CUSTOM[name][0] == "number":
            value = _CUSTOM[name][1](info)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found[name] = float(value)
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


def _skill_folders() -> list[Path]:
    home = Path(os.path.expanduser("~"))
    return [home / ".claude" / "skills", home / ".claude" / "plugins", home / ".codex" / "skills"]


@lru_cache(maxsize=1)
def skill_descriptions() -> dict[str, str]:
    """Each installed skill's description, by name, read from the skills' own SKILL.md files."""
    found: dict[str, str] = {}
    for folder in _skill_folders():
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("SKILL.md")):
            try:
                head = path.read_text(errors="replace")[:4000]
            except OSError:
                continue
            match = re.search(r"^description:\s*(.+)$", head, re.MULTILINE)
            if not match:
                continue
            description = match.group(1).strip().strip("\"'")
            if re.fullmatch(r"[|>][-+]?", description):
                description = _block(head[match.end() :], description[0])
            named = re.search(r"^name:\s*(.+)$", head, re.MULTILINE)
            for key in {path.parent.name, named.group(1).strip().strip("\"'") if named else path.parent.name}:
                found.setdefault(key, description)
    return found


def _block(rest: str, style: str) -> str:
    """The indented lines after a YAML | or >, as one text. | keeps the line breaks, > joins the lines with spaces."""
    lines = []
    for line in rest.split("\n")[1:]:
        if line.strip() and not line[0].isspace():
            break
        lines.append(line)
    if style == ">":
        return " ".join(line.strip() for line in lines if line.strip())
    return textwrap.dedent("\n".join(lines)).strip()


def skills_text(skills: list[str]) -> str:
    """The skills as one text: each skill's name and description. A plugin's skill is found by its own name."""
    known = skill_descriptions()
    lines = []
    for skill in sorted(set(skills)):
        if ": " in skill:
            lines.append(skill)
            continue
        description = known.get(skill) or known.get(skill.split(":")[-1], "")
        lines.append(f"{skill}: {description}" if description else skill)
    return "\n".join(lines)[:SKILLS_CHARS]
