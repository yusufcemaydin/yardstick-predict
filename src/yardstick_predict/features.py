"""Features: what the model knows about a task before it runs.

The prompt is always used. These can be added:

- context: the conversation before the prompt, embedded.
- skills: the skills the agent used, embedded by their descriptions.
- benchmarks: public facts about each AI model, such as its prices and context size.

Every task also brings its AI model, thinking level, agent, who started it and its prompt's length.

Developers can add their own features with register_feature(). A text feature is embedded like the
prompt. A number feature goes to the model as it is.
"""

from __future__ import annotations

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
    "skills": "the skills the agent used, embedded by their descriptions",
    "benchmarks": "public facts about each AI model: prices, context size, longest answer and release date",
}
DEFAULT = ("prompt", "context")
TEXT = {"prompt", "context", "skills"}
CONTEXT_CHARS = 1_500
"""How much of the conversation, from its end, the context feature embeds."""
SKILLS_CHARS = 4_000

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
_CUSTOM: dict[str, tuple[str, Callable[[TaskInfo], object]]] = {}


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
    project: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def of(cls, task: Task) -> TaskInfo:
        return cls(
            prompt=task.prompt,
            model=task.model,
            effort=task.effort,
            agent=task.agent,
            origin=task.origin,
            context=task.context,
            skills=[s for s in task.skills.split("\n") if s],
            project=task.project,
        )


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
    value = _CUSTOM[name][1](info)
    return str(value)[:TEXT_CHARS] if value else ""


def numbers(chosen: list[str], info: TaskInfo, table: dict | None) -> dict[str, float]:
    """The number features of a task."""
    found: dict[str, float] = {}
    for name in chosen:
        if name == "benchmarks":
            found.update(benchmarks.facts(table or {}, info.model))
        elif name in _CUSTOM and _CUSTOM[name][0] == "number":
            value = _CUSTOM[name][1](info)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found[name] = float(value)
    return found


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
