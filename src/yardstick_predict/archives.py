"""Read Claude Code and Codex records into tasks.

A task is one turn that a person or a program started. It runs from the prompt until the agent
stops and hands control back. Turns that an automatic event started, such as a CI event, are not
tasks. A turn that reports back on a background job, such as a command or a helper that finished,
counts toward the task that started the job. So does the work of the helper agents a task started.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import prices

FORMATS = ("claude-code", "codex")
DEFAULT_ARCHIVES = (
    {"name": "claude-code", "format": "claude-code", "path": "~/.claude/projects"},
    {"name": "codex", "format": "codex", "path": "~/.codex/sessions"},
)

READER_VERSION = 2
"""The version of the rules that read records into tasks. A predictor read with other rules reads every record again."""

TEXT_CHARS = 24_000
"""How much of the prompt and of the context each task keeps. The context keeps its newest part."""

IDLE_GAP_SECONDS = 30 * 60
"""A task with a longer silence was left waiting, not working. It is kept but not trained on."""

SETTLE_SECONDS = 15 * 60
"""A record file written to more recently than this may still be in use. Its last task, or a helper, may still be
running."""

CODEX_TOOL_ITEMS = {
    "function_call",
    "custom_tool_call",
    "local_shell_call",
    "web_search_call",
    "tool_search_call",
}


@dataclass
class Task:
    id: str
    archive: str
    agent: str
    agent_version: str
    session: str
    project: str
    origin: str
    started_at: float
    model: str
    effort: str
    prompt: str
    context: str
    context_tokens: int
    seconds: float
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    usd: float | None
    calls: int
    direct: bool
    stopped: bool
    idle: bool
    skills: str = ""
    """The skills the agent used, one name per line."""
    finished_at: float | None = None
    """When the task and its helpers really finished, time spent waiting for the person included. None when it
    isn't known."""
    prior_skills: str = ""
    """The skills known before the task ran, one name per line: those its prompt invokes, such as a typed
    /command, and those used earlier in its session."""
    source: str = ""
    """The record file the task was read from. Empty for a task stored by an older version."""

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens + self.output_tokens


@dataclass(slots=True)
class _Call:
    model: str
    new_input: int
    cache_write_5m: int
    cache_write_1h: int
    cache_read: int
    output: int
    fast: bool = False

    def cost(self) -> float | None:
        return prices.call_cost(
            self.model,
            self.new_input,
            self.cache_write_5m,
            self.cache_write_1h,
            self.cache_read,
            self.output,
            self.fast,
        )


def _totals(calls: Iterable[_Call]) -> tuple[int, int, int, int, float | None]:
    new_input = cache_write = cache_read = output = 0
    usd: float | None = 0.0
    for call in calls:
        new_input += call.new_input
        cache_write += call.cache_write_5m + call.cache_write_1h
        cache_read += call.cache_read
        output += call.output
        cost = call.cost()
        usd = None if usd is None or cost is None else usd + cost
    return new_input, cache_write, cache_read, output, usd


def _timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except (ValueError, OverflowError, OSError):
            return None
    return None


def _number(value: object) -> int:
    """A token count. Anything that isn't a whole number counts as none."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _name(value: object) -> str | None:
    """An id or a name written as text or as a number. None when there isn't one."""
    if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "":
        return str(value)
    return None


def _shared(value: object) -> str | None:
    """A short text that many lines repeat, such as a session id or a model's name, kept once in memory."""
    found = _name(value)
    return sys.intern(found) if found is not None and len(found) < 200 else found


_NO_SKILLS: frozenset[str] = frozenset()


def _skill_set(skills: set[str]) -> frozenset[str]:
    return frozenset(skills) if skills else _NO_SKILLS


def _lines(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            try:
                value = json.loads(raw)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield value


class _History:
    """The conversation so far in one session, kept short enough to embed."""

    def __init__(self) -> None:
        self.text = ""

    def add(self, who: str, words: str) -> None:
        words = words.strip()
        if words:
            self.text = (self.text + f"\n\n{who}: {words}")[-2 * TEXT_CHARS :]

    def context(self) -> str:
        return self.text[-TEXT_CHARS:].strip()


_KEEP_CHARS = 2 * TEXT_CHARS + 1024
"""How much of one reply's text the reader keeps. The conversation keeps less than this of it anyway."""


def archive_path(archive: dict) -> Path:
    return Path(os.path.expanduser(archive["path"]))


def record_files(archive: dict) -> dict[str, tuple[int, float]]:
    """Every record file in an archive, with its size and modification time. The only walk through its folders."""
    root = archive_path(archive)
    if not root.is_dir():
        return {}
    files = {}
    for path in root.rglob("*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        files[str(path)] = (stat.st_size, stat.st_mtime)
    return files


@dataclass
class Reading:
    """What reading some of an archive's record files found."""

    tasks: list[Task]
    held: set[str] = field(default_factory=set)
    """Tasks that may still be running, such as one whose helper is still working. They're read again next time."""
    open_files: set[str] = field(default_factory=set)
    """The record files of tasks that may still be running. They must be read again next time."""
    sessions: set[str] = field(default_factory=set)
    """The sessions whose own record files were read."""


def read_archive(archive: dict, only: set[str] | None = None, now: float | None = None) -> list[Task]:
    """Read the archive's tasks. With `only`, read just those record files, with their sessions' helpers."""
    files = record_files(archive)
    if only is not None:
        files = {path: state for path, state in files.items() if path in only or _helper_session(path) in only}
    return read_records(archive, files, now).tasks


def read_records(archive: dict, files: dict[str, tuple[int, float]], now: float | None = None) -> Reading:
    """Read the tasks in some of an archive's record files, listed with their sizes and modification times.

    The list is the one walk through the archive's folders. Nothing else is looked up on disk.
    """
    now = time.time() if now is None else now
    if archive["format"] == "claude-code":
        return _read_claude_code(archive, files, now)
    if archive["format"] == "codex":
        return _read_codex(archive, files, now)
    raise ValueError(f"Unknown archive format {archive['format']!r}. Use one of: {', '.join(FORMATS)}.")


def families(archive: dict, paths: Iterable[str]) -> dict[str, str]:
    """Each record file's family. A family is read together, because its files add to the same tasks."""
    if archive["format"] == "codex":
        return codex_families(paths)
    return claude_families(paths)


def completeness(task: Task) -> tuple[int, bool]:
    """How complete a copy of a task is: its model calls first, then whether it finished rather than stopped."""
    return task.calls, not task.stopped


def keep_most_complete(tasks: dict[str, Task], task: Task) -> None:
    """Keep the copy of a task with the most model calls. With as many calls, a finished copy beats a stopped one.

    This decides only between copies from different record files or archives, such as a backup made before
    a helper finished. A copy that's more complete wins. A first read of a record whose last task was cut off
    looks stopped, so its finished copy wins.
    """
    kept = tasks.get(task.id)
    if kept is None or completeness(task) > completeness(kept):
        tasks[task.id] = task


def codex_families(paths: Iterable[str]) -> dict[str, str]:
    """Each Codex record file's family: the main thread at the top of its chain of helper threads.

    A helper's tokens count toward a task in that main thread, so a family is read together.
    """
    links: dict[str, tuple[str, str | None]] = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                first = json.loads(handle.readline() or "{}")
        except (OSError, ValueError):
            first = {}
        meta = first.get("payload") if isinstance(first, dict) and first.get("type") == "session_meta" else None
        meta = meta if isinstance(meta, dict) else {}
        links[path] = (str(meta.get("id") or Path(path).stem), _codex_spawn(meta)[1])
    parents = dict(links.values())
    found = {}
    for path, (top, _) in links.items():
        seen = {top}
        while parents.get(top) and parents[top] not in seen:
            top = parents[top]
            seen.add(top)
        found[path] = top
    return found


def _codex_spawn(meta: dict) -> tuple[bool, str | None]:
    """From a Codex thread's session_meta: whether it's a helper thread, and the thread it works for.

    A helper that another thread spawned names that thread. A helper that names none, such as one that
    reviews an approval, works for the session it names.
    """
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    parent = _name(spawn.get("parent_thread_id")) if isinstance(spawn, dict) else None
    helper = isinstance(source, dict) and "subagent" in source
    if helper and parent is None:
        session = _name(meta.get("session_id"))
        parent = session if session != _name(meta.get("id")) else None
    return helper, parent


# Claude Code ------------------------------------------------------------------------------------------

_NOT_AGENT_WORK = ("<bash-input", "<bash-stdout", "<bash-stderr", "<local-command")
_AUTOMATIC_TAGS = ("task-notification", "system-reminder", "ci-monitor-event", "artifact-", "local-command", "bash-")
"""Wrappers Claude Code puts around messages that no person or program typed. A prompt that merely
starts with some other tag, such as an orchestrator's <task>, is still a prompt."""

_BUILT_IN_COMMANDS = frozenset(
    """add-dir agents allowed-tools approved-tools bashes bug chrome clear color compact config context cost
    doctor effort exit export extra-usage fast feedback help hooks ide install-github-app keybindings login logout
    mcp memory migrate-installer model output-style permissions plugin plugins privacy-settings quit release-notes
    remote-env rename resume rewind sandbox skills status statusline stickers tasks terminal-setup theme todos
    upgrade usage vim""".split()
)
"""Claude Code's own slash commands, which change Claude Code itself rather than give the agent a skill."""

_COMMAND_NAME = re.compile(r"<command-name>\s*/?([^<>\s]+)\s*</command-name>")
"""How Claude Code records a typed /command in its record."""

_TYPED_COMMAND = re.compile(r"\s*/([A-Za-z0-9][\w.:-]*)(?=\s|$)")
"""A /command as the person typed it. A path such as /Users/... isn't one."""

PERSON_TOOLS = {"AskUserQuestion", "ExitPlanMode"}
"""Tools that wait for the person to answer or approve."""

_RUN_TIMES = (("durationMs", 0.001), ("totalDurationMs", 0.001), ("durationSeconds", 1.0))
"""Fields in a tool's result that say how long the tool ran, and the seconds in each of their units."""

_STOP_TOOLS = {"TaskStop", "KillShell"}
"""Tools that stop a background job."""

_COMMANDS = {"Bash", "PowerShell"}
"""Tools whose background job is a command, not a helper's paid work."""

_NOTICE = re.compile(r"<task-notification>(.*?)</task-notification>", re.S)


def _part_text(part: dict) -> str:
    """A text part's words. A part with no words, such as one whose text is null, has none."""
    text = part.get("text")
    return text if isinstance(text, str) else ""


def _claude_text(message: object) -> str | None:
    """The text of a user message, or None when it only carries tool results."""
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content):
            return None
        return "\n".join(_part_text(part) for part in content if isinstance(part, dict) and part.get("type") == "text")
    return None


def _claude_turn_start(line: dict) -> str | None:
    """The prompt text when this line starts a turn, else None.

    Claude Code marks lines it writes itself as meta, such as a skill's instructions. A meta line that
    says who started its turn starts one: a scheduled task, a message from another session, or a notice.
    """
    if line.get("type") != "user" or line.get("isSidechain"):
        return None
    if line.get("isMeta") and not line.get("turnOrigin"):
        return None
    if line.get("isCompactSummary") or line.get("isVisibleInTranscriptOnly"):
        return None
    text = _claude_text(line.get("message"))
    if text is None or not text.strip() or text.lstrip().startswith("[Request interrupted"):
        return None
    return text


def _claude_origin(line: dict, text: str) -> str | None:
    """Who started the turn: 'person', 'program', or None for an automatic event.

    Newer records say who started each turn. A person's ! shell command that the agent then works on is a
    person's task. A message from another session and a scheduled task are a program's. A notice from Claude
    Code itself, such as about a message to another session, is automatic. Older records don't say, so the
    prompt's wrapper decides, and a headless run, such as claude -p, is a program's.
    """
    stripped = text.lstrip()
    origin = line.get("turnOrigin")
    if origin == "human":
        return None if stripped.startswith("<local-command") else "person"
    if origin in ("sdk", "peer", "scheduled"):
        return "program"
    if origin is not None:
        return None
    if stripped.startswith(_NOT_AGENT_WORK):
        return None
    tag = re.match(r"<([A-Za-z_-]+)", stripped)
    if tag and tag.group(1) == "scheduled-task":
        return "program"
    if tag and tag.group(1).startswith(_AUTOMATIC_TAGS):
        return None
    if str(line.get("entrypoint") or "").startswith("sdk"):
        return "program"
    return "person"


def _claude_call(line: dict) -> tuple[tuple, _Call] | None:
    message = line.get("message")
    if not isinstance(message, dict):
        return None
    model = message.get("model")
    if not isinstance(model, str) or not model or model == "<synthetic>":
        return None
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    writes = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    write_total = _number(usage.get("cache_creation_input_tokens"))
    write_1h = _number(writes.get("ephemeral_1h_input_tokens"))
    write_5m = writes.get("ephemeral_5m_input_tokens")
    write_5m = (write_total - write_1h) if write_5m is None else _number(write_5m)
    call = _Call(
        model=_shared(model) or model,
        new_input=_number(usage.get("input_tokens")),
        cache_write_5m=max(write_5m, 0),
        cache_write_1h=write_1h,
        cache_read=_number(usage.get("cache_read_input_tokens")),
        output=_number(usage.get("output_tokens")),
        fast=usage.get("speed") == "fast",
    )
    return (message.get("id"), line.get("requestId")), call


def _claude_failed(line: dict) -> bool:
    """Whether a reply is a failed call that Claude Code wrote itself, such as a server error or a usage limit.

    Its placeholder for a message that needs no answer isn't a failure, and neither is a real model reply.
    """
    message = line.get("message")
    if not isinstance(message, dict) or message.get("model") != "<synthetic>":
        return False
    return bool(line.get("isApiErrorMessage") or line.get("error"))


def prompt_skills(text: str) -> set[str]:
    """The skills a prompt invokes: a /command typed at its start, or one Claude Code recorded as a command.

    Claude Code's own commands, such as /model or /compact, aren't skills.
    """
    if not isinstance(text, str):
        return set()
    names = set(_COMMAND_NAME.findall(text))
    typed = _TYPED_COMMAND.match(text)
    if typed:
        names.add(typed.group(1))
    return {name for name in names if name.lower() not in _BUILT_IN_COMMANDS}


def line_skills(line: dict) -> set[str]:
    """The skills a Claude Code assistant line used: the skill its reply is credited to, and any it loads."""
    if line.get("type") != "assistant":
        return set()
    found = set()
    if line.get("attributionSkill"):
        found.add(str(line["attributionSkill"]))
    message = line.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and part.get("type") == "tool_use" and part.get("name") == "Skill":
            given = part.get("input")
            if isinstance(given, dict) and given.get("skill"):
                found.add(str(given["skill"]))
    return found


def _notices(text: str) -> tuple[tuple[str, str, str], ...]:
    """Each background job report in a text: the job's id, the id of the tool call that started it, and its status.

    A monitor's event has no status. It tells what the monitor saw, not that a job finished.
    """
    if "<task-notification>" not in text:
        return ()
    found = []
    for block in _NOTICE.findall(text) or [text]:
        tags = {name: re.search(rf"<{name}>([^<]*)</{name}>", block) for name in ("task-id", "tool-use-id", "status")}
        found.append(tuple(match.group(1).strip() if match else "" for match in tags.values()))
    return tuple(found)


def _helper_session(path: str) -> str | None:
    """The session record that a helper's record belongs to, or None for a session's own record.

    Claude Code keeps a session's helpers under <session>/subagents, at any depth below it, such as a
    workflow's helpers in subagents/workflows/<workflow>.
    """
    parts = Path(path).parts
    if "subagents" not in parts[:-1]:
        return None
    where = parts.index("subagents")
    return str(Path(*parts[:where])) + ".jsonl" if where > 0 else None


def _first_session(path: str) -> str | None:
    """The session id of the first line in a Claude Code record that names one."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for _, raw in zip(range(50), handle):
                try:
                    value = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(value, dict) and _name(value.get("sessionId")):
                    return _name(value.get("sessionId"))
    except OSError:
        return None
    return None


def claude_families(paths: Iterable[str]) -> dict[str, str]:
    """Each Claude Code record file's family: the session it starts with.

    A resumed session copies the lines before it into a new record, and the copies keep their own session's
    id. So a copy and its original share a family. A helper's record joins its session's family.
    """
    paths = list(paths)
    starts = {path: _first_session(path) or Path(path).stem for path in paths if _helper_session(path) is None}
    found = {}
    for path in paths:
        session = _helper_session(path)
        found[path] = starts[path] if session is None else starts.get(session, Path(session).stem)
    return found


@dataclass(slots=True)
class _Line:
    """What the reader needs from one line of a Claude Code record. The rest, such as tool output, is dropped."""

    kind: str
    """'user', 'assistant', or 'other' for a line that only marks a moment."""
    uuid: str | None = None
    session: str | None = None
    moment: float | None = None
    sidechain: bool = False
    start: str | None = None
    """The prompt, when the line starts a turn."""
    origin: str | None = None
    prompt_id: str | None = None
    cwd: str = ""
    version: str = ""
    call: tuple[tuple, _Call] | None = None
    effort: str | None = None
    failed: bool = False
    tools: tuple[tuple[str, str], ...] = ()
    """The reply's tool calls: each call's id and the tool's name."""
    stops: tuple[str, ...] = ()
    """The background jobs the line stops."""
    texts: tuple[str, ...] = ()
    skills: frozenset[str] = frozenset()
    """The skills the line uses or invokes."""
    results: tuple[str, ...] = ()
    """The tool calls the line answers."""
    interrupt: bool = False
    asked: str = ""
    """Who allowed the tool to run, such as the person or the settings."""
    ran: float | None = None
    """How long the tool ran, when its result says."""
    launched: str | None = None
    """The background job the tool's result says it started."""
    helper: str | None = None
    """The helper agent the tool's result names."""
    notices: tuple[tuple[str, str, str], ...] = ()
    """Reports on background jobs that the line carries."""


def _claude_line(line: dict) -> _Line:
    kind = line.get("type")
    slim = _Line(
        kind if kind in ("user", "assistant") else "other",
        uuid=_name(line.get("uuid")),
        session=_shared(line.get("sessionId")),
        moment=_timestamp(line.get("timestamp")),
        sidechain=bool(line.get("isSidechain")),
    )
    message = line.get("message") if isinstance(line.get("message"), dict) else {}
    content = message.get("content") if isinstance(message.get("content"), list) else []
    if kind == "user":
        text = _claude_turn_start(line)
        if text is not None:
            slim.start, slim.origin = text, _claude_origin(line, text)
            slim.prompt_id = _name(line.get("promptId"))
            slim.cwd, slim.version = _shared(line.get("cwd")) or "", _shared(line.get("version")) or ""
            slim.skills = _skill_set(prompt_skills(text))
            slim.notices = _notices(text)
            return slim
        words = _claude_text(message)
        if words is not None and words.lstrip().startswith("[Request interrupted"):
            slim.interrupt = True
        elif words is not None and "<command-name>" in words:
            slim.skills = _skill_set(prompt_skills(words))  # a skill Claude Code loaded
        slim.results = tuple(
            str(part.get("tool_use_id") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "tool_result"
        )
        decision = line.get("permissionDecision")
        slim.asked = (_shared(decision.get("source")) or "") if isinstance(decision, dict) else ""
        result = line.get("toolUseResult")
        if isinstance(result, dict):
            ran = [result[name] * unit for name, unit in _RUN_TIMES if type(result.get(name)) in (int, float)]
            slim.ran = ran[0] if ran else None
            started = result.get("status") == "async_launched" or result.get("isAsync")
            slim.launched = _name(result.get("backgroundTaskId")) or (
                _name(result.get("agentId") or result.get("taskId")) if started else None
            )
            slim.helper = _name(result.get("agentId"))
    elif kind == "assistant":
        found = _claude_call(line)
        if found is None:
            slim.failed = _claude_failed(line)
            return slim
        slim.call, slim.effort = found, _shared(line.get("effort"))
        slim.skills = _skill_set(line_skills(line))
        tools, stops, texts = [], [], []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "tool_use":
                name = _shared(part.get("name")) or ""
                tools.append((str(part.get("id") or ""), name))
                given = part.get("input") if isinstance(part.get("input"), dict) else {}
                job = _name(given.get("task_id") or given.get("shell_id") or given.get("bash_id"))
                if name in _STOP_TOOLS and job:
                    stops.append(job)
            elif part.get("type") == "text" and _part_text(part):
                texts.append(_part_text(part)[-_KEEP_CHARS:])
        slim.tools, slim.stops, slim.texts = tuple(tools), tuple(stops), tuple(texts)
    elif kind == "attachment":
        attachment = line.get("attachment")
        if isinstance(attachment, dict) and attachment.get("type") == "queued_command":
            queued = attachment.get("prompt")
            slim.notices = _notices(queued) if isinstance(queued, str) else ()
    return slim


def _claude_lines(path: Path) -> list[_Line]:
    """A record's lines, slimmed as they're read. A line that marks no moment and says nothing is left out."""
    found = []
    for line in _lines(path):
        slim = _claude_line(line)
        if slim.kind != "other" or slim.moment is not None or slim.notices:
            found.append(slim)
    return found


def _helper_calls(path: Path) -> Iterator[tuple[str | None, str | None, tuple, _Call, float | None]]:
    """A helper's model calls, each with the id of the prompt it worked for, the helper's id, and when it ended.

    A helper's first instruction carries the id of the prompt that started it. Its tool results carry the id
    of whatever prompt the main session is on, so they say nothing about whom the work is for. A later
    instruction, such as a message the main agent sent it from another prompt, carries that prompt's id.
    """
    owner = agent = None
    waiting: list[tuple[tuple, _Call, float | None]] = []
    for line in _lines(path):
        agent = agent or _name(line.get("agentId"))
        if line.get("type") == "user":
            prompt_id = _name(line.get("promptId"))
            words = _claude_text(line.get("message"))
            instruction = words is not None and not words.lstrip().startswith("[Request interrupted")
            if prompt_id and (owner is None or (instruction and prompt_id != owner)):
                owner = prompt_id
                for key, call, moment in waiting:
                    yield owner, agent, key, call, moment
                waiting = []
            continue
        if line.get("type") != "assistant":
            continue
        found = _claude_call(line)
        if found is None:
            continue
        if owner is None:
            waiting.append((*found, _timestamp(line.get("timestamp"))))
        else:
            yield owner, agent, *found, _timestamp(line.get("timestamp"))
    for key, call, moment in waiting:
        yield owner, agent, key, call, moment


@dataclass
class _ClaudeFile:
    path: str
    stem: str
    mtime: float
    lines: list[_Line]


@dataclass
class _Turn:
    """One turn of a Claude Code session: the line that started it, and the lines after it until the next turn."""

    index: int
    line: _Line
    source: str
    """The record file the turn's first line was written to."""
    lines: list[_Line] = field(default_factory=list)
    files: set[str] = field(default_factory=set)
    """The record files that hold the turn's first line."""
    followed: bool = False
    """Whether another turn came after it in a record that holds it."""


def _written_first(record: _ClaudeFile) -> tuple[float, float, str]:
    """Sorts a family's records by when their own lines began, so an original comes before its copies."""
    own = [line.moment for line in record.lines if line.session == record.stem and line.moment is not None]
    return min(own, default=math.inf), record.mtime, record.path


def _claude_turns(records: list[_ClaudeFile]) -> list[_Turn]:
    """The turns of one family of records, each line counted once.

    A copy keeps its lines' ids, and a record can repeat its own earlier lines, so a line already seen is
    skipped. A copy can also leave out lines, such as the prompt that ended a turn. Its lines then belong where
    the record that has them all put them. A line seen for the first time goes to the turn its own record is on.
    """
    turns: dict[str, _Turn] = {}
    seen: set[str] = set()
    for record in records:
        current: _Turn | None = None
        for number, line in enumerate(record.lines):
            if line.start is not None:
                key = line.uuid or f"{record.path}:{number}"
                turn = turns.get(key)
                if turn is None:
                    turn = turns[key] = _Turn(len(turns), line, record.path)
                elif line.session == record.stem:
                    turn.source = record.path
                turn.files.add(record.path)
                if current is not None:
                    current.followed = True
                if line.uuid:
                    seen.add(line.uuid)
                current = turn
                continue
            if line.uuid:
                if line.uuid in seen:
                    continue
                seen.add(line.uuid)
            elif line.session and line.session != record.stem:
                continue  # a copied line that has no id to tell it by
            if current is not None:
                current.lines.append(line)
    return list(turns.values())


@dataclass
class _Part:
    """What one turn adds to its task: its own model calls, and when the agent worked."""

    start: float | None
    calls: list[_Call] = field(default_factory=list)
    first_input: int | None = None
    model: str = ""
    effort: str = ""
    last: float | None = None
    moments: list[float] = field(default_factory=list)
    waits: list[tuple[float, float]] = field(default_factory=list)
    used_tools: bool = False
    stopped: bool = False
    failed: bool = False
    open_call: bool = False
    """Whether it ended on a tool call that never got its result."""
    skills: set[str] = field(default_factory=set)

    def worked(self) -> tuple[float, bool]:
        """How long the agent worked in this turn, waits for the person left out, and whether it was left waiting."""
        if self.start is None or self.last is None:
            return 0.0, False
        clipped = [(max(began, self.start), min(ended, self.last)) for began, ended in self.waits]
        waits = [(began, ended) for began, ended in clipped if began < ended]
        seconds = max(self.last - self.start - _covered(waits), 0.0)
        return seconds, _longest_gap(self.start, self.last, self.moments, waits) > IDLE_GAP_SECONDS


def _claude_part(turn: _Turn, owned: dict[tuple, int]) -> _Part:
    part = _Part(turn.line.moment)
    counted: set[tuple] = set()
    tools_started: dict[str, tuple[str, float]] = {}
    pending: set[str] = set()
    for line in turn.lines:
        if line.moment is not None:
            part.moments.append(line.moment)
        if line.sidechain:
            continue
        if line.kind == "user":
            part.stopped = part.stopped or line.interrupt
            pending.difference_update(line.results)
            wait = _person_wait(line, tools_started)
            if wait is not None:
                part.waits.append(wait)
            continue
        if line.kind != "assistant":
            continue
        if line.call is None:
            # Claude Code writes a failed call, such as a server error or a usage limit, as a reply of its
            # own. The task stopped there, unless a real reply came after it, as when Claude Code retried.
            part.failed = part.failed or line.failed
            continue
        key, call = line.call
        if owned.get(key) != turn.index:
            continue  # the call belongs to another turn
        part.failed = False
        if key not in counted:
            counted.add(key)
            part.calls.append(call)
            if part.first_input is None:
                part.first_input = call.new_input + call.cache_write_5m + call.cache_write_1h + call.cache_read
        part.model = part.model or call.model
        part.effort = part.effort or line.effort or ""
        part.skills |= line.skills
        if line.moment is not None:
            part.last = line.moment if part.last is None else max(part.last, line.moment)
        for tool_id, name in line.tools:
            part.used_tools = True
            if tool_id:
                pending.add(tool_id)
                if line.moment is not None:
                    tools_started[tool_id] = (name, line.moment)
    part.open_call = bool(pending) and not part.stopped
    return part


def _read_claude_code(archive: dict, files: dict[str, tuple[int, float]], now: float) -> Reading:
    helpers: dict[str, list[str]] = {}
    sessions = []
    for path in files:
        session = _helper_session(path)
        if session is None:
            sessions.append(path)
        else:
            helpers.setdefault(session, []).append(path)
    groups: dict[str, list[str]] = {}
    for path, family in claude_families(sessions).items():
        groups.setdefault(family, []).append(path)
    reading = Reading(tasks=[])
    found: dict[str, Task] = {}
    for paths in groups.values():
        part = _read_claude_family(paths, helpers, files, archive["name"], now)
        for task in part.tasks:
            keep_most_complete(found, task)  # families share no lines, so this only matters for odd records
        reading.held |= part.held
        reading.open_files |= part.open_files
        reading.sessions |= part.sessions
    reading.tasks = sorted(found.values(), key=lambda t: (t.started_at, t.id))
    return reading


def _read_claude_family(
    paths: list[str], helpers: dict[str, list[str]], stats: dict[str, tuple[int, float]], archive_name: str, now: float
) -> Reading:
    """The tasks in one family of Claude Code records: a session, its copies, and their helpers."""
    records = []
    for path in paths:
        try:
            records.append(_ClaudeFile(path, Path(path).stem, stats[path][1], _claude_lines(Path(path))))
        except OSError:
            continue  # removed since the folder was listed, as Claude Code does with old sessions
    reading = Reading(tasks=[], sessions={record.stem for record in records})
    records.sort(key=_written_first)
    turns = _claude_turns(records)
    helper_paths = [helper for record in records for helper in sorted(helpers.get(record.path, []))]

    def fresh(path: str) -> bool:
        return now - stats[path][1] < SETTLE_SECONDS

    family_fresh = any(fresh(path) for path in [record.path for record in records] + helper_paths)

    # Which turn started each tool call, and each background job and helper.
    by_prompt: dict[str, _Turn] = {}
    tool_turn: dict[str, tuple[_Turn, str]] = {}
    for turn in turns:
        if turn.line.prompt_id:
            by_prompt.setdefault(turn.line.prompt_id, turn)
        for line in turn.lines:
            for tool_id, name in line.tools:
                if tool_id:
                    tool_turn.setdefault(tool_id, (turn, name))
    jobs: dict[str, tuple[_Turn, str]] = {}  # each background job: the turn that started it, and the call's id
    helper_turn: dict[str, _Turn] = {}  # each helper agent: the turn whose call started it
    reported: dict[str, float | None] = {}  # each job's id or call's id: when it was reported done, or stopped
    for turn in turns:
        for line in (turn.line, *turn.lines):
            for job, tool_id, status in line.notices:
                for name in (job, tool_id) if status else ():
                    if name and name not in reported:
                        reported[name] = line.moment
            for job in line.stops:
                reported.setdefault(job, line.moment)
            launch = tool_turn.get(line.results[0]) if line.results else None
            if launch is not None and line.launched and launch[1] != "Monitor":
                jobs.setdefault(line.launched, (launch[0], line.results[0]))
            if launch is not None and line.helper:
                helper_turn.setdefault(line.helper, launch[0])

    # Each turn's task: its own, a task whose background job it reports back on, or none.
    owner: dict[int, _Turn | None] = {}
    for turn in turns:
        if turn.line.origin is not None:
            owner[turn.index] = turn
            continue
        boss = None
        for job, tool_id, status in turn.line.notices:
            launch = tool_turn.get(tool_id or jobs.get(job, (None, ""))[1])
            if status and launch is not None and launch[1] != "Monitor":
                boss = owner.get(launch[0].index)
                break
        owner[turn.index] = boss

    # One model call belongs to one turn: the first that holds it.
    owned: dict[tuple, int] = {}
    for turn in turns:
        for line in turn.lines:
            if line.call is not None and not line.sidechain:
                owned.setdefault(line.call[0], turn.index)
    helper_work: dict[int, list[_Call]] = {}
    helper_end: dict[int, float] = {}
    counted = set(owned)
    for path in helper_paths:
        try:
            for prompt_id, agent, key, call, moment in _helper_calls(Path(path)):
                started = by_prompt.get(prompt_id or "") or helper_turn.get(agent or "")
                boss = owner.get(started.index) if started is not None else None
                if boss is None or key in counted:
                    continue
                counted.add(key)
                helper_work.setdefault(boss.index, []).append(call)
                if moment is not None:
                    helper_end[boss.index] = max(helper_end.get(boss.index, moment), moment)
        except OSError:
            continue

    # The conversation before each turn, and the skills known by then.
    history = _History()
    known: set[str] = set()
    contexts: dict[int, str] = {}
    priors: dict[int, set[str]] = {}
    members: dict[int, list[_Turn]] = {}
    for turn in turns:
        contexts[turn.index] = history.context()
        priors[turn.index] = known | turn.line.skills
        if turn.line.origin is not None:
            history.add("User", turn.line.start or "")
        known |= turn.line.skills
        for line in turn.lines:
            if not line.sidechain:  # a helper's lines are another agent's, as in the conversation
                known |= line.skills
            if line.kind == "assistant" and line.call is not None and not line.sidechain:
                for words in line.texts:
                    history.add("Assistant", words)
        boss = owner.get(turn.index)
        if boss is not None:
            members.setdefault(boss.index, []).append(turn)
    jobs_of: dict[int, list[tuple[str, str]]] = {}
    for job, (launch, tool_id) in jobs.items():
        boss = owner.get(launch.index)
        if boss is not None:
            jobs_of.setdefault(boss.index, []).append((job, tool_id))

    for turn in turns:
        if owner.get(turn.index) is not turn:
            continue
        line, segments = turn.line, members[turn.index]
        parts = [_claude_part(segment, owned) for segment in segments]
        own = parts[0]
        if not own.calls or own.start is None or own.last is None:
            continue
        task_id = f"claude-code:{line.uuid or f'{Path(turn.source).stem}:{turn.index}'}"
        started_jobs = jobs_of.get(turn.index, [])
        open_jobs = [tool_id for job, tool_id in started_jobs if job not in reported and tool_id not in reported]
        still = any(not s.followed and any(fresh(path) for path in s.files) for s in segments)
        if still or ((open_jobs or parts[-1].open_call) and family_fresh):
            reading.held.add(task_id)  # it may still be running: read it again next time
            continue
        # A helper or workflow that went quiet without reporting back may not have finished its paid work. A
        # background command that never reports, such as a server, often never ends, and isn't paid work.
        cut_short = parts[-1].open_call or any(tool_turn[tool_id][1] not in _COMMANDS for tool_id in open_jobs)
        calls = [call for part in parts for call in part.calls] + helper_work.get(turn.index, [])
        new_input, cache_write, cache_read, output, usd = _totals(calls)
        worked = [part.worked() for part in parts]
        ends = [part.last for part in parts if part.last is not None]
        ends += [helper_end[turn.index]] if turn.index in helper_end else []
        ends += [reported[name] for job in started_jobs for name in job if reported.get(name) is not None]
        reading.tasks.append(
            Task(
                id=task_id,
                archive=archive_name,
                agent="claude-code",
                agent_version=line.version,
                session=line.session or Path(turn.source).stem,
                project=Path(line.cwd).name,
                origin=line.origin or "",
                started_at=own.start,
                model=own.model,
                effort=own.effort or "default",
                prompt=(line.start or "").strip()[:TEXT_CHARS],
                context=contexts[turn.index],
                context_tokens=own.first_input or 0,
                seconds=max(sum(seconds for seconds, _ in worked), 0.5),
                input_tokens=new_input,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                output_tokens=output,
                usd=usd,
                calls=len(calls),
                direct=not any(part.used_tools for part in parts),
                # An error stops the task unless a reply came after it, such as in a later report's turn.
                stopped=any(part.stopped for part in parts) or parts[-1].failed or cut_short,
                idle=any(left for _, left in worked),
                skills="\n".join(sorted(set().union(*(part.skills for part in parts)))),
                finished_at=max(ends),
                prior_skills="\n".join(sorted(priors[turn.index])),
                source=turn.source,
            )
        )
    if reading.held:
        reading.open_files = {record.path for record in records} | set(helper_paths)
    return reading


def _person_wait(line: _Line, tools_started: dict[str, tuple[str, float]]) -> tuple[float, float] | None:
    """When a tool's result waited for the person: a question, a plan to approve, or a tool to allow.

    A question or a plan waits from the call to the answer. A tool the person allowed waited only until it
    began to run. When its result says how long it ran, that run is agent time. Most results don't say, such
    as a shell command's, and then the whole wait is taken out. None means the result didn't wait for the person.
    """
    if line.moment is None or not line.results:
        return None
    asked = line.asked.startswith("user")
    began, ended = line.moment, line.moment
    for tool_id in line.results:
        name, called = tools_started.get(tool_id, ("", line.moment))
        if name in PERSON_TOOLS or asked:
            began = min(began, called)
        if asked and name not in PERSON_TOOLS and line.ran:
            ended = min(ended, line.moment - line.ran)
    return (began, ended) if began < ended else None


def _covered(spans: list[tuple[float, float]]) -> float:
    """How long some spans of time cover together. Two questions asked at once overlap, and count once."""
    total, reach = 0.0, float("-inf")
    for began, ended in sorted(spans):
        total += max(ended - max(began, reach), 0.0)
        reach = max(reach, ended)
    return total


def _longest_gap(start: float, last: float, moments: list[float], waits: list[tuple[float, float]]) -> float:
    """The longest silence while the agent worked, not counting time spent waiting for the person.

    Other lines can land inside a wait, such as a helper's progress or a hook. They split the wait
    into shorter silences, and none of those count either.
    """
    busy = sorted({start, last, *(m for m in moments if start <= m <= last)})
    return max(
        (b - a for a, b in zip(busy, busy[1:]) if not any(began <= a and b <= ended for began, ended in waits)),
        default=0.0,
    )


# Codex ------------------------------------------------------------------------------------------------


def _codex_text(payload: dict) -> str:
    content = payload.get("content")
    parts = content if isinstance(content, list) else []
    return "\n".join(
        _part_text(part)
        for part in parts
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")
    )


_CODEX_INJECTED = (
    "<environment_context",
    "<user_instructions",
    "<user_shell_command",
    "<external_codex_apps",
    "# AGENTS.md instructions",
)
"""How the messages Codex adds to a turn begin. A prompt that starts with some other tag is still a prompt."""


def _codex_injected(text: str) -> bool:
    """Instructions and environment details that Codex adds to a turn, not the prompt itself."""
    return text.lstrip().startswith(_CODEX_INJECTED)


def _codex_call(usage: dict, model: str) -> _Call:
    total_input = _number(usage.get("input_tokens"))
    cached = _number(usage.get("cached_input_tokens"))
    written = _number(usage.get("cache_write_input_tokens"))
    return _Call(
        model=model,
        new_input=max(total_input - cached - written, 0),
        cache_write_5m=written,
        cache_write_1h=0,
        cache_read=cached,
        output=_number(usage.get("output_tokens")),
    )


@dataclass
class _CodexTurn:
    turn_id: str
    started_at: float
    model: str
    effort: str
    prompt: list[str] = field(default_factory=list)
    calls: list[_Call] = field(default_factory=list)
    first_input: int | None = None
    used_tools: bool = False
    longest_gap: float = 0.0
    last_moment: float | None = None
    context: str = ""


@dataclass
class _ThreadTurn:
    """One turn of a Codex thread, for finding the task its model calls count toward."""

    id: str
    root: str
    """The main thread's turn behind this one, as newer versions of Codex write it. Empty in older records."""
    started: float
    calls: list[_Call] = field(default_factory=list)
    last: float | None = None
    """When its last model call ended."""
    done: bool = False
    """Whether it finished, or another turn started after it."""
    ended: float | None = None
    """When it finished, when the record says."""


@dataclass
class _CodexThread:
    """One Codex thread as one record file holds it: its id, the thread it works for, and its turns."""

    id: str
    parent: str | None
    turns: list[_ThreadTurn]
    """Calls before the first turn count from when the thread began."""
    path: str = ""
    fresh: bool = False
    """Whether its record was written to too recently to be sure it's done."""


def _read_codex(archive: dict, files: dict[str, tuple[int, float]], now: float) -> Reading:
    tasks: dict[str, Task] = {}
    threads: list[_CodexThread] = []
    unfinished: dict[str, str] = {}  # each main turn still running: its thread
    reading = Reading(tasks=[])
    for path in sorted(files, key=lambda p: (files[p][1], p)):
        try:
            found, thread, still = _read_codex_file(Path(path), archive["name"], now, files[path][1])
        except OSError:
            continue  # removed since the folder was listed
        if thread is not None:
            threads.append(thread)
            if thread.parent is None:
                reading.sessions.add(thread.id)
            unfinished.update((task_id, thread.id) for task_id in still)
        for task in found:
            keep_most_complete(tasks, task)
    # A main turn still running in one record may be finished in another, such as the original of a copy.
    busy = {thread_id for task_id, thread_id in unfinished.items() if task_id not in tasks}
    reading.held = {task_id for task_id in unfinished if task_id not in tasks}
    for task_id in _attach_codex_helpers(tasks, threads):
        reading.held.add(task_id)
        busy.add(tasks.pop(task_id).session)
    if busy:
        parents = {thread.id: thread.parent for thread in threads}
        tops = {_codex_top(thread_id, parents) for thread_id in busy}
        reading.open_files = {thread.path for thread in threads if _codex_top(thread.id, parents) in tops}
    reading.tasks = sorted(tasks.values(), key=lambda t: (t.started_at, t.id))
    return reading


def _codex_top(thread_id: str, parents: dict[str, str | None]) -> str:
    """The main thread at the top of a thread's chain of helpers."""
    seen = {thread_id}
    while parents.get(thread_id) and parents[thread_id] not in seen:
        thread_id = parents[thread_id]
        seen.add(thread_id)
    return thread_id


def _read_codex_file(
    path: Path, archive_name: str, now: float, mtime: float
) -> tuple[list[Task], _CodexThread | None, set[str]]:
    """A Codex record's tasks, its thread, and the tasks still running in it."""
    meta: dict = {}
    model, effort = "", "default"
    turn: _CodexTurn | None = None
    current: _ThreadTurn | None = None
    last_total: dict | None = None
    history = _History()
    tasks: list[Task] = []
    turns: list[_ThreadTurn] = []
    began: float | None = None
    for line in _lines(path):
        kind = line.get("type")
        payload = line.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        moment = _timestamp(line.get("timestamp"))
        starts = kind == "event_msg" and payload.get("type") == "task_started"  # the next turn's time isn't this one's
        if turn is not None and moment is not None and not starts:
            if turn.last_moment is not None:
                turn.longest_gap = max(turn.longest_gap, moment - turn.last_moment)
            turn.last_moment = moment if turn.last_moment is None else max(turn.last_moment, moment)
        if kind == "session_meta" and not meta:
            meta, began = payload, moment
            continue
        if kind == "turn_context":
            model = payload.get("model") or model
            effort = payload.get("effort") or effort
            if turn is not None:
                turn.model, turn.effort = model, effort
            continue
        if kind == "response_item":
            item = payload.get("type")
            if item == "message" and payload.get("role") == "user":
                text = _codex_text(payload)
                if turn is not None and text.strip() and not _codex_injected(text):
                    turn.prompt.append(text)
                    history.add("User", text)
            elif item == "message" and payload.get("role") == "assistant":
                history.add("Assistant", _codex_text(payload))
            elif item in CODEX_TOOL_ITEMS and turn is not None:
                turn.used_tools = True
            continue
        if kind != "event_msg":
            continue
        event = payload.get("type")
        if event == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            total = info.get("total_token_usage")
            if not total or total == last_total:
                continue
            last_total = total
            usage = info.get("last_token_usage")
            call = _codex_call(usage if isinstance(usage, dict) else {}, model)
            if not turns:
                turns.append(_ThreadTurn("", "", began or moment or 0.0, done=True))
            turns[-1].calls.append(call)
            if moment is not None:
                turns[-1].last = moment if turns[-1].last is None else max(turns[-1].last, moment)
            if turn is not None:
                turn.calls.append(call)
                if turn.first_input is None:
                    turn.first_input = call.new_input + call.cache_read + call.cache_write_5m
        elif event == "task_started":
            if turn is not None:
                # A turn left open when the next one starts was cut short, such as by a crash. It's kept, stopped.
                task = _codex_task(turn, {}, "cut_short", meta, path, archive_name)
                if task is not None:
                    tasks.append(task)
            if current is not None:
                current.done = True
            started = _timestamp(payload.get("started_at")) or moment
            turn = _CodexTurn(str(payload.get("turn_id") or ""), started or 0.0, model, effort)
            turn.last_moment = moment
            turn.context = history.context()
            current = _ThreadTurn(turn.turn_id, str(payload.get("root_turn_id") or ""), turn.started_at)
            turns.append(current)
        elif event in ("task_complete", "turn_aborted") and turn is not None:
            task = _codex_task(turn, payload, event, meta, path, archive_name)
            if task is not None:
                tasks.append(task)
            if current is not None:
                current.done = True
                current.ended = _timestamp(payload.get("completed_at")) or moment
            turn = current = None
    fresh = now - mtime < SETTLE_SECONDS
    still: set[str] = set()
    if turn is not None and not fresh:
        task = _codex_task(turn, {}, "never_finished", meta, path, archive_name)
        if task is not None:
            tasks.append(task)
    elif turn is not None and turn.turn_id:
        still.add(f"codex:{turn.turn_id}")
    thread_id = str(meta.get("id") or path.stem)
    helper, parent = _codex_spawn(meta)
    thread = _CodexThread(thread_id, parent, turns, str(path), fresh)
    if helper:
        return [], thread if parent else None, set()
    return tasks, thread, still


def _codex_task(turn: _CodexTurn, payload: dict, event: str, meta: dict, path: Path, archive_name: str) -> Task | None:
    if not turn.prompt or not turn.calls or not turn.turn_id:
        return None
    finished = _timestamp(payload.get("completed_at")) or turn.last_moment or turn.started_at
    duration = payload.get("duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        seconds = duration / 1000
    else:
        seconds = finished - turn.started_at
    new_input, cache_write, cache_read, output, usd = _totals(turn.calls)
    return Task(
        id=f"codex:{turn.turn_id}",
        archive=archive_name,
        agent="codex",
        agent_version=str(meta.get("cli_version") or ""),
        session=str(meta.get("id") or path.stem),
        project=Path(str(meta.get("cwd") or "")).name,
        origin="program" if meta.get("originator") == "codex_exec" else "person",
        started_at=turn.started_at,
        model=turn.model,
        effort=str(turn.effort or "default"),
        prompt="\n\n".join(turn.prompt).strip()[:TEXT_CHARS],
        context=turn.context,
        context_tokens=turn.first_input or 0,
        seconds=max(seconds, 0.5),
        input_tokens=new_input,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
        output_tokens=output,
        usd=usd,
        calls=len(turn.calls),
        direct=not turn.used_tools,
        stopped=event != "task_complete" or bool(payload.get("error")),
        idle=turn.longest_gap > IDLE_GAP_SECONDS,
        finished_at=finished,
        source=str(path),
    )


def _attach_codex_helpers(tasks: dict[str, Task], threads: list[_CodexThread]) -> set[str]:
    """Add each helper turn's tokens to the task of the main thread's turn behind it. Returns the tasks whose
    helpers are still working: their outcome isn't known yet.

    Newer versions of Codex name that turn in each helper turn. So a helper, or a helper spawned by a
    helper, still counts toward the turn that started it when it works on after the person's next prompt.
    Older records name no turn. There a helper's turn counts toward its parent thread's turn that started
    last before it, as that turn spawned the helper or sent it more work, and so on up to the main thread.
    A helper's tokens then count even when its first reply comes after that turn finished.

    A thread can be spread over several files, such as a session resumed into a new file, and a file can
    be copied. A helper forked from its parent also starts with a copy of the parent's turn. So each turn
    belongs to the shallowest thread that holds it, counts once from its fullest copy, and a copy of a main
    thread's turn adds nothing: it's in its task.

    A helper turn that never finished leaves its task's outcome unknown. While its record is still being
    written, the helper may be working, and the task waits. Once its record has gone quiet, the task is stopped.
    """
    parents: dict[str, str | None] = {}
    for thread in threads:
        if thread.parent or thread.id not in parents:
            parents[thread.id] = thread.parent

    def depth(thread_id: str) -> int:
        seen, steps = {thread_id}, 0
        while parents.get(thread_id) and parents[thread_id] not in seen:
            thread_id = parents[thread_id]
            seen.add(thread_id)
            steps += 1
        return steps

    def key(thread_id: str, turn: _ThreadTurn) -> str:
        return turn.id or f"{thread_id} {turn.started}"  # calls before a thread's first turn have no turn id

    owner: dict[str, str] = {}  # each turn's thread: the shallowest one that holds it
    fullest: dict[str, _ThreadTurn] = {}
    done: dict[str, bool] = {}
    working: dict[str, bool] = {}
    holds: dict[str, list[_ThreadTurn]] = {}  # every turn each thread holds, copies included
    for thread in sorted(threads, key=lambda t: depth(t.id)):
        for turn in thread.turns:
            name = key(thread.id, turn)
            holds.setdefault(thread.id, []).append(turn)
            owner.setdefault(name, thread.id)
            if name not in fullest or len(turn.calls) > len(fullest[name].calls):
                fullest[name] = turn
            done[name] = done.get(name, False) or turn.done
            working[name] = working.get(name, False) or thread.fresh

    def main_turn(thread_id: str, turn: _ThreadTurn) -> str | None:
        """The main thread's turn behind a helper's turn. None when the records don't hold it."""
        seen = set()
        while parents.get(thread_id):
            if key(thread_id, turn) in seen:
                return None
            seen.add(key(thread_id, turn))
            if turn.root:
                if turn.root not in owner:
                    return None
                thread_id, turn = owner[turn.root], fullest[turn.root]
                continue
            parent = parents[thread_id]
            earlier = [t for t in holds.get(parent, []) if t.started <= turn.started]
            if not earlier:
                return None
            thread_id, turn = parent, max(earlier, key=lambda t: (t.started, t.id))
        return turn.id

    running: set[str] = set()
    for name, thread_id in owner.items():
        if not parents.get(thread_id):
            continue
        turn = fullest[name]
        sent = main_turn(thread_id, turn)
        task = tasks.get(f"codex:{sent}") if sent else None
        if task is None:
            continue
        new_input, cache_write, cache_read, output, usd = _totals(turn.calls)
        task.input_tokens += new_input
        task.cache_write_tokens += cache_write
        task.cache_read_tokens += cache_read
        task.output_tokens += output
        task.usd = None if task.usd is None or usd is None else task.usd + usd
        task.calls += len(turn.calls)
        end = turn.ended or turn.last
        if end is not None and task.finished_at is not None:
            task.finished_at = max(task.finished_at, end)  # its outcome is known once the helper is done
        if not done[name]:
            if working[name]:
                running.add(task.id)
            else:
                task.stopped = True
    return running
