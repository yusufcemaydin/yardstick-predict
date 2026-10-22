"""Read Claude Code and Codex records into tasks.

A task is one turn that a person or a program started. It runs from the prompt until the agent
stops and hands control back. Turns that an automatic event started, such as a finished
background job, are not tasks. Tokens used by helper agents count toward the task that started them.
"""

from __future__ import annotations

import json
import os
import re
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

TEXT_CHARS = 24_000
"""How much of the prompt and of the context each task keeps. The context keeps its newest part."""

IDLE_GAP_SECONDS = 30 * 60
"""A task with a longer silence was left waiting, not working. It is kept but not trained on."""

SETTLE_SECONDS = 15 * 60
"""The last task in a file written to more recently than this may still be running."""

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

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens + self.output_tokens


@dataclass
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
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


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


def archive_path(archive: dict) -> Path:
    return Path(os.path.expanduser(archive["path"]))


def record_files(archive: dict) -> dict[str, tuple[int, float]]:
    """Every record file in an archive, with its size and modification time."""
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


def read_archive(archive: dict, only: set[str] | None = None, now: float | None = None) -> list[Task]:
    """Read the archive's tasks. With `only`, read just those session files."""
    now = time.time() if now is None else now
    if archive["format"] == "claude-code":
        return _read_claude_code(archive, only, now)
    if archive["format"] == "codex":
        return _read_codex(archive, only, now)
    raise ValueError(f"Unknown archive format {archive['format']!r}. Use one of: {', '.join(FORMATS)}.")


def _keep_most_complete(tasks: dict[str, Task], task: Task) -> None:
    """Keep the copy of a task with the most model calls.

    A helper that finishes later adds calls to its parent. A copy of a task in a resumed session
    can't see the original's helpers, so it has fewer calls and never replaces the original.
    """
    kept = tasks.get(task.id)
    if kept is None or task.calls > kept.calls:
        tasks[task.id] = task


def codex_parent(path: str) -> str | None:
    """The parent thread of a Codex helper thread's record file, or None for a main thread."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            first = json.loads(handle.readline() or "{}")
    except (OSError, ValueError):
        return None
    source = (first.get("payload") or {}).get("source") if isinstance(first, dict) else None
    spawn = source.get("subagent", {}).get("thread_spawn") if isinstance(source, dict) else None
    return str(spawn["parent_thread_id"]) if isinstance(spawn, dict) and spawn.get("parent_thread_id") else None


# Claude Code ------------------------------------------------------------------------------------------

_NOT_AGENT_WORK = ("<bash-input", "<bash-stdout", "<bash-stderr", "<local-command")
_AUTOMATIC_TAGS = ("task-notification", "system-reminder", "ci-monitor-event", "artifact-", "local-command", "bash-")
"""Wrappers Claude Code puts around messages that no person or program typed. A prompt that merely
starts with some other tag, such as an orchestrator's <task>, is still a prompt."""


def _claude_text(message: dict) -> str | None:
    """The text of a user message, or None when it only carries tool results."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(part, dict) and part.get("type") == "tool_result" for part in content):
            return None
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return None


def _claude_turn_start(line: dict) -> str | None:
    """The prompt text when this line starts a turn, else None."""
    if line.get("type") != "user" or line.get("isSidechain") or line.get("isMeta"):
        return None
    if line.get("isCompactSummary") or line.get("isVisibleInTranscriptOnly"):
        return None
    text = _claude_text(line.get("message") or {})
    if text is None or not text.strip() or text.lstrip().startswith("[Request interrupted"):
        return None
    return text


def _claude_origin(line: dict, text: str) -> str | None:
    """Who started the turn: 'person', 'program', or None for an automatic event."""
    stripped = text.lstrip()
    if stripped.startswith(_NOT_AGENT_WORK):
        return None
    origin = line.get("turnOrigin")
    if origin == "human":
        return "person"
    if origin in ("sdk", "peer", "scheduled"):
        return "program"
    if origin is not None:
        return None
    tag = re.match(r"<([A-Za-z_-]+)", stripped)
    if tag and tag.group(1).startswith(_AUTOMATIC_TAGS):
        return None
    return "person"


def _claude_call(line: dict) -> tuple[tuple, _Call] | None:
    message = line.get("message") or {}
    model = message.get("model")
    if not model or model == "<synthetic>":
        return None
    usage = message.get("usage") or {}
    writes = usage.get("cache_creation") or {}
    write_total = usage.get("cache_creation_input_tokens") or 0
    write_1h = writes.get("ephemeral_1h_input_tokens") or 0
    write_5m = writes.get("ephemeral_5m_input_tokens")
    write_5m = (write_total - write_1h) if write_5m is None else write_5m
    call = _Call(
        model=model,
        new_input=usage.get("input_tokens") or 0,
        cache_write_5m=max(write_5m, 0),
        cache_write_1h=write_1h,
        cache_read=usage.get("cache_read_input_tokens") or 0,
        output=usage.get("output_tokens") or 0,
        fast=usage.get("speed") == "fast",
    )
    return (message.get("id"), line.get("requestId")), call


def _claude_helpers(session_dir: Path) -> dict[str, dict[tuple, _Call]]:
    """Helper-agent calls grouped by the prompt that started them."""
    by_prompt: dict[str, dict[tuple, _Call]] = {}
    helper_dir = session_dir / "subagents"
    if not helper_dir.is_dir():
        return by_prompt
    for path in sorted(helper_dir.glob("*.jsonl")):
        for line in _lines(path):
            if line.get("type") != "assistant" or not line.get("promptId"):
                continue
            found = _claude_call(line)
            if found:
                key, call = found
                by_prompt.setdefault(line["promptId"], {}).setdefault(key, call)
    return by_prompt


def _read_claude_code(archive: dict, only: set[str] | None, now: float) -> list[Task]:
    root = archive_path(archive)
    files = [p for p in root.rglob("*.jsonl") if p.parent.name != "subagents"] if root.is_dir() else []
    if only is not None:
        files = [p for p in files if str(p) in only]
    files.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    tasks: dict[str, Task] = {}
    for path in files:
        for task in _read_claude_file(path, archive["name"], now):
            _keep_most_complete(tasks, task)
    return list(tasks.values())


def _read_claude_file(path: Path, archive_name: str, now: float) -> list[Task]:
    lines = list(_lines(path))
    starts = [(i, text) for i, line in enumerate(lines) if (text := _claude_turn_start(line)) is not None]
    if not starts:
        return []
    helpers = _claude_helpers(path.with_suffix(""))
    settled = now - path.stat().st_mtime >= SETTLE_SECONDS
    history = _History()
    tasks = []
    for k, (i, text) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        line = lines[i]
        origin = _claude_origin(line, text)
        context = history.context()
        calls: dict[tuple, _Call] = {}
        model = effort = None
        first_input = None
        used_tools = stopped = False
        start = _timestamp(line.get("timestamp"))
        last = start
        moments: list[float] = []
        tools_started: dict[str, tuple[str, float]] = {}
        skills: set[str] = set()
        waited = 0.0
        wait_ends: set[float] = set()
        for event in lines[i + 1 : end]:
            moment = _timestamp(event.get("timestamp"))
            if moment is not None:
                moments.append(moment)
            if event.get("isSidechain"):
                continue
            if event.get("type") == "user":
                message = event.get("message") or {}
                if (_claude_text(message) or "").lstrip().startswith("[Request interrupted"):
                    stopped = True
                wait = _person_wait(message, event, tools_started, moment)
                if wait > 0 and moment is not None:
                    waited += wait
                    wait_ends.add(moment)
                continue
            if event.get("type") != "assistant":
                continue
            found = _claude_call(event)
            if not found:
                continue
            key, call = found
            if key not in calls:
                calls[key] = call
                if first_input is None:
                    first_input = call.new_input + call.cache_write_5m + call.cache_write_1h + call.cache_read
            model = model or call.model
            effort = effort or event.get("effort")
            if event.get("attributionSkill"):
                skills.add(str(event["attributionSkill"]))
            if moment is not None:
                last = moment if last is None else max(last, moment)
            for part in (event.get("message") or {}).get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool_use":
                    used_tools = True
                    if part.get("id") and moment is not None:
                        tools_started[part["id"]] = (str(part.get("name")), moment)
                    if (
                        part.get("name") == "Skill"
                        and isinstance(part.get("input"), dict)
                        and part["input"].get("skill")
                    ):
                        skills.add(str(part["input"]["skill"]))
                elif part.get("type") == "text":
                    history.add("Assistant", part.get("text") or "")
        if origin is not None:
            history.add("User", text)
        finished = k + 1 < len(starts) or settled
        if origin is None or not calls or start is None or last is None or not finished:
            continue
        helper_calls = helpers.get(line.get("promptId") or "", {})
        all_calls = list(calls.values()) + [c for key, c in helper_calls.items() if key not in calls]
        new_input, cache_write, cache_read, output, usd = _totals(all_calls)
        tasks.append(
            Task(
                id=f"claude-code:{line.get('uuid') or f'{path.stem}:{i}'}",
                archive=archive_name,
                agent="claude-code",
                agent_version=str(line.get("version") or ""),
                session=str(line.get("sessionId") or path.stem),
                project=Path(line.get("cwd") or "").name,
                origin=origin,
                started_at=start,
                model=model or "",
                effort=str(effort or "default"),
                prompt=text.strip()[:TEXT_CHARS],
                context=context,
                context_tokens=first_input or 0,
                seconds=max(last - start - waited, 0.5),
                input_tokens=new_input,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
                output_tokens=output,
                usd=usd,
                calls=len(all_calls),
                direct=not used_tools,
                stopped=stopped,
                idle=_longest_gap(start, last, moments, wait_ends) > IDLE_GAP_SECONDS,
                skills="\n".join(sorted(skills)),
            )
        )
    return tasks


PERSON_TOOLS = {"AskUserQuestion", "ExitPlanMode"}
"""Tools that wait for the person to answer or approve."""


def _person_wait(message: dict, line: dict, tools_started: dict[str, tuple[str, float]], moment: float | None) -> float:
    """Seconds a tool result spent waiting for the person: a question, a plan approval or a permission prompt."""
    if moment is None or not isinstance(message.get("content"), list):
        return 0.0
    asked_person = str((line.get("permissionDecision") or {}).get("source") or "").startswith("user")
    wait = 0.0
    for part in message["content"]:
        if not isinstance(part, dict) or part.get("type") != "tool_result":
            continue
        name, began = tools_started.get(part.get("tool_use_id") or "", ("", moment))
        if name in PERSON_TOOLS or asked_person:
            wait = max(wait, moment - began)
    return wait


def _longest_gap(start: float, last: float, moments: list[float], wait_ends: set[float]) -> float:
    """The longest silence while the agent worked, not counting time spent waiting for the person."""
    busy = sorted({start, last, *(m for m in moments if start <= m <= last)})
    return max((b - a for a, b in zip(busy, busy[1:]) if b not in wait_ends), default=0.0)


# Codex ------------------------------------------------------------------------------------------------


def _codex_text(payload: dict) -> str:
    return "\n".join(
        part.get("text", "")
        for part in payload.get("content") or []
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
    total_input = usage.get("input_tokens") or 0
    cached = usage.get("cached_input_tokens") or 0
    written = usage.get("cache_write_input_tokens") or 0
    return _Call(
        model=model,
        new_input=max(total_input - cached - written, 0),
        cache_write_5m=written,
        cache_write_1h=0,
        cache_read=cached,
        output=usage.get("output_tokens") or 0,
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
class _CodexHelper:
    parent: str
    started_at: float
    calls: list[_Call]


def _read_codex(archive: dict, only: set[str] | None, now: float) -> list[Task]:
    root = archive_path(archive)
    files = list(root.rglob("*.jsonl")) if root.is_dir() else []
    if only is not None:
        files = [p for p in files if str(p) in only]
    files.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    tasks: dict[str, Task] = {}
    helpers: list[_CodexHelper] = []
    for path in files:
        found, helper = _read_codex_file(path, archive["name"], now)
        if helper:
            helpers.append(helper)
        for task in found:
            _keep_most_complete(tasks, task)
    _attach_codex_helpers(list(tasks.values()), helpers)
    return list(tasks.values())


def _read_codex_file(path: Path, archive_name: str, now: float) -> tuple[list[Task], _CodexHelper | None]:
    meta: dict = {}
    model, effort = "", "default"
    turn: _CodexTurn | None = None
    last_total: dict | None = None
    history = _History()
    tasks: list[Task] = []
    helper_calls: list[_Call] = []
    helper_start: float | None = None
    for line in _lines(path):
        kind = line.get("type")
        payload = line.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        moment = _timestamp(line.get("timestamp"))
        if turn is not None and moment is not None:
            if turn.last_moment is not None:
                turn.longest_gap = max(turn.longest_gap, moment - turn.last_moment)
            turn.last_moment = moment if turn.last_moment is None else max(turn.last_moment, moment)
        if kind == "session_meta" and not meta:
            meta = payload
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
            elif item == "message" and payload.get("role") == "assistant":
                history.add("Assistant", _codex_text(payload))
            elif item in CODEX_TOOL_ITEMS and turn is not None:
                turn.used_tools = True
            continue
        if kind != "event_msg":
            continue
        event = payload.get("type")
        if event == "token_count":
            info = payload.get("info") or {}
            total = info.get("total_token_usage")
            if not total or total == last_total:
                continue
            last_total = total
            call = _codex_call(info.get("last_token_usage") or {}, model)
            if helper_start is None and moment is not None:
                helper_start = moment
            helper_calls.append(call)
            if turn is not None:
                turn.calls.append(call)
                if turn.first_input is None:
                    turn.first_input = call.new_input + call.cache_read + call.cache_write_5m
        elif event == "task_started":
            started = _timestamp(payload.get("started_at")) or moment
            turn = _CodexTurn(str(payload.get("turn_id") or ""), started or 0.0, model, effort)
            turn.last_moment = moment
            turn.context = history.context()
        elif event in ("task_complete", "turn_aborted") and turn is not None:
            task = _codex_task(turn, payload, event, meta, path, archive_name)
            for words in turn.prompt:
                history.add("User", words)
            if task is not None:
                tasks.append(task)
            turn = None
    if turn is not None and now - path.stat().st_mtime >= SETTLE_SECONDS:
        task = _codex_task(turn, {}, "never_finished", meta, path, archive_name)
        if task is not None:
            tasks.append(task)
    source = meta.get("source")
    spawn = source.get("subagent", {}).get("thread_spawn") if isinstance(source, dict) else None
    if isinstance(source, dict) and "subagent" in source:
        parent = spawn.get("parent_thread_id") if isinstance(spawn, dict) else None
        helper = _CodexHelper(str(parent), helper_start or 0.0, helper_calls) if parent else None
        return [], helper
    return tasks, None


def _codex_task(turn: _CodexTurn, payload: dict, event: str, meta: dict, path: Path, archive_name: str) -> Task | None:
    if not turn.prompt or not turn.calls or not turn.turn_id:
        return None
    duration = payload.get("duration_ms")
    if isinstance(duration, (int, float)) and duration > 0:
        seconds = duration / 1000
    else:
        finished = _timestamp(payload.get("completed_at")) or turn.last_moment or turn.started_at
        seconds = finished - turn.started_at
    new_input, cache_write, cache_read, output, usd = _totals(turn.calls)
    return Task(
        id=f"codex:{turn.turn_id}",
        archive=archive_name,
        agent="codex",
        agent_version=str(meta.get("cli_version") or ""),
        session=str(meta.get("id") or path.stem),
        project=Path(meta.get("cwd") or "").name,
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
    )


def _attach_codex_helpers(tasks: list[Task], helpers: list[_CodexHelper]) -> None:
    """Add each helper thread's tokens to the parent task that was running when the helper started."""
    by_session: dict[str, list[Task]] = {}
    for task in tasks:
        by_session.setdefault(task.session, []).append(task)
    for helper in helpers:
        for task in by_session.get(helper.parent, []):
            if task.started_at <= helper.started_at <= task.started_at + task.seconds:
                new_input, cache_write, cache_read, output, usd = _totals(helper.calls)
                task.input_tokens += new_input
                task.cache_write_tokens += cache_write
                task.cache_read_tokens += cache_read
                task.output_tokens += output
                task.usd = None if task.usd is None or usd is None else task.usd + usd
                task.calls += len(helper.calls)
                break
