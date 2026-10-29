"""Each task's share of the weekly limit, for Claude Code's status line on a Claude subscription.

Claude Code tells the status line how much of the week's limit is used. It doesn't say what each task
used. So yardstick works out how much of the limit a dollar of work uses: the share used this week, over
what every Claude Code session on this computer spent since the week began, at list prices. A task's share
is what it spent so far times that. A forecast's share is its cost times that.

The spend comes from Claude Code's records, each reply counted once, as training counts it. Chats on
claude.ai count toward the limit too. They leave no record here. While you use them, the shares come out
a little high.

The status line keeps the latest reading. The work in the background, after a forecast, divides it by the
week's spend, so the status line never waits for a week of records to be read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .claude_code import _number, _read, _session_file, _write, folder

WEEK = 7 * 86400
READING = "weekly.json"
"""The latest reading of the weekly limit: the share used, and when the week resets."""
RATE = "weekly-rate.json"
"""How much of the weekly limit a dollar uses, as last worked out."""
SPENT = "weekly-spent.json"
"""What each minute of the past week cost, and how far into each record that was read."""
LEAST_PERCENT = 1.0
"""Below this share of the week, Claude Code's reading, which has one decimal, is too rough to divide."""
LEAST_USD = 1.0
"""Below this spend since the week began, there's too little to divide by."""
CALIBRATE_SECONDS = 300.0
"""How often, at most, the work in the background works out the rate again."""
RECENT = 2000
"""How many replies it remembers, so a reply whose lines come across two looks counts once."""


def note(event: dict) -> dict | None:
    """The weekly limit's reading in what Claude Code sends the status line. It's kept when it changes.

    None when Claude Code sends none, as with an API key, or before a session's first reply.
    """
    limits = event.get("rate_limits")
    week = limits.get("seven_day") if isinstance(limits, dict) else None
    if not isinstance(week, dict):
        return None
    percent, resets = _number(week.get("used_percentage")), _number(week.get("resets_at"))
    if percent is None or resets is None:
        return None
    reading = {"percent": percent, "resets_at": resets}
    path = folder() / READING
    saved = _read(path)
    if {key: saved.get(key) for key in reading} != reading:
        try:
            _write(path, reading)
        except OSError:
            pass  # the share still shows. The rate waits for a reading it can keep.
    return reading


def rate() -> float | None:
    """How many percent of the weekly limit a dollar at list prices uses. None until it's been worked out."""
    found = _number(_read(folder() / RATE).get("percent_per_usd"))
    return found if found is not None and found > 0 else None


def share(usd: float, percent_per_usd: float) -> str:
    """A share of the weekly limit, with four decimals."""
    return f"{max(usd, 0.0) * percent_per_usd:.4f}%"


def calibrate(records: Path, now: float | None = None) -> None:
    """Work out again how much of the weekly limit a dollar uses, from the latest reading.

    `records` is the folder of Claude Code's records. It does nothing when it last did this a few minutes
    ago, or when the reading is too rough to divide.
    """
    now = time.time() if now is None else now
    reading = _read(folder() / READING)
    percent, resets = _number(reading.get("percent")), _number(reading.get("resets_at"))
    if percent is None or resets is None or percent < LEAST_PERCENT or resets <= now:
        return
    last = _read(folder() / RATE)
    if now - (_number(last.get("at")) or 0.0) < CALIBRATE_SECONDS:
        return
    usd = spent_since(records, resets - WEEK, now)
    if usd >= LEAST_USD:
        _write(folder() / RATE, {"percent_per_usd": percent / usd, "resets_at": resets, "at": now})


def spent_since(records: Path, since: float, now: float | None = None) -> float:
    """What every Claude Code session in `records` spent since `since`, at list prices.

    It reads only what was added to each record since it last looked, and keeps a sum for each minute of
    the past eight days.
    """
    from . import locks

    now = time.time() if now is None else now
    keep = now - WEEK - 86400
    path = folder() / SPENT
    with locks.held(folder() / "weekly-spent.lock"):
        cache = _read(path)
        files = _offsets(cache.get("files"))
        minutes = _sums(cache.get("minutes"))
        recent = _marks(cache.get("recent"))
        seen = set(recent)
        try:
            records_found = list(records.rglob("*.jsonl"))
        except OSError:
            records_found = []
        for record in records_found:
            try:
                if record.stat().st_mtime < keep:
                    continue
            except OSError:
                continue
            for moment, usd in _new_costs(record, files, keep, seen, recent):
                minute = str(int(moment // 60) * 60)
                minutes[minute] = minutes.get(minute, 0.0) + usd
        minutes = {minute: usd for minute, usd in minutes.items() if float(minute) >= keep}
        files = {name: offset for name, offset in files.items() if Path(name).exists()}
        _write(path, {"files": files, "minutes": minutes, "recent": recent[-RECENT:]})
    return sum(usd for minute, usd in minutes.items() if float(minute) + 60 > since)


def task_spent(session: str, state: dict, transcript: str | None = None) -> float | None:
    """What the session's task has spent so far, at list prices: its record from its prompt on, and its helpers'.

    `transcript` is where the record is now, as Claude Code tells the status line. Claude Code moves it when
    a session moves to another folder, such as a worktree. Without it, the record is where it was at the
    prompt. Each look reads only what was added since the one before. None when the record isn't known.
    """
    started = _number(state.get("prompt_at"))
    transcript = transcript or state.get("transcript")
    if started is None or not isinstance(transcript, str) or not transcript:
        return None
    record = Path(transcript)
    start = state.get("offset") if isinstance(state.get("offset"), int) else 0
    path = _session_file(session, ".spent.json")
    cache = _read(path)
    if cache.get("prompt_at") != started:
        cache = {"prompt_at": started, "files": {str(record): start}, "usd": 0.0, "recent": []}
    files = _offsets(cache.get("files"))
    recent = _marks(cache.get("recent"))
    usd = _number(cache.get("usd")) or 0.0
    before = dict(files)
    helpers = record.with_suffix("") / "subagents"  # Claude Code keeps a session's helpers there, at any depth
    try:
        paths = [record, *sorted(helpers.rglob("*.jsonl"))] if helpers.is_dir() else [record]
    except OSError:
        paths = [record]
    seen = set(recent)
    for found in paths:
        usd += sum(cost for _, cost in _new_costs(found, files, started, seen, recent))
    if files != before:
        try:
            _write(path, {"prompt_at": started, "files": files, "usd": usd, "recent": recent[-RECENT:]})
        except OSError:
            pass
    return usd


def _new_costs(record: Path, files: dict[str, int], since: float, seen: set[str], recent: list[str]):
    """The time and cost of each reply added to a record since the offset in `files`, from `since` on.

    It moves the offset past each whole line it reads. A line still being written waits for the next look.
    """
    from .archives import _claude_call, _timestamp

    name = str(record)
    offset = files.get(name, 0)
    try:
        if record.stat().st_size < offset:
            offset = 0  # the record was written again from the start
        with open(record, "rb") as handle:
            handle.seek(offset)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                offset += len(raw)
                if b'"usage"' not in raw or b'"assistant"' not in raw:
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(line, dict) or line.get("type") != "assistant":
                    continue
                moment = _timestamp(line.get("timestamp"))
                found = _claude_call(line) if moment is not None and moment >= since else None
                if found is None:
                    continue
                key, call = found
                if key != (None, None):  # Claude Code writes one reply across several lines
                    mark = json.dumps(key)
                    if mark in seen:
                        continue
                    seen.add(mark)
                    recent.append(mark)
                cost = call.cost()
                if cost is not None:
                    yield moment, cost
    except OSError:
        pass
    files[name] = offset


def _offsets(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, int) and v >= 0}


def _sums(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    found = {}
    for minute, usd in value.items():
        try:
            float(minute)
        except (TypeError, ValueError):
            continue
        if _number(usd) is not None:
            found[minute] = float(usd)
    return found


def _marks(value: object) -> list[str]:
    return [mark for mark in value if isinstance(mark, str)] if isinstance(value, list) else []
