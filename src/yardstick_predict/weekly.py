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

import hashlib
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
"""For each record: how far into it was read, and what each ten minutes of its past week cost."""
BUCKET = 600
"""The week's spend is summed ten minutes at a time."""
LEAST_PERCENT = 1.0
"""Below this share of the week, Claude Code's reading, which has one decimal, is too rough to divide."""
LEAST_USD = 1.0
"""Below this spend since the week began, there's too little to divide by."""
CALIBRATE_SECONDS = 300.0
"""How often, at most, the work in the background works out the rate again."""


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

    It reads only what was added to each record since it last looked, and keeps each record's sums for the
    past eight days. A record is known by its session, not its folder: Claude Code moves a session's
    records when the session moves to another folder, and a moved record is read on from where it was.

    Each reply counts once, however many records hold it. A resumed or forked session's record starts with
    a copy of the conversation it came from, and Claude Code writes one reply across several lines.
    """
    from . import locks

    now = time.time() if now is None else now
    keep = now - WEEK - 86400
    path = folder() / SPENT
    with locks.held(folder() / "weekly-spent.lock"):
        entries = _entries(_read(path).get("files"))
        seen = _seen(entries)
        newest: dict[str, tuple[float, Path]] = {}  # a record left in two folders is read from the newer
        try:
            found = list(records.rglob("*.jsonl"))
        except OSError:
            found = []
        for record in found:
            try:
                changed = record.stat().st_mtime
            except OSError:
                continue
            key = _record_key(record, records)
            if changed >= keep and (key not in newest or changed > newest[key][0]):
                newest[key] = (changed, record)
        for key, (_, record) in newest.items():
            entry = entries.setdefault(key, _entry())
            for line in _new_lines(record, entry, seen, prompts=False):
                reply = _reply(line, seen, keep)
                if reply is not None:
                    moment, cost, mark = reply
                    _add(entry, str(int(moment // BUCKET) * BUCKET), cost, mark)
        for key, entry in list(entries.items()):
            for group in [group for group in entry["spent"] if float(group) < keep]:
                entry["spent"].pop(group, None)
                entry["replies"].pop(group, None)
            if not entry["spent"] and key not in newest:
                del entries[key]  # nothing of this week, and not written to since
        _write(path, {"files": entries})
    return sum(
        usd for entry in entries.values() for bucket, usd in entry["spent"].items() if float(bucket) + BUCKET > since
    )


def task_spent(session: str, state: dict, transcript: str | None = None) -> float | None:
    """What the session's task has spent so far, at list prices: its record from its prompt on, and its helpers'.

    `transcript` is where the record is now, as Claude Code tells the status line. Claude Code moves it when
    a session moves to another folder, such as a worktree. Without it, the record is where it was at the
    prompt. Each look reads only what was added since the one before. None when the record isn't known.

    As in training, a reply belongs to the prompt it worked for. In the session's record, that's the prompt
    before it. A helper works for the prompt that gave it its latest instruction, so a helper an earlier task
    started, still at work after this prompt, counts toward that task, not this one. The task's prompts are
    its own and those that went on with it, such as a background job's report, but not a prompt that
    started no task.
    """
    from .claude_code import _others

    started = _number(state.get("prompt_at"))
    transcript = transcript or state.get("transcript")
    if started is None or not isinstance(transcript, str) or not transcript:
        return None
    record = Path(transcript)
    own = state.get("prompt_id") if isinstance(state.get("prompt_id"), str) else ""
    path = _session_file(session, ".spent.json")
    cache = _read(path)
    if cache.get("prompt_at") != started:
        start = state.get("offset") if isinstance(state.get("offset"), int) else 0
        cache = {"prompt_at": started, "files": {"": _entry(start, own)}, "prompts": [own] if own else []}
    entries = _entries(cache.get("files"))
    prompts = _marks(cache.get("prompts"))
    before = json.dumps(entries, sort_keys=True)
    seen = _seen(entries)
    main = entries.setdefault("", _entry(0, own))
    for line in _new_lines(record, main, seen):
        if line.get("type") == "user":
            prompt_id = _name(line.get("promptId"))
            if prompt_id:
                main["owner"] = prompt_id
                if prompt_id not in prompts:
                    prompts.append(prompt_id)
            continue
        reply = _reply(line, seen, started)
        if reply is not None:
            _add(main, main.get("owner") or own, reply[1], reply[2])
    helpers = record.with_suffix("") / "subagents"  # Claude Code keeps a session's helpers there, at any depth
    try:
        found = sorted(helpers.rglob("*.jsonl")) if helpers.is_dir() else []
    except OSError:
        found = []
    for helper in found:
        try:
            if helper.stat().st_mtime < started:
                continue  # it did no work since the prompt
        except OSError:
            continue
        entry = entries.setdefault(str(helper.relative_to(helpers)), _entry())
        for line in _new_lines(helper, entry, seen):
            _helper_line(line, entry, seen)
    if json.dumps(entries, sort_keys=True) != before or prompts != _marks(cache.get("prompts")):
        try:
            _write(path, {"prompt_at": started, "files": entries, "prompts": prompts})
        except OSError:
            pass
    ours = {prompt_id for prompt_id in prompts if prompt_id not in set(_others(state))} | {own}
    return sum(usd for entry in entries.values() for owner, usd in entry["spent"].items() if owner in ours)


def _helper_line(line: dict, entry: dict, seen: set[str]) -> None:
    """Add a helper's line to whom it works for, as training reads a helper's record.

    Its first instruction carries the id of the prompt that started it. Its tool results carry the id of
    whatever prompt the main session is on, so they say nothing about whom the work is for. A later
    instruction, such as a message from another prompt, carries that prompt's id. A reply before the first
    instruction waits under no one until that instruction comes.
    """
    from .archives import _claude_text

    if line.get("type") == "user":
        prompt_id = _name(line.get("promptId"))
        words = _claude_text(line.get("message"))
        instruction = words is not None and not words.lstrip().startswith("[Request interrupted")
        owner = entry.get("owner")
        if prompt_id and (not owner or (instruction and prompt_id != owner)):
            if not owner and "" in entry["spent"]:
                _add(entry, prompt_id, entry["spent"].pop(""), *entry["replies"].pop("", []))
            entry["owner"] = prompt_id
        return
    reply = _reply(line, seen, 0.0)
    if reply is not None:
        _add(entry, entry.get("owner") or "", reply[1], reply[2])


def _record_key(record: Path, records: Path) -> str:
    """A record's name, wherever Claude Code keeps it: its path below the project's folder of records.

    That's the session's own file, or a helper's below the session's folder. Session ids are unique, so the
    name stays one record's when Claude Code moves the session's records to another project's folder.
    """
    try:
        parts = record.relative_to(records).parts
    except ValueError:
        return str(record)
    return str(Path(*parts[1:])) if len(parts) > 1 else str(Path(*parts))


def _entry(offset: int = 0, owner: str = "") -> dict:
    """What's kept of one record: how far it was read and whom its latest line worked for. Then, in groups,
    what its replies cost and a mark for each reply, so it counts once."""
    return {"offset": offset, "owner": owner, "spent": {}, "replies": {}}


def _add(entry: dict, group: str, usd: float, *marks: str) -> None:
    entry["spent"][group] = entry["spent"].get(group, 0.0) + usd
    entry["replies"].setdefault(group, []).extend(marks)


def _seen(entries: dict[str, dict]) -> set[str]:
    """The marks of every reply already counted."""
    return {mark for entry in entries.values() for marks in entry["replies"].values() for mark in marks}


def _new_lines(record: Path, entry: dict, seen: set[str], prompts: bool = True):
    """Each whole line added to a record since the offset in `entry` that may hold a reply, or, with
    `prompts`, a prompt's id.

    It moves the offset past each whole line it reads. A line still being written waits for the next look.
    When the record was written again from the start, what it counted before goes, and it's read afresh.
    """
    offset = entry["offset"]
    try:
        if record.stat().st_size < offset:
            offset = 0
            seen.difference_update(mark for marks in entry["replies"].values() for mark in marks)
            entry.update(_entry())
        with open(record, "rb") as handle:
            handle.seek(offset)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                offset += len(raw)
                entry["offset"] = offset
                reply = b'"assistant"' in raw and b'"usage"' in raw
                if not reply and not (prompts and b'"promptId"' in raw):
                    continue
                try:
                    line = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(line, dict):
                    yield line
    except OSError:
        return


def _reply(line: dict, seen: set[str], since: float) -> tuple[float, float, str] | None:
    """The time, list price and mark of a reply from `since` on, the first time it comes. Else None."""
    from .archives import _claude_call, _timestamp

    if line.get("type") != "assistant":
        return None
    moment = _timestamp(line.get("timestamp"))
    found = _claude_call(line) if moment is not None and moment >= since else None
    if found is None:
        return None
    key, call = found
    # A short mark of the reply's ids. A reply with none can't be told from another, so each line counts.
    mark = hashlib.blake2b(json.dumps(key).encode(), digest_size=8).hexdigest() if key != (None, None) else ""
    if mark and mark in seen:
        return None
    seen.add(mark)
    cost = call.cost()
    return None if cost is None else (moment, cost, mark)


def _entries(value: object) -> dict[str, dict]:
    """The kept records, with anything damaged left out."""
    if not isinstance(value, dict):
        return {}
    found = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        offset = entry.get("offset")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            continue
        spent = entry.get("spent") if isinstance(entry.get("spent"), dict) else {}
        replies = entry.get("replies") if isinstance(entry.get("replies"), dict) else {}
        owner = entry.get("owner") if isinstance(entry.get("owner"), str) else ""
        found[key] = {
            "offset": offset,
            "owner": owner,
            "spent": {k: float(v) for k, v in spent.items() if isinstance(k, str) and _number(v) is not None},
            "replies": {k: _marks(v) for k, v in replies.items() if isinstance(k, str)},
        }
    return found


def _name(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _marks(value: object) -> list[str]:
    return [mark for mark in value if isinstance(mark, str)] if isinstance(value, list) else []
