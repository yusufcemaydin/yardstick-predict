from __future__ import annotations

import os

from builders import ClaudeSession, CodexSession, usage

from yardstick_predict.pipeline import read_all, save_scan, scan
from yardstick_predict.store import Predictor

START = 1_780_000_000.0
LATER = START + 30 * 86400


def predictor_for(archive: dict) -> Predictor:
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    save_scan(predictor, read_all([archive], now=LATER))
    return predictor


def stored(predictor: Predictor, prompt: str):
    return next(t for t in predictor.tasks() if t.prompt == prompt)


def test_a_helper_that_finishes_after_a_refresh_is_added_to_its_parent(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Start a background helper")
    session.reply(5, calls_usage=usage(new_input=100, read=1000, output=10))
    session.helper(line["promptId"], calls=1, call_usage=usage(new_input=10, read=500, output=5))
    session.save(mtime=START)
    predictor = predictor_for({"name": "claude-code", "format": "claude-code", "path": str(tmp_path)})
    before = stored(predictor, "Start a background helper")
    assert before.calls == 2

    # The helper keeps working after the refresh and writes another record.
    session.helper(line["promptId"], calls=3, call_usage=usage(new_input=10, read=500, output=5))
    for path in (session.folder / "s1" / "subagents").glob("*.jsonl"):
        os.utime(path, (START + 60, START + 60))
    found = scan(predictor, now=LATER)
    assert not found.new_tasks
    assert [t.calls for t in found.updated_tasks] == [5]
    save_scan(predictor, found)
    after = stored(predictor, "Start a background helper")
    assert after.calls == 5
    assert after.input_tokens == before.input_tokens + 3 * 10
    assert predictor.count_tasks() == 1


def test_a_copy_in_a_resumed_session_never_replaces_the_fuller_original(tmp_path):
    original = ClaudeSession(tmp_path, "s1", START)
    line = original.prompt("Original task", uuid="same-uuid")
    original.reply(5)
    original.helper(line["promptId"], calls=2)
    original.save(mtime=START)
    predictor = predictor_for({"name": "claude-code", "format": "claude-code", "path": str(tmp_path)})
    assert stored(predictor, "Original task").calls == 3

    resumed = ClaudeSession(tmp_path, "s2", START)
    resumed.prompt("Original task", uuid="same-uuid", prompt_id=line["promptId"])
    resumed.reply(5)
    resumed.prompt("A new task")
    resumed.reply(2)
    resumed.save(mtime=START + 100)
    found = scan(predictor, now=LATER)
    assert [t.prompt for t in found.new_tasks] == ["A new task"]
    assert not found.updated_tasks
    save_scan(predictor, found)
    assert stored(predictor, "Original task").calls == 3


def test_a_codex_helper_thread_that_appears_later_is_added_to_its_parent(tmp_path):
    parent = CodexSession(tmp_path, "parent-thread", START)
    parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    predictor = predictor_for({"name": "codex", "format": "codex", "path": str(tmp_path)})
    assert stored(predictor, "Split the work").calls == 1

    helper = CodexSession(tmp_path, "helper-thread", START + 10, parent="parent-thread")
    helper.turn("h1", "Helper part", calls=[(200, 0, 20), (300, 0, 30)], seconds=5)
    helper.save(mtime=START + 200)
    found = scan(predictor, now=LATER)
    assert [t.calls for t in found.updated_tasks] == [3]
    save_scan(predictor, found)
    assert stored(predictor, "Split the work").input_tokens == 1500
