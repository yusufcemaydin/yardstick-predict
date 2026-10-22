from __future__ import annotations

import os
import sqlite3
from dataclasses import fields

from builders import ClaudeSession, CodexSession, usage

from yardstick_predict import model_types
from yardstick_predict.archives import Task
from yardstick_predict.pipeline import embedding_items, read_all, save_scan, scan, train
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


def test_when_one_codex_helper_changes_its_unchanged_siblings_still_count(tmp_path):
    parent = CodexSession(tmp_path, "parent-thread", START)
    parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    first = CodexSession(tmp_path, "first-helper", START + 10, parent="parent-thread")
    first.turn("h1", "First part", calls=[(200, 0, 20)], seconds=5)
    first.save(mtime=START)
    second = CodexSession(tmp_path, "second-helper", START + 20, parent="parent-thread")
    second.turn("h2", "Second part", calls=[(300, 0, 30), (300, 0, 30)], seconds=5)
    second.save(mtime=START)
    predictor = predictor_for({"name": "codex", "format": "codex", "path": str(tmp_path)})
    assert stored(predictor, "Split the work").calls == 4

    # The first helper keeps working after the refresh. The second one is done and doesn't change.
    first = CodexSession(tmp_path, "first-helper", START + 10, parent="parent-thread")
    first.turn("h1", "First part", calls=[(200, 0, 20)] * 4, seconds=5)
    first.save(mtime=START + 200)
    found = scan(predictor, now=LATER)
    assert [t.calls for t in found.updated_tasks] == [7]
    save_scan(predictor, found)
    assert stored(predictor, "Split the work").input_tokens == 1000 + 4 * 200 + 2 * 300


def test_a_codex_helper_that_a_helper_spawns_later_is_added_to_the_main_task(tmp_path):
    parent = CodexSession(tmp_path, "parent-thread", START)
    parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    helper = CodexSession(tmp_path, "helper-thread", START + 10, parent="parent-thread")
    helper.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5)
    helper.save(mtime=START)
    predictor = predictor_for({"name": "codex", "format": "codex", "path": str(tmp_path)})
    assert stored(predictor, "Split the work").calls == 2

    nested = CodexSession(tmp_path, "nested-thread", START + 12, parent="helper-thread")
    nested.turn("n1", "Nested part", calls=[(50, 0, 5)], seconds=2)
    nested.save(mtime=START + 200)
    found = scan(predictor, now=LATER)
    assert [t.calls for t in found.updated_tasks] == [3]
    save_scan(predictor, found)
    assert stored(predictor, "Split the work").input_tokens == 1250


def test_a_codex_task_read_before_it_finished_is_repaired_when_it_finishes(tmp_path):
    live = {"name": "codex", "format": "codex", "path": str(tmp_path / "live")}
    session = CodexSession(tmp_path / "live", "c1", START)
    session.turn("t1", "Run the long build", calls=[(1000, 0, 100)], seconds=60)
    finished = session.lines.pop()  # the completion hasn't been written yet
    session.save(mtime=START)
    predictor = predictor_for(live)
    assert stored(predictor, "Run the long build").stopped  # quiet so long that it looked abandoned

    session.lines.append(finished)
    session.save(mtime=START + 3600)
    found = scan(predictor, now=LATER)
    assert [t.stopped for t in found.updated_tasks] == [False]
    save_scan(predictor, found)
    assert not stored(predictor, "Run the long build").stopped

    # A copy saved before it finished, such as in a backup, loses to the finished one.
    early = CodexSession(tmp_path / "backup", "c1", START)
    early.lines = session.lines[:-1]
    early.save(mtime=START)
    backup = {"name": "backup", "format": "codex", "path": str(tmp_path / "backup")}
    assert [t.stopped for t in read_all([backup, live], now=LATER).new_tasks] == [False]


def test_when_two_archives_hold_the_same_task_the_most_complete_copy_wins(tmp_path):
    archives = [
        {"name": "backup", "format": "claude-code", "path": str(tmp_path / "backup")},
        {"name": "live", "format": "claude-code", "path": str(tmp_path / "live")},
    ]

    def write_both(session: str, prompt: str):
        # The backup was copied before the helper finished. The live folder has the whole task.
        for folder, helper_done in (("backup", False), ("live", True)):
            builder = ClaudeSession(tmp_path / folder, session, START)
            builder.prompt(prompt, uuid=f"{session}-prompt", prompt_id=f"{session}-p")
            builder.reply(5)
            if helper_done:
                builder.helper(f"{session}-p", calls=2)
            builder.save(mtime=START)

    write_both("s1", "First shared task")
    found = read_all(archives, now=LATER)
    assert [(t.prompt, t.calls) for t in found.new_tasks] == [("First shared task", 3)]
    assert found.per_archive == {"backup": 0, "live": 1}
    predictor = Predictor.create("default", archives, {"model": "voyage-4", "dimensions": 256})
    save_scan(predictor, found)

    write_both("s2", "Second shared task")
    found = scan(predictor, now=LATER)
    assert [(t.prompt, t.calls) for t in found.new_tasks] == [("Second shared task", 3)]
    assert found.per_archive == {"backup": 0, "live": 1}


def test_max_tasks_trains_on_the_same_tasks_it_embeds(tmp_path, monkeypatch):
    session = ClaudeSession(tmp_path, "s1", START)
    for prompt in ("Oldest", "Middle", "Newest finished"):
        session.prompt(prompt)
        session.reply(5)
        session.clock += 60
    session.prompt("Newest, stopped")
    session.reply(5, tool="Bash")
    session.interrupt()
    session.save(mtime=START)
    predictor = predictor_for({"name": "claude-code", "format": "claude-code", "path": str(tmp_path)})
    predictor.config.update(max_tasks=2, model={"type": "capture"})

    given: list = []

    class Captured:
        meta = {"tasks": 0, "report": {}}

        def save(self, folder):
            pass

    def capture(tasks, rows, settings):
        given.extend(tasks)
        return Captured()

    monkeypatch.setitem(model_types._TYPES, "capture", (capture, lambda folder: Captured()))
    train(predictor)
    embedded = [text for _, _, text in embedding_items(predictor.tasks(), ["prompt"], limit=2)]
    assert sorted(t.prompt for t in given) == sorted(embedded) == ["Middle", "Newest finished"]


def test_an_older_dataset_with_its_columns_in_another_order_keeps_each_value_in_its_column(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Add a login page")
    session.reply(5)
    session.save(mtime=START)
    archive = {"name": "claude-code", "format": "claude-code", "path": str(tmp_path)}
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    # An older version made the table without these columns. Adding them later puts them at the end.
    older = [f.name for f in fields(Task) if f.name not in ("agent_version", "effort")]
    db = sqlite3.connect(predictor.dir / "dataset.sqlite")
    db.execute(f"CREATE TABLE tasks ({', '.join(older)}, PRIMARY KEY (id))")
    db.commit()
    db.close()

    found = read_all([archive], now=LATER)
    save_scan(predictor, found)
    assert predictor.tasks() == found.new_tasks

    session.helper(line["promptId"], calls=2)
    for path in (session.folder / "s1" / "subagents").glob("*.jsonl"):
        os.utime(path, (START + 60, START + 60))
    found = scan(predictor, now=LATER)
    assert [t.calls for t in found.updated_tasks] == [3]
    save_scan(predictor, found)
    assert predictor.tasks() == found.updated_tasks
