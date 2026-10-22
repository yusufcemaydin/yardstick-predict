from __future__ import annotations

import dataclasses
import os
import shutil
import sqlite3
from dataclasses import fields

import numpy as np
import pytest
from builders import ClaudeSession, CodexSession, usage

from yardstick_predict import model_types
from yardstick_predict.archives import READER_VERSION, Task, read_archive, record_files
from yardstick_predict.pipeline import embedding_items, learnable, read_all, save_scan, scan, train
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


def test_when_two_archives_hold_the_same_codex_helper_it_counts_once(tmp_path):
    archives = [
        {"name": "backup", "format": "codex", "path": str(tmp_path / "backup")},
        {"name": "live", "format": "codex", "path": str(tmp_path / "live")},
    ]
    for folder in ("backup", "live"):
        parent = CodexSession(tmp_path / folder, "parent-thread", START)
        parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
        parent.save(mtime=START)
        helper = CodexSession(tmp_path / folder, "helper-thread", START + 10, parent="parent-thread")
        helper.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5)
        helper.save(mtime=START)
    # The live folder also keeps a stray copy of the helper's file.
    shutil.copy(helper.path, helper.path.with_name("rollout-helper-thread-copy.jsonl"))

    found = read_all(archives, now=LATER)
    assert [(t.prompt, t.calls, t.input_tokens) for t in found.new_tasks] == [("Split the work", 2, 1200)]


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


@pytest.mark.parametrize("ending", ["error", "interrupt"])
def test_a_task_that_turns_out_stopped_after_a_refresh_is_corrected(tmp_path, ending):
    archive = {"name": "claude-code", "format": "claude-code", "path": str(tmp_path)}
    session = ClaudeSession(tmp_path, "long-build", START)
    line = session.prompt("Run the long build")
    tool = session.reply(5, tool="Bash")
    session.tool_result(tool, 1000, line["promptId"])  # the build took a while, and the next reply hasn't come
    session.save(mtime=START + 1005)
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    save_scan(predictor, read_all([archive], now=START + 1906))  # quiet for 15 minutes, so it looks finished
    assert len(learnable(predictor.tasks())) == 1

    if ending == "error":
        session.synthetic(3, error="rate_limit")  # the next call failed, with no paid call
    else:
        session.interrupt()  # the person pressed Esc before the reply came
    session.save(mtime=START + 1010)
    found = scan(predictor, now=START + 2000)
    assert [t.stopped for t in found.updated_tasks] == [True]
    save_scan(predictor, found)
    assert stored(predictor, "Run the long build").stopped
    assert not learnable(predictor.tasks())


def test_a_changed_workflow_helper_record_makes_its_session_read_again(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Run the review workflow")
    session.reply(5)
    session.save(mtime=START)
    predictor = predictor_for({"name": "claude-code", "format": "claude-code", "path": str(tmp_path)})
    assert stored(predictor, "Run the review workflow").calls == 1

    session.helper(line["promptId"], calls=2, folder="workflows/wf_1")
    session.save(mtime=START, helpers_mtime=START + 60)
    found = scan(predictor, now=LATER)
    assert [t.calls for t in found.updated_tasks] == [3]


def test_a_task_whose_helper_is_still_working_waits_and_its_records_are_read_again(tmp_path):
    archive = {"name": "claude-code", "format": "claude-code", "path": str(tmp_path)}
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Research the options in the background")
    tool_id, job = session.background(line["promptId"], tool="Agent")
    session.reply(2, text="A helper is on it.")
    helper = session.helper(line["promptId"], calls=1, agent=job)
    session.save(mtime=START, helpers_mtime=START + 3500)
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    found = read_all([archive], now=START + 3560)
    # The helper is still working, so the task isn't saved, and the session's record isn't remembered as read.
    assert not found.new_tasks and str(session.path) not in found.files
    save_scan(predictor, found)

    session.helper_more(helper, None, calls=2)
    session.notification(tool_id, job, seconds=600)
    session.reply(5)
    session.save(mtime=START + 3700, helpers_mtime=START + 3600)
    found = scan(predictor, now=LATER)
    assert [(t.prompt, t.calls, t.stopped) for t in found.new_tasks] == [
        ("Research the options in the background", 6, False)
    ]


def test_a_codex_task_whose_helper_is_still_working_is_saved_once_the_helper_finishes(tmp_path):
    archive = {"name": "codex", "format": "codex", "path": str(tmp_path)}
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("p1", "Launch background build", calls=[(1000, 0, 100)], seconds=10, root="p1")
    parent.save(mtime=START + 10)
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "Background build", calls=[(200, 0, 20)], seconds=50, root="p1", finished=False)
    helper.save(mtime=START + 900)
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    found = read_all([archive], now=START + 960)
    assert not found.new_tasks and str(parent.path) not in found.files
    save_scan(predictor, found)

    helper.clock = START + 950
    helper.call(400, 0, 40)
    helper.finish("h1", 5)
    helper.save(mtime=START + 1000)
    found = scan(predictor, now=LATER)
    assert [(t.input_tokens, t.calls, t.stopped) for t in found.new_tasks] == [(1600, 3, False)]


def test_a_dataset_read_by_an_older_reader_is_read_again_and_tidied(tmp_path):
    archive = {"name": "claude-code", "format": "claude-code", "path": str(tmp_path)}
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Research the options")
    session.reply(5)
    session.helper(line["promptId"], calls=2)
    session.prompt("Check the deploy", origin="scheduled", meta=True)
    session.reply(5)
    session.save(mtime=START)
    fresh = {t.prompt: t for t in read_archive(archive, now=LATER)}
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    # What an older version stored: no record file, its helper missed and the scheduled task's call in it, a
    # turn the new reader doesn't count as a task, and a task whose session Claude Code has since deleted.
    older = dataclasses.replace(fresh["Research the options"], calls=2, source="", prior_skills="")
    ghost = dataclasses.replace(older, id="claude-code:ghost", prompt="Not a task")
    gone = dataclasses.replace(older, id="claude-code:gone", session="deleted", prompt="From a deleted session")
    predictor.add_tasks([older, ghost, gone])
    predictor.put_embeddings("prompt", {"claude-code:ghost": np.ones(256)})
    predictor.remember_files(record_files(archive))  # the older version had read every file
    assert "reader" not in predictor.config

    found = scan(predictor, now=LATER)
    assert found.reader == READER_VERSION
    assert [t.prompt for t in found.new_tasks] == ["Check the deploy"]
    assert [(t.prompt, t.calls, t.source) for t in found.updated_tasks] == [
        ("Research the options", 3, str(session.path))
    ]
    assert found.removed == ["claude-code:ghost"]
    save_scan(predictor, found)
    assert predictor.config["reader"] == READER_VERSION and predictor.config["needs_training"]
    assert sorted(t.prompt for t in predictor.tasks()) == [
        "Check the deploy",
        "From a deleted session",
        "Research the options",
    ]
    assert "claude-code:ghost" not in predictor.embedded("prompt")
    # The next refresh reads only the files that changed, and none did.
    again = scan(predictor, now=LATER)
    assert (again.new_tasks, again.updated_tasks, again.removed, again.reader) == ([], [], [], None)
