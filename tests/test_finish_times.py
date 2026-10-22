from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import asdict, fields

import pytest
from builders import ClaudeSession, CodexSession

from yardstick_predict import model as M
from yardstick_predict.archives import Task, read_archive
from yardstick_predict.store import Predictor

START = 1_780_000_000.0
LATER = START + 30 * 86400


def claude_archive(path):
    return {"name": "claude-code", "format": "claude-code", "path": str(path)}


def codex_archive(path):
    return {"name": "codex", "format": "codex", "path": str(path)}


def test_a_task_that_waited_30_minutes_for_the_person_finishes_30_minutes_after_its_start_plus_its_seconds(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Pick a database for me")
    tool = session.reply(5, tool="AskUserQuestion")
    session.tool_result(tool, 30 * 60, line["promptId"])
    session.reply(6)
    session.save(mtime=START)

    [task] = read_archive(claude_archive(tmp_path), now=LATER)
    assert task.seconds == pytest.approx(11)
    assert task.finished_at == pytest.approx(START + 30 * 60 + 11)
    assert M.finish_times([task])[0] == pytest.approx(task.started_at + task.seconds + 30 * 60)


def test_a_codex_task_finishes_at_its_completion_or_else_at_its_last_moment(tmp_path):
    session = CodexSession(tmp_path, "c1", START)
    session.turn("t1", "Fix the failing test", calls=[(1000, 0, 100)], seconds=40)
    session.clock = START + 100
    session.turn("t2", "Never finished", calls=[(100, 0, 10), (100, 0, 10)], seconds=20)
    session.lines.pop()  # the completion was never written
    session.save(mtime=START)

    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    assert tasks["Fix the failing test"].finished_at == START + 40
    assert tasks["Never finished"].finished_at == START + 120
    assert tasks["Never finished"].stopped


def test_a_task_from_an_older_dataset_without_a_finish_time_still_loads(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Add a login page")
    session.reply(5)
    session.save(mtime=START)
    archive = claude_archive(tmp_path)
    [task] = read_archive(archive, now=LATER)
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    # Version 0.2.0 made the table without the finish time, and stored the task without it.
    older = [f.name for f in fields(Task) if f.name != "finished_at"]
    db = sqlite3.connect(predictor.dir / "dataset.sqlite")
    db.execute(f"CREATE TABLE tasks ({', '.join(older)}, PRIMARY KEY (id))")
    values = asdict(task)
    db.execute(
        f"INSERT INTO tasks ({', '.join(older)}) VALUES ({', '.join('?' for _ in older)})", [values[n] for n in older]
    )
    db.commit()
    db.close()

    [loaded] = predictor.tasks()
    assert loaded.finished_at is None
    assert loaded.prompt == "Add a login page" and loaded.seconds == task.seconds
    assert M.finish_times([loaded])[0] == loaded.started_at + loaded.seconds


def history(n: int) -> list[Task]:
    """Tasks whose time and tokens grow with the prompt, so the model has something to learn."""
    rng = random.Random(0)
    tasks = []
    for i in range(n):
        words = rng.randint(1, 80)
        seconds = math.exp(math.log(30 + 4 * words) + rng.gauss(0, 0.4))
        tokens = int(math.exp(math.log(50_000 * (1 + words / 10)) + rng.gauss(0, 0.4)))
        tasks.append(
            Task(
                id=f"t{i}",
                archive="test",
                agent="claude-code",
                agent_version="1",
                session=f"s{i // 5}",
                project="demo",
                origin="person",
                started_at=START + i * 600,
                model="claude-opus-5-5",
                effort="high",
                prompt="word " * words,
                context="",
                context_tokens=20_000,
                seconds=seconds,
                input_tokens=tokens // 10,
                cache_write_tokens=0,
                cache_read_tokens=tokens - tokens // 10,
                output_tokens=0,
                usd=tokens / 1e6,
                calls=5,
                direct=False,
                stopped=False,
                idle=False,
            )
        )
    return tasks


def test_a_test_window_doesnt_learn_from_a_task_still_waiting_for_the_person_when_it_opens(monkeypatch):
    tasks = history(150)
    waited = {t.id: t.started_at + t.seconds + 5 * 3600 for t in tasks[::4]}  # five hours waiting for the person
    for t in tasks:
        if t.id in waited:
            t.finished_at = waited[t.id]
    learned, windows = [], []
    real_fit, real_score = M.fit, M._score_window

    def fit(train_tasks, train_rows, settings):
        learned.append(train_tasks)
        return real_fit(train_tasks, train_rows, settings)

    def score(trial, train_tasks, test_tasks, test_rows):
        windows.append(test_tasks)
        return real_score(trial, train_tasks, test_tasks, test_rows)

    monkeypatch.setattr(M, "fit", fit)
    monkeypatch.setattr(M, "_score_window", score)
    M.test(tasks, [M.Features(t.model, t.effort, t.agent, t.origin, len(t.prompt)) for t in tasks], {})
    assert len(learned) == len(windows) == 3
    for fitted, window in zip(learned, windows):
        opens = min(t.started_at for t in window)
        finished = [t.id for t in tasks if t.started_at < opens and waited.get(t.id, t.started_at + t.seconds) <= opens]
        assert [t.id for t in fitted] == finished
        assert any(t.id in waited for t in fitted)  # a task that waited is learned from once it finished
