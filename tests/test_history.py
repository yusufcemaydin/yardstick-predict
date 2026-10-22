from __future__ import annotations

import math
from dataclasses import replace

from builders import ClaudeSession

from yardstick_predict import pipeline
from yardstick_predict.archives import Task
from yardstick_predict.history import Past, size_numbers
from yardstick_predict.store import Predictor

START = 1_780_000_000.0


def task(i: int, start: float, seconds: float = 60.0, project: str = "app", session: str = "s1", **changes) -> Task:
    found = Task(
        id=f"t{i}",
        archive="test",
        agent="claude-code",
        agent_version="1",
        session=session,
        project=project,
        origin="person",
        started_at=start,
        model="claude-opus-5-5",
        effort="high",
        prompt=f"Task {i}",
        context="",
        context_tokens=10_000,
        seconds=seconds,
        input_tokens=1_000,
        cache_write_tokens=0,
        cache_read_tokens=9_000,
        output_tokens=500,
        usd=0.25,
        calls=3,
        direct=False,
        stopped=False,
        idle=False,
    )
    return replace(found, **changes)


def test_a_project_shows_its_last_finished_tasks_and_nothing_still_running():
    done = [task(i, START + i * 600, seconds=60 * (i + 1)) for i in range(7)]
    running = task(7, START + 4_190, seconds=600)  # started before, still running at the moment asked about
    past = Past([*done, running])
    found = past.project_numbers("app", START + 4_200)
    # Only the 7 finished tasks count. Their last 5 took 3 to 7 minutes, so the middle one took 5.
    assert found["project_tasks_before"] == math.log1p(7)
    assert math.isclose(found["project_seconds"], math.log(300))
    assert found["project_direct"] == 0.0
    assert math.isclose(found["project_days_since"], math.log1p((4_200 - 3_600 - 420) / 86400))


def test_a_task_that_waited_for_the_person_counts_only_once_it_really_finished():
    waited = task(1, START, seconds=60, finished_at=START + 3_600)  # one minute of work, an hour of waiting
    past = Past([waited])
    assert "project_seconds" not in past.project_numbers("app", START + 600)
    assert "project_seconds" in past.project_numbers("app", START + 3_600)


def test_the_conversation_so_far_is_its_last_finished_task():
    first = task(1, START, seconds=120, usd=0.5)
    second = task(2, START + 300, seconds=30, session="other")
    past = Past([first, second])
    found = past.session_numbers("s1", START + 600)
    assert found["session_tasks_before"] == math.log1p(1)
    assert math.isclose(found["previous_seconds"], math.log(120))
    assert math.isclose(found["previous_usd"], math.log(0.5))
    assert math.isclose(found["minutes_since_previous"], math.log1p(8))
    assert past.session_numbers("new", START + 600) == {"session_tasks_before": 0.0}  # a new one
    assert past.project_numbers("", START + 600) == {"project_tasks_before": 0.0}


def test_a_stopped_task_still_counts_as_the_conversation_so_far():
    stopped = task(1, START, seconds=40, stopped=True)
    found = Past([stopped]).session_numbers("s1", START + 100)
    assert math.isclose(found["previous_seconds"], math.log(40))
    # It's no help to the project's typical outcomes, which only learnable tasks make.
    assert "project_seconds" not in Past([stopped]).project_numbers("app", START + 100)


def test_a_programs_conversation_size_has_its_own_number():
    assert size_numbers(20_000, "person") == {"conversation_tokens": math.log1p(20_000)}
    assert size_numbers(20_000, "program") == {"conversation_tokens_program": math.log1p(20_000)}
    assert size_numbers(0, "person") == {}


def test_a_new_conversation_gets_the_size_new_conversations_usually_start_at():
    first = task(1, START, context_tokens=20_000)
    second = task(2, START + 600, context_tokens=90_000)
    other = task(3, START + 1_200, session="s2", context_tokens=24_000)
    past = Past([first, second, other])
    assert past.new_sizes == {"claude-code": 22_000}  # from each conversation's first task only
    found = past.numbers("app", "s3", START + 2_000, "person", 0, "claude-code", "x" * 400)
    assert math.isclose(found["conversation_tokens"], math.log1p(22_100))
    assert past.numbers("app", "s1", START + 2_000, "person", 50_000)["conversation_tokens"] == math.log1p(50_000)


def test_a_forecast_reads_only_its_own_project_folders_new_tasks(tmp_path):
    records = tmp_path / "projects"
    archive = {"name": "claude-code", "format": "claude-code", "path": str(records)}
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    here = ClaudeSession(records, "s1", START, project="shop")
    here.prompt("Add a login page")
    here.reply(30)
    here.prompt("Add tests")  # still running, so it's read again next time
    here.reply(30)
    record = here.save()
    elsewhere = ClaudeSession(records, "s2", START, project="blog")
    elsewhere.prompt("Fix the header")
    elsewhere.reply(30)
    elsewhere.save()
    found = pipeline.recent_tasks(predictor, str(record))
    assert [(t.project, t.prompt) for t in found] == [("shop", "Add a login page")]
    predictor.remember_files({str(record): (record.stat().st_size, record.stat().st_mtime)})
    assert pipeline.recent_tasks(predictor, str(record)) == []  # nothing changed since the last refresh
