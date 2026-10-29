"""Open Runs and the runs format: public runs that come with the tool, and records any program can write."""

from __future__ import annotations

import gzip
import json
import lzma
import sqlite3
from array import array
from dataclasses import replace

import pytest
from test_llm import FIELDS, answer_for, ids_in
from test_privacy import private_data

from yardstick_predict import archives, cli, embeddings, features, llm, pipeline, store
from yardstick_predict import model as M
from yardstick_predict.archives import OPEN_RUNS, Task

HOW = {"provider": "anthropic", "model": "claude-opus-5-5"}


def runs_folder(tmp_path, *lines, name="runs.jsonl"):
    folder = tmp_path / "runs"
    folder.mkdir(exist_ok=True)
    text = "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n"
    path = folder / name
    if name.endswith(".xz"):
        path.write_bytes(lzma.compress(text.encode()))
    elif name.endswith(".gz"):
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return {"name": "mine", "format": "runs", "path": str(folder)}


def test_a_runs_file_gives_one_task_per_line(tmp_path):
    archive = runs_folder(
        tmp_path,
        {
            "id": "a1",
            "prompt": "Add a login page",
            "model": "claude-opus-5-5",
            "agent": "my-agent",
            "effort": "high",
            "started_at": "2026-10-01T09:00:00Z",
            "seconds": 240,
            "input_tokens": 1000,
            "cache_read_tokens": 50_000,
            "output_tokens": 3000,
        },
        {
            "prompt": "Fix the build",
            "model": "gpt-6.1-sol",
            "started_at": 1.79e9,
            "seconds": 60.5,
            "tokens": 90_000,
            "usd": 0.4,
        },
        {"prompt": "No time given", "model": "gpt-6.1-sol", "started_at": 1.79e9, "tokens": 10},
        {"prompt": "No start given", "model": "gpt-6.1-sol", "seconds": 5, "tokens": 10},
        {"prompt": "", "model": "gpt-6.1-sol", "started_at": 1.79e9, "seconds": 5, "tokens": 10},
        "not json",
    )
    tasks = {task.prompt: task for task in archives.read_archive(archive)}
    assert sorted(tasks) == ["Add a login page", "Fix the build"]
    first, second = tasks["Add a login page"], tasks["Fix the build"]
    assert first.id == "mine:a1" and first.archive == "mine" and first.agent == "my-agent" and first.effort == "high"
    assert first.seconds == 240 and first.tokens == 54_000 and first.origin == "program"
    assert first.usd is not None and first.usd > 0  # worked out from the AI model's list price
    assert first.finished_at == first.started_at + 240 and first.source.endswith("runs.jsonl")
    assert second.agent == "other" and second.effort == "default" and second.input_tokens == 90_000
    assert second.usd == 0.4 and second.id.startswith("mine:")


def test_compressed_runs_files_are_read_too(tmp_path):
    line = {"prompt": "Write tests", "model": "claude-opus-5-5", "started_at": 1.79e9, "seconds": 30, "tokens": 5000}
    for name in ("a.jsonl.gz", "b.jsonl.xz"):
        archive = runs_folder(tmp_path, {**line, "id": name}, name=name)
    assert sorted(task.id for task in archives.read_archive(archive)) == ["mine:a.jsonl.gz", "mine:b.jsonl.xz"]
    files = archives.record_files(archive)
    assert archives.families(archive, files) == {path: path for path in files}


def test_open_runs_comes_with_the_tool():
    folder = archives.archive_path(OPEN_RUNS)
    assert folder.parent.name == "yardstick_predict" and archives.record_files(OPEN_RUNS)
    tasks = archives.read_archive(OPEN_RUNS)
    assert len(tasks) > 1000 and all(task.archive == "open-runs" for task in tasks)
    assert len({task.prompt for task in tasks}) > 100 and len({task.model for task in tasks}) > 20
    assert all(task.seconds > 0 and task.tokens > 0 and not task.project and not task.session for task in tasks)


def test_no_open_runs_prompt_holds_private_data():
    prompts = {task.prompt for task in archives.read_archive(OPEN_RUNS)}
    found = {problem for prompt in prompts for line in prompt.splitlines() for problem in private_data(line)}
    assert found == set()


def test_setup_offers_open_runs(monkeypatch, open_runs):
    monkeypatch.setattr(cli, "DEFAULT_ARCHIVES", ())  # no records of your own here
    asked = []
    replies = ["y", ""]

    def fake_input(question: str = "") -> str:
        asked.append(question.strip())
        return replies.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.choose_archives(True, []) == [OPEN_RUNS]
    assert "Start from Open Runs" in asked[0] and asked[1].startswith("(Optional) Add a folder")
    assert cli.choose_archives(False, []) == [OPEN_RUNS]
    assert cli.choose_archives(False, [], open_runs=False) == []
    assert cli.parse_archive("open-runs") == OPEN_RUNS


# Vectors and answers that come with the tool ------------------------------------------------------------------


def vector_file(path, model: str, texts: dict[str, float]):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE vectors (model TEXT, dimensions INTEGER, hash TEXT, vector BLOB)")
    for text, value in texts.items():
        vector = array("f", [value] * embeddings.DIMENSIONS).tobytes()
        db.execute(
            "INSERT INTO vectors VALUES (?, ?, ?, ?)",
            (model, embeddings.DIMENSIONS, embeddings.text_hash(text), vector),
        )
    db.commit()
    db.close()


def test_vectors_that_come_with_the_tool_are_read_after_your_own(tmp_path, monkeypatch):
    shipped = tmp_path / "vectors.sqlite"
    vector_file(shipped, "voyage-4", {"public prompt": 0.5, "shared prompt": 0.25})
    monkeypatch.setattr(embeddings, "_shipped_cache", lambda: sqlite3.connect(shipped))
    embeddings.remember("voyage-4", {"shared prompt": [1.0] * embeddings.DIMENSIONS})
    found = embeddings.cached("voyage-4", ["public prompt", "shared prompt", "new prompt"])
    assert found["public prompt"][0] == 0.5 and found["shared prompt"][0] == 1.0 and "new prompt" not in found
    assert embeddings.in_cache("voyage-4", ["public prompt", "new prompt"]) == {"public prompt"}
    assert embeddings.cached("voyage-4-lite", ["public prompt"]) == {}  # another Voyage model's vectors differ


def test_every_open_runs_prompt_has_its_vectors(open_runs):
    texts = sorted({features.text("prompt", features.TaskInfo.of(t)) for t in archives.read_archive(OPEN_RUNS)})
    for model in embeddings.MODELS:
        assert embeddings.in_cache(model, texts) == set(texts), model


def task(n: int, prompt: str, archive: str = "claude-code", started: float = 1.79e9, model: str = "m1") -> Task:
    return Task(
        id=f"{archive}:{n}",
        archive=archive,
        agent="claude-code",
        agent_version="",
        session="",
        project="",
        origin="program",
        started_at=started + n * 60,
        model=model,
        effort="default",
        prompt=prompt,
        context="",
        context_tokens=0,
        seconds=30.0,
        input_tokens=100,
        cache_write_tokens=0,
        cache_read_tokens=0,
        output_tokens=50,
        usd=0.01,
        calls=3,
        direct=False,
        stopped=False,
        idle=False,
    )


def test_analysis_reads_each_text_once(monkeypatch):
    sent = []

    def fake(model, system, text, shape):
        sent.extend(ids_in(text))
        return answer_for(ids_in(text))

    monkeypatch.setattr(llm, "_anthropic", fake)
    config = {"features": ["prompt", "llm"], "llm": HOW, "options": {"llm": ["kind", "description"]}}
    tasks = [task(1, "Fix the build"), task(2, "Add tests"), task(3, "Fix the build")]
    asks = pipeline.analysis_plan(tasks, config, {})
    assert len(asks) == 2 and sorted(ask.also for ask in asks) == [(), ("claude-code:1",)]
    found = llm.analyze(asks, HOW, FIELDS).answers
    assert sorted(sent) == sorted(ask.id for ask in asks)  # the shared text went once
    assert sorted(found) == sorted(t.id for t in tasks) and found["claude-code:1"]["kind"] == "feature"


def test_a_shared_text_never_shares_a_request_with_a_later_task_of_any_of_its_conversations():
    config = {"features": ["prompt", "llm"], "llm": HOW, "options": {"llm": ["kind"]}}
    a1 = replace(task(3, "Fix the build"), session="a")
    b1 = replace(task(1, "Fix the build"), session="b")
    b2 = replace(task(2, "Add tests", started=1.79e9), session="b")  # its context may show how b1 ended
    asks = pipeline.analysis_plan([b1, b2, a1], config, {})
    shared = next(ask for ask in asks if ask.also)
    assert shared.conversation in ("a", "b") and set((shared.conversation, *shared.also_in)) == {"a", "b"}
    assert all(len(batch) == 1 for batch in llm.batches(asks, 10))


def test_answers_that_come_with_the_tool_count_as_read(monkeypatch):
    public = task(1, "Implement a sampler", archive="open-runs")
    mine = task(2, "Implement a sampler")
    answer = answer_for(["x"])["tasks"][0]
    shipped = {"by": "anthropic:claude-opus-5-5", "fields": FIELDS, "answers": {llm.read_of(public.prompt, ""): answer}}
    monkeypatch.setattr(llm, "_shipped_file", lambda: {"sets": [shipped]})
    config = {"features": ["prompt", "llm"], "llm": HOW, "options": {"llm": ["kind", "description"]}}
    labels = store.llm_answers(config, tasks=[public, mine])
    assert labels[public.id]["kind"] == answer["kind"] and mine.id not in labels  # only public runs get them
    assert [ask.id for ask in pipeline.analysis_plan([public, mine], config, labels)] == [mine.id]
    other = {**config, "llm": {"provider": "anthropic", "model": "claude-haiku-5-5"}}
    assert public.id not in store.llm_answers(other, tasks=[public])  # another AI model reads it itself


# Learning from public runs -----------------------------------------------------------------------------------


def mixed(own: int, public: int) -> list[Task]:
    tasks = [task(n, f"public {n}", archive="open-runs", started=1.70e9) for n in range(public)]
    tasks += [task(n, f"mine {n}", started=1.79e9) for n in range(own)]
    return tasks


def test_public_runs_are_never_tested_when_you_have_your_own():
    tasks = mixed(own=100, public=300)
    windows = M._test_windows(tasks, M.settings_for({}))
    assert windows and all(tasks[i].archive == "claude-code" for _, part in windows for i in part)


def test_with_only_public_runs_the_public_runs_are_tested():
    tasks = mixed(own=0, public=300)
    assert M._test_windows(tasks, M.settings_for({}))


@pytest.mark.parametrize("own", [0, 100])
def test_trees_stop_on_your_own_newest_tasks(own, monkeypatch):
    seen = {}

    def grow(model, key, matrix, models, z, weight, mask, fit_rows, valid_rows, efforts):
        seen["valid"] = valid_rows

    monkeypatch.setattr(M, "_grow_part", grow)
    monkeypatch.setattr(M, "_grow_direct", lambda *args: None)
    monkeypatch.setattr(M, "_fit_width_scale", lambda *args: None)
    monkeypatch.setattr(M, "_refit", lambda *args: None)
    tasks = mixed(own=own, public=300)
    rows = [M.Features(t.model, t.effort, t.agent, t.origin, len(t.prompt), {}, {}, t.started_at) for t in tasks]
    M.fit(tasks, rows, M.settings_for({}))
    kinds = {tasks[i].archive for i in seen["valid"]}
    assert kinds == ({"claude-code"} if own else {"open-runs"})
