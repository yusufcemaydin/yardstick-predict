"""The line setup and refresh say before they go ahead: how long the rest takes, and the room it needs."""

from __future__ import annotations

import json
import os
import re

import pytest
from builders import ClaudeSession, flat, history
from conftest import KEY
from test_cli import QuickModel, fake_analysis, quick_train
from test_model import synthetic

from yardstick_predict import estimate, llm, model_types, ui
from yardstick_predict import model as M
from yardstick_predict.cli import main
from yardstick_predict.pipeline import learnable

START = 1_780_000_000.0
ANSI = re.compile(r"\033\[[0-9;]*[mK]")


def test_a_stretch_of_time_is_said_in_rough_words():
    assert estimate.span(10, 50) == "under a minute"
    assert estimate.span(40, 80) == "about a minute"
    assert estimate.span(250, 540) == "about 4 to 9 minutes"
    assert estimate.span(520, 560) == "about 9 minutes"
    assert estimate.span(2 * 3600, 3.5 * 3600) == "about 2 to 3.5 hours"
    assert [estimate.rough(s) for s in (59, 60, 150, 2 * 3600)] == ["under a minute", "1 min", "2 min", "2 h"]


def test_the_line_says_a_range_and_what_its_made_of():
    six = estimate.Estimate({"LLM analysis": 120, "embedding": 180, "training": 60}, disk=120e6, memory=1e9)
    assert estimate.line(six) == (
        "It takes about 4 to 9 minutes: LLM analysis 2 min, embedding 3 min, training 1 min. "
        "The dataset takes about 120 MB on disk."
    )
    # A range from 70% to 150% of the estimate.
    assert estimate.RANGE == (0.7, 1.5)
    alone = estimate.Estimate({"training": 100}, disk=52e6, memory=1e9)
    assert estimate.line(alone) == "Training takes about 1 to 2 minutes. The dataset takes about 52 MB on disk."
    # Under a minute in all, the steps are named without times of their own.
    short = estimate.Estimate({"embedding": 5, "training": 20}, disk=900_000, memory=1e9)
    assert estimate.line(short) == "Embedding and training take under a minute. The dataset takes about 900 KB on disk."
    every = estimate.Estimate({"LLM analysis": 10, "embedding": 5, "training": 20}, disk=900_000, memory=1e9)
    assert estimate.line(every).startswith("LLM analysis, embedding and training take under a minute.")


def test_each_step_takes_what_its_constants_say():
    tasks = synthetic(400)
    config = {"features": ["prompt", "llm"], "llm": {"provider": "claude", "model": "claude-opus-5-5"}}
    found = estimate.of(tasks, config, tokens=400_000, vectors=400, asks=400)
    assert list(found.steps) == ["LLM analysis", "embedding", "training"]
    assert found.steps["LLM analysis"] == 400 * estimate.LLM_SECONDS_PER_TASK["claude"]
    # LLM analysis describes each task, and the descriptions are embedded too.
    tokens = 400_000 + 400 * llm.DESCRIPTION_TOKENS
    assert found.steps["embedding"] == tokens / estimate.EMBED_TOKENS_A_SECOND
    # Training fits a model for each test window, then the final one. With LLM analysis, each is fitted twice.
    sizes = M.fit_sizes(tasks, {})
    once = sum(estimate.TRAIN_SECONDS_PER_MODEL + estimate.TRAIN_SECONDS_PER_TASK * size for size in sizes)
    assert len(sizes) > 1 and found.steps["training"] == pytest.approx(2 * once)
    texts = sum(len(t.prompt) + len(t.context) + len(t.skills) + len(t.prior_skills) for t in tasks)
    assert found.disk == pytest.approx(
        estimate.DISK_PER_TASK * 400
        + estimate.DISK_PER_TEXT_BYTE * texts
        + estimate.DISK_PER_VECTOR * 800  # the prompts' and the descriptions'
        + estimate.DISK_PER_ANSWER * 400
    )
    assert found.memory == estimate.MEMORY_BASE + estimate.MEMORY_PER_TASK * 400 + estimate.MEMORY_PER_TEXT_BYTE * texts

    # A subscription is slower than an API key.
    paid = {name: estimate.of(tasks, {**config, "llm": {"provider": name}}, 0, 0, 400) for name in llm.PROVIDERS}
    seconds = {name: found.steps["LLM analysis"] for name, found in paid.items()}
    assert seconds["anthropic"] == seconds["openai"] < min(seconds["claude"], seconds["codex"])

    # Without LLM analysis, nothing to embed and only the newest 100 tasks to learn from, training is all.
    plain = estimate.of(tasks, {"features": ["prompt"], "max_tasks": 100}, tokens=0, vectors=0, asks=0)
    newest = learnable(tasks, 100)
    assert list(plain.steps) == ["training"]
    assert plain.steps["training"] == pytest.approx(
        sum(estimate.TRAIN_SECONDS_PER_MODEL + estimate.TRAIN_SECONDS_PER_TASK * s for s in M.fit_sizes(newest, {}))
    )
    assert plain.memory == estimate.MEMORY_BASE + estimate.MEMORY_PER_TASK * 100 + estimate.MEMORY_PER_TEXT_BYTE * texts


def test_a_refresh_adds_the_new_tasks_to_the_dataset_it_has():
    tasks = synthetic(200)
    config = {"features": ["prompt"]}
    first = estimate.of(tasks[:190], config, tokens=0, vectors=190, asks=0)
    later = estimate.of(tasks, config, tokens=0, vectors=10, asks=0, new_tasks=tasks[190:], dataset=5_000_000)
    new = estimate.of(tasks[190:], config, tokens=0, vectors=10, asks=0)
    assert later.disk == pytest.approx(5_000_000 + new.disk) and later.disk > first.disk


def test_memory_is_mentioned_only_when_it_may_pass_a_quarter_of_the_computers():
    found = estimate.Estimate({"training": 60}, disk=0, memory=2e9)  # up to 3 GB, at the top of the range
    assert estimate.memory_note(found, int(16e9)) is None
    assert estimate.memory_note(found, int(12e9)) is None  # a quarter is 3 GB: it can't pass that
    assert estimate.memory_note(found, int(8e9)) == (
        "Training may use up to about 3.0 GB of memory, more than a quarter of this computer's."
    )
    assert estimate.memory_note(found, None) is None  # when the computer doesn't say, never


def test_the_computers_memory_comes_from_the_operating_system(monkeypatch):
    found = estimate.total_memory()
    assert found is None or (isinstance(found, int) and found > 0)

    def unknown(name):
        raise ValueError(name)

    monkeypatch.setattr(estimate.sys, "platform", "linux")
    monkeypatch.setattr(estimate.os, "sysconf", unknown, raising=False)  # Windows' Python has no sysconf
    assert estimate.total_memory() is None
    monkeypatch.setattr(estimate.sys, "platform", "win32")
    if os.name == "nt":
        assert estimate.total_memory() > 0  # Windows says
    else:
        assert estimate.total_memory() is None  # no Windows to ask here


# What the commands say ----------------------------------------------------------------------------------------


@pytest.fixture
def records(tmp_path):
    """40 Claude Code tasks on one AI model: too few to test, so training fits only the final model."""
    root = tmp_path / "records"
    history(root, tasks=40, start=START)
    return root


@pytest.fixture
def ready(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    return voyage


def said(capsys) -> str:
    return flat(capsys.readouterr().out)


def test_setup_and_refresh_say_once_how_long_they_take_before_going_ahead(capsys, monkeypatch, records, ready):
    monkeypatch.setattr(estimate, "total_memory", lambda: None)
    flags = ["--features", "", "--archive", f"claude-code={records}"]
    assert main(["create", *flags]) == 2  # a program sees it, then must agree
    out = said(capsys)
    assert out.count("The dataset takes about") == 1
    assert out.index("Embedding and training take under a minute.") < out.index("Add --yes to go ahead.")

    assert main(["create", "--yes", *flags]) == 0
    out = said(capsys)
    assert out.count("Embedding and training take under a minute. The dataset takes about") == 1
    assert out.index("The dataset takes about") < out.index("Embedded")
    assert "memory" not in out

    session = ClaudeSession(records, "more", START + 90 * 86400)
    session.prompt("one more task")
    session.reply(5)
    session.save(mtime=START)
    monkeypatch.setattr(estimate, "total_memory", lambda: 1_000_000_000)  # a computer with very little memory
    assert main(["refresh", "--yes"]) == 0
    out = said(capsys)
    assert out.count("Embedding and training take under a minute. The dataset takes about") == 1
    assert out.count("Training may use up to about") == 1 and "more than a quarter of this computer's" in out


def test_the_line_names_llm_analysis_by_the_way_its_paid(capsys, monkeypatch, records, ready):
    monkeypatch.setattr(llm, "analyze", fake_analysis([]))
    flags = ["--features", "llm", "--llm-provider", "claude", "--archive", f"claude-code={records}"]
    assert main(["create", "--yes", *flags]) == 0
    # 40 tasks through a subscription, at 2 seconds a task, with a little embedding and training.
    expected = "It takes about 1 to 2 minutes: LLM analysis 1 min, embedding under a minute, training under a minute."
    assert expected in said(capsys)


def test_every_long_step_shows_a_bar_in_a_terminal(capsys, monkeypatch, records, ready):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(ui, "_bar", None)
    monkeypatch.setattr(llm, "analyze", fake_analysis([]))
    flags = ["--features", "llm", "--llm-provider", "claude", "--archive", f"claude-code={records}"]
    assert main(["create", "--yes", *flags]) == 0
    out = ANSI.sub("", said(capsys))
    assert re.search(r" (\d+) KB / \1 KB", out)  # reading, by the bytes of the records
    assert " 40 / 40 tasks" in out  # LLM analysis
    assert re.search(r" (\d+) / \1 texts", out)  # embedding
    assert " 2 / 2 steps" in out  # training: the final model, with LLM analysis and without it

    # A model type from a plugin says nothing about how far it has got, so it keeps the working line.
    load = lambda folder: QuickModel(json.loads((folder / "model.json").read_text()))  # noqa: E731
    monkeypatch.setitem(model_types._TYPES, "quick", (quick_train, load))
    flags = ["--model-type", "quick", "--features", "", "--archive", f"claude-code={records}"]
    assert main(["create", "other", "--yes", *flags]) == 0
    out = ANSI.sub("", said(capsys))
    assert "Training, then checking its forecasts" in out and " steps" not in out


def test_text_outside_english_counts_by_its_bytes():
    from dataclasses import replace

    task = replace(synthetic(1)[0], context="", skills="", prior_skills="")
    sizes = [estimate.text_bytes([replace(task, prompt=prompt)]) for prompt in ("ab", "日本", "\ud83d")]
    assert sizes == [2, 6, 1]  # half an emoji, as records can hold, counts and never fails
