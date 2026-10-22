from __future__ import annotations

import json
import os

import pytest
from builders import ClaudeSession, CodexSession, usage

from yardstick_predict import prices
from yardstick_predict.archives import read_archive

START = 1_780_000_000.0  # early June 2026
LATER = START + 30 * 86400


def claude_archive(path):
    return {"name": "claude-code", "format": "claude-code", "path": str(path)}


def codex_archive(path):
    return {"name": "codex", "format": "codex", "path": str(path)}


def read_claude(path, now=LATER):
    return {t.prompt: t for t in read_archive(claude_archive(path), now=now)}


def test_a_task_counts_each_reply_once_and_adds_helpers(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Add a login page")
    tool = session.reply(
        5, tool="Bash", calls_usage=usage(new_input=100, write=1000, read=20_000, output=300), repeat=3
    )
    session.tool_result(tool, 2, line["promptId"])
    session.reply(4, calls_usage=usage(new_input=50, read=21_000, output=100))
    session.helper(line["promptId"], calls=2, call_usage=usage(new_input=10, read=5_000, output=50))
    session.save(mtime=START)

    task = read_claude(tmp_path)["Add a login page"]
    assert task.calls == 4
    assert task.input_tokens == 100 + 50 + 2 * 10
    assert task.cache_write_tokens == 1000
    assert task.cache_read_tokens == 20_000 + 21_000 + 2 * 5_000
    assert task.output_tokens == 300 + 100 + 2 * 50
    assert task.seconds == pytest.approx(11)
    assert not task.direct
    assert task.model == "claude-opus-5-5"
    assert task.effort == "high"
    assert task.origin == "person"
    assert task.context_tokens == 100 + 1000 + 20_000


def test_cost_is_priced_per_call_and_per_model(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Price this")
    session.reply(1, calls_usage=usage(new_input=1_000_000, write=0, read=0, output=0))
    session.helper(
        line["promptId"], model="claude-haiku-5-5", calls=1, call_usage=usage(new_input=1_000_000, read=0, output=0)
    )
    session.save(mtime=START)

    task = read_claude(tmp_path)["Price this"]
    opus = prices.price("claude-opus-5-5")["input"]
    haiku_long = prices.price("claude-haiku-5-5")["long_context"]["input"]
    assert task.usd == pytest.approx(opus + haiku_long)


def test_automatic_turns_are_not_tasks_but_end_the_one_before(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Run the tests")
    session.reply(10, tool="Bash")
    session.prompt("<task-notification>Background job finished</task-notification>", origin="task_notification")
    session.reply(500, text="The tests passed.")
    session.prompt("<bash-input>ls</bash-input>", origin=None)
    session.prompt("Thanks, now ship it", origin="sdk")
    session.reply(3)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert set(tasks) == {"Run the tests", "Thanks, now ship it"}
    assert tasks["Run the tests"].seconds == pytest.approx(10)
    assert tasks["Thanks, now ship it"].origin == "program"
    assert "The tests passed." in tasks["Thanks, now ship it"].context


def test_waiting_for_the_person_is_not_agent_time(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Pick a database for me")
    tool = session.reply(5, tool="AskUserQuestion")
    session.tool_result(tool, 3000, line["promptId"])
    session.reply(6)
    session.save(mtime=START)

    task = read_claude(tmp_path)["Pick a database for me"]
    assert task.seconds == pytest.approx(11)
    assert not task.idle


def test_late_bookkeeping_lines_dont_make_a_task_look_left_waiting(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Quick question")
    session.reply(4)
    session.bookkeeping(7200)
    session.prompt("Another one")
    session.reply(3)
    session.save(mtime=START)

    task = read_claude(tmp_path)["Quick question"]
    assert task.seconds == pytest.approx(4)
    assert task.direct
    assert not task.idle


def test_a_long_silence_while_working_marks_the_task_left_waiting(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Long job")
    tool = session.reply(5, tool="Bash")
    session.tool_result(tool, 4000, line["promptId"])
    session.reply(5)
    session.save(mtime=START)
    assert read_claude(tmp_path)["Long job"].idle


def test_stopped_tasks_are_marked(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Do something big")
    session.reply(5, tool="Bash")
    session.interrupt()
    session.prompt("Never mind")
    session.reply(2)
    session.save(mtime=START)
    tasks = read_claude(tmp_path)
    assert tasks["Do something big"].stopped
    assert not tasks["Never mind"].stopped


def test_a_task_copied_into_a_resumed_session_counts_once(tmp_path):
    original = ClaudeSession(tmp_path, "s1", START)
    line = original.prompt("Original task", uuid="same-uuid")
    original.reply(5)
    original.save(mtime=START)
    resumed = ClaudeSession(tmp_path, "s2", START)
    resumed.prompt("Original task", uuid="same-uuid", prompt_id=line["promptId"])
    resumed.reply(5)
    resumed.prompt("New task in the resumed session")
    resumed.reply(2)
    resumed.save(mtime=START + 100)

    tasks = read_archive(claude_archive(tmp_path), now=LATER)
    assert sorted(t.prompt for t in tasks) == ["New task in the resumed session", "Original task"]


def test_the_last_task_of_a_file_still_being_written_waits(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("First")
    session.reply(2)
    session.prompt("Maybe still running")
    session.reply(2)
    session.save(mtime=START)
    assert set(read_claude(tmp_path, now=START + 60)) == {"First"}
    assert set(read_claude(tmp_path, now=LATER)) == {"First", "Maybe still running"}


def test_codex_turns_become_tasks(tmp_path):
    session = CodexSession(tmp_path, "c1", START)
    session.turn("t1", "Fix the failing test", calls=[(1000, 9000, 200), (500, 11_000, 100)], seconds=40)
    session.turn("t2", "Explain the error", calls=[(300, 20_000, 80)], seconds=5, tools=False)
    session.turn("t3", "Abandoned", calls=[(10, 10, 10)], seconds=1, aborted=True)
    session.save(mtime=START)

    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    fix = tasks["Fix the failing test"]
    assert fix.seconds == pytest.approx(40)
    assert fix.input_tokens == 1500
    assert fix.cache_read_tokens == 20_000
    assert fix.output_tokens == 300
    assert fix.calls == 2
    assert fix.model == "gpt-6.1-sol" and fix.effort == "xhigh"
    assert fix.origin == "program"
    assert not fix.direct
    assert "AGENTS.md" not in fix.prompt
    assert tasks["Explain the error"].direct
    assert "Fixed." in tasks["Explain the error"].context
    assert tasks["Abandoned"].stopped
    expected = prices.call_cost("gpt-6.1-sol", 1000, 0, 0, 9000, 200) + prices.call_cost(
        "gpt-6.1-sol", 500, 0, 0, 11_000, 100
    )
    assert fix.usd == pytest.approx(expected)


def test_prompts_wrapped_in_tags_are_still_prompts(tmp_path):
    claude = ClaudeSession(tmp_path / "claude", "s1", START)
    claude.prompt("<task>Fix the parser</task>", origin=None)
    claude.reply(3)
    claude.prompt("<task-notification>A job finished</task-notification>", origin=None)
    claude.reply(3)
    claude.prompt("<html>Make this page faster</html>", origin="sdk")
    claude.reply(3)
    claude.save(mtime=START)
    assert set(read_claude(tmp_path / "claude")) == {
        "<task>Fix the parser</task>",
        "<html>Make this page faster</html>",
    }

    codex = CodexSession(tmp_path / "codex", "c1", START)
    codex.add("event_msg", {"type": "task_started", "turn_id": "t1", "started_at": int(START)})
    codex.add(
        "response_item",
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "<environment_context>cwd=/work</environment_context>"}],
        },
    )
    codex.add("turn_context", {"model": "gpt-6.1-sol", "effort": "xhigh", "turn_id": "t1"})
    codex.add(
        "response_item",
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "<task>Fix the flaky test</task>"}],
        },
    )
    usage_line = {
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 10,
        "reasoning_output_tokens": 0,
        "total_tokens": 110,
    }
    codex.add(
        "event_msg", {"type": "token_count", "info": {"total_token_usage": usage_line, "last_token_usage": usage_line}}
    )
    codex.add("event_msg", {"type": "task_complete", "turn_id": "t1", "duration_ms": 3000})
    codex.save(mtime=START)
    tasks = read_archive(codex_archive(tmp_path / "codex"), now=LATER)
    assert [t.prompt for t in tasks] == ["<task>Fix the flaky test</task>"]


def test_codex_helper_threads_add_to_the_parent_task(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    helper = CodexSession(tmp_path, "helper", START + 10, parent="parent")
    helper.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5)
    helper.save(mtime=START)

    tasks = read_archive(codex_archive(tmp_path), now=LATER)
    assert [t.prompt for t in tasks] == ["Split the work"]
    assert tasks[0].input_tokens == 1200
    assert tasks[0].calls == 2


def test_a_codex_helper_whose_first_reply_comes_after_its_parent_finished_still_counts(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Start a helper and finish", calls=[(1000, 0, 100)], seconds=10)
    parent.save(mtime=START)
    # The helper starts during the parent's turn, but its first reply takes a minute.
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "Slow helper part", calls=[(200, 0, 20)], seconds=60)
    helper.save(mtime=START)

    tasks = read_archive(codex_archive(tmp_path), now=LATER)
    assert [t.prompt for t in tasks] == ["Start a helper and finish"]
    assert tasks[0].calls == 2
    assert tasks[0].input_tokens == 1200


def test_a_reused_codex_helper_charges_each_turn_to_the_task_that_sent_it(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=100)
    parent.clock = START + 200
    parent.turn("t2", "Ask the same helper for more", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    helper = CodexSession(tmp_path, "helper", START + 10, parent="parent")
    helper.turn("h1", "First job", calls=[(200, 0, 20)], seconds=5)
    helper.clock = START + 210
    helper.turn("h2", "Second job", calls=[(300, 0, 30), (300, 0, 30)], seconds=5)
    helper.save(mtime=START)

    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    first, second = tasks["Start a helper"], tasks["Ask the same helper for more"]
    assert (first.calls, first.input_tokens) == (2, 1200)
    assert (second.calls, second.input_tokens) == (3, 1600)


def test_the_context_keeps_the_conversation_in_order(tmp_path):
    claude = ClaudeSession(tmp_path / "claude", "s1", START)
    claude.prompt("First question")
    claude.reply(2, text="First answer")
    claude.prompt("Second question")
    claude.reply(2)
    claude.save(mtime=START)
    task = read_claude(tmp_path / "claude")["Second question"]
    assert task.context == "User: First question\n\nAssistant: First answer"

    codex = CodexSession(tmp_path / "codex", "c1", START)
    codex.turn("t1", "First question", calls=[(100, 0, 10)], seconds=2)
    codex.turn("t2", "Second question", calls=[(100, 0, 10)], seconds=2)
    codex.save(mtime=START)
    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path / "codex"), now=LATER)}
    assert tasks["Second question"].context == "User: First question\n\nAssistant: Fixed."


def test_broken_lines_are_skipped(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Still readable")
    session.reply(2)
    path = session.save(mtime=START)
    with open(path, "a") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps([1, 2, 3]) + "\n")
    os.utime(path, (START, START))
    assert set(read_claude(tmp_path)) == {"Still readable"}


def test_prices():
    assert prices.normalize("claude-opus-4-5-20251101") == "claude-opus-4-5"
    assert prices.call_cost("unknown-model", 1, 0, 0, 0, 1) is None
    assert prices.call_cost("claude-opus-5-5", 1_000_000, 0, 0, 0, 0) == pytest.approx(4.0)
    assert prices.call_cost("claude-opus-5-5", 0, 0, 0, 1_000_000, 0, fast=True) == pytest.approx(0.4)
    assert prices.call_cost("gpt-6.1-sol", 300_000, 0, 0, 0, 0) == pytest.approx(1.2)
