from __future__ import annotations

import json
import os
import shutil

import pytest
from builders import ClaudeSession, CodexSession, usage

from yardstick_predict import prices
from yardstick_predict.archives import line_skills, prompt_skills, read_archive, read_records, record_files

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
    session.prompt("Push the branch")
    session.reply(10)
    session.prompt("<ci-monitor-event>The checks failed.</ci-monitor-event>", origin=None)
    session.reply(500, text="A check failed.")
    session.prompt("<bash-input>ls</bash-input>", origin=None)  # an older record's ! command, with no reply
    session.prompt("Thanks, now ship it", origin="sdk")
    session.reply(3)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert set(tasks) == {"Push the branch", "Thanks, now ship it"}
    assert (tasks["Push the branch"].seconds, tasks["Push the branch"].calls) == (pytest.approx(10), 1)
    assert tasks["Thanks, now ship it"].origin == "program"
    assert "A check failed." in tasks["Thanks, now ship it"].context


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


@pytest.mark.parametrize("kind", ["helper", "progress", "hook"])
@pytest.mark.parametrize("tool, source", [("AskUserQuestion", "config"), ("Bash", "user_temporary")])
def test_a_line_written_while_waiting_for_the_person_doesnt_make_the_task_look_left_waiting(
    tmp_path, kind, tool, source
):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Pick a database for me")
    tool_id = session.reply(5, tool=tool)
    session.side_line(1900, kind)
    session.tool_result(tool_id, 200, line["promptId"], source=source)
    session.reply(6)
    session.save(mtime=START)

    task = read_claude(tmp_path)["Pick a database for me"]
    assert task.seconds == pytest.approx(11)
    assert not task.idle


def test_a_long_silence_after_waiting_for_the_person_still_marks_the_task_left_waiting(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Pick a database, then set it up")
    tool = session.reply(5, tool="AskUserQuestion")
    session.side_line(1900, "progress")
    session.tool_result(tool, 200, line["promptId"])
    tool = session.reply(6, tool="Bash")
    session.tool_result(tool, 2400, line["promptId"])
    session.reply(5)
    session.save(mtime=START)
    assert read_claude(tmp_path)["Pick a database, then set it up"].idle


@pytest.mark.parametrize("kind", ["helper", "progress"])
def test_a_helper_working_for_a_long_time_doesnt_make_the_task_look_left_waiting(tmp_path, kind):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Compare these three libraries")
    tool = session.reply(5, tool="Task")
    for _ in range(4):
        session.side_line(600, kind)
    session.tool_result(tool, 600, line["promptId"])
    session.reply(5)
    session.save(mtime=START)

    task = read_claude(tmp_path)["Compare these three libraries"]
    assert task.seconds == pytest.approx(3010)
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


@pytest.mark.parametrize("error", ["rate_limit", "server_error", "authentication_failed"])
def test_a_task_that_ends_on_an_api_error_is_stopped(tmp_path, error):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Refactor the parser")
    tool = session.reply(5, tool="Bash")
    session.tool_result(tool, 2, line["promptId"])
    session.synthetic(3, error=error)
    session.prompt("Carry on")
    session.reply(2)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert tasks["Refactor the parser"].stopped
    assert not tasks["Carry on"].stopped


def test_an_api_error_followed_by_a_retry_that_worked_doesnt_stop_the_task(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Refactor the parser")
    tool = session.reply(5, tool="Bash")
    session.tool_result(tool, 2, line["promptId"])
    session.synthetic(3, error="server_error")
    session.reply(4)  # Claude Code retried, and the agent carried on
    session.save(mtime=START)

    task = read_claude(tmp_path)["Refactor the parser"]
    assert not task.stopped
    assert task.calls == 2


def test_an_api_error_followed_only_by_a_placeholder_reply_still_stops_the_task(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Refactor the parser")
    session.reply(5)
    session.synthetic(3, error="rate_limit")
    session.synthetic(60, error=None)  # "No response requested." No model call came after the error.
    session.save(mtime=START)
    assert read_claude(tmp_path)["Refactor the parser"].stopped


def test_an_api_error_in_a_helper_doesnt_stop_the_task(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Compare these libraries")
    session.reply(5)
    session.synthetic(1, error="server_error", sidechain=True)
    session.save(mtime=START)
    assert not read_claude(tmp_path)["Compare these libraries"].stopped


def test_a_synthetic_reply_is_never_a_model_call_and_never_sets_the_model(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Price this")
    session.synthetic(1, error="server_error")
    session.synthetic(1, error=None)
    session.reply(5, model="claude-sonnet-5", calls_usage=usage(new_input=100, read=0, output=10))
    session.prompt("Only an error came back")
    session.synthetic(1, error="rate_limit")
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert set(tasks) == {"Price this"}
    task = tasks["Price this"]
    assert (task.calls, task.model, task.input_tokens, task.output_tokens) == (1, "claude-sonnet-5", 100, 10)
    assert task.usd == pytest.approx(prices.call_cost("claude-sonnet-5", 100, 0, 0, 0, 10))


def test_a_task_copied_into_a_resumed_session_counts_once(tmp_path):
    original = ClaudeSession(tmp_path, "s1", START)
    original.prompt("Original task")
    original.reply(5)
    original.save(mtime=START)
    # A resumed session copies the lines before it into its own record. The copies keep their session's id.
    resumed = ClaudeSession(tmp_path, "s2", START + 100)
    resumed.lines = [dict(line) for line in original.lines]
    resumed.prompt("New task in the resumed session")
    resumed.reply(2)
    resumed.save(mtime=START + 200)

    tasks = read_archive(claude_archive(tmp_path), now=LATER)
    assert sorted((t.prompt, t.calls, t.source) for t in tasks) == [
        ("New task in the resumed session", 1, str(resumed.path)),
        ("Original task", 1, str(original.path)),
    ]


def test_a_copy_that_lost_a_prompt_doesnt_swallow_the_next_task_or_count_it_twice(tmp_path):
    original = ClaudeSession(tmp_path, "s1", START)
    original.prompt("Task X", uuid="ux")
    original.reply(5)
    original.prompt("Task Y", uuid="uy")
    original.reply(50, tool="Bash")
    original.reply(50)
    original.save(mtime=START)
    # A copy of the session that's missing Y's prompt would put Y's replies in X.
    copy = ClaudeSession(tmp_path, "s1-copy", START)
    copy.lines = [dict(line) for line in original.lines if line["uuid"] != "uy"]
    copy.save(mtime=START + 10)

    tasks = read_claude(tmp_path)
    assert {prompt: (t.calls, t.seconds) for prompt, t in tasks.items()} == {"Task X": (1, 5), "Task Y": (2, 100)}


def test_a_record_that_writes_its_history_again_counts_each_task_once(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("First")
    session.reply(5)
    session.prompt("Second")
    session.reply(5)
    session.lines += [dict(line) for line in session.lines]  # Claude Code wrote the conversation out again
    session.prompt("Third")
    session.reply(2)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert {prompt: (t.calls, t.seconds) for prompt, t in tasks.items()} == {
        "First": (1, 5),
        "Second": (1, 5),
        "Third": (1, 2),
    }


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


def test_a_codex_field_of_another_type_is_passed_over_and_the_rest_is_read(tmp_path):
    session = CodexSession(tmp_path, "c1", START)
    session.turn("t1", "Fix the failing test", calls=[(1000, 9000, 200)], seconds=40)
    session.turn("t2", "Explain the error", calls=[(300, 20_000, 80)], seconds=5, model={"name": "odd"}, effort=7)
    session.save(mtime=START)
    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    assert tasks["Fix the failing test"].model == "gpt-6.1-sol"
    # The odd fields are passed over, as missing ones are: the turn keeps the model and effort it had.
    assert tasks["Explain the error"].model == "gpt-6.1-sol" and tasks["Explain the error"].effort == "xhigh"


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


def test_a_codex_helper_spawned_by_a_helper_counts_toward_the_main_task(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    helper = CodexSession(tmp_path, "helper", START + 10, parent="parent")
    helper.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5)
    helper.save(mtime=START)
    nested = CodexSession(tmp_path, "nested", START + 12, parent="helper")
    nested.turn("n1", "Nested part", calls=[(50, 0, 5)], seconds=2)
    nested.save(mtime=START)

    tasks = read_archive(codex_archive(tmp_path), now=LATER)
    assert [t.prompt for t in tasks] == ["Split the work"]
    assert tasks[0].calls == 3
    assert tasks[0].input_tokens == 1250


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


def two_prompts(tmp_path, roots=True):
    """A main thread with two prompts, 100 seconds apart. Newer versions of Codex name each turn's own root."""
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=10, root="t1" if roots else None)
    parent.clock = START + 100
    parent.turn("t2", "Something else", calls=[(1000, 0, 100)], seconds=10, root="t2" if roots else None)
    parent.save(mtime=START)


def first_and_second(tmp_path):
    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    first, second = tasks["Start a helper"], tasks["Something else"]
    return (first.calls, first.input_tokens), (second.calls, second.input_tokens)


def test_a_codex_helper_turn_that_starts_after_the_next_prompt_counts_toward_the_turn_codex_names(tmp_path):
    two_prompts(tmp_path)
    # The helper works on after the person's next prompt. Codex says its second turn still belongs to the first.
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "First part", calls=[(200, 0, 20)], seconds=5, root="t1")
    helper.clock = START + 150
    helper.turn("h2", "Second part", calls=[(300, 0, 30)], seconds=5, root="t1")
    helper.save(mtime=START)
    assert first_and_second(tmp_path) == ((3, 1500), (1, 1000))


@pytest.mark.parametrize("nested_root", ["t1", "h1"])
def test_a_nested_codex_helper_spawned_after_the_next_prompt_counts_toward_the_turn_that_started_the_chain(
    tmp_path, nested_root
):
    two_prompts(tmp_path)
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "Long part", calls=[(200, 0, 20)], seconds=300, root="t1")
    helper.save(mtime=START)
    # The helper spawns its own helper after the person's next prompt. Its root is the main thread's
    # turn, or the helper turn that spawned it, which leads up to the same main thread's turn.
    nested = CodexSession(tmp_path, "nested", START + 150, parent="helper")
    nested.turn("n1", "Nested part", calls=[(50, 0, 5)], seconds=2, root=nested_root)
    nested.save(mtime=START)
    assert first_and_second(tmp_path) == ((3, 1250), (1, 1000))


def test_without_named_turns_a_nested_codex_helper_counts_toward_the_turn_its_parent_helper_worked_for(tmp_path):
    two_prompts(tmp_path, roots=False)
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "Long part", calls=[(200, 0, 20)], seconds=300)
    helper.save(mtime=START)
    # Older records name no turns. The nested helper's turn started during the helper's turn, which
    # started during the first prompt's turn.
    nested = CodexSession(tmp_path, "nested", START + 150, parent="helper")
    nested.turn("n1", "Nested part", calls=[(50, 0, 5)], seconds=2)
    nested.save(mtime=START)
    assert first_and_second(tmp_path) == ((3, 1250), (1, 1000))


def test_a_codex_helper_turn_naming_a_turn_missing_from_the_records_isnt_charged_to_another_task(tmp_path):
    parent = CodexSession(tmp_path, "parent", START + 100)
    parent.turn("t2", "Something else", calls=[(1000, 0, 100)], seconds=10, root="t2")
    parent.save(mtime=START)
    # The helper's turn belongs to an earlier turn whose record is gone, not to the one running now.
    helper = CodexSession(tmp_path, "helper", START + 150, parent="parent")
    helper.turn("h1", "Part of an earlier task", calls=[(200, 0, 20)], seconds=5, root="t1")
    helper.save(mtime=START)

    [task] = read_archive(codex_archive(tmp_path), now=LATER)
    assert (task.calls, task.input_tokens) == (1, 1000)


def one_helper(tmp_path, helper_calls=1):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    parent.save(mtime=START)
    helper = CodexSession(tmp_path, "helper", START + 10, parent="parent")
    helper.turn("h1", "Helper part", calls=[(200, 0, 20)] * helper_calls, seconds=5)
    helper.save(mtime=START)
    return helper


def test_a_codex_helper_file_copied_into_another_folder_counts_once(tmp_path):
    helper = one_helper(tmp_path)
    (tmp_path / "backup").mkdir()
    shutil.copy(helper.path, tmp_path / "backup" / helper.path.name)

    [task] = read_archive(codex_archive(tmp_path), now=LATER)
    assert (task.calls, task.input_tokens) == (2, 1200)


def test_a_codex_helper_copy_saved_before_it_finished_adds_nothing_to_the_whole_one(tmp_path):
    one_helper(tmp_path, helper_calls=3)
    early = CodexSession(tmp_path / "backup", "helper", START + 10, parent="parent")
    early.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5)
    early.lines.pop()  # saved while the helper was still working
    early.save(mtime=START)

    [task] = read_archive(codex_archive(tmp_path), now=LATER)
    assert (task.calls, task.input_tokens) == (4, 1600)


def test_a_codex_thread_resumed_into_a_new_file_keeps_the_turns_of_both_files(tmp_path):
    first = CodexSession(tmp_path, "parent", START)
    first.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=100)
    first.save(mtime=START)
    second = CodexSession(tmp_path, "parent", START + 200, name="parent-resumed")
    second.turn("t2", "Ask the same helper for more", calls=[(1000, 0, 100)], seconds=100)
    second.save(mtime=START + 300)
    # The helper's thread is spread over two files the same way, one job in each.
    helper = CodexSession(tmp_path, "helper", START + 10, parent="parent")
    helper.turn("h1", "First job", calls=[(200, 0, 20)], seconds=5)
    helper.save(mtime=START)
    resumed = CodexSession(tmp_path, "helper", START + 210, parent="parent", name="helper-resumed")
    resumed.turn("h2", "Second job", calls=[(300, 0, 30), (300, 0, 30)], seconds=5)
    resumed.save(mtime=START + 300)

    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    first_task, second_task = tasks["Start a helper"], tasks["Ask the same helper for more"]
    assert (first_task.calls, first_task.input_tokens) == (2, 1200)
    assert (second_task.calls, second_task.input_tokens) == (3, 1600)


def test_a_forked_codex_helper_doesnt_count_its_copy_of_the_parent_turn_again(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=10, root="t1")
    parent.save(mtime=START)
    # A helper forked from its parent starts with a copy of the parent's turn so far.
    helper = CodexSession(tmp_path, "helper", START, parent="parent")
    helper.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=0, root="t1")
    helper.lines.pop()
    helper.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5, root="t1")
    helper.save(mtime=START)

    [task] = read_archive(codex_archive(tmp_path), now=LATER)
    assert (task.calls, task.input_tokens) == (2, 1200)


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


def test_two_waits_for_the_person_at_once_are_taken_off_once(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Ask me two things, then do it")
    first = session.reply(5, tool="AskUserQuestion")
    second = session.reply(1, tool="AskUserQuestion")
    session.tool_result(first, 1200, line["promptId"])  # answered after 20 minutes
    session.tool_result(second, 600, line["promptId"])  # answered after 30 minutes, the two waits overlap
    session.reply(9)
    session.save(mtime=START)
    # 5 seconds before the questions and 9 after. Counting both waits in full left 0.5 seconds, the least.
    assert read_claude(tmp_path)["Ask me two things, then do it"].seconds == pytest.approx(14)


# Helpers ------------------------------------------------------------------------------------------------


def test_a_helper_counts_toward_the_prompt_that_started_it_not_the_prompt_its_tool_results_carry(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    first = session.prompt("Research the options")
    session.reply(5, calls_usage=usage(new_input=100, read=0, output=10))
    second = session.prompt("Meanwhile, fix the typo")
    session.reply(3, calls_usage=usage(new_input=100, read=0, output=10))
    # The helper works on after the next prompt. Its tool results carry that prompt's id, its instruction doesn't.
    session.helper(
        first["promptId"],
        calls=3,
        call_usage=usage(new_input=10, read=0, output=1),
        results_prompt_id=second["promptId"],
    )
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert (tasks["Research the options"].calls, tasks["Research the options"].input_tokens) == (4, 130)
    assert (tasks["Meanwhile, fix the typo"].calls, tasks["Meanwhile, fix the typo"].input_tokens) == (1, 100)


def test_a_helper_sent_more_work_from_a_later_prompt_counts_that_work_toward_the_later_task(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    first = session.prompt("Research the options")
    session.reply(5, calls_usage=usage(new_input=100, read=0, output=10))
    helper = session.helper(first["promptId"], calls=1, call_usage=usage(new_input=10, read=0, output=1))
    second = session.prompt("Ask the same helper to compare prices")
    session.reply(3, calls_usage=usage(new_input=100, read=0, output=10))
    session.helper_more(helper, second["promptId"], calls=2, call_usage=usage(new_input=20, read=0, output=1))
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert (tasks["Research the options"].calls, tasks["Research the options"].input_tokens) == (2, 110)
    assert tasks["Ask the same helper to compare prices"].input_tokens == 100 + 2 * 20


def test_a_workflow_helper_deep_under_subagents_counts_and_is_never_read_as_a_session(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Run the review workflow")
    session.reply(5)
    session.helper(line["promptId"], calls=2, folder="workflows/wf_1")
    session.save(mtime=START)
    journal = session.folder / "s1" / "subagents" / "workflows" / "wf_1" / "journal.jsonl"
    journal.write_text(json.dumps({"type": "result", "agentId": "a1", "key": "k", "result": "ok"}) + "\n")
    os.utime(journal, (START, START))

    tasks = read_archive(claude_archive(tmp_path), now=LATER)
    assert [(t.prompt, t.calls) for t in tasks] == [("Run the review workflow", 3)]


def test_the_reader_reads_only_the_record_files_it_is_given(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Research the options")
    session.reply(5)
    session.helper(line["promptId"], calls=2)
    session.save(mtime=START)
    archive = claude_archive(tmp_path)
    files = record_files(archive)
    [task] = read_records(archive, {str(session.path): files[str(session.path)]}, now=LATER).tasks
    assert task.calls == 1  # the helper's record wasn't listed, so it isn't read
    assert read_records(archive, files, now=LATER).tasks[0].calls == 3


# Who started a turn ------------------------------------------------------------------------------------


def test_a_scheduled_task_or_another_sessions_message_starts_a_program_task_of_its_own(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Quick question")
    session.reply(4, calls_usage=usage(new_input=100, read=0, output=10))
    session.clock += 3 * 3600
    session.prompt("Check the deploy", origin="scheduled", meta=True)  # Claude Code writes these as meta lines
    session.reply(5, tool="Bash", calls_usage=usage(new_input=5000, read=0, output=900))
    session.reply(5, calls_usage=usage(new_input=5000, read=0, output=900))
    session.prompt("A message from another session", origin="peer", meta=True)
    session.reply(2)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    question = tasks["Quick question"]
    assert (question.calls, question.input_tokens, question.seconds, question.direct, question.idle) == (
        1,
        100,
        pytest.approx(4),
        True,
        False,
    )
    assert (tasks["Check the deploy"].origin, tasks["Check the deploy"].calls) == ("program", 2)
    assert tasks["A message from another session"].origin == "program"


def test_a_notice_from_claude_code_itself_is_automatic_and_ends_the_task_before_it(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Send a note to the other session")
    session.reply(4)
    session.prompt("[Cross-session message] Your message was delivered.", origin="system", meta=True)
    session.reply(30, text="It was delivered.")
    session.save(mtime=START)

    [task] = read_archive(claude_archive(tmp_path), now=LATER)
    assert (task.prompt, task.calls, task.seconds) == ("Send a note to the other session", 1, pytest.approx(4))


def test_a_persons_shell_command_that_the_agent_works_on_is_a_task(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    command = "<bash-input>npm test</bash-input><bash-stdout>3 failed</bash-stdout><bash-stderr></bash-stderr>"
    session.prompt(command, origin="human")
    session.reply(5, tool="Bash")
    session.lines[-1]["message"]["content"][0]["id"] = "fix"
    session.tool_result("fix", 1, "p")
    session.reply(20)
    session.save(mtime=START)

    [task] = read_archive(claude_archive(tmp_path), now=LATER)
    assert (task.prompt, task.origin, task.calls) == (command, "person", 2)


def test_a_headless_run_and_a_scheduled_task_are_programs_in_older_records(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Summarize the diff", origin=None, entrypoint="sdk-cli")
    session.reply(3)
    session.prompt("<scheduled-task name='nightly'>Run the checks</scheduled-task>", origin=None)
    session.reply(3)
    session.prompt("Thanks", origin=None, entrypoint="claude-desktop")
    session.reply(3)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    assert {prompt[:9]: t.origin for prompt, t in tasks.items()} == {
        "Summarize": "program",
        "<schedule": "program",
        "Thanks": "person",
    }


# Background jobs ---------------------------------------------------------------------------------------


def background_task(tmp_path, tool="Bash"):
    """A task that starts a background job, says so and stops, and later gets the job's report."""
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Run the full test suite, then fix what fails")
    tool_id, job = session.background(line["promptId"], seconds=5, tool=tool, calls_usage=usage(100, 0, 1000, 10))
    session.reply(2, text="The tests are running.", calls_usage=usage(100, 0, 1000, 10))
    return session, line, tool_id, job


def test_a_background_jobs_report_counts_toward_the_task_that_started_it(tmp_path):
    session, line, tool_id, job = background_task(tmp_path)
    session.clock += 600  # the tests run for ten minutes in the background
    session.notification(tool_id, job)
    session.reply(20, calls_usage=usage(300, 0, 2000, 30))
    session.prompt("Thanks")
    session.reply(3)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    task = tasks["Run the full test suite, then fix what fails"]
    assert (task.calls, task.input_tokens, task.cache_read_tokens) == (3, 500, 4000)
    # The agent worked 7.1 seconds before the report and 20 after it. The ten minutes between aren't its work.
    assert task.seconds == pytest.approx(27.1)
    assert task.finished_at == pytest.approx(START + 627.1, abs=0.01)
    assert not task.stopped and not task.idle and not task.direct
    assert (tasks["Thanks"].calls, tasks["Thanks"].input_tokens) == (1, 100)


def test_a_report_on_a_job_reaches_a_turn_still_working_without_moving_its_calls(tmp_path):
    session, line, tool_id, job = background_task(tmp_path)
    session.prompt("Meanwhile, rename the module")
    session.reply(30, tool="Bash")
    session.lines[-1]["message"]["content"][0]["id"] = "rename"
    session.notification(tool_id, job, queued=True, seconds=5)  # the tests finished while the agent was working
    session.tool_result("rename", 1, "p")
    session.reply(10)
    session.save(mtime=START)

    tasks = read_claude(tmp_path)
    first = tasks["Run the full test suite, then fix what fails"]
    assert (first.calls, first.stopped, first.finished_at) == (2, False, pytest.approx(START + 42.1, abs=0.01))
    assert tasks["Meanwhile, rename the module"].calls == 2


def test_a_monitors_events_and_a_ci_event_stay_uncounted(tmp_path):
    session, line, tool_id, job = background_task(tmp_path, tool="Monitor")
    session.notification(tool_id, job, status=None, seconds=60)  # a line the monitor saw
    session.reply(10)
    session.prompt("<ci-monitor-event>The checks passed.</ci-monitor-event>", origin=None)
    session.reply(10)
    session.save(mtime=START)

    [task] = read_archive(claude_archive(tmp_path), now=LATER)
    assert (task.calls, task.seconds) == (2, pytest.approx(7.1))


def test_a_task_waits_while_its_background_helper_works_and_is_stopped_if_it_never_reports(tmp_path):
    session, line, tool_id, job = background_task(tmp_path, tool="Agent")
    session.prompt("Meanwhile, fix the typo")
    session.reply(3)
    session.helper(line["promptId"], calls=2, agent=job)
    session.save(mtime=START, helpers_mtime=START + 3500)
    archive = claude_archive(tmp_path)
    task_id = f"claude-code:{line['uuid']}"

    # The helper wrote a minute ago and hasn't reported back. It may still be working, so the task waits.
    reading = read_records(archive, record_files(archive), now=START + 3560)
    assert [t.prompt for t in reading.tasks] == ["Meanwhile, fix the typo"]
    assert task_id in reading.held and str(session.path) in reading.open_files
    # Once it has been quiet a long time without reporting, the task's outcome isn't known: it's stopped.
    assert read_claude(tmp_path)["Run the full test suite, then fix what fails"].stopped

    session.notification(tool_id, job, seconds=10)
    session.reply(5)
    session.save(mtime=START, helpers_mtime=START + 3500)
    task = read_claude(tmp_path)["Run the full test suite, then fix what fails"]
    assert (task.calls, task.stopped) == (5, False)


def test_a_background_command_that_never_reports_doesnt_stop_its_task(tmp_path):
    session, line, tool_id, job = background_task(tmp_path)  # such as a server, which runs until it's stopped
    session.save(mtime=START)
    archive = claude_archive(tmp_path)
    assert read_records(archive, record_files(archive), now=START + 60).held == {f"claude-code:{line['uuid']}"}
    [task] = read_archive(archive, now=LATER)
    assert (task.calls, task.stopped) == (2, False)


def test_a_quiet_record_whose_last_tool_call_never_got_a_result_is_stopped(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Run the migration")
    session.reply(5, tool="Bash")
    session.save(mtime=START)
    archive = claude_archive(tmp_path)
    assert read_records(archive, record_files(archive), now=START + 60).held  # may still be running
    [task] = read_archive(archive, now=LATER)
    assert task.stopped


def test_an_allowed_tool_whose_result_says_how_long_it_ran_counts_the_run_as_agent_time(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Look it up")
    tool = session.reply(5, tool="WebSearch")
    # The person allowed the search after 60 seconds. Then it ran for 8 seconds.
    session.tool_result(tool, 68, line["promptId"], source="user_temporary", result={"durationSeconds": 8.0})
    session.reply(2)
    session.save(mtime=START)
    assert read_claude(tmp_path)["Look it up"].seconds == pytest.approx(5 + 8 + 2)


def test_the_task_keeps_the_record_file_it_was_read_from(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Add a login page")
    session.reply(5)
    session.save(mtime=START)
    codex = CodexSession(tmp_path / "codex", "c1", START)
    codex.turn("t1", "Fix the failing test", calls=[(100, 0, 10)], seconds=5)
    codex.save(mtime=START)
    assert [t.source for t in read_archive(claude_archive(tmp_path), now=LATER)] == [str(session.path)]
    assert [t.source for t in read_archive(codex_archive(tmp_path / "codex"), now=LATER)] == [str(codex.path)]


# Odd records -------------------------------------------------------------------------------------------


def test_a_text_part_without_text_and_a_record_removed_while_reading_dont_crash(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Still readable")
    session.reply(2)
    session.lines.append({"type": "user", "uuid": "odd", "message": {"content": [{"type": "text", "text": None}]}})
    session.lines.append({"type": "user", "uuid": "odd2", "message": "not a message"})
    session.save(mtime=START)
    codex = CodexSession(tmp_path / "codex", "c1", START)
    codex.turn("t1", "Fix the failing test", calls=[(100, 0, 10)], seconds=5)
    codex.lines.insert(
        2,
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": None}]},
        },
    )
    codex.save(mtime=START)
    archive = claude_archive(tmp_path)
    files = {**record_files(archive), str(tmp_path / "-work-demo" / "gone.jsonl"): (10, START)}
    assert [t.prompt for t in read_records(archive, files, now=LATER).tasks] == ["Still readable"]
    codex_files = {
        **record_files(codex_archive(tmp_path / "codex")),
        str(tmp_path / "codex" / "gone.jsonl"): (10, START),
    }
    assert [t.prompt for t in read_records(codex_archive(tmp_path / "codex"), codex_files, now=LATER).tasks] == [
        "Fix the failing test"
    ]


# Skills ------------------------------------------------------------------------------------------------


def test_prompt_skills_finds_typed_and_recorded_commands_but_not_claude_codes_own():
    assert prompt_skills("/deploy the staging site") == {"deploy"}
    assert prompt_skills("<command-message>deploy</command-message>\n<command-name>/deploy</command-name>") == {
        "deploy"
    }
    assert prompt_skills("/superpowers:brainstorming") == {"superpowers:brainstorming"}
    assert prompt_skills("<command-name>/model</command-name>") == set()
    assert prompt_skills("/compact") == set()
    assert prompt_skills("/Users/me/notes.md says what to do") == set()
    assert prompt_skills("Fix the bug, see /deploy") == set()


def test_line_skills_finds_the_skill_a_reply_is_credited_to_and_the_skills_it_loads():
    line = {
        "type": "assistant",
        "attributionSkill": "frontend-design",
        "message": {"content": [{"type": "tool_use", "name": "Skill", "input": {"skill": "artifact-design"}}]},
    }
    assert line_skills(line) == {"frontend-design", "artifact-design"}
    assert line_skills({"type": "user", "attributionSkill": "x"}) == set()


def test_a_task_knows_the_skills_its_prompt_invokes_and_those_used_before_it(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Design the page")
    tool = session.reply(3, tool="Skill", tool_input={"skill": "artifact-design"})
    session.tool_result(tool, 1, "p")
    session.reply(2)
    session.prompt("<command-message>deploy</command-message>\n<command-name>/deploy</command-name>")
    session.reply(4)
    session.prompt("Thanks")
    session.reply(1)
    session.save(mtime=START)

    tasks = {t.prompt[:6]: t for t in read_archive(claude_archive(tmp_path), now=LATER)}
    assert (tasks["Design"].prior_skills, tasks["Design"].skills) == ("", "artifact-design")
    assert tasks["<comma"].prior_skills == "artifact-design\ndeploy"
    assert tasks["Thanks"].prior_skills == "artifact-design\ndeploy"


# Codex -------------------------------------------------------------------------------------------------


def test_a_codex_turn_left_open_when_the_next_one_starts_is_kept_as_stopped(tmp_path):
    session = CodexSession(tmp_path, "c1", START)
    session.turn("t1", "Crashed turn", calls=[(1000, 0, 100), (1000, 0, 100)], seconds=40, finished=False)
    session.clock += 3600
    session.turn("t2", "Next turn", calls=[(10, 0, 1)], seconds=2)
    session.save(mtime=START)

    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    crashed = tasks["Crashed turn"]
    assert (crashed.calls, crashed.stopped, crashed.finished_at) == (2, True, START + 40)
    assert not tasks["Next turn"].stopped


def test_a_codex_helper_that_reviews_an_approval_counts_toward_the_turn_it_reviewed(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Deploy it", calls=[(1000, 0, 100)], seconds=10, root="t1")
    parent.save(mtime=START)
    guardian = CodexSession(tmp_path, "guardian", START + 5, guardian_of="parent")
    guardian.turn("g1", "Review this approval", calls=[(300, 0, 30)], seconds=2, root="t1")
    guardian.save(mtime=START)

    [task] = read_archive(codex_archive(tmp_path), now=LATER)
    assert (task.calls, task.input_tokens) == (2, 1300)


def test_a_nested_codex_helpers_copy_of_its_parents_turn_doesnt_take_that_turn_from_its_parent(tmp_path):
    main = CodexSession(tmp_path, "main", START)
    main.turn("t1", "Split the work", calls=[(1000, 0, 100)], seconds=100)
    main.save(mtime=START + 300)
    first = CodexSession(tmp_path, "first", START + 10, parent="main")
    first.turn("h1", "Part one", calls=[(200, 0, 20)], seconds=50)
    first.save(mtime=START + 200)
    # A helper forked from the first helper starts with a copy of its turn, and finished before it.
    nested = CodexSession(tmp_path, "nested", START + 20, parent="first", depth=2)
    nested.turn("h1", "Part one", calls=[(200, 0, 20)], seconds=0, finished=False)
    nested.turn("h2", "Part two", calls=[(50, 0, 5)], seconds=2)
    nested.save(mtime=START + 100)

    [task] = read_archive(codex_archive(tmp_path), now=LATER)
    assert (task.calls, task.input_tokens) == (3, 1250)


def test_a_codex_helper_without_named_turns_finds_its_parents_turn_when_another_main_thread_holds_a_copy(tmp_path):
    # A branched session has its own id and a copy of the parent's first turn. Its record was written first.
    branch = CodexSession(tmp_path, "branch", START)
    branch.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=10)
    branch.clock = START + 100
    branch.turn("t2", "Go another way", calls=[(500, 0, 50)], seconds=10)
    branch.save(mtime=START + 1)
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("t1", "Start a helper", calls=[(1000, 0, 100)], seconds=10)
    parent.save(mtime=START + 2)
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "Helper part", calls=[(200, 0, 20)], seconds=5)
    helper.save(mtime=START + 3)

    tasks = {t.prompt: t for t in read_archive(codex_archive(tmp_path), now=LATER)}
    assert (tasks["Start a helper"].calls, tasks["Start a helper"].input_tokens) == (2, 1200)
    assert tasks["Go another way"].calls == 1


def test_a_codex_task_waits_while_its_helper_works_and_is_stopped_if_the_helper_goes_quiet(tmp_path):
    parent = CodexSession(tmp_path, "parent", START)
    parent.turn("p1", "Launch background build", calls=[(1000, 0, 100)], seconds=10, root="p1")
    parent.save(mtime=START + 10)
    helper = CodexSession(tmp_path, "helper", START + 5, parent="parent")
    helper.turn("h1", "Background build", calls=[(200, 0, 20)], seconds=50, root="p1", finished=False)
    helper.save(mtime=START + 55)
    archive = codex_archive(tmp_path)

    # The root finished, but its helper is still running. Its outcome isn't known yet, so it isn't read.
    reading = read_records(archive, record_files(archive), now=START + 60)
    assert (reading.tasks, reading.held) == ([], {"codex:p1"})
    assert reading.open_files == {str(parent.path), str(helper.path)}
    # A helper whose record went quiet without finishing leaves the outcome unknown: the task is stopped.
    [quiet] = read_records(archive, record_files(archive), now=START + 55 + 901).tasks
    assert (quiet.calls, quiet.stopped) == (2, True)

    helper.clock = START + 95
    helper.call(400, 0, 40)  # the helper made another call, then finished
    helper.finish("h1", 5)
    helper.save(mtime=START + 100)
    [task] = read_records(archive, record_files(archive), now=START + 120).tasks
    assert (task.input_tokens, task.calls, task.stopped) == (1600, 3, False)
    assert task.finished_at == START + 100
