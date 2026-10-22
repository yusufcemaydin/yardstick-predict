from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from yardstick_predict import llm

FIELDS = llm.parts(["kind", "description"])


def answer_for(ids: list[str], **changes) -> dict:
    entry = {
        "kind": "feature",
        "files": 4,
        "steps": 3,
        "runs_code": True,
        "looks_up": False,
        "continues": False,
        "clarity": "clear",
        "minutes": 12.5,
        "description": "Add a login page.",
    }
    return {"tasks": [{"id": i, **entry, **changes} for i in ids]}


def ids_in(text: str) -> list[str]:
    return [line.split()[1] for line in text.splitlines() if line.startswith("Task ")]


def test_the_answer_shape_asks_for_exactly_the_chosen_fields():
    shape = llm.schema(llm.parts(["kind"]))
    task = shape["properties"]["tasks"]["items"]
    assert task["required"] == ["id", *llm.KIND_FIELDS]
    assert "description" not in task["properties"] and task["additionalProperties"] is False
    assert llm.parts(["description"]) == ["description"]


def test_an_answer_that_makes_no_sense_is_dropped():
    good = answer_for(["a"])["tasks"][0]
    assert llm._clean(good, FIELDS)["minutes"] == 12.5
    assert llm._clean({**good, "kind": "cooking"}, FIELDS) is None
    assert llm._clean({**good, "files": True}, FIELDS) is None
    assert llm._clean({**good, "description": "  "}, FIELDS) is None
    assert llm._clean({**good, "steps": -3}, FIELDS)["steps"] == 0.0


def test_tasks_go_in_batches_and_each_answer_says_who_gave_it(monkeypatch):
    calls = []

    def fake(model, system, text, shape):
        calls.append(ids_in(text))
        return answer_for(ids_in(text))

    monkeypatch.setattr(llm, "_anthropic", fake)
    asks = [llm.Ask(f"t{i}", f"Prompt {i}") for i in range(20)]
    progress = []
    found = llm.analyze(
        asks, {"provider": "anthropic", "model": "claude-opus-5-5"}, FIELDS, lambda f, done: progress.append(done)
    ).answers
    assert sorted(len(batch) for batch in calls) == [4, 8, 8]  # the batches run side by side
    assert len(found) == 20 and len(progress) == 3 and progress[-1] == 20  # one update per request
    assert found["t3"]["by"] == "anthropic:claude-opus-5-5" and found["t3"]["fields"] == FIELDS


def test_a_request_that_failed_is_left_for_later_and_says_why(monkeypatch):
    def flaky(model, system, text, shape):
        if "t0" in ids_in(text):
            raise llm.LLMError("Anthropic's API said: overloaded")
        return answer_for(ids_in(text))

    monkeypatch.setattr(llm, "_anthropic", flaky)
    asks = [llm.Ask(f"t{i}", "x") for i in range(12)]
    found = llm.analyze(asks, {"provider": "anthropic", "model": "claude-opus-5-5"}, FIELDS)
    assert sorted(found.answers) == sorted(f"t{i}" for i in range(8, 12)) and "overloaded" in found.problem

    def broken(model, system, text, shape):
        raise llm.LLMError("No Anthropic API key was found. Set ANTHROPIC_API_KEY to your key.")

    monkeypatch.setattr(llm, "_anthropic", broken)
    found = llm.analyze(asks, {"provider": "anthropic", "model": "claude-opus-5-5"}, FIELDS)
    assert found.answers == {} and "ANTHROPIC_API_KEY" in found.problem


def test_a_refusal_or_a_skipped_task_is_noted_and_tried_at_most_twice(monkeypatch):
    def picky(model, system, text, shape):
        if "t0" in ids_in(text):
            raise llm.LLMError("The AI model declined to describe these tasks.", lasting=True)
        return answer_for([i for i in ids_in(text) if i != "t9"])  # it leaves t9 out

    monkeypatch.setattr(llm, "_anthropic", picky)
    how = {"provider": "anthropic", "model": "claude-opus-5-5"}
    found = llm.analyze([llm.Ask(f"t{i}", "x") for i in range(10)], how, FIELDS).answers
    assert found["t0"]["failed"] and found["t9"]["failed"] and not found["t8"].get("failed")
    assert llm.needs(found["t0"], how, FIELDS)  # tried once: one more try
    assert not llm.needs({**found["t0"], "tries": 2}, how, FIELDS)
    assert llm.needs({**found["t0"], "tries": 2}, {**how, "model": "claude-haiku-5-5"}, FIELDS)


def test_two_tasks_of_one_conversation_never_share_a_request():
    asks = [llm.Ask(f"t{i}", "x", conversation="long chat") for i in range(5)] + [
        llm.Ask("u", "y", conversation="other")
    ]
    batches = llm.batches(asks, 8)
    assert len(batches) == 5 and all(len({a.conversation for a in b}) == len(b) for b in batches)


def test_only_a_few_requests_run_or_wait_at_once_so_stopping_is_quick(monkeypatch):
    import threading

    calls, lock = [], threading.Lock()

    def slow(model, system, text, shape):
        with lock:
            calls.append(text)
        return answer_for(ids_in(text))

    monkeypatch.setattr(llm, "_anthropic", slow)

    def stop(found, done):
        raise KeyboardInterrupt  # Ctrl+C right after the first answer

    asks = [llm.Ask(f"t{i}", "x") for i in range(400)]
    with pytest.raises(KeyboardInterrupt):
        llm.analyze(asks, {"provider": "anthropic", "model": "claude-opus-5-5"}, FIELDS, stop)
    assert len(calls) <= 2 * llm.PROVIDERS["anthropic"].parallel  # not all 50 requests


def test_answers_for_tasks_it_was_never_asked_about_are_ignored(monkeypatch):
    monkeypatch.setattr(llm, "_openai", lambda *args: answer_for(["t0", "made-up"]))
    found = llm.analyze([llm.Ask("t0", "x")], {"provider": "openai", "model": "gpt-6.1-sol"}, FIELDS).answers
    assert list(found) == ["t0"]


def test_a_task_is_read_again_only_with_another_model_or_new_fields():
    how = {"provider": "anthropic", "model": "claude-opus-5-5"}
    labels = {"by": "anthropic:claude-opus-5-5", "fields": FIELDS}
    assert not llm.needs(labels, how, FIELDS)
    assert llm.needs(None, how, FIELDS)
    assert llm.needs(labels, {**how, "model": "claude-haiku-5-5"}, FIELDS)
    assert llm.needs({**labels, "fields": ["description"]}, how, FIELDS)


def test_what_it_said_becomes_numbers_the_trees_can_read():
    found = llm.numbers(llm._clean(answer_for(["a"])["tasks"][0], FIELDS))
    assert found["llm_kind_feature"] == 1.0 and found["llm_kind_question"] == 0.0
    assert math.isclose(found["llm_files"], math.log1p(4)) and found["llm_runs_code"] == 1.0
    assert found["llm_clarity"] == 0.0 and llm.numbers({}) == {}


def test_names_and_ways_to_pay():
    assert llm.title("claude-opus-5-5") == "Claude Opus 5.5"
    assert llm.title("gpt-6.1-sol") == "GPT 6.1 Sol"
    assert llm.access(None, None) == {"provider": "anthropic", "model": "claude-opus-5-5"}
    assert llm.access("chatgpt", None) == {"provider": "codex", "model": llm.DEFAULT_CODEX}
    with pytest.raises(ValueError, match="one of"):
        llm.access("gemini", None)
    assert math.isclose(llm.cost(1000, "claude-opus-5-5"), 8.0)


def test_the_claude_subscription_runs_with_no_tools_no_settings_and_no_saved_session(monkeypatch):
    seen = {}

    def run(arguments, **kwargs):
        seen.update(arguments=arguments, env=kwargs["env"], text=kwargs["input"])
        reply = {"type": "result", "is_error": False, "structured_output": answer_for(["t1"])}
        return subprocess.CompletedProcess(arguments, 0, stdout=json.dumps(reply), stderr="")

    monkeypatch.setattr(llm.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(llm.subprocess, "run", run)
    found = llm.analyze(
        [llm.Ask("t1", "Fix the typo")], {"provider": "claude", "model": "claude-opus-5-5"}, FIELDS
    ).answers
    arguments = seen["arguments"]
    assert arguments[:2] == ["/bin/claude", "-p"] and "--no-session-persistence" in arguments
    assert arguments[arguments.index("--tools") + 1] == "" and arguments[arguments.index("--setting-sources") + 1] == ""
    assert seen["env"][llm.INSIDE] == "1" and "Fix the typo" in seen["text"]
    assert found["t1"]["kind"] == "feature"


def test_the_chatgpt_subscription_runs_in_an_empty_folder_and_saves_no_session(monkeypatch):
    seen = {}

    def run(arguments, **kwargs):
        seen.update(arguments=arguments, cwd=kwargs["cwd"], env=kwargs["env"])
        Path(arguments[arguments.index("-o") + 1]).write_text(json.dumps(answer_for(["t1"])))
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")

    monkeypatch.setattr(llm.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(llm.subprocess, "run", run)
    found = llm.analyze([llm.Ask("t1", "x")], {"provider": "codex", "model": llm.DEFAULT_CODEX}, FIELDS).answers
    arguments = seen["arguments"]
    assert arguments[:2] == ["/bin/codex", "exec"] and arguments[-1] == "-"
    assert {"--ephemeral", "--ignore-user-config", "--skip-git-repo-check"} <= set(arguments) and "-m" not in arguments
    assert arguments[arguments.index("--sandbox") + 1] == "read-only" and seen["env"][llm.INSIDE] == "1"
    assert not Path(seen["cwd"]).exists()  # the folder it ran in is gone
    assert found["t1"]["minutes"] == 12.5


def test_a_missing_command_says_what_to_install(monkeypatch):
    monkeypatch.setattr(llm.shutil, "which", lambda name: None)
    found = llm.analyze([llm.Ask("t1", "x")], {"provider": "codex", "model": llm.DEFAULT_CODEX}, FIELDS)
    assert found.answers == {} and "needs Codex" in found.problem


def fake_anthropic(monkeypatch, create):
    """A stand-in for Anthropic's library, enough to see what yardstick sends and how it reads errors."""
    import sys
    import types

    module = types.ModuleType("anthropic")
    for name in ("AuthenticationError", "APIStatusError", "APIConnectionError"):
        setattr(module, name, type(name, (Exception,), {}))
    messages = types.SimpleNamespace(create=lambda **request: create(False, request))
    beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=lambda **request: create(True, request)))
    module.Anthropic = lambda: types.SimpleNamespace(messages=messages, beta=beta)
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_an_api_key_call_sends_the_answer_shape_and_retries_a_decline_on_another_model(monkeypatch):
    import types

    sent = []

    def create(is_beta, request):
        sent.append((is_beta, request))
        reply = types.SimpleNamespace(type="text", text=json.dumps(answer_for(["t1"])))
        return types.SimpleNamespace(stop_reason="end_turn", content=[reply])

    fake_anthropic(monkeypatch, create)
    for model in ("claude-opus-5-5", "claude-haiku-5-5"):
        found = llm.analyze([llm.Ask("t1", "x")], {"provider": "anthropic", "model": model}, FIELDS)
        assert found.answers["t1"]["kind"] == "feature"
    (opus_beta, opus), (haiku_beta, haiku) = sent
    assert opus_beta and opus["fallbacks"] == "default" and opus["betas"] == ["server-side-fallback-2026-07-01"]
    assert not haiku_beta and "fallbacks" not in haiku  # Haiku has no model to fall back to
    assert opus["output_config"]["format"]["schema"] == llm.schema(FIELDS)


def test_no_api_key_is_a_plain_reason_not_a_crash(monkeypatch):
    def create(is_beta, request):
        raise TypeError("Could not resolve authentication method. Expected one of api_key, auth_token")

    fake_anthropic(monkeypatch, create)
    found = llm.analyze([llm.Ask("t1", "x")], {"provider": "anthropic", "model": "claude-opus-5-5"}, FIELDS)
    assert found.answers == {} and "ANTHROPIC_API_KEY" in found.problem


def test_yardstick_hooks_stay_out_of_the_commands_llm_analysis_starts(monkeypatch):
    from yardstick_predict import claude_code

    monkeypatch.setenv(llm.INSIDE, "1")
    monkeypatch.setattr(claude_code.sys, "stdin", None)  # reading the event would fail
    assert claude_code.hook() == 0
