from __future__ import annotations

import io
import json
import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from builders import ClaudeSession

from yardstick_predict import claude_code as cc
from yardstick_predict import cli
from yardstick_predict.archives import read_archive
from yardstick_predict.store import Predictor

THEIRS = {
    "model": "opus",
    "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]},
    "permissions": {"allow": ["Bash(ls:*)"]},
}


def test_adding_keeps_your_own_settings_and_removing_puts_them_back():
    added = cc.added(THEIRS, "/py")
    assert added["model"] == "opus" and added["permissions"] == THEIRS["permissions"]
    stop = [entry["command"] for group in added["hooks"]["Stop"] for entry in group["hooks"]]
    assert stop[0] == "say done" and stop[1].endswith("-m yardstick_predict.claude_code hook")
    assert added["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == '"/py" -m yardstick_predict.claude_code hook'
    assert added["statusLine"]["command"].endswith("-m yardstick_predict.claude_code status")
    assert added["statusLine"]["refreshInterval"] == 1  # the clock ticks once a second
    assert "StopFailure" in added["hooks"]  # a failed request to the model ends the task too
    assert cc.added(added, "/py") == added  # adding twice changes nothing
    assert cc.removed(added) == THEIRS
    assert cc.other_status_line({"statusLine": {"command": "my-line"}}) == "my-line"
    assert cc.other_status_line(added) is None


def test_the_status_line_follows_a_task_from_prompt_to_stop(monkeypatch):
    monkeypatch.setattr(cc, "_start_forecast", lambda session, prompt_at: None)
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "Add a login page"})
    state_path, forecast_path = cc._paths("s1")
    started = json.loads(state_path.read_text())["prompt_at"]
    assert cc._prompt_path("s1", started).read_text() == "Add a login page"
    assert cc.status_line("s1", "claude-opus-5-5", started + 1).endswith("· forecasting…")

    forecasts = [
        {"model": "claude-opus-5-5", "effort": "max", "seconds": 240.0, "tokens": 9, "usd": 1.234},
        {"model": "claude-opus-5-5", "effort": "high", "seconds": 120.0, "tokens": 9, "usd": 0.5},
        {"model": "claude-sonnet-5", "seconds": None, "tokens": None, "usd": None, "reason": "not_enough_data"},
    ]
    forecast_path.write_text(json.dumps({"prompt_at": started, "forecasts": forecasts}))
    line = plain(cc.status_line("s1", "claude-opus-5-5[1m]", started + 72))
    assert line == "yardstick · 1:12 of ~4 min · ~$1.23"
    at_high = plain(cc.status_line("s1", "claude-opus-5-5", started + 72, effort="high"))
    assert at_high == "yardstick · 1:12 of ~2 min · ~$0.50"  # Claude Code says which effort level is on
    missing = plain(cc.status_line("s1", "claude-sonnet-5", started + 5))
    assert missing == "yardstick · no forecast for claude-sonnet-5 yet"

    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop"})
    done = json.loads(state_path.read_text())["done_at"]
    line = plain(cc.status_line("s1", "claude-opus-5-5", done + 999))
    assert line.startswith("yardstick · took 0:0") and line.endswith("· forecast ~4 min · ~$1.23")


def test_an_older_forecast_never_replaces_a_newer_one(monkeypatch):
    commands = forecasts_by_hand(monkeypatch)

    def answer(predictor, tasks):
        if tasks[0].prompt == "First prompt":
            # While the first forecast runs, a second prompt is sent, and its forecast finishes first.
            send(monkeypatch, {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "Second prompt"})
            cc.main(commands[-1])
        seconds = 60.0 if tasks[0].prompt == "First prompt" else 120.0
        return {"forecasts": [{"model": "m", "seconds": seconds, "tokens": 9, "usd": None}]}

    monkeypatch.setattr("yardstick_predict.forecast.answer", answer)
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "First prompt"})
    assert cc.main(commands[0]) == 0
    latest = json.loads(cc._paths("s1")[0].read_text())["prompt_at"]
    assert plain(cc.status_line("s1", "m", latest + 1)) == "yardstick · 0:01 of ~2 min"  # the second prompt's


def test_each_forecast_reads_its_own_prompt(monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list[list[str]] = []

    def answer(predictor, tasks):
        asked[-1].append(tasks[0].prompt)
        seconds = 60.0 if tasks[0].prompt == "First prompt" else 120.0
        return {"forecasts": [{"model": "m", "seconds": seconds, "tokens": 9, "usd": None}]}

    monkeypatch.setattr("yardstick_predict.forecast.answer", answer)
    # The second prompt is sent before the first prompt's forecast has started.
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "First prompt"})
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "Second prompt"})
    for command in commands:
        asked.append([])
        assert cc.main(command) == 0
    assert "Second prompt" not in asked[0] and asked[1] == ["Second prompt"]
    assert not list(cc.folder().glob("*.prompt"))  # each forecast took its prompt away
    latest = json.loads(cc._paths("s1")[0].read_text())["prompt_at"]
    assert plain(cc.status_line("s1", "m", latest + 1)) == "yardstick · 0:01 of ~2 min"


def test_the_forecast_says_a_person_typed_the_prompt_after_the_conversation_so_far(tmp_path, monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked = []

    def answer(predictor, tasks):
        asked.extend(tasks)
        return {"forecasts": []}

    monkeypatch.setattr("yardstick_predict.forecast.answer", answer)
    sent = 1_780_000_000  # when the hook runs, by the clock of forecasts_by_hand
    session = ClaudeSession(tmp_path, "s1", sent - 600)
    session.prompt("Add a login page")
    session.reply(5, text="Added the login page.")
    session.prompt("continue")
    session.reply(5, text="Styled the form too.")
    session.clock = sent - 1
    session.prompt("continue")  # the prompt being forecast, recorded just before the hook ran
    session.reply(60, text="Checking the tests.")  # the agent's answer, after it
    record = session.save()
    tasks = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=time.time() + 86400)
    trained = max(tasks, key=lambda t: t.started_at).context  # the context training gives this prompt
    assert trained.endswith("User: continue\n\nAssistant: Styled the form too.")

    for lines in (session.lines, session.lines[:-2]):  # with the prompt already in the record, and not yet
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        event = {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "continue"}
        send(monkeypatch, {**event, "transcript_path": str(record)})
        assert cc.main(commands[-1]) == 0
    assert [(t.origin, t.agent, t.context) for t in asked] == [("person", "claude-code", trained)] * 2


def test_the_session_effort_level_picks_the_forecast(tmp_path):
    record = tmp_path / "session.jsonl"
    lines = [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant", "effort": "high", "message": {"model": "claude-opus-5-5"}},
        {"type": "assistant", "effort": "max", "message": {"model": "claude-opus-5-5"}},
    ]
    record.write_text("\n".join(json.dumps(line) for line in lines))
    assert cc.last_model_and_effort(str(record)) == ("claude-opus-5-5", "max")
    assert cc.last_model_and_effort(str(tmp_path / "missing.jsonl")) == ("", "")
    forecasts = [{"model": "m", "effort": "high", "seconds": 1}, {"model": "m", "effort": "max", "seconds": 2}]
    assert cc._pick(forecasts, "m", "max")["seconds"] == 2
    assert cc._pick(forecasts, "m-20260101", "")["seconds"] == 1  # a dated name, and the usual level


def test_a_model_matches_its_dated_name_both_ways_and_never_another_model():
    dated = [{"model": "claude-opus-4-5-20251101", "effort": "high", "seconds": 60.0}]
    assert cc._pick(dated, "claude-opus-4-5", "high") == dated[0]  # Claude Code may name the model without a date
    alias = [{"model": "claude-opus-4-5", "seconds": 30.0}]
    assert cc._pick(alias, "claude-opus-4-5-20251101", "") == alias[0]
    assert cc._pick([{"model": "claude-opus-4", "seconds": 9.0}], "claude-opus-4-5", "") is None
    assert cc._pick([{"model": "claude-opus-4-5-20250101", "seconds": 8.0}], "claude-opus-4-5-20251101", "") is None
    state_path, forecast_path = cc._paths("s1")
    state_path.write_text(json.dumps({"prompt_at": 100.0}))
    forecast_path.write_text(json.dumps({"prompt_at": 100.0, "forecasts": dated}))
    assert plain(cc.status_line("s1", "claude-opus-4-5[1m]", 172.0)) == "yardstick · 1:12 of ~60 s"
    assert plain(cc.status_line("s1", "claude-opus-4", 172.0)) == "yardstick · no forecast for claude-opus-4 yet"


def test_pressing_esc_stops_the_clock(tmp_path):
    record = tmp_path / "session.jsonl"
    stopped = {
        "type": "user",
        "timestamp": "2026-10-22T10:00:30Z",
        "message": {"content": [{"type": "text", "text": "[Request interrupted by user]"}]},
    }
    mention = {"type": "user", "message": {"content": [{"type": "tool_result", "content": "[Request interrupted"}]}}
    lines = [{"type": "assistant", "message": {"model": "m"}}, stopped]
    record.write_text("\n".join(json.dumps(line) for line in lines))
    started = datetime.fromisoformat("2026-10-22T10:00:00+00:00").timestamp()
    assert cc.interrupted_at(str(record), started) == started + 30
    assert cc.interrupted_at(str(record), started + 60) is None  # an older stop doesn't count
    record.write_text("\n".join(json.dumps(line) for line in [*lines, {"type": "assistant", "message": {}}]))
    assert cc.interrupted_at(str(record), started) is None  # the agent went on after it
    record.write_text(json.dumps(mention))
    assert cc.interrupted_at(str(record), started) is None  # only mentioning it isn't a stop


def test_install_and_remove_in_this_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps(THEIRS))
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 1  # no predictor yet
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert cli.main(["claude-code", "install", "--here"]) == 2  # a program must say yes
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert "statusLine" in json.loads(settings.read_text())
    assert json.loads(settings.with_name("settings.local.json.before-yardstick").read_text()) == THEIRS
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert json.loads(settings.read_text()) == THEIRS
    assert "Removed" in capsys.readouterr().out


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")


def send(monkeypatch, event: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert cc.hook() == 0


def forecasts_by_hand(monkeypatch) -> list[list[str]]:
    """Keep each background forecast the hook would start, to run it here by hand with cc.main.

    The forecasts use a stand-in predictor that forecasts model m. The clock moves 10 seconds at each
    look, so two prompts never share a time.
    """
    commands: list[list[str]] = []
    monkeypatch.setattr(cc.subprocess, "Popen", lambda command, **options: commands.append(command[3:]))
    clock = iter(range(1_780_000_000, 1_790_000_000, 10))
    monkeypatch.setattr(cc, "time", SimpleNamespace(time=lambda: float(next(clock))))
    monkeypatch.setattr(cli, "request_predictor", lambda chosen: SimpleNamespace(name="default"))
    monkeypatch.setattr(cli, "forecast_models", lambda predictor: ["m"])
    monkeypatch.setattr(cli, "effort_pairs", lambda predictor, models, effort: [(model, None) for model in models])
    return commands


def plain(text: str) -> str:
    return text.replace("\033[2m", "").replace("\033[22m", "")
