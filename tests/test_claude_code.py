from __future__ import annotations

import io
import json
from datetime import datetime

import pytest

from yardstick_predict import claude_code as cc
from yardstick_predict import cli
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
    monkeypatch.setattr(cc, "_start_forecast", lambda session: None)
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "Add a login page"})
    state_path, forecast_path, prompt_path = cc._paths("s1")
    started = json.loads(state_path.read_text())["prompt_at"]
    assert prompt_path.read_text() == "Add a login page"
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


def plain(text: str) -> str:
    return text.replace("\033[2m", "").replace("\033[22m", "")
