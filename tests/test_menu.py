from __future__ import annotations

import math

import pytest

from yardstick_predict import features, menu


def answers(monkeypatch, *replies: str) -> None:
    replies_left = list(replies)
    monkeypatch.setattr("builtins.input", lambda prompt="": replies_left.pop(0))


def test_flags_name_features_and_their_options():
    picked = menu.from_flags("skills:named,llm:description,benchmarks", "codex", None)
    assert picked.features == ["prompt", "skills", "llm", "benchmarks"]
    assert picked.options == {"skills": ["named"], "llm": ["description"]}
    assert picked.llm == {"provider": "codex", "model": "codex-default"}
    assert menu.from_flags(None).features == ["prompt", "conversation", "context"]
    assert menu.from_flags("context").features == ["prompt", "context"]  # a feature not named is left out
    assert menu.from_flags("").features == ["prompt"]
    assert menu.from_flags("llm").llm == {"provider": "anthropic", "model": "claude-opus-5-5"}


def test_the_time_feature_takes_either_option_or_both():
    for given, chosen in (
        ("time", ["trend", "clock"]),
        ("time:trend", ["trend"]),
        ("time:clock", ["clock"]),
        ("time:trend+clock", ["trend", "clock"]),
    ):
        picked = menu.from_flags(given)
        assert picked.features == ["prompt", "time"] and picked.options == {"time": chosen}, given
    with pytest.raises(ValueError, match="time has no option 'season'. Its options: trend, clock."):
        menu.from_flags("time:season")


def test_flags_that_make_no_sense_say_why():
    with pytest.raises(ValueError, match="has no option"):
        menu.from_flags("skills:everything")
    with pytest.raises(ValueError, match="go with the llm feature"):
        menu.from_flags("context", "claude", None)
    with pytest.raises(ValueError, match="Unknown feature"):
        menu.from_flags("horoscope")


def test_a_person_turns_features_and_options_on_and_off(monkeypatch, capsys):
    # Turn on one skills option and all of LLM analysis, turn the conversation off, then pay with Codex.
    answers(monkeypatch, "6b 9", "4", "1", "", "4")
    picked = menu.ask()
    out = capsys.readouterr().out
    assert picked.features == ["prompt", "context", "skills", "llm"]
    assert picked.options == {"skills": ["earlier"], "llm": ["kind", "description"]}
    assert picked.llm == {"provider": "codex", "model": "codex-default"}
    assert "Task is always on" in out and "How should it be paid for?" not in out  # the prompt shows on input


def test_all_turns_on_every_feature_with_all_its_options(monkeypatch, capsys):
    every = [group for group in features.MENU if not group.always]
    answers(monkeypatch, "all", "", "4")  # every feature, then pay for LLM analysis with Codex
    picked = menu.ask()
    assert picked.features == ["prompt", *(group.key for group in every)]
    assert picked.options == {group.key: [o.key for o in group.options] for group in every if len(group.options) > 1}
    assert picked.llm == {"provider": "codex", "model": "codex-default"}
    assert "Type all to turn on every feature with all its options." in " ".join(capsys.readouterr().out.split())
    # Typed with others, each goes in turn: here every feature, then model facts off again.
    answers(monkeypatch, "all 7", "", "4")
    assert "benchmarks" not in menu.ask().features


def test_the_features_screen_shows_the_time_with_its_two_options(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    answers(monkeypatch, "8b", "")  # only the clock
    picked = menu.ask()
    out = capsys.readouterr().out
    assert picked.features == ["prompt", "conversation", "context", "time"]
    assert picked.options == {"time": ["clock"]}
    assert " 8  Time" in out and "when the task started" in out and " 9  LLM analysis" in out
    assert "days since 1 January 2020" in out and "hour of day and day of week, in your time zone" in out


def test_a_key_paid_analysis_shows_each_models_price(monkeypatch, capsys):
    answers(monkeypatch, "9", "", "1", "3")
    picked = menu.ask()
    out = capsys.readouterr().out
    assert picked.llm == {"provider": "anthropic", "model": "claude-haiku-5-5"}
    assert "Claude Opus 5.5" in out and "per 1,000 tasks" in out


def test_options_saved_before_options_existed_mean_all_of_them():
    assert features.options({}, "skills") == ["named", "earlier"]
    assert features.options({"options": {"skills": ["named"]}}, "skills") == ["named"]


def test_the_prompt_shape_and_effort_level_are_always_numbers():
    info = features.TaskInfo(prompt="Fix these:\n- src/app.py\n- docs/README.md\nWhy?", model="m", effort="xhigh")
    found = features.numbers([], info, None)
    assert math.isclose(found["prompt_list_items"], math.log1p(2))
    assert math.isclose(found["prompt_file_names"], math.log1p(2))
    assert found["prompt_asks"] == 1.0 and found["effort_level"] == 4.0
    assert "effort_level" not in features.numbers([], features.TaskInfo(prompt="x", model="m"), None)


def test_the_skills_options_keep_the_skills_the_prompt_invokes_or_the_ones_used_before_it():
    known = ["review", "tdd"]  # what the prompt invokes, and what the conversation used before it
    assert features.known_skills(known, "/review this, please") == ["review", "tdd"]
    assert features.known_skills(known, "/review this, please", ["named"]) == ["review"]
    assert features.known_skills(known, "/review this, please", ["earlier"]) == ["tdd"]
    assert features.known_skills([], "<command-name>/review</command-name>", ["named"]) == ["review"]
    assert features.known_skills([], "Move /src/app.py to /lib") == []  # a path isn't a skill
