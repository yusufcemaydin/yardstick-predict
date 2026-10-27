from __future__ import annotations

import http.client
import json
import os
import sys
import textwrap
import time
from datetime import UTC, datetime
from fractions import Fraction

import numpy as np
import pytest
from builders import ClaudeSession, history, usage
from conftest import KEY
from test_cli import ONE_WINDOW

from yardstick_predict import benchmarks, features, model_types, pipeline, prices
from yardstick_predict.archives import TEXT_CHARS, read_archive
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor

START = 1_780_000_000.0
TIME_NUMBERS = {"days_since_2020", "hour_of_day", "day_of_week"}


@pytest.fixture
def skills_home(tmp_path, monkeypatch):
    """A home folder of its own, where a test installs skills."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows finds the home folder here, not in HOME
    features.skill_descriptions.cache_clear()
    yield tmp_path / ".claude" / "skills"
    features.skill_descriptions.cache_clear()


def install(folder, name: str, text: str):
    (folder / name).mkdir(parents=True, exist_ok=True)
    path = folder / name / "SKILL.md"
    path.write_bytes(text.encode())
    return path


def test_the_prompt_is_always_first_and_unknown_features_are_refused():
    assert features.check([]) == ["prompt"]
    assert features.check(["context", "prompt", "context"]) == ["prompt", "context"]
    with pytest.raises(ValueError, match="Unknown feature"):
        features.check(["horoscope"])


def test_context_embeds_only_the_end_of_the_conversation():
    info = features.TaskInfo(prompt="Fix it", model="m", context="old " * 2000 + "the latest exchange")
    text = features.text("context", info)
    assert text.endswith("the latest exchange") and len(text) == features.CONTEXT_CHARS


def test_skills_are_embedded_by_their_descriptions(tmp_path, monkeypatch):
    skill = tmp_path / ".claude" / "skills" / "tdd"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: tdd\ndescription: Write the test first, then the code.\n---\nBody")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows finds the home folder here, not in HOME
    features.skill_descriptions.cache_clear()
    try:
        text = features.skills_text(["superpowers:tdd", "unknown-skill", "custom: does a thing"])
    finally:
        features.skill_descriptions.cache_clear()
    assert "superpowers:tdd: Write the test first, then the code." in text
    assert "unknown-skill" in text and "custom: does a thing" in text


def test_a_skill_description_can_span_several_lines(tmp_path, monkeypatch):
    blocks = {
        "fold": "description: >\n  Plan the work\n  before you start.\n",
        "fold-keep": "description: >+\n  Review the change\n  line by line.\n\n",
        "lines": "description: |\n  Write the test first.\n  Then the code.\n",
        "lines-strip": "description: |-\n  Keep it short.\n\n  Keep it clear.\n",
    }
    for name, block in blocks.items():
        skill = tmp_path / ".claude" / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(f"---\n{block}name: {name}\n---\nBody")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    features.skill_descriptions.cache_clear()
    try:
        text = features.skills_text(list(blocks))
    finally:
        features.skill_descriptions.cache_clear()
    assert text == (
        "fold: Plan the work before you start.\n"
        "fold-keep: Review the change line by line.\n"
        "lines: Write the test first.\nThen the code.\n"
        "lines-strip: Keep it short.\n\nKeep it clear."
    )


def test_the_reader_records_which_skills_a_task_used(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Design the page")
    tool = session.reply(3, tool="Skill")
    session.lines[-1]["message"]["content"][0]["input"] = {"skill": "artifact-design"}
    session.lines[-1]["attributionSkill"] = "frontend-design"
    session.tool_result(tool, 1, line["promptId"])
    session.reply(2, calls_usage=usage())
    session.save(mtime=START)
    task = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=START + 86400 * 30)[0]
    assert task.skills == "artifact-design\nfrontend-design"


def test_the_skills_feature_knows_only_the_skills_known_before_a_task_runs(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    line = session.prompt("Design the page")
    tool = session.reply(3, tool="Skill")
    session.lines[-1]["message"]["content"][0]["input"] = {"skill": "artifact-design"}
    session.tool_result(tool, 1, line["promptId"])
    session.reply(2, calls_usage=usage())
    session.prompt("/review the page")
    session.reply(4, calls_usage=usage())
    session.save(mtime=START)
    first, second = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=START + 86400)
    # The first task used a skill, but nothing said so before it ran.
    assert first.skills == "artifact-design" and features.TaskInfo.of(first).skills == []
    # The second task's prompt invokes a skill, and the conversation used another one before it.
    assert features.TaskInfo.of(second).skills == ["artifact-design", "review"]


def test_benchmarks_come_from_a_public_list_with_one_spelling_for_every_source():
    listing = {
        "data": [
            {
                "id": "anthropic/claude-opus-5.5",
                "created": 1790094732,
                "context_length": 1_000_000,
                "pricing": {"prompt": "0.000004", "completion": "0.00002", "input_cache_read": "0.0000002"},
                "top_provider": {"max_completion_tokens": 128000},
                "supported_parameters": ["reasoning"],
            },
            {"id": "anthropic/claude-opus-5.5:batch", "pricing": {"prompt": "0.000002"}},
        ]
    }
    table = benchmarks.parse(listing)
    assert list(table) == ["claude-opus-5-5"]
    facts = benchmarks.facts(table, "claude-opus-5-5")
    assert facts["benchmark_price_input"] == pytest.approx(0.69897, abs=1e-4)  # log10(1 + $4 per million)
    assert facts["benchmark_context_window"] == pytest.approx(6.0)
    assert facts["benchmark_reasons"] == 1.0
    assert benchmarks.facts(table, "some-unlisted-model") == {}
    assert benchmarks.normalize("openai/gpt-6.1-sol") == benchmarks.normalize("gpt-6.1-sol") == "gpt-6-1-sol"


@pytest.fixture
def time_zone():
    """A way to set this computer's time zone for one test. The old one comes back after it."""
    if not hasattr(time, "tzset"):
        pytest.skip("Windows can't change a running program's time zone")
    before = os.environ.get("TZ")

    def set_to(zone: str) -> None:
        os.environ["TZ"] = zone
        time.tzset()

    yield set_to
    if before is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = before
    time.tzset()


def utc(*when: int) -> float:
    """A moment in UTC, in seconds since 1970."""
    return datetime(*when, tzinfo=UTC).timestamp()


def test_the_trend_counts_days_since_2020_the_same_in_every_time_zone(time_zone):
    for zone in ("UTC0", "JST-9", "EST+5"):
        time_zone(zone)
        assert features.time_numbers(utc(2020, 1, 1), ["trend"]) == {"days_since_2020": 0.0}
        assert features.time_numbers(utc(2020, 1, 2, 12), ["trend"]) == {"days_since_2020": 1.5}
        found = features.time_numbers(utc(2026, 10, 18, 23, 30), ["trend"])
        assert found["days_since_2020"] == pytest.approx(2482 + 23.5 / 24)


def test_the_clock_reads_the_hour_and_the_day_of_the_week_in_your_time_zone(time_zone):
    sunday_night, monday_morning = utc(2026, 10, 18, 23, 30), utc(2026, 1, 12, 9, 15)
    london = "GMT0BST,M3.5.0/1,M10.5.0"  # an hour ahead from the last Sunday in March to the last in October
    zones = {
        "UTC0": [(23.5, 6.0), (9.25, 0.0)],  # 0 is Monday and 6 is Sunday
        "EST+5": [(18.5, 6.0), (4.25, 0.0)],  # five hours behind
        "JST-9": [(8.5, 0.0), (18.25, 0.0)],  # nine hours ahead, so Sunday night is Monday morning
        "IST-5:30": [(5.0, 0.0), (14.75, 0.0)],  # five and a half hours ahead
        london: [(0.5, 0.0), (9.25, 0.0)],  # on summer time in October, not in January
    }
    for zone, expected in zones.items():
        time_zone(zone)
        found = [features.time_numbers(moment, ["clock"]) for moment in (sunday_night, monday_morning)]
        assert found == [{"hour_of_day": hour, "day_of_week": day} for hour, day in expected], zone


def test_the_time_options_pick_their_numbers_from_when_the_task_started(tmp_path):
    session = ClaudeSession(tmp_path, "s1", START)
    session.prompt("Fix the flaky test")
    session.reply(30, calls_usage=usage())
    session.save(mtime=START)
    task = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=START + 86400)[0]
    info = features.TaskInfo.of(task)
    assert info.started_at == task.started_at == START  # a recorded task comes with its start time

    def chosen(config: dict | None) -> set[str]:
        return set(features.numbers(["time"], info, None, config)) & TIME_NUMBERS

    assert chosen({"options": {"time": ["trend"]}}) == {"days_since_2020"}
    assert chosen({"options": {"time": ["clock"]}}) == {"hour_of_day", "day_of_week"}
    assert chosen({"options": {"time": ["trend", "clock"]}}) == chosen({}) == chosen(None) == TIME_NUMBERS
    assert not set(features.numbers([], info, None)) & TIME_NUMBERS  # the feature is off
    unknown = features.TaskInfo(prompt="Fix it", model="m")
    assert not set(features.numbers(["time"], unknown, None)) & TIME_NUMBERS  # no start time, no numbers


def test_a_model_trains_and_forecasts_with_the_time(tmp_path, capsys, voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    records = tmp_path / "records"
    history(records, tasks=180, start=START, models=("claude-opus-5-5", "claude-sonnet-5-5"), efforts=("high", "max"))
    code = main(["create", "--yes", "--archive", f"claude-code={records}", "--features", "time", *ONE_WINDOW])
    assert code == 0, capsys.readouterr().out
    predictor = Predictor("main")
    assert predictor.config["features"] == ["prompt", "time"]
    assert predictor.config["options"] == {"time": ["trend", "clock"]}
    meta = json.loads((predictor.model_dir / "model.json").read_text())
    assert TIME_NUMBERS <= set(meta["numbers"])
    capsys.readouterr()
    assert main(["predict", "--model", "claude-opus-5-5", "--prompt", "fix the api"]) == 0
    assert json.loads(capsys.readouterr().out)["forecasts"][0]["seconds"] > 0
    # Training reads only the options the predictor chose.
    predictor.config["options"] = {"time": ["clock"]}
    row = pipeline.feature_rows(predictor, predictor.tasks()[:1])[0]
    assert set(row.numbers) & TIME_NUMBERS == {"hour_of_day", "day_of_week"}


PLUGIN = '''
from yardstick_predict import features, model_types
from yardstick_predict import model as xgboost_model

features.register_feature("words", "number", lambda info: float(len(info.prompt.split())))
features.register_feature("summary", "text", lambda info: info.extra.get("summary") or info.prompt[:40])
features.register_feature("ticket", "number", lambda info: info.extra.get("ticket"))  # past tasks have no extra


class Steady:
    """A model type that forecasts the same minute for every task, for testing the plug-in point."""

    def __init__(self):
        self.meta = {"report": {"trained_on": 0, "tested_on": 0, "windows": 0, "models": {}, "pit": {}}, "tasks": 0}

    def forecast(self, rows):
        return [{"shape": None, "direct_probability": None, "seconds": {"expected": 60.0, "distribution": None},
                 "tokens": None, "usd": None} for _ in rows]

    def served(self, model):
        return True, None

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text("{}")


model_types.register_model_type("steady", lambda tasks, rows, settings: Steady(), lambda folder: Steady())
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "my_plugin.py").write_text(textwrap.dedent(PLUGIN))
    monkeypatch.syspath_prepend(str(tmp_path / "plugins"))
    saved = dict(features._CUSTOM)
    sys.modules.pop("my_plugin", None)
    yield "my_plugin"
    sys.modules.pop("my_plugin", None)  # each command is a fresh process in real use; tests share one
    features._CUSTOM.clear()
    features._CUSTOM.update(saved)
    model_types._TYPES.pop("steady", None)


def test_your_own_features_train_and_forecast_through_a_plugin(tmp_path, capsys, voyage, monkeypatch, plugin):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    records = tmp_path / "records"
    history(records, tasks=300, start=START)
    chosen = "words,summary,ticket"
    flags = ["--plugin", plugin, "--features", chosen, *ONE_WINDOW]
    code = main(["create", "--yes", "--archive", f"claude-code={records}", *flags])
    out = capsys.readouterr().out
    assert code == 0, out
    predictor = Predictor("main")
    assert predictor.config["features"] == ["prompt", "words", "summary", "ticket"]
    assert predictor.config["plugins"] == [plugin]
    assert len(predictor.embedded("summary")) == 300
    meta = json.loads((predictor.model_dir / "model.json").read_text())
    assert "words" in meta["numbers"] and "prompt_lines" in meta["numbers"]  # with the ones always used
    assert meta["text_fields"] == ["prompt", "summary"]
    # A feature that reads only a request's extra has no value for any past task, so nothing can be learned from it.
    assert meta["features_without_values"] == ["ticket"]

    request = {
        "version": 1,
        "tasks": [
            {"id": "a", "prompt": "fix the api", "models": ["claude-opus-5-5"], "extra": {"summary": "a small fix"}}
        ],
    }
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(request)))
    assert main(["predict"]) == 0
    assert json.loads(capsys.readouterr().out)["forecasts"][0]["seconds"] > 0
    assert any("a small fix" in r["input"] for r in voyage.requests)


def test_your_own_model_type_plugs_in(tmp_path, capsys, voyage, monkeypatch, plugin):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    records = tmp_path / "records"
    history(records, tasks=60, start=START)
    code = main(
        ["create", "--yes", "--archive", f"claude-code={records}", "--plugin", plugin, "--model-type", "steady"]
    )
    assert code == 0, capsys.readouterr().out
    capsys.readouterr()
    assert main(["predict", "--model", "anything", "--prompt", "fix it"]) == 0
    forecast = json.loads(capsys.readouterr().out)["forecasts"][0]
    assert forecast["seconds"] == 60.0 and forecast["tokens"] is None
    with pytest.raises(ValueError, match="Unknown model type"):
        model_types.get("nonexistent")


def test_a_skill_description_is_read_from_the_frontmatter_in_every_yaml_form(skills_home):
    files = {
        "plain": "---\nname: plain\ndescription: Starts here\n  and goes on.\n---\nBody",
        "next-line": "---\ndescription:\n  Written on the next line\n  and the one after.\nname: next-line\n---\n",
        "double": '---\nname: double\ndescription: "Quoted, with a\\ttab and\n  a second line"\n---\n',
        "single": "---\nname: single\ndescription: 'It''s quoted\n  over two lines'\n---\n",
        "indented": "---\nname: indented\ndescription: |2-\n    Two more spaces.\n---\n",
        "comment": "---\nname: comment\ndescription: Plain text # a note, not part of it\n---\n",
        "crlf": "---\r\nname: crlf\r\ndescription: >\r\n  Folded over\r\n  two lines.\r\n---\r\n",
        "empty": "---\ndescription:\nname: empty\n---\n",
        "body-only": "---\nname: body-only\n---\ndescription: in the body, not the frontmatter\n",
        "no-frontmatter": "description: there is no frontmatter here\n",
    }
    for name, text in files.items():
        install(skills_home, name, text)
    assert features.skill_descriptions() == {
        "plain": "Starts here and goes on.",
        "next-line": "Written on the next line and the one after.",
        "double": "Quoted, with a\ttab and a second line",
        "single": "It's quoted over two lines",
        "indented": "Two more spaces.",
        "comment": "Plain text",
        "crlf": "Folded over two lines.",
    }


def test_skills_installed_as_links_are_found_and_a_link_that_loops_back_ends(skills_home, tmp_path):
    real = tmp_path / "elsewhere"
    install(real, "linked", "---\nname: linked\ndescription: Installed as a link.\n---\n")
    install(skills_home, "plain", "---\nname: plain\ndescription: In its own folder.\n---\n")
    try:
        os.symlink(real / "linked", skills_home / "linked", target_is_directory=True)
        os.symlink(skills_home, skills_home / "plain" / "loop", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this computer doesn't let a test make links")
    assert features.skill_descriptions() == {"linked": "Installed as a link.", "plain": "In its own folder."}


def test_no_skills_means_no_text_and_no_look_through_the_skill_folders(skills_home, monkeypatch):
    monkeypatch.setattr(features, "_skill_files", lambda roots: pytest.fail("the skill folders were read"))
    assert features.skills_text([]) == "" and features.skills_text(["", "  "]) == ""
    assert features.text("skills", features.TaskInfo(prompt="Fix it", model="m")) == ""


def test_another_process_reuses_what_the_skill_folders_held_until_they_change(skills_home, monkeypatch):
    path = install(skills_home, "tdd", "---\nname: tdd\ndescription: Write the test first.\n---\n")
    assert features.skills_text(["tdd"]) == "tdd: Write the test first."
    real = features._skill_files
    features.skill_descriptions.cache_clear()  # as in another process
    monkeypatch.setattr(features, "_skill_files", lambda roots: pytest.fail("the skill folders were read again"))
    assert features.skills_text(["tdd"]) == "tdd: Write the test first."

    path.write_text("---\nname: tdd\ndescription: Write the failing test first.\n---\n")
    changed = os.stat(path).st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(changed, changed))
    monkeypatch.setattr(features, "_skill_files", real)
    features.skill_descriptions.cache_clear()
    assert features.skills_text(["tdd"]) == "tdd: Write the failing test first."
    install(skills_home, "review", "---\nname: review\ndescription: Read the change.\n---\n")
    features.skill_descriptions.cache_clear()
    assert features.skills_text(["review"]) == "review: Read the change."  # a new skill is found too


def test_number_features_take_numpy_numbers_but_not_bools_or_numbers_that_arent_finite(monkeypatch):
    monkeypatch.setattr(features, "_CUSTOM", {})
    values = {
        "an_int": np.int64(3),
        "a_float": np.float32(1.5),
        "a_fraction": Fraction(1, 4),
        "a_bool": True,
        "a_numpy_bool": np.True_,
        "not_a_number": float("nan"),
        "endless": np.inf,
        "a_text": "4",
    }
    for name, value in values.items():
        features.register_feature(name, "number", lambda info, value=value: value)
    found = features.numbers(list(values), features.TaskInfo(prompt="Fix it", model="m"), None)
    assert {name: value for name, value in found.items() if name in values} == {
        "an_int": 3.0,
        "a_float": 1.5,
        "a_fraction": 0.25,
    }
    assert all(type(value) is float for value in found.values())


def test_a_prompt_is_embedded_as_the_records_keep_it(monkeypatch):
    def text(prompt: str) -> str:
        return features.text("prompt", features.TaskInfo(prompt=prompt, model="m"))

    assert text("  fix the login page \n") == "fix the login page"
    assert len(text("x" * (TEXT_CHARS + 500))) == TEXT_CHARS
    cut_at_a_space = "a" * (TEXT_CHARS - 1) + "  and more"
    stored = cut_at_a_space.strip()[:TEXT_CHARS]  # as the reader keeps a task's prompt
    assert text(cut_at_a_space) == text(stored) == "a" * (TEXT_CHARS - 1)


def test_sonnet_4_and_4_5_cost_more_above_200000_input_tokens():
    below = prices.call_cost("claude-sonnet-4-5-20250929", 1000, 0, 0, 199_000, 1000)
    assert below == pytest.approx((1000 * 3 + 199_000 * 0.3 + 1000 * 15) / 1e6)
    above = prices.call_cost("claude-sonnet-4-5[1m]", 1000, 0, 0, 600_000, 1000)
    assert above == pytest.approx((1000 * 6 + 600_000 * 0.6 + 1000 * 22.5) / 1e6)
    writes = prices.call_cost("claude-sonnet-4", 100_000, 60_000, 40_001, 0, 0)  # cache writes count toward it
    assert writes == pytest.approx((100_000 * 6 + 60_000 * 7.5 + 40_001 * 12) / 1e6)
    later = prices.call_cost("claude-sonnet-4-6", 1000, 0, 0, 600_000, 1000)  # 4.6 and later have no premium
    assert later == pytest.approx((1000 * 3 + 600_000 * 0.3 + 1000 * 15) / 1e6)


def test_codex_models_have_their_list_prices():
    listed = {
        "gpt-5-codex": (1.25, 10, 0.125),
        "gpt-5.1-codex": (1.25, 10, 0.125),
        "gpt-5.1-codex-max": (1.25, 10, 0.125),
        "gpt-5.1-codex-mini": (0.25, 2, 0.025),
        "gpt-5.2-codex": (1.75, 14, 0.175),
    }
    for model, (new_input, output, cache_read) in listed.items():
        price = prices.price(model)
        assert (price["input"], price["output"], price["cache_read"]) == (new_input, output, cache_read), model


def test_a_model_list_in_an_odd_shape_is_refused_or_skipped_cleanly():
    for listing in ([], "text", {"data": None}, {"data": "models"}):
        with pytest.raises(ValueError):
            benchmarks.parse(listing)
    odd = {
        "id": "openai/gpt-5",
        "pricing": ["not", "a", "price"],
        "top_provider": "somewhere",
        "created": True,
        "supported_parameters": "reasoning",
        "context_length": float("inf"),
    }
    table = benchmarks.parse({"data": [None, "x", {"id": 5}, odd]})
    assert table == {"gpt-5": {"benchmark_reasons": 0.0}}


def test_a_model_list_download_cut_off_midway_is_a_connection_problem(monkeypatch):
    class CutOff:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            raise http.client.IncompleteRead(b'{"data": [')

    monkeypatch.setattr(benchmarks.urllib.request, "urlopen", lambda request, timeout: CutOff())
    with pytest.raises(OSError, match="broke"):
        benchmarks.download()


def test_a_plugin_that_named_its_own_feature_time_or_conversation_keeps_it(monkeypatch):
    # Both names became built-in features after 0.3.0, which let plugins use them.
    monkeypatch.setattr(features, "_CUSTOM", {})
    features.register_feature("time", "number", lambda info: 7)
    features.register_feature("conversation", "text", lambda info: "a summary")
    info = features.TaskInfo(prompt="Fix it", model="m", started_at=START)
    found = features.numbers(["time"], info, None)
    assert found["time"] == 7.0 and "days_since_2020" not in found  # the plugin's, not the built-in one
    assert not features.built_in("conversation", ["conversation"])
    assert features.is_text("conversation") and features.text("conversation", info) == "a summary"
    with pytest.raises(ValueError):
        features.register_feature("context", "text", lambda info: "x")  # built in since before plugins came
