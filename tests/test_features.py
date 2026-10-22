from __future__ import annotations

import json
import sys
import textwrap

import pytest
from builders import ClaudeSession, history, usage
from conftest import KEY

from yardstick_predict import benchmarks, features, model_types
from yardstick_predict.archives import read_archive
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor

START = 1_780_000_000.0


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


PLUGIN = '''
from yardstick_predict import features, model_types
from yardstick_predict import model as xgboost_model

features.register_feature("words", "number", lambda info: float(len(info.prompt.split())))
features.register_feature("summary", "text", lambda info: info.extra.get("summary") or info.prompt[:40])


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
    code = main(
        ["create", "--yes", "--archive", f"claude-code={records}", "--plugin", plugin, "--features", "words,summary"]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt", "words", "summary"] and predictor.config["plugins"] == [plugin]
    assert len(predictor.embedded("summary")) == 300
    meta = json.loads((predictor.model_dir / "model.json").read_text())
    assert meta["numbers"] == ["words"] and meta["text_fields"] == ["prompt", "summary"]

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
