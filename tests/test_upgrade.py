"""Upgrading: a model the previous release saved keeps forecasting, as that release did, until it's trained again.

The predictor in fixtures/predictor-0.3.0 was saved by 0.3.0's own code, and fixtures/forecasts-0.3.0.json
holds what 0.3.0 answered to some requests. fixtures/README.md says how they were made.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from test_cli import quick, quick_predictor  # noqa: F401 (quick is a fixture: a model type that trains at once)
from test_forecast import schema

from yardstick_predict import cli, embeddings, features, forecast, llm, model_0_3, model_types, pipeline, store
from yardstick_predict import model as M
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KEPT = json.loads((FIXTURES / "forecasts-0.3.0.json").read_text())
"""The requests 0.3.0 answered, what it answered, and the stand-ins it answered with."""


@pytest.fixture
def saved_by_0_3() -> Predictor:
    """The predictor 0.3.0 saved, in this test's own yardstick folder. Its records are nowhere, so a refresh
    finds no new tasks."""
    shutil.copytree(FIXTURES / "predictor-0.3.0", store.predictor_folder("default"))
    return Predictor("default")


@pytest.fixture
def as_0_3_forecast(monkeypatch):
    """The stand-ins 0.3.0 forecast with: the same moment, the same vectors and the same LLM analysis.

    A text 0.3.0 didn't embed has no vector here, so a forecast that needs one fails for want of a Voyage key.
    """
    vectors, labels = KEPT["vectors"], KEPT["labels"]
    monkeypatch.setattr(forecast, "time", SimpleNamespace(time=lambda: KEPT["now"]))
    monkeypatch.setattr(embeddings, "cached", lambda model, texts: {t: vectors[t] for t in texts if t in vectors})
    monkeypatch.setattr(
        llm, "analyze_requests", lambda predictor, tasks: {t.id: labels[t.id] for t in tasks if t.id in labels}
    )


def same(found, expected) -> None:
    """Fail unless two answers match: every field, with each number equal or one unit apart in its last digit.

    The answer rounds each number: seconds to a tenth, tokens to a whole number and the rest to 6 decimals. On the
    computer that made the fixture they're equal. Another kind of processor may add up the last digits of a sum
    differently, so a number that lands just on a rounding edge comes out one unit apart. It says every field
    that differs, not just the first.
    """
    differ = list(apart(found, expected, "answer"))
    assert not differ, "\n".join(differ)


def apart(found, expected, path: str):
    """Where two answers differ, one line each."""
    if isinstance(expected, dict):
        if not isinstance(found, dict) or sorted(found) != sorted(expected):
            yield f"{path}: the fields {sorted(found) if isinstance(found, dict) else found!r} != {sorted(expected)}"
            return
        for key in expected:
            yield from apart(found[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(found, list) or len(found) != len(expected):
            yield f"{path}: {found!r} != {expected!r}"
            return
        for n, (one, other) in enumerate(zip(found, expected)):
            yield from apart(one, other, f"{path}[{n}]")
    elif isinstance(expected, float) or (isinstance(expected, int) and path.endswith(".tokens")):
        unit = 0.1 if path.endswith(".seconds") else 1 if path.endswith(".tokens") else 1e-6
        close = isinstance(found, (int, float)) and math.isclose(found, expected, rel_tol=1e-6, abs_tol=unit * 1.000001)
        if not close:
            yield f"{path}: {found!r} != {expected!r}"
    elif found != expected:
        yield f"{path}: {found!r} != {expected!r}"


def ask_what_0_3_was_asked(predictor: Predictor) -> dict:
    """This version's answer to the requests 0.3.0 answered."""
    tasks, distribution, _ = forecast.parse(KEPT["request"])
    return forecast.answer(predictor, tasks, distribution)


def test_a_model_saved_by_0_3_forecasts_exactly_as_0_3_did(saved_by_0_3, as_0_3_forecast):
    model = M.load(saved_by_0_3.model_dir)
    # 0.3 trained versions of the model. Each AI model uses its own, and a task LLM analysis couldn't read
    # goes to the same version without it.
    assert isinstance(model, model_0_3.Choice)
    assert sorted(model.versions) == ["conversation", "conversation-llm", "plain"]
    uses = model.meta["uses"]
    assert uses["claude-opus-5-5"] == "conversation-llm" and uses["gpt-6.1-sol"] == "plain"
    assert model.meta["without_llm"]["claude-opus-5-5"]["version"] == "conversation"
    answered = ask_what_0_3_was_asked(saved_by_0_3)
    jsonschema.validate(answered, schema("forecast-response.v1.json"))
    same(answered, KEPT["answer"])
    given = {(f["task"], f["model"]) for f in answered["forecasts"] if f["seconds"] is not None}
    assert {("read", "claude-opus-5-5"), ("unread", "claude-opus-5-5"), ("read", "gpt-6.1-sol")} <= given


@pytest.mark.parametrize("listed", [True, False])
@pytest.mark.parametrize("kind", ["number", "text"])
def test_a_model_saved_by_0_3_gets_the_conversation_whatever_plugin_takes_its_name(
    saved_by_0_3, as_0_3_forecast, monkeypatch, kind, listed
):
    # 0.3 gave every model the conversation's numbers, and let a plugin name a feature of its own conversation.
    # Today a plugin's feature of that name turns the built-in one off, but only for models trained today.
    monkeypatch.setattr(features, "_CUSTOM", {})
    features.register_feature("conversation", kind, lambda info: 2.0 if kind == "number" else "a summary")
    if listed:  # the predictor chose the plugin's feature in 0.3
        saved_by_0_3.config["features"] = ["prompt", "context", "llm", "conversation"]
    assert not features.built_in("conversation", pipeline.chosen_features(saved_by_0_3))
    same(ask_what_0_3_was_asked(saved_by_0_3), KEPT["answer"])


def test_todays_rows_carry_every_number_and_embedding_a_model_saved_by_0_3_reads(saved_by_0_3):
    model = M.load(saved_by_0_3.model_dir)
    # Its settings get the conversation feature, since every predictor read the conversation then.
    assert pipeline.chosen_features(saved_by_0_3) == ["prompt", "conversation", "context", "llm"]
    rows = pipeline.feature_rows(saved_by_0_3, saved_by_0_3.tasks())
    for name, version in model.versions.items():
        prepared = version.prepare(rows)
        assert set(version.meta["numbers"]) <= {n for r in prepared for n in r.numbers}, name
        assert set(version.meta["text_fields"]) <= {n for r in prepared for n in r.vectors}, name


def test_a_single_version_saved_by_0_3_loads_as_one_model(saved_by_0_3):
    # When every AI model used the same version, 0.3 saved only that one, as each version's own folder is saved.
    choice = M.load(saved_by_0_3.model_dir)
    alone = M.load(saved_by_0_3.model_dir / "plain")
    assert isinstance(alone, M.Model) and alone.meta["format"] == model_types.PREVIOUS_FORMAT
    rows = [r for r in pipeline.feature_rows(saved_by_0_3, saved_by_0_3.tasks()) if r.model == "gpt-6.1-sol"]
    assert alone.forecast(rows[-10:]) == choice.forecast(rows[-10:])  # GPT's tasks use the plain version


def test_a_model_saved_by_0_3_is_saved_again_as_0_3_saved_it(saved_by_0_3, tmp_path):
    model = M.load(saved_by_0_3.model_dir)
    model.save(tmp_path / "model")
    again = M.load(tmp_path / "model")
    assert isinstance(again, model_0_3.Choice) and again.meta == model.meta
    rows = pipeline.feature_rows(saved_by_0_3, saved_by_0_3.tasks()[-20:])
    assert again.forecast(rows) == model.forecast(rows)


def test_status_says_a_model_saved_by_0_3_still_forecasts(saved_by_0_3, capsys):
    assert pipeline.from_previous_version(saved_by_0_3) and not pipeline.outdated(saved_by_0_3)
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "made by the previous version, and it still forecasts" in out and "yardstick-predict refresh" in out
    assert "Trained on 192 tasks." in out and re.search(r"claude-opus-5-5\s+forecasts on", out)
    assert main(["status", "--json"]) == 0
    entry = json.loads(capsys.readouterr().out)["predictors"][0]
    assert entry["report"]["models"]["claude-opus-5-5"]["served"]


def test_a_refresh_trains_a_model_saved_by_0_3_again_in_this_versions_format(saved_by_0_3, as_0_3_forecast, capsys):
    assert main(["refresh", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "made by the previous version. It still forecasts, and training it again brings" in out
    assert "Trained on 192 tasks." in out
    again = Predictor("default")
    assert not pipeline.from_previous_version(again) and not pipeline.outdated(again)
    assert json.loads((again.model_dir / "model.json").read_text())["format"] == model_types.BUILT_IN_FORMAT
    assert isinstance(M.load(again.model_dir), M.WithoutLLM)  # and without LLM analysis, for tasks it can't read
    saved = json.loads((again.dir / "predictor.json").read_text())
    assert saved["format"] == store.FORMAT and saved["features"] == ["prompt", "conversation", "context", "llm"]
    forecasts = {(f["task"], f["model"]): f for f in ask_what_0_3_was_asked(again)["forecasts"]}
    assert forecasts[("read", "claude-opus-5-5")]["seconds"] > 0
    assert main(["refresh", "--yes"]) == 0 and "Nothing new. The model is up to date." in capsys.readouterr().out


def test_settings_a_newer_version_saved_are_read_and_never_saved(saved_by_0_3, as_0_3_forecast, capsys, monkeypatch):
    path = saved_by_0_3.dir / "predictor.json"
    newer = json.loads(path.read_text())
    newer.update(format=store.FORMAT + 1, features=["prompt", "conversation", "context", "llm", "weather"])
    newer["something_new"] = {"this version": "doesn't know it"}
    path.write_text(json.dumps(newer, indent=2) + "\n")
    before = path.read_text()
    predictor = Predictor("default")
    assert predictor.newer and pipeline.unknown_features(predictor) == ["weather"]
    predictor.config["needs_training"] = True
    predictor.save_config()
    assert path.read_text() == before

    # The feature this version doesn't know is skipped, and the model forecasts as before.
    same(ask_what_0_3_was_asked(predictor), KEPT["answer"])
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert out.count("A newer version of yardstick-predict saved this predictor.") == 1
    assert out.count("Skipping weather: this version doesn't know it.") == 1
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    assert main(["predict", "add a search page with tests and docs"]) == 0
    out = capsys.readouterr().out
    assert "A newer version of yardstick-predict saved this predictor." in out
    assert "Skipping weather: this version doesn't know it." in out

    # A refresh would save it the way this older version does, so it changes nothing.
    assert main(["refresh", "--yes"]) == 1
    error = capsys.readouterr().err
    assert "saved by a newer version" in error and "Upgrade with: uv tool upgrade yardstick-predict" in error
    assert path.read_text() == before and M.load(predictor.model_dir).meta["format"] == model_types.PREVIOUS_FORMAT


@pytest.mark.usefixtures("quick")
def test_a_feature_this_version_doesnt_know_is_skipped_with_a_note(capsys):
    predictor = quick_predictor()
    predictor.config["features"] = ["prompt", "weather", "context"]
    predictor.save_config()
    assert pipeline.chosen_features(predictor) == ["prompt", "context"]
    assert main(["refresh", "--yes"]) == 0
    out = capsys.readouterr().out
    assert out.count("Skipping weather: this version doesn't know it. A newer version or a plugin added it.") == 1
    assert "Nothing new" in out
    assert Predictor("default").config["features"] == ["prompt", "weather", "context"]  # kept for what knows it


def test_settings_this_version_cant_read_at_all_ask_for_an_upgrade(capsys):
    folder = store.predictor_folder("default")
    folder.mkdir(parents=True)
    (folder / "predictor.json").write_text(json.dumps({"format": store.FORMAT + 1, "sources": []}))
    assert main(["status"]) == 1
    out = capsys.readouterr().out
    assert "saved by a newer version" in out and "uv tool upgrade yardstick-predict" in out
    assert "delete the folder" not in out  # nothing about it is broken
