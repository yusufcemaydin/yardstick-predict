from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from conftest import KEY, fake_vector
from test_model import ONE_WINDOW, rows_for, synthetic

from yardstick_predict import features, pipeline
from yardstick_predict import model as M
from yardstick_predict.forecast import RequestError, answer, parse
from yardstick_predict.store import Predictor

SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


def schema(name: str) -> dict:
    return json.loads((SCHEMAS / name).read_text())


def request(**task) -> dict:
    return {"version": 1, "tasks": [{"id": "t1", "prompt": "Add a login page", "models": ["claude-opus-5-5"], **task}]}


@pytest.mark.parametrize(
    "bad, words",
    [
        ({"tasks": []}, "version"),
        ({"version": 2, "tasks": [{"id": "a", "prompt": "x", "models": ["m"]}]}, "version"),
        ({"version": 1, "tasks": []}, "at least one task"),
        ({"version": 1, "tasks": [{"id": "a", "prompt": "", "models": ["m"]}]}, "prompt"),
        ({"version": 1, "tasks": [{"id": "a", "prompt": "x", "models": []}]}, "models"),
        ({"version": 1, "tasks": [{"id": "a", "prompt": "x", "models": ["m"]}] * 2}, "unique"),
        ({"version": 1, "tasks": [{"id": "a", "prompt": "x", "models": ["m"], "origin": "robot"}]}, "origin"),
        ({"version": 1, "tasks": [{"id": "a", "prompt": "x", "models": ["m"], "skills": "tdd"}]}, "skills"),
        ({"version": 1, "tasks": [{"id": "a", "prompt": "x", "models": ["m"], "extra": [1]}]}, "extra"),
    ],
)
def test_bad_requests_get_a_plain_reason(bad, words):
    with pytest.raises(RequestError) as caught:
        parse(bad)
    assert caught.value.code == "invalid_request"
    assert words in caught.value.message


def test_good_requests_match_the_published_schema():
    full = request(context="Earlier we added a signup page.", effort="max", skills=["tdd"], extra={"repo_size": 3})
    jsonschema.validate(full, schema("forecast-request.v1.json"))
    tasks, distribution, predictor = parse(request(origin="person", skills=["tdd"]))
    assert tasks[0].origin == "person" and tasks[0].skills == ["tdd"] and not distribution and predictor is None


TASK = {"id": "a", "prompt": "Fix the flaky test", "models": ["m"]}


@pytest.mark.parametrize(
    "changes, task_changes",
    [
        ({"version": True}, {}),  # JSON's true isn't the number 1
        ({"version": 1.0}, {}),
        ({"version": "1"}, {}),
        ({"distribution": None}, {}),
        ({"distribution": 1}, {}),
        ({"predictor": None}, {}),
        ({"predictor": ""}, {}),
        ({"predictor": 3}, {}),
        ({}, {"prompt": "   \n"}),
        ({}, {"prompt": " Fix it "}),
        ({}, {"models": ["m", "m"]}),
        ({}, {"models": [""]}),
        ({}, {"extra": []}),
        ({}, {"extra": 0}),
        ({}, {"extra": None}),
        ({}, {"extra": {"repo_size": 3}}),
        ({}, {"skills": ""}),
        ({}, {"skills": {}}),
        ({}, {"skills": None}),
        ({}, {"skills": [3]}),
        ({}, {"context": None, "effort": None, "agent": None, "project": None, "origin": None}),
        ({}, {"context": 3}),
        ({}, {"effort": False}),
        ({}, {"origin": "robot"}),
        ({}, {"origin": "person"}),
        ({}, {"id": ""}),
    ],
)
def test_the_request_checks_and_the_request_schema_agree(changes, task_changes):
    sent = {"version": 1, "tasks": [{**TASK, **task_changes}], **changes}
    try:
        jsonschema.validate(sent, schema("forecast-request.v1.json"))
        schema_takes_it = True
    except jsonschema.ValidationError:
        schema_takes_it = False
    try:
        parse(sent)
        code_takes_it = True
    except RequestError:
        code_takes_it = False
    assert code_takes_it == schema_takes_it


def test_a_prompt_and_its_context_are_prepared_as_training_prepares_them(stub):
    from yardstick_predict.archives import TEXT_CHARS

    predictor, model = stub
    long = "  " + "word " * TEXT_CHARS
    context = "Earlier we talked. " * 2000 + "\n\n"
    tasks, _, _ = parse(request(prompt=long, context=context))
    assert tasks[0].prompt == long.strip()[:TEXT_CHARS]
    assert tasks[0].context == context[-TEXT_CHARS:].strip()
    answer(predictor, tasks)
    assert model.rows[0].prompt_chars == TEXT_CHARS  # the prompt's length feature counts what training would keep


def test_a_model_named_twice_is_forecast_once():
    tasks, _, _ = parse(request(models=["claude-opus-5-5", "gpt-6.1-sol", "claude-opus-5-5"]))
    assert tasks[0].models == ["claude-opus-5-5", "gpt-6.1-sol"]


def test_every_error_code_is_in_the_answer_schema_and_the_format_doc():
    from yardstick_predict.forecast import ERRORS

    codes = schema("forecast-response.v1.json")["properties"]["error"]["properties"]["code"]["enum"]
    assert sorted(codes) == sorted(ERRORS)
    doc = (SCHEMAS.parent / "docs" / "format.md").read_text()
    assert all(f"| `{code}` |" in doc for code in ERRORS)


@pytest.fixture(scope="module")
def trained_model():
    """A model trained once. The tests only forecast with it, so they can share it."""
    tasks = synthetic(900)
    vectors = {"prompt": {t.id: np.array(fake_vector(t.prompt), dtype=np.float32) for t in tasks}}
    return M.train(tasks, rows_for(tasks, vectors), ONE_WINDOW)


@pytest.fixture
def trained(voyage, monkeypatch, trained_model):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.config["features"] = ["prompt"]
    predictor.save_config()
    trained_model.save(predictor.model_dir)
    return predictor


def test_answers_give_expected_values_and_say_why_when_they_cant(trained):
    tasks, _, _ = parse(request(models=["claude-opus-5-5", "gpt-6.1-sol", "brand-new-model"]))
    result = answer(trained, tasks)
    jsonschema.validate(result, schema("forecast-response.v1.json"))
    by_model = {f["model"]: f for f in result["forecasts"]}
    opus = by_model["claude-opus-5-5"]
    assert "reason" not in opus and opus["effort"] == "high"
    assert opus["seconds"] > 0 and opus["tokens"] > 0 and opus["usd"] > 0
    assert isinstance(opus["tokens"], int)
    assert by_model["brand-new-model"] == {
        "task": "t1",
        "model": "brand-new-model",
        "seconds": None,
        "tokens": None,
        "usd": None,
        "reason": "unknown_model",
    }


def test_distributions_come_on_request_and_agree_with_the_expected_values(trained):
    tasks, distribution, _ = parse({**request(), "distribution": True})
    result = answer(trained, tasks, distribution)
    jsonschema.validate(result, schema("forecast-response.v1.json"))
    forecast = result["forecasts"][0]
    shape = forecast["distribution"]
    assert shape["shape"] == "two_part_lognormal"
    curve = shape["seconds"]
    again = M.expected(
        shape["direct_probability"],
        (curve["direct"]["mu"], curve["direct"]["sigma"]),
        (curve["work"]["mu"], curve["work"]["sigma"]),
    )
    assert forecast["seconds"] == pytest.approx(again, rel=1e-3)


def test_a_longer_prompt_is_forecast_to_take_longer(trained):
    short, _, _ = parse(request(prompt="fix"))
    long, _, _ = parse(request(prompt="word " * 70))
    assert answer(trained, long)["forecasts"][0]["seconds"] > answer(trained, short)["forecasts"][0]["seconds"]


class Stub:
    """A stand-in for a trained model. It forecasts every AI model, keeps the rows it gets and answers `result`."""

    def __init__(self):
        self.meta: dict = {"defaults": {}}
        self.rows: list = []
        self.result: dict = {"shape": None, "direct_probability": None, "seconds": None, "tokens": None, "usd": None}

    def served(self, model):
        return True, None

    def forecast(self, rows):
        self.rows += rows
        return [self.result for _ in rows]


@pytest.fixture
def stub(voyage, monkeypatch):
    """A predictor whose model is a Stub."""
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.model_dir.mkdir()
    (predictor.model_dir / "model.json").write_text("{}")
    model = Stub()
    monkeypatch.setattr(pipeline, "load_model", lambda _: model)
    return predictor, model


def test_your_own_text_features_see_each_rows_model_effort_and_agent(stub, voyage, monkeypatch):
    predictor, model = stub
    model.meta["defaults"] = {
        "claude-opus-5-5": {"effort": "high", "agent": "claude-code"},
        "gpt-6.1-sol": {"effort": "medium", "agent": "codex"},
    }
    monkeypatch.setattr(features, "_CUSTOM", {})
    features.register_feature("setup", "text", lambda info: f"{info.model} {info.effort} {info.agent}")
    predictor.config["features"] = ["prompt", "setup"]
    predictor.save_config()
    first = {"id": "a", "prompt": "Add a login page", "models": ["claude-opus-5-5", "gpt-6.1-sol"]}
    second = {**first, "id": "b", "models": ["claude-opus-5-5"], "effort": "max", "agent": "pi"}
    tasks, _, _ = parse({"version": 1, "tasks": [first, second]})
    answer(predictor, tasks)
    # Like in training: the row's own model, and the request's effort and agent or the model's usual ones.
    setups = ["claude-opus-5-5 high claude-code", "gpt-6.1-sol medium codex", "claude-opus-5-5 max pi"]
    assert len(model.rows) == len(setups)
    for row, setup in zip(model.rows, setups):
        assert np.array_equal(row.vectors["setup"], np.array(fake_vector(setup), dtype=np.float32))
    sent = [text for r in voyage.requests for text in r["input"]]
    assert sorted(sent) == sorted(["Add a login page", *setups])  # each text is sent once, the prompt too


def test_a_forecast_reads_the_conversation_only_with_the_conversation_feature(stub):
    predictor, model = stub
    tasks, _, _ = parse(request(session="s1", conversation_tokens=40_000, origin="person"))
    for chosen, reads in ((["prompt"], False), (["prompt", "conversation"], True)):
        predictor.config["features"] = chosen
        model.rows.clear()
        answer(predictor, tasks)
        numbers = model.rows[0].numbers
        assert ("session_tasks_before" in numbers) == reads and ("conversation_tokens" in numbers) == reads
        assert "project_tasks_before" in numbers  # the project's recent tasks are always read


def test_numbers_that_json_cant_hold_become_null(stub):
    predictor, model = stub
    curve = {"direct": {"mu": 1.0, "sigma": 0.5}, "work": {"mu": 4.0, "sigma": 1.0}}
    model.result = {
        "shape": "two_part_lognormal",
        "direct_probability": math.nan,
        "seconds": {"expected": math.inf, "distribution": {**curve, "direct": {"mu": math.nan, "sigma": 0.5}}},
        "tokens": {"expected": 1000.0, "distribution": {**curve, "work": {"mu": 4.0, "sigma": math.inf}}},
        "usd": {"expected": math.nan, "distribution": curve},
    }
    tasks, distribution, _ = parse({**request(), "distribution": True})
    result = answer(predictor, tasks, distribution)
    json.dumps(result, allow_nan=False)  # strict JSON readers reject NaN and infinity
    jsonschema.validate(result, schema("forecast-response.v1.json"))
    forecast = result["forecasts"][0]
    assert (forecast["seconds"], forecast["tokens"], forecast["usd"]) == (None, 1000, None)
    shape = forecast["distribution"]
    assert shape["direct_probability"] is None and shape["seconds"] is None and shape["tokens"] is None
    assert shape["usd"] == curve


def test_a_forecast_embeds_only_the_texts_the_model_can_use(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.config["features"] = ["prompt", "context", "skills"]
    predictor.save_config()
    tasks = synthetic(300)
    few = tasks[: M.MIN_NEIGHBORS - 1]  # too few past contexts to compare with, and no task used a skill
    vectors = {
        "prompt": {t.id: np.array(fake_vector(t.prompt), dtype=np.float32) for t in tasks},
        "context": {t.id: np.array(fake_vector("we talked before"), dtype=np.float32) for t in few},
    }
    M.train(tasks, rows_for(tasks, vectors), ONE_WINDOW).save(predictor.model_dir)
    found, _, _ = parse(request(context="Earlier we added a signup page.", skills=["tdd"]))
    forecasts = answer(predictor, found)["forecasts"]
    assert [r["input"] for r in voyage.requests] == [["Add a login page"]]  # only the prompt goes to Voyage
    assert forecasts[0]["seconds"] > 0 and forecasts == answer(predictor, parse(request())[0])["forecasts"]


def test_an_untrained_predictor_says_so():
    predictor = Predictor.create("empty", [], {"model": "voyage-4", "dimensions": 256})
    tasks, _, _ = parse(request())
    with pytest.raises(RequestError) as caught:
        answer(predictor, tasks)
    assert caught.value.code == "not_trained"
    jsonschema.validate(caught.value.as_answer(), schema("forecast-response.v1.json"))


def test_an_effort_level_reaches_a_model_that_takes_one_and_a_model_of_your_own_needs_none(stub):
    predictor, model = stub
    asked = []

    class Levels(Stub):
        def served(self, model, effort=None):
            asked.append(effort)
            return (False, "not_enough_data") if effort == "max" else (True, None)

    levels = Levels()
    tasks, _, _ = parse({"version": 1, "tasks": [{**TASK, "effort": "max"}, {**TASK, "id": "b"}]})
    forecasts = answer(predictor, tasks, model=levels)["forecasts"]
    assert asked == ["max", None]  # a request with no level is forecast at the model's usual one
    assert forecasts[0]["reason"] == "not_enough_data" and forecasts[0]["effort"] == "max"
    assert "reason" not in forecasts[1]
    assert len(answer(predictor, tasks)["forecasts"]) == 2  # Stub's served takes only the model's name


def test_a_feature_no_plugin_adds_anymore_is_a_plugin_problem(stub):
    predictor, _ = stub
    predictor.config["features"] = ["prompt", "a_feature_from_a_plugin"]
    with pytest.raises(RequestError) as caught:
        answer(predictor, parse(request())[0])
    assert caught.value.code == "plugin_failed" and "a_feature_from_a_plugin" in caught.value.message


def test_a_free_model_costs_nothing_and_a_model_with_no_price_has_no_cost(stub, monkeypatch):
    from yardstick_predict import prices

    predictor, model = stub
    listed = {
        "free-model": {"input": 0, "output": 0, "cache_read": 0},
        "free-until-long-prompts": {"input": 0, "output": 0, "long_context": {"above": 200_000, "input": 1}},
    }
    monkeypatch.setattr(prices, "price", lambda name: listed.get(name))
    curve = {"direct": {"mu": 1.0, "sigma": 0.5}, "work": {"mu": 4.0, "sigma": 1.0}}
    # The model drops a free model's usd head, since it can't learn $0 on a log scale.
    model.result = {
        "shape": "two_part_lognormal",
        "direct_probability": 0.2,
        "seconds": {"expected": 60.0, "distribution": curve},
        "tokens": None,
        "usd": None,
    }
    tasks, _, _ = parse(request(models=["free-model", "free-until-long-prompts", "unlisted-model"]))
    result = answer(predictor, tasks, distribution=True)
    jsonschema.validate(result, schema("forecast-response.v1.json"))
    by_model = {f["model"]: f for f in result["forecasts"]}
    assert by_model["free-model"]["usd"] == 0 and by_model["free-model"]["distribution"]["usd"] is None
    assert by_model["free-until-long-prompts"]["usd"] is None and by_model["unlisted-model"]["usd"] is None


def test_a_task_llm_analysis_couldnt_read_gets_the_verdict_of_the_model_without_it(stub, monkeypatch):
    predictor, _ = stub
    predictor.config["features"] = ["prompt", "llm"]
    predictor.save_config()
    asked = []

    class WithLLM(Stub):
        def served_without_llm(self, model, effort=None):
            asked.append(model)
            return False, "not_accurate_enough"

    model = WithLLM()
    model.result = {**model.result, "seconds": {"expected": 90.0}}
    # The AI model read the first task, and couldn't be reached for the second.
    read = {"a": {"kind": "bug_fix", "files": 2, "steps": 3, "minutes": 10}}
    monkeypatch.setattr("yardstick_predict.forecast._llm_labels", lambda predictor, tasks, chosen: read)
    tasks, _, _ = parse({"version": 1, "tasks": [TASK, {**TASK, "id": "b"}]})
    first, second = answer(predictor, tasks, model=model)["forecasts"]
    assert first["seconds"] == 90.0 and "reason" not in first
    assert second["seconds"] is None and second["reason"] == "not_accurate_enough"
    assert asked == ["m"]  # only the task it couldn't read asks the version without it
