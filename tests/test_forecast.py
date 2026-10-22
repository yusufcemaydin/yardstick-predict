from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from conftest import KEY, fake_vector
from test_model import rows_for, synthetic

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


@pytest.fixture
def trained(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.config["features"] = ["prompt"]
    predictor.save_config()
    tasks = synthetic(900)
    vectors = {"prompt": {t.id: np.array(fake_vector(t.prompt), dtype=np.float32) for t in tasks}}
    M.train(tasks, rows_for(tasks, vectors)).save(predictor.model_dir)
    return predictor


def test_answers_give_expected_values_and_say_why_when_they_cant(trained):
    tasks, _, _ = parse(request(models=["claude-opus-5-5", "gpt-6.1-sol", "brand-new-model"]))
    result = answer(trained, tasks)
    jsonschema.validate(result, schema("forecast-response.v1.json"))
    by_model = {f["model"]: f for f in result["forecasts"]}
    opus = by_model["claude-opus-5-5"]
    assert "reason" not in opus and opus["seconds"] > 0 and opus["tokens"] > 0 and opus["usd"] > 0
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


def test_an_untrained_predictor_says_so():
    predictor = Predictor.create("empty", [], {"model": "voyage-4", "dimensions": 256})
    tasks, _, _ = parse(request())
    with pytest.raises(RequestError) as caught:
        answer(predictor, tasks)
    assert caught.value.code == "not_trained"
    jsonschema.validate(caught.value.as_answer(), schema("forecast-response.v1.json"))
