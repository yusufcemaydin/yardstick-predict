"""Answer forecast requests in the fixed, versioned format.

A request (version 1):

    {"version": 1,
     "tasks": [{"id": "t1", "prompt": "Add a login page", "models": ["claude-opus-5-5"]}]}

The answer (version 1):

    {"version": 1, "predictor": "default", "embedding_model": "voyage-4",
     "forecasts": [{"task": "t1", "model": "claude-opus-5-5",
                    "seconds": 245.0, "tokens": 1830000, "usd": 0.92}]}

The numbers are expected values. When the tool can't stand behind a forecast, the numbers are null
and a reason says why. Within version 1, fields may be added but are never renamed or removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import embeddings, features, pipeline, prices
from .store import Predictor

FORMAT_VERSION = 1
MAX_TASKS = 1000
MAX_MODELS = 50
TARGETS = ("seconds", "tokens", "usd")
REASONS = {
    "unknown_model": "This predictor has never seen a task on this model.",
    "not_enough_data": "Too few tasks on this model to test its forecasts.",
    "not_accurate_enough": "Its forecasts on this model didn't pass the test.",
}


class RequestError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_answer(self) -> dict:
        return {"version": FORMAT_VERSION, "error": {"code": self.code, "message": self.message}}


@dataclass
class TaskRequest:
    id: str
    prompt: str
    models: list[str]
    context: str = ""
    skills: list[str] = field(default_factory=list)
    effort: str | None = None
    agent: str | None = None
    origin: str = "program"
    project: str = ""
    extra: dict = field(default_factory=dict)


def parse(data: object) -> tuple[list[TaskRequest], bool, str | None]:
    """Check a request and read its tasks. Raises RequestError with a plain message when it's wrong."""

    def bad(message: str) -> RequestError:
        return RequestError("invalid_request", message)

    if not isinstance(data, dict):
        raise bad("The request must be a JSON object.")
    if data.get("version") != FORMAT_VERSION:
        raise bad(f'Set "version" to {FORMAT_VERSION}. This tool reads format version {FORMAT_VERSION}.')
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise bad('"tasks" must be a list with at least one task.')
    if len(tasks) > MAX_TASKS:
        raise bad(f"Send at most {MAX_TASKS} tasks in one request.")
    found: list[TaskRequest] = []
    seen: set[str] = set()
    for n, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            raise bad(f"Task {n} must be a JSON object.")
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise bad(f'Task {n} needs an "id" string.')
        if task_id in seen:
            raise bad(f"Two tasks share the id {task_id!r}. Each id must be unique.")
        seen.add(task_id)
        prompt = task.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise bad(f'Task {task_id!r} needs a "prompt" string.')
        models = task.get("models")
        if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
            raise bad(f'Task {task_id!r} needs "models": a list of model names.')
        if len(models) > MAX_MODELS:
            raise bad(f"Task {task_id!r} lists more than {MAX_MODELS} models.")
        for name in ("context", "effort", "agent", "origin", "project"):
            if task.get(name) is not None and not isinstance(task.get(name), str):
                raise bad(f'In task {task_id!r}, "{name}" must be a string.')
        if task.get("origin") not in (None, "program", "person"):
            raise bad(f'In task {task_id!r}, "origin" must be "program" or "person".')
        skills = task.get("skills") or []
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            raise bad(f'In task {task_id!r}, "skills" must be a list of skill names.')
        extra = task.get("extra") or {}
        if not isinstance(extra, dict):
            raise bad(f'In task {task_id!r}, "extra" must be an object.')
        found.append(
            TaskRequest(
                id=task_id,
                prompt=prompt,
                models=list(dict.fromkeys(models)),
                context=task.get("context") or "",
                skills=skills,
                effort=task.get("effort"),
                agent=task.get("agent"),
                origin=task.get("origin") or "program",
                project=task.get("project") or "",
                extra=extra,
            )
        )
    distribution = data.get("distribution", False)
    if not isinstance(distribution, bool):
        raise bad('"distribution" must be true or false.')
    predictor = data.get("predictor")
    if predictor is not None and not isinstance(predictor, str):
        raise bad('"predictor" must be a string.')
    return found, distribution, predictor


def _info(task: TaskRequest, model: str, defaults: dict) -> features.TaskInfo:
    known = defaults.get(model, {})
    return features.TaskInfo(
        prompt=task.prompt,
        model=model,
        effort=task.effort or known.get("effort", "default"),
        agent=task.agent or known.get("agent", ""),
        origin=task.origin,
        context=task.context,
        skills=task.skills,
        project=task.project,
        extra=task.extra,
    )


def _usable_texts(model, chosen: list[str]) -> list[str]:
    """The text features the model can use. Each needs enough past tasks with that text to compare with.

    With fewer, the model sees no similar tasks for it, so embedding it would cost tokens for nothing.
    A model type of your own gets every text feature, since only it knows what it uses.
    """
    from .model import MIN_NEIGHBORS, Model

    names = [name for name in chosen if features.is_text(name)]
    if not isinstance(model, Model):
        return names
    return [name for name in names if name in model.pools and len(model.pools[name][0]) >= MIN_NEIGHBORS]


def _embed(predictor: Predictor, items: list[tuple[int, str, str]]) -> dict[str, dict[int, np.ndarray]]:
    """Embed each row's texts with the predictor's own embedding model."""
    vectors: dict[str, dict[int, np.ndarray]] = {}
    if not items:
        return vectors
    key, _ = embeddings.find_key()
    if not key:
        raise RequestError(
            "no_embedding_key",
            "This predictor uses Voyage embeddings. Set VOYAGE_API_KEY or run yardstick-predict to save a key.",
        )
    # The same text, such as one prompt forecast on several models or effort levels, is sent once.
    texts = list(dict.fromkeys(text for _, _, text in items))
    found: dict[int, np.ndarray] = {}
    try:
        for positions, batch in embeddings.embed(texts, predictor.embedding_model, key):
            for position, vector in zip(positions, batch):
                found[position] = np.asarray(vector, dtype=np.float32)
    except embeddings.VoyageError as error:
        raise RequestError("embedding_failed", str(error)) from None
    where = {text: position for position, text in enumerate(texts)}
    for row, name, text in items:
        vectors.setdefault(name, {})[row] = found[where[text]]
    return vectors


def answer(predictor: Predictor, tasks: list[TaskRequest], distribution: bool = False) -> dict:
    """Forecast every task on every model it lists."""
    from .model import Features

    if not (predictor.model_dir / "model.json").is_file():
        raise RequestError("not_trained", f"Predictor {predictor.name!r} has no model yet. Run yardstick-predict.")
    model = pipeline.load_model(predictor)
    chosen = pipeline.chosen_features(predictor)
    defaults = model.meta.get("defaults", {})
    pairs = [(task, name, *model.served(name)) for task in tasks for name in task.models]
    # A row for each model to forecast. As in training, its features see its model, effort level and agent.
    # A task with no model to forecast has no row, so its texts aren't sent to Voyage. Neither are texts
    # the model can't use.
    infos = [_info(task, name, defaults) for task, name, served, _ in pairs if served]
    usable = _usable_texts(model, chosen)
    items = [
        (row, name, text)
        for row, info in enumerate(infos)
        for name in usable
        if (text := features.text(name, info)).strip()
    ]
    vectors = _embed(predictor, items)
    table = pipeline.benchmark_table(predictor)
    rows = []
    for row, info in enumerate(infos):
        rows.append(
            Features(
                model=info.model,
                effort=info.effort,
                agent=info.agent,
                origin=info.origin,
                prompt_chars=len(info.prompt),
                vectors={field: found[row] for field, found in vectors.items() if row in found},
                numbers=features.numbers(chosen, info, table),
            )
        )
    results = iter(model.forecast(rows))
    forecasts = []
    for task, name, served, reason in pairs:
        forecast: dict = {"task": task.id, "model": name}
        effort = task.effort or defaults.get(name, {}).get("effort")
        if effort:
            forecast["effort"] = effort  # the request's, or the level this model ran at most often
        forecast.update(seconds=None, tokens=None, usd=None)
        if reason:
            forecast["reason"] = reason  # only when the numbers are null
        result = next(results) if served else None
        if result is not None:
            shape: dict = {
                "shape": result.get("shape"),
                "direct_probability": _number(result.get("direct_probability")),
            }
            for target in TARGETS:
                head = result.get(target)
                if head is None or (target == "usd" and prices.price(name) is None):
                    shape[target] = None
                    continue
                forecast[target] = _round(target, head["expected"])
                shape[target] = _curve(head.get("distribution"))
            if distribution:
                forecast["distribution"] = shape if shape["shape"] else None
        elif distribution:
            forecast["distribution"] = None
        forecasts.append(forecast)
    return {
        "version": FORMAT_VERSION,
        "predictor": predictor.name,
        "embedding_model": predictor.embedding_model,
        "forecasts": forecasts,
    }


def _number(value) -> float | None:
    """A number for the answer. JSON has no NaN or infinity, so those become None, which is null."""
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def _curve(distribution: dict | None) -> dict | None:
    """A curve for the answer. A curve needs all its numbers, so one with NaN or infinity in it becomes null."""
    if not distribution:
        return None
    curve = {part: {k: _number(v) for k, v in values.items()} for part, values in distribution.items()}
    return None if any(v is None for values in curve.values() for v in values.values()) else curve


def _round(target: str, value: float) -> float | int | None:
    if not math.isfinite(value):
        return None
    if target == "tokens":
        return int(round(value))
    if target == "seconds":
        return round(value, 1)
    return round(value, 6)
