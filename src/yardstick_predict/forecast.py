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

import inspect
import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import embeddings, features, model_types, pipeline, prices
from .archives import TEXT_CHARS, Task
from .history import Past
from .store import Predictor

FORMAT_VERSION = 1
KEY_WAIT = 10.0
"""How many seconds a forecast gives the password store to answer when it needs the Voyage key."""
MAX_TASKS = 1000
MAX_MODELS = 50
TARGETS = ("seconds", "tokens", "usd")
REASONS = {
    "unknown_model": "This predictor has never seen a task on this model.",
    "not_enough_data": "Too few tasks on this model to test its forecasts.",
    "not_accurate_enough": "Its forecasts on this model didn't pass the test.",
}
ERRORS = {
    "invalid_request": "The request breaks a rule of the format.",
    "no_predictor": "No predictor exists yet.",
    "no_default_predictor": "There are several predictors, none is the default, and the request names none.",
    "unknown_predictor": "The request names a predictor that doesn't exist.",
    "broken_predictor": "The predictor's settings file can't be read, or is empty or broken.",
    "plugin_failed": "A plugin the predictor uses couldn't be loaded.",
    "not_trained": "The predictor has no model yet, or its model is from a version before the previous release.",
    "no_embedding_key": "No Voyage key was found.",
    "embedding_failed": "Voyage couldn't embed the request's text.",
    "missing_library": "A library the model needs is missing.",
    "model_too_new": "The predictor's model was saved by a newer version of yardstick-predict.",
    "internal_error": "Something went wrong that the tool didn't expect.",
}
"""Every error code an answer can have, as the answer schema and docs/format.md list them."""


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
    session: str = ""
    """The conversation the task is in, as the agent's records name it, for what came before it."""
    conversation_tokens: int = 0
    """How big the conversation is so far, in tokens. 0 when unknown or new."""
    extra: dict = field(default_factory=dict)


def parse(data: object) -> tuple[list[TaskRequest], bool, str | None]:
    """Check a request and read its tasks. Raises RequestError with a plain message when it's wrong.

    It checks what the request schema checks. A field that may be left out may also be null, which means
    the same. Each prompt and context is prepared as training prepares a task's: the prompt without the
    spaces at its ends, cut to its first TEXT_CHARS characters, and the context's last TEXT_CHARS.
    """

    def bad(message: str) -> RequestError:
        return RequestError("invalid_request", message)

    if not isinstance(data, dict):
        raise bad("The request must be a JSON object.")
    version = data.get("version")
    if isinstance(version, bool) or version != FORMAT_VERSION:  # JSON's true would equal 1 in Python
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
        if not isinstance(prompt, str):
            raise bad(f'Task {task_id!r} needs a "prompt" string.')
        if not prompt.strip():
            raise bad(f'Task {task_id!r} needs a "prompt" with some text in it, not only spaces.')
        models = task.get("models")
        if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
            raise bad(f'Task {task_id!r} needs "models": a list of model names.')
        if len(models) > MAX_MODELS:
            raise bad(f"Task {task_id!r} lists more than {MAX_MODELS} models.")
        for name in ("context", "effort", "agent", "origin", "project", "session"):
            if task.get(name) is not None and not isinstance(task.get(name), str):
                raise bad(f'In task {task_id!r}, "{name}" must be a string.')
        if task.get("origin") not in (None, "program", "person"):
            raise bad(f'In task {task_id!r}, "origin" must be "program" or "person".')
        size = task.get("conversation_tokens")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise bad(f'In task {task_id!r}, "conversation_tokens" must be a whole number, 0 or more.')
        skills = task.get("skills")
        skills = [] if skills is None else skills
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            raise bad(f'In task {task_id!r}, "skills" must be a list of skill names.')
        extra = task.get("extra")
        extra = {} if extra is None else extra
        if not isinstance(extra, dict):
            raise bad(f'In task {task_id!r}, "extra" must be an object.')
        found.append(
            TaskRequest(
                id=task_id,
                prompt=prompt.strip()[:TEXT_CHARS],
                models=list(dict.fromkeys(models)),  # a model named twice is forecast once
                context=(task.get("context") or "")[-TEXT_CHARS:].strip(),
                skills=list(skills),
                effort=task.get("effort"),
                agent=task.get("agent"),
                origin=task.get("origin") or "program",
                project=task.get("project") or "",
                session=task.get("session") or "",
                conversation_tokens=size or 0,
                extra=extra,
            )
        )
    distribution = data.get("distribution")
    distribution = False if distribution is None else distribution
    if not isinstance(distribution, bool):
        raise bad('"distribution" must be true or false.')
    predictor = data.get("predictor")
    if predictor is not None and (not isinstance(predictor, str) or not predictor):
        raise bad('"predictor" must be the name of a predictor.')
    return found, distribution, predictor


def _info(task: TaskRequest, model: str, defaults: dict, skills: list[str], llm: dict, now: float) -> features.TaskInfo:
    """The task as features see it on one AI model. It starts `now`, the moment the forecast is asked for."""
    known = defaults.get(model, {})
    return features.TaskInfo(
        prompt=task.prompt,
        model=model,
        effort=task.effort or known.get("effort", "default"),
        agent=task.agent or known.get("agent", ""),
        origin=task.origin,
        context=task.context,
        skills=skills,
        project=task.project,
        extra=task.extra,
        llm=llm,
        started_at=now,
    )


def _known_skills(task: TaskRequest, config: dict) -> list[str]:
    """The skills known before the task starts that the predictor's options keep.

    The request lists the ones its conversation used before it. The ones its prompt invokes are read from
    the prompt, as training reads them.
    """
    return features.known_skills(task.skills, task.prompt, features.options(config, "skills"))


def _usable_texts(model, chosen: list[str]) -> list[str]:
    """The text features the model can use. Each needs enough past tasks with that text to compare with.

    With fewer, the model sees no similar tasks for it, so embedding it would cost tokens for nothing.
    A model type of your own gets every text feature, since only it knows what it uses.
    """
    from .model import MIN_NEIGHBORS, Model, WithoutLLM
    from .model_0_3 import Choice  # a model saved by 0.3. This goes when model_0_3 does.

    names = [name for name in chosen if features.is_text(name)]
    if not isinstance(model, (Model, WithoutLLM, Choice)):
        return names
    return [name for name in names if name in model.pools and len(model.pools[name][0]) >= MIN_NEIGHBORS]


def _embed(predictor: Predictor, items: list[tuple[int, str, str]]) -> dict[str, dict[int, np.ndarray]]:
    """Embed each row's texts with the predictor's own embedding model.

    A text embedded before, for training or for an earlier forecast, comes from the cache and costs
    nothing. The rest go to Voyage once each, with little patience: when Voyage can't be reached, a
    forecast says so within seconds.
    """
    vectors: dict[str, dict[int, np.ndarray]] = {}
    if not items:
        return vectors
    model = predictor.embedding_model
    # The same text, such as one prompt forecast on several models or effort levels, is sent once.
    texts = list(dict.fromkeys(text for _, _, text in items))
    found = {text: np.asarray(vector, dtype=np.float32) for text, vector in embeddings.cached(model, texts).items()}
    missing = [text for text in texts if text not in found]
    if missing:
        # Only now is the key needed. A password store that doesn't answer, such as a locked one, holds the
        # forecast up for KEY_WAIT seconds at most.
        key, source = embeddings.find_key(wait=KEY_WAIT)
        if source == embeddings.NO_ANSWER:
            raise RequestError(
                "no_embedding_key", "Your password store didn't answer in time. Unlock it, or set VOYAGE_API_KEY."
            )
        if not key:
            raise RequestError(
                "no_embedding_key",
                "This predictor uses Voyage embeddings. Set VOYAGE_API_KEY or run yardstick-predict to save a key.",
            )
        try:
            for positions, batch in embeddings.embed(missing, model, key, patience=embeddings.QUICK):
                for position, vector in zip(positions, batch):
                    found[missing[position]] = np.asarray(vector, dtype=np.float32)
        except embeddings.VoyageError as error:
            raise RequestError("embedding_failed", str(error)) from None
    for row, name, text in items:
        vectors.setdefault(name, {})[row] = found[text]
    return vectors


def load(predictor: Predictor):
    """The predictor's trained model, with its problems as RequestErrors in the answer format."""
    from .model import NewerFormat, OutdatedModel

    if not pipeline.model_ready(predictor):
        raise RequestError("not_trained", f"Predictor {predictor.name!r} has no model yet. Run yardstick-predict.")
    try:
        return pipeline.load_model(predictor)
    except model_types.PluginError as error:
        raise RequestError("plugin_failed", str(error)) from None
    except NewerFormat as error:  # saved by a newer version of this tool
        raise RequestError("model_too_new", str(error)) from None
    except OutdatedModel as error:  # trained by a version before the previous release, which read other features
        raise RequestError("not_trained", str(error)) from None


RATES = ("input", "output", "cache_write_5m", "cache_write_1h", "cache_read")
"""The rates in a model's list price, in dollars per million tokens."""


def _free(model: str) -> bool:
    """Whether an AI model costs nothing: every rate in its list price is 0, for long prompts too."""
    price = prices.price(model)
    if price is None:
        return False
    tiers = [price, price.get("long_context") or {}]
    return all(tier.get(rate, 0) == 0 for tier in tiers for rate in RATES)


def _verdicts(model):
    """A function that says whether the model forecasts an AI model at an effort level, and why not.

    The built-in model can say no for a level the AI model never ran at. A model type of your own may
    take only the AI model's name.
    """
    try:
        takes_effort = "effort" in inspect.signature(model.served).parameters
    except (TypeError, ValueError):
        takes_effort = False
    if takes_effort:
        return lambda name, effort: model.served(name, effort=effort or None)
    return lambda name, effort: model.served(name)


def answer(
    predictor: Predictor,
    tasks: list[TaskRequest],
    distribution: bool = False,
    model=None,
    recent: list[Task] | None = None,
) -> dict:
    """Forecast every task on every model it lists. `model` is the predictor's model, when it's loaded already.

    What came before each task, in its project and, with the conversation feature, in its conversation, comes
    from the predictor's tasks plus `recent`: tasks read from the records just now, newer than the last refresh.
    Each task starts now, the moment the forecast is asked for.
    """
    from .model import Features

    now = time.time()
    model = load(predictor) if model is None else model
    # A feature this version doesn't know, such as one a plugin added before and doesn't add now, is skipped.
    # The model still forecasts, without that feature's values.
    chosen = pipeline.chosen_features(predictor)
    defaults = model.meta.get("defaults", {})
    verdict = _verdicts(model)
    pairs = [(task, name, *verdict(name, task.effort)) for task in tasks for name in task.models]
    labels = _llm_labels(predictor, tasks, chosen) if any(served for _, _, served, _ in pairs) else {}
    without = getattr(model, "served_without_llm", None)
    if "llm" in chosen and callable(without):
        # A task LLM analysis couldn't read is forecast by the model without it, which had its own test.
        unread = {task.id for task in tasks if not _read(labels.get(task.id))}
        pairs = [
            (task, name, *(without(name, task.effort or None) if served and task.id in unread else (served, reason)))
            for task, name, served, reason in pairs
        ]
    # A row for each model to forecast. As in training, its features see its model, effort level and agent.
    # A task with no model to forecast has no row, so its texts aren't sent to Voyage. Neither are texts
    # the model can't use.
    served_pairs = [(task, name) for task, name, served, _ in pairs if served]
    infos = [
        _info(task, name, defaults, _known_skills(task, predictor.config), labels.get(task.id, {}), now)
        for task, name in served_pairs
    ]
    usable = _usable_texts(model, chosen)
    items = [
        (row, name, text)
        for row, info in enumerate(infos)
        for name in usable
        if (text := features.text(name, info)).strip()
    ]
    vectors = _embed(predictor, items)
    table = pipeline.benchmark_table(predictor)
    if served_pairs:
        newer = {t.id: t for t in recent or []}
        past = Past([t for t in predictor.tasks(texts=False) if t.id not in newer] + list(newer.values()))
    rows = []
    for row, ((task, _), info) in enumerate(zip(served_pairs, infos)):
        before = past.numbers(
            task.project,
            task.session,
            now,
            task.origin,
            task.conversation_tokens,
            info.agent,
            task.prompt,
            conversation="conversation" in chosen,
        )
        rows.append(
            Features(
                model=info.model,
                effort=info.effort,
                agent=info.agent,
                origin=info.origin,
                prompt_chars=len(info.prompt),
                vectors={field: found[row] for field, found in vectors.items() if row in found},
                numbers={**features.numbers(chosen, info, table, predictor.config), **before},
                started_at=now,
            )
        )
    results = iter(model.forecast(rows))
    # A free model's cost is known without forecasting it: nothing. Its usd head can't learn $0 on a log
    # scale, so the model drops it. A predictor set to leave usd out still leaves it out.
    gives_usd = "usd" in ((model.meta.get("settings") or {}).get("heads") or TARGETS)
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
                if target == "usd" and gives_usd and _free(name):
                    forecast["usd"], shape["usd"] = 0.0, None
                    continue
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


def _llm_labels(predictor: Predictor, tasks: list[TaskRequest], chosen: list[str]) -> dict[str, dict]:
    """What LLM analysis says about each task, when the predictor uses it. Nothing when it can't be reached."""
    if "llm" not in chosen:
        return {}
    from . import llm

    return llm.analyze_requests(predictor, tasks)


def _read(labels: dict | None) -> bool:
    """Whether LLM analysis said something about a task that the model can use."""
    from . import llm

    return bool(labels) and bool(llm.numbers(labels) or str(labels.get("description") or "").strip())


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
