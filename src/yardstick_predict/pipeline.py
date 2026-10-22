"""The steps behind setup and refresh: read the records, add new tasks, embed them and train."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import astuple, dataclass, field, fields

from . import benchmarks, embeddings, features, locks, model_types
from .archives import READER_VERSION, SETTLE_SECONDS, Task, completeness, families, read_records, record_files
from .store import Predictor


@dataclass
class Scan:
    """What a look through the archives found."""

    new_tasks: list[Task]
    files: dict[str, tuple[int, float]] = field(default_factory=dict)
    """The record files read, to remember as read. One that may still be in use is left out, so it's read again."""
    per_archive: dict[str, int] = field(default_factory=dict)
    updated_tasks: list[Task] = field(default_factory=list)
    """Stored tasks whose newer copy differs. A re-read of the record file a task came from replaces it, whatever
    it says, as when the task turned out to be stopped. A copy from another file or archive, such as a backup,
    replaces it only when it's more complete: more model calls, or as many and finished rather than stopped."""
    removed: list[str] = field(default_factory=list)
    """Stored tasks that a re-read of their record file no longer finds."""
    reader: int | None = None
    """The reader's version, when this look read every record file with it."""


@dataclass
class _Stored:
    """What a refresh needs to know about a stored task."""

    archive: str
    session: str
    source: str
    calls: int
    stopped: bool


_TASK_FIELDS = [f.name for f in fields(Task)]


def _stored(predictor: Predictor) -> dict[str, _Stored]:
    rows = predictor.db.execute("SELECT id, archive, session, source, calls, stopped FROM tasks")
    return {
        row[0]: _Stored(str(row[1] or ""), str(row[2] or ""), str(row[3] or ""), int(row[4] or 0), bool(row[5]))
        for row in rows
    }


def _plain(values: tuple) -> tuple:
    """A task's values in a form that compares equal however they were stored."""
    return tuple(None if name == "finished_at" and value == "" else value for name, value in zip(_TASK_FIELDS, values))


def _stored_values(predictor: Predictor, ids: list[str]) -> dict[str, tuple]:
    """The stored values of some tasks, by id."""
    found = {}
    for at in range(0, len(ids), 500):
        chunk = ids[at : at + 500]
        query = f"SELECT {', '.join(_TASK_FIELDS)} FROM tasks WHERE id IN ({', '.join('?' for _ in chunk)})"
        for row in predictor.db.execute(query, chunk):
            found[row[0]] = _plain(row)
    return found


def _pick(copies: list[Task], source: str) -> Task:
    """The copy of a task to keep: the most complete one. A re-read of the record file a stored task came from
    wins, unless another file's copy is more complete."""
    best = copies[0]
    for task in copies[1:]:
        if completeness(task) > completeness(best):
            best = task
    same = next((task for task in copies if source and task.source == source), None)
    return same if same is not None and completeness(same) >= completeness(best) else best


def scan(predictor: Predictor, now: float | None = None) -> Scan:
    """Find tasks the dataset doesn't have yet, and stored tasks whose records changed.

    Only changed record files are read, each with its family, such as a session with its copies and helpers.
    A dataset read by an older version of the reader has every record file read again.
    """
    now = time.time() if now is None else now
    stored = _stored(predictor)
    known_files = predictor.known_files()
    everything = predictor.config.get("reader") != READER_VERSION
    result = Scan(new_tasks=[], reader=READER_VERSION if everything else None)
    # A task can be in more than one archive, such as in a backup. Each copy is kept until one is picked.
    copies: dict[str, list[Task]] = {}
    held: set[str] = set()
    reread: set[str] = set()
    sessions: dict[str, set[str]] = {}
    for archive in predictor.config["archives"]:
        current = record_files(archive)
        changed = {path for path, state in current.items() if everything or known_files.get(path) != state}
        if changed and not everything:
            # A file adds to tasks in the other files of its family, so when one changes, its family is read again.
            family = families(archive, current)
            touched = {family[path] for path in changed}
            changed = {path for path in current if family[path] in touched}
        result.per_archive[archive["name"]] = 0
        if not changed:
            continue
        reading = read_records(archive, {path: current[path] for path in changed}, now)
        for task in reading.tasks:
            copies.setdefault(task.id, []).append(task)
        held |= reading.held
        reread |= changed
        sessions[archive["name"]] = reading.sessions
        # A file that may still be in use, such as one with a task that may still be running, is read again next time.
        settled = [path for path in changed if now - current[path][1] >= SETTLE_SECONDS]
        result.files.update({path: current[path] for path in settled if path not in reading.open_files})
    before = _stored_values(predictor, [task_id for task_id in copies if task_id in stored])
    for task_id, found in copies.items():
        kept = stored.get(task_id)
        task = _pick(found, kept.source if kept else "")
        if kept is None:
            result.new_tasks.append(task)
            result.per_archive[task.archive] = result.per_archive.get(task.archive, 0) + 1
        elif _plain(astuple(task)) != before.get(task_id) and (
            everything or task.source == kept.source or completeness(task) > (kept.calls, not kept.stopped)
        ):
            result.updated_tasks.append(task)
    for task_id, kept in stored.items():
        if task_id in copies or task_id in held:
            continue  # found again, or may still be running
        if kept.source in reread:
            result.removed.append(task_id)  # its record file was read again and no longer holds it
        elif everything and not kept.source and kept.session in sessions.get(kept.archive, set()):
            result.removed.append(task_id)  # stored by an older version, and its session's record no longer holds it
    return result


def save_scan(predictor: Predictor, found: Scan) -> int:
    # The model is out of date until training succeeds, even if this run stops before then.
    predictor.config["needs_training"] = True
    predictor.save_config()
    added = predictor.add_tasks(found.new_tasks)
    predictor.replace_tasks(found.updated_tasks)
    predictor.remove_tasks(found.removed)
    predictor.remember_files(found.files)
    if found.reader is not None:
        predictor.config["reader"] = found.reader  # every record file was read with this version of the reader
        predictor.save_config()
    return added


def read_all(archives: list[dict], now: float | None = None) -> Scan:
    """Read every task in the archives, for a new predictor."""
    now = time.time() if now is None else now
    result = Scan(new_tasks=[], reader=READER_VERSION)
    # A task can be in more than one archive, such as in a backup. Only its most complete copy counts.
    copies: dict[str, list[Task]] = {}
    for archive in archives:
        current = record_files(archive)
        reading = read_records(archive, current, now)
        for task in reading.tasks:
            copies.setdefault(task.id, []).append(task)
        result.per_archive[archive["name"]] = 0
        settled = [path for path, state in current.items() if now - state[1] >= SETTLE_SECONDS]
        result.files.update({path: current[path] for path in settled if path not in reading.open_files})
    result.new_tasks = [_pick(found, "") for found in copies.values()]
    for task in result.new_tasks:
        result.per_archive[task.archive] = result.per_archive.get(task.archive, 0) + 1
    return result


@dataclass
class EmbedPlan:
    """The texts still to embed, and what embedding them costs.

    `items` are the tasks' texts that need a vector, newest tasks first. `texts` are the ones that really
    go to Voyage: each text once, however many tasks share it, and none that the cache already holds.
    `counts` are their tokens. Only they cost anything.
    """

    items: list[tuple[str, str, str]]
    counts: list[int]
    cost: embeddings.Cost
    texts: list[str] = field(default_factory=list)

    @property
    def tasks(self) -> int:
        return len({task_id for task_id, _, _ in self.items})


def chosen_features(predictor: Predictor) -> list[str]:
    """The features the predictor uses. The prompt is always first."""
    return features.check(predictor.config.get("features") or list(features.DEFAULT))


def settings(predictor: Predictor) -> dict:
    """The predictor's model settings. Anything left out uses the default."""
    return dict(predictor.config.get("model") or {})


def load_plugins(predictor: Predictor) -> None:
    """Load the plugins the predictor names. Raises model_types.PluginError with a plain reason."""
    model_types.load_plugins(predictor.config.get("plugins") or [])


def learnable(tasks: list[Task], limit: int | None = None) -> list[Task]:
    """The tasks the model learns from, newest first. Stopped tasks and tasks left waiting are left out.

    With `limit`, only the newest that many. Embedding and training both pick tasks here, so they agree.
    """
    from .model import usable

    kept = usable(tasks)[::-1]  # the model's own rule, so nothing embedded is left out of training
    return kept[: int(limit)] if limit else kept


def _texts(tasks: list[Task], chosen: list[str], limit: int | None):
    """Each learnable task's text for each chosen embedded feature, newest tasks first. A text may be empty."""
    fields = [name for name in chosen if features.is_text(name)]
    for task in learnable(tasks, limit):
        info = features.TaskInfo.of(task)
        for name in fields:
            yield task.id, name, features.text(name, info)


def _made_from(stored, task_id: str, text: str) -> bool:
    """Whether the task's stored vector, when it has one, was made from this text."""
    if stored is None or task_id not in stored:
        return False
    hashes = getattr(stored, "hashes", None)
    if hashes is None:
        return True  # a plain set says only which tasks have a vector
    return hashes.get(task_id) is not None and hashes[task_id] == embeddings.text_hash(text)


def embedding_items(
    tasks: list[Task], chosen: list[str], have: dict[str, set[str]] | None = None, limit: int | None = None
) -> list[tuple[str, str, str]]:
    """Each task's texts that need a vector, one for each chosen embedded feature, newest tasks first.

    Only tasks the model learns from are embedded, so stopped tasks and tasks left waiting are skipped.
    A task with no text for a feature, such as no context, has nothing to embed for it. `have` says which
    tasks have vectors, as predictor.embedded gives them, with the hash of the text each was made from. A
    vector made from another text is made again, as when a fuller copy of the task replaced it or a
    skill's description changed.
    """
    have = have or {}
    return [
        (task_id, name, text)
        for task_id, name, text in _texts(tasks, chosen, limit)
        if text.strip() and not _made_from(have.get(name), task_id, text)
    ]


def price(items: list[tuple[str, str, str]], model: str, key: str | None) -> EmbedPlan:
    """Count the tokens of the texts that really go to Voyage, with the model's own tokenizer, and price them.

    A text goes once, however many tasks share it, as when many prompts say "continue". A text the cache
    already holds, such as a prompt a forecast embedded, doesn't go at all.
    """
    if not items:
        return EmbedPlan([], [], embeddings.Cost(0, True, 0.0, None))
    unique = list(dict.fromkeys(text for _, _, text in items))
    held = embeddings.in_cache(model, unique)
    texts = [text for text in unique if text not in held]
    counts, exact = embeddings.count_tokens(texts, model)
    return EmbedPlan(items, counts, embeddings.cost(sum(counts), exact, model, key), texts)


def plan_embeddings(predictor: Predictor, key: str | None) -> EmbedPlan:
    """What the predictor still has to embed, and what it costs.

    A vector whose task no longer has a text for its feature is deleted here, so training doesn't use it.
    """
    chosen = chosen_features(predictor)
    have = {name: predictor.embedded(name) for name in chosen if features.is_text(name)}
    tasks, limit = predictor.tasks(), predictor.config.get("max_tasks")
    emptied: dict[str, list[str]] = {}
    for task_id, name, text in _texts(tasks, chosen, limit):
        if not text.strip() and task_id in have.get(name, ()):
            emptied.setdefault(name, []).append(task_id)
    for name, task_ids in emptied.items():
        predictor.forget_embeddings(name, task_ids)
    items = embedding_items(tasks, chosen, have, limit)
    return price(items, predictor.embedding_model, key)


def newest_that_fit(plan: EmbedPlan, tokens: int) -> int:
    """How many of the newest tasks fit in a number of tokens, with all their texts.

    A text that several tasks share is paid for once, by the newest of them. A text the cache holds is free.
    """
    # A task counts only when all its texts fit. The plan lists the newest tasks first, and so does this.
    if not plan.texts and len(plan.counts) == len(plan.items):
        costs = list(plan.counts)  # a plan made with one count for each item, as plans were before
    else:
        count_of = dict(zip(plan.texts, plan.counts))
        paid: set[str] = set()
        costs = []
        for _, _, text in plan.items:
            costs.append(count_of[text] if text in count_of and text not in paid else 0)
            paid.add(text)
    per_task: dict[str, int] = {}
    for (task_id, _, _), count in zip(plan.items, costs):
        per_task[task_id] = per_task.get(task_id, 0) + count
    used, tasks = 0, 0
    for count in per_task.values():
        if used + count > tokens:
            break
        used, tasks = used + count, tasks + 1
    return tasks


def run_embeddings(
    predictor: Predictor,
    plan: EmbedPlan,
    key: str,
    on_progress: Callable[[int, int], None] | None = None,
    on_wait: Callable[[str], None] | None = None,
) -> None:
    """Give each task in the plan its vectors. Each request's results are saved at once, so a stopped run resumes.

    A text goes to Voyage once, however many tasks share it, and not at all when the cache holds it. Each
    vector is saved with the hash of its text, so a task whose text changes later gets a new one.
    """
    model = predictor.embedding_model
    users: dict[str, list[tuple[str, str]]] = {}
    for task_id, name, text in plan.items:
        users.setdefault(text, []).append((task_id, name))

    def save(vectors: dict[str, list[float]]) -> None:
        by_field: dict[str, dict] = {}
        hashes: dict[str, dict] = {}
        for text, vector in vectors.items():
            digest = embeddings.text_hash(text)
            for task_id, name in users.get(text, []):
                by_field.setdefault(name, {})[task_id] = vector
                hashes.setdefault(name, {})[task_id] = digest
        for name, found in by_field.items():
            predictor.put_embeddings(name, found, hashes[name])

    held = embeddings.cached(model, list(users))  # paid for already, by this predictor, another or a forecast
    save(held)
    texts = [text for text in users if text not in held]
    if plan.texts or len(plan.counts) != len(plan.items):
        count_of = dict(zip(plan.texts, plan.counts))
    else:  # a plan made with one count for each item, as plans were before
        count_of = {text: count for (_, _, text), count in zip(plan.items, plan.counts)}
    counts = [count_of.get(text) or embeddings.most_tokens(text) for text in texts]
    done = 0
    for positions, vectors in embeddings.embed(texts, model, key, counts, on_wait):
        save({texts[position]: vector for position, vector in zip(positions, vectors)})
        done += len(positions)
        if on_progress:
            on_progress(done, len(texts))


def refresh_benchmarks(predictor: Predictor) -> str | None:
    """Download the model benchmarks again, when the predictor uses them. Returns a warning, or None."""
    if "benchmarks" not in chosen_features(predictor):
        return None
    try:
        table = benchmarks.download()
    except (OSError, ValueError) as error:
        kept = "Keeping the ones saved before." if (predictor.dir / "benchmarks.json").is_file() else ""
        return f"Couldn't download the model benchmarks ({error}). {kept}".strip()
    from . import files

    # Saved in one step, so a forecast reading meanwhile finds the old table or the new one, never half of one.
    files.write_text(predictor.dir / "benchmarks.json", json.dumps(table) + "\n")
    return None


def benchmark_table(predictor: Predictor) -> dict:
    try:
        found = json.loads((predictor.dir / "benchmarks.json").read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def feature_rows(predictor: Predictor, tasks: list[Task]) -> list:
    """Each task's features, as the model reads them.

    A task's vector counts only when it was made from the task's text as it is now. One made from an older
    text, such as before a fuller copy of the task replaced it, describes another task, so it's left out.
    """
    from .model import Features

    chosen = chosen_features(predictor)
    names = [name for name in chosen if features.is_text(name)]
    stored = {name: predictor.get_embeddings(name) for name in names}
    hashes = {name: predictor.embedded(name).hashes for name in names}
    table = benchmark_table(predictor)
    rows = []
    for task in tasks:
        info = features.TaskInfo.of(task)
        vectors = {}
        for name in names:
            vector, made_from = stored[name].get(task.id), hashes[name].get(task.id)
            if vector is not None and made_from == embeddings.text_hash(features.text(name, info)):
                vectors[name] = vector
        rows.append(
            Features(
                model=task.model,
                effort=task.effort,
                agent=task.agent,
                origin=task.origin,
                prompt_chars=len(task.prompt),
                vectors=vectors,
                numbers=features.numbers(chosen, info, table),
            )
        )
    return rows


def features_without_values(predictor: Predictor, rows: list) -> list[str]:
    """Your own features that have a value for no past task, so the model has nothing to learn from them.

    A feature that reads only a forecast request's "extra" is one: past tasks have no extra.
    """
    own = [name for name in chosen_features(predictor) if name not in features.TEXT and name not in features.BUILT_IN]
    return [
        name
        for name in own
        if not any(name in row.vectors if features.is_text(name) else name in row.numbers for row in rows)
    ]


def train(predictor: Predictor):
    """Train the predictor's model on its whole dataset and save it."""
    load_plugins(predictor)
    tasks = predictor.tasks()
    if predictor.config.get("max_tasks"):
        # The same newest tasks that were embedded, oldest first.
        tasks = learnable(tasks, predictor.config["max_tasks"])[::-1]
    model_settings = settings(predictor)
    trainer, _ = model_types.get(model_settings.get("type", "xgboost"))
    rows = feature_rows(predictor, tasks)
    model = trainer(tasks, rows, model_settings)
    if isinstance(getattr(model, "meta", None), dict):
        model.meta["features_without_values"] = features_without_values(predictor, rows)
    with _model_lock(predictor):
        # A save that was stopped halfway leaves its folder behind. No save runs while this lock is held.
        for unfinished in predictor.dir.glob(".model-*"):
            shutil.rmtree(unfinished, ignore_errors=True)
        model.save(predictor.model_dir)
    predictor.config["refreshed_at"] = time.time()
    predictor.config["needs_training"] = False
    predictor.save_config()
    return model


def _model_lock(predictor: Predictor, shared: bool = False):
    """Held alone while a model is saved, and shared while one is read.

    Saving moves the old model aside and the new one into place. Without the lock, a forecast reading in
    between, such as the status line's, could find no model, or read the old model's settings and the new
    one's trees and mix them.
    """
    return locks.held(predictor.dir / "model.lock", shared=shared)


def model_ready(predictor: Predictor) -> bool:
    """Whether the predictor has a model. While a new one is being swapped in, it waits for it."""
    with _model_lock(predictor, shared=True):
        return (predictor.model_dir / "model.json").is_file()


def model_meta(predictor: Predictor) -> dict | None:
    """What the saved model says about itself, such as its test report, without loading its trees. None without one."""
    with _model_lock(predictor, shared=True):
        try:
            found = json.loads((predictor.model_dir / "model.json").read_text())
        except (OSError, ValueError):
            return None
    return found if isinstance(found, dict) else None


def load_model(predictor: Predictor):
    """Load the predictor's model, all of it from one save."""
    load_plugins(predictor)
    _, loader = model_types.get(settings(predictor).get("type", "xgboost"))
    with _model_lock(predictor, shared=True):
        return loader(predictor.model_dir)
