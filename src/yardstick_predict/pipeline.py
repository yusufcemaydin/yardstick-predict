"""The steps behind setup and refresh: read the records, add new tasks, embed them and train."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import astuple, dataclass, field, fields
from functools import cache
from pathlib import Path

from . import benchmarks, embeddings, features, locks, model_types
from .archives import (
    READER_VERSION,
    SETTLE_SECONDS,
    Reading,
    Task,
    archive_path,
    completeness,
    families,
    read_records,
    record_files,
)
from .history import Past
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


@dataclass
class _Bytes:
    """How many bytes of record files have been read, for one progress bar over every archive.

    It passes on a step only once another thousandth of the whole is read, so reading many small files isn't
    slowed down by drawing the bar each time.
    """

    total: int
    on_progress: Callable[[int, int], None] | None
    done: int = 0
    told: int = 0

    def read(self, size: int) -> None:
        self.done = min(self.done + size, self.total)
        if self.done - self.told >= self.total / 1000 or self.done == self.total:
            self.reach(self.done)

    def reach(self, done: int) -> None:
        self.done = self.told = min(done, self.total)
        if self.on_progress is not None and self.total:
            self.on_progress(self.done, self.total)


def _sizes(files: dict[str, tuple[int, float]]) -> int:
    return sum(size for size, _ in files.values())


@dataclass
class Reads:
    """The record files a scan reads in one archive, found from their sizes and times without reading them."""

    archive: dict
    files: dict[str, tuple[int, float]]
    chats: set[str] = field(default_factory=set)
    """The conversations those files hold, by family: a session with its copies and helpers is one. Counted only
    when asked for."""

    @property
    def size(self) -> int:
        return _sizes(self.files)


def to_read(predictor: Predictor, count_chats: bool = False) -> list[Reads]:
    """What a scan of the predictor reads in each of its archives, one for each archive, with no files when nothing
    in it changed.

    Only sizes and times are compared. When any file changed, the first lines of the archive's files say which
    conversation each belongs to, as families tells. A dataset read by an older version of the reader has every
    record file read again. With `count_chats`, the conversations are counted then too.
    """
    known_files = predictor.known_files()
    everything = predictor.config.get("reader") != READER_VERSION
    found = []
    for archive in predictor.config["archives"]:
        current = record_files(archive)
        changed = {path for path, state in current.items() if everything or known_files.get(path) != state}
        family = families(archive, current) if changed and (count_chats or not everything) else {}
        if changed and not everything:
            # A file adds to tasks in the other files of its family, so when one changes, its family is read again.
            touched = {family[path] for path in changed}
            changed = {path for path in current if family[path] in touched}
        chats = {family[path] for path in changed} if family else set()
        found.append(Reads(archive, {path: current[path] for path in changed}, chats))
    return found


class Readings:
    """The record files read in this session, the last reading of each archive, so a second look at the same
    files reads none of them again. A reading of more files serves a look at fewer of them."""

    def __init__(self) -> None:
        self._kept: dict[tuple, tuple[dict[str, tuple[int, float]], Reading]] = {}

    @staticmethod
    def _which(archive: dict) -> tuple:
        return archive["name"], archive["format"], str(archive_path(archive))

    def get(self, archive: dict, files: dict[str, tuple[int, float]]) -> Reading | None:
        """The reading kept of these files as they are now, or None when there's none."""
        kept = self._kept.get(self._which(archive))
        if kept is None or any(kept[0].get(path) != state for path, state in files.items()):
            return None
        return kept[1]

    def put(self, archive: dict, files: dict[str, tuple[int, float]], reading: Reading) -> None:
        self._kept[self._which(archive)] = (dict(files), reading)


def scan(
    predictor: Predictor,
    now: float | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    reads: list[Reads] | None = None,
    readings: Readings | None = None,
) -> Scan:
    """Find tasks the dataset doesn't have yet, and stored tasks whose records changed.

    Only changed record files are read, each with its family, such as a session with its copies and helpers.
    A dataset read by an older version of the reader has every record file read again. `on_progress` is told
    how many bytes of them are read, and how many there are. `reads` are what to_read found, when it was asked
    already. Files that `readings` holds aren't read again, and the ones read are added to it.
    """
    now = time.time() if now is None else now
    stored = _stored(predictor)
    everything = predictor.config.get("reader") != READER_VERSION
    result = Scan(new_tasks=[], reader=READER_VERSION if everything else None)
    # A task can be in more than one archive, such as in a backup. Each copy is kept until one is picked.
    copies: dict[str, list[Task]] = {}
    held: set[str] = set()
    reread: set[str] = set()
    sessions: dict[str, set[str]] = {}
    # Every archive's changed files are found first, so the progress bar knows how much there is to read.
    reads = to_read(predictor) if reads is None else reads
    todo = [(found.archive, found.files) for found in reads if found.files]
    for found in reads:
        result.per_archive[found.archive["name"]] = 0
    unread = [(archive, files) for archive, files in todo if readings is None or readings.get(archive, files) is None]
    progress = _Bytes(sum(_sizes(files) for _, files in unread), on_progress)
    progress.reach(0)
    for archive, files in todo:
        reading = readings.get(archive, files) if readings is not None else None
        if reading is None:
            start = progress.done
            reading = read_records(archive, files, now, progress.read)
            # The files the reader passed over count as read too, such as a helper's whose session is gone.
            progress.reach(start + _sizes(files))
            if readings is not None:
                readings.put(archive, files, reading)
        for task in reading.tasks:
            if task.source and task.source not in files:
                continue  # a reading of more files than this scan needs holds other files' tasks too
            copies.setdefault(task.id, []).append(task)
        held |= reading.held
        reread |= set(files)
        sessions[archive["name"]] = reading.sessions
        # A file that may still be in use, such as one with a task that may still be running, is read again next time.
        settled = [path for path in files if now - files[path][1] >= SETTLE_SECONDS]
        result.files.update({path: files[path] for path in settled if path not in reading.open_files})
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


def recent_tasks(predictor: Predictor, transcript: str) -> list[Task]:
    """The tasks in a Claude Code session's project folder that changed since the last refresh: read, not saved.

    A forecast in Claude Code uses them, so the project's and the conversation's newest tasks count, as in
    training. Only that one folder is read, so it stays quick.
    """
    folder = Path(transcript).parent
    for archive in predictor.config.get("archives") or []:
        if archive.get("format") != "claude-code" or archive_path(archive) not in folder.parents:
            continue
        current = {}
        for path in folder.rglob("*.jsonl"):
            try:
                stat = path.stat()
            except OSError:
                continue
            current[str(path)] = (stat.st_size, stat.st_mtime)
        known = predictor.known_files()
        changed = {path for path, state in current.items() if known.get(path) != state}
        if not changed:
            return []
        try:
            # A file adds to tasks in the other files of its family, such as its session's helpers, so they're read too.
            family = families(archive, current)
            touched = {family[path] for path in changed}
            return read_records(archive, {path: at for path, at in current.items() if family[path] in touched}).tasks
        except (OSError, ValueError):
            return []
    return []


def read_all(
    archives: list[dict], now: float | None = None, on_progress: Callable[[int, int], None] | None = None
) -> Scan:
    """Read every task in the archives, for a new predictor.

    `on_progress` is told how many bytes of record files are read, and how many there are.
    """
    now = time.time() if now is None else now
    result = Scan(new_tasks=[], reader=READER_VERSION)
    # A task can be in more than one archive, such as in a backup. Only its most complete copy counts.
    copies: dict[str, list[Task]] = {}
    # Every archive is listed first, so the progress bar knows how much there is to read.
    listed = [(archive, record_files(archive)) for archive in archives]
    progress = _Bytes(sum(_sizes(current) for _, current in listed), on_progress)
    progress.reach(0)
    for archive, current in listed:
        start = progress.done
        reading = read_records(archive, current, now, progress.read)
        # The files the reader passed over count as read too, such as a helper's whose session is gone.
        progress.reach(start + _sizes(current))
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
    """The features the predictor uses. The prompt is always first.

    A feature this version doesn't know is skipped, so a predictor that lists one still trains and forecasts,
    without that feature's values. unknown_features names them.
    """
    unknown = set(unknown_features(predictor))
    return features.check([name for name in _listed(predictor) if name not in unknown])


def unknown_features(predictor: Predictor) -> list[str]:
    """The features the predictor's settings list that this version doesn't know, such as one a newer version
    added, or one from a plugin that no longer adds it. Load the predictor's plugins first."""
    known = features.available()
    return [name for name in dict.fromkeys(_listed(predictor)) if name != features.REQUIRED and name not in known]


def _listed(predictor: Predictor) -> list[str]:
    """The features the predictor's settings list, or the default ones when they list none."""
    listed = predictor.config.get("features")
    if not isinstance(listed, list) or not listed:
        return list(features.DEFAULT)
    return [name if isinstance(name, str) else json.dumps(name) for name in listed]


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


def task_info(task: Task, config: dict, labels: dict | None = None) -> features.TaskInfo:
    """A recorded task as features see it: the skills its options keep, and what LLM analysis said about it."""
    return features.TaskInfo.of(task, features.options(config, "skills"), (labels or {}).get(task.id))


def _texts(tasks: list[Task], chosen: list[str], limit: int | None, config: dict | None = None, labels=None):
    """Each learnable task's text for each chosen embedded feature, newest tasks first. A text may be empty.

    `config` holds the predictor's feature options and `labels` what LLM analysis said about each task.
    """
    fields = [name for name in chosen if features.is_text(name)]
    for task in learnable(tasks, limit):
        info = task_info(task, config or {}, labels)
        for name in fields:
            yield task.id, name, features.text(name, info)


def _made_from(stored, task_id: str, text: str) -> bool:
    """Whether the task has a stored vector made from this text."""
    if stored is None or task_id not in stored:
        return False
    made_from = getattr(stored, "made_from", None)
    if made_from is None:
        return True  # a plain set says only which tasks have a vector
    return made_from(task_id, text)


def embedding_items(
    tasks: list[Task],
    chosen: list[str],
    have: dict[str, set[str]] | None = None,
    limit: int | None = None,
    config: dict | None = None,
    labels: dict | None = None,
) -> list[tuple[str, str, str]]:
    """Each task's texts that need a vector, one for each chosen embedded feature, newest tasks first.

    Only tasks the model learns from are embedded, so stopped tasks and tasks left waiting are skipped.
    A task with no text for a feature, such as no context, has nothing to embed for it. `have` says which
    tasks have vectors, as predictor.embedded gives them, with the hash of the text each was made from. A
    vector made from another text is made again, as when a fuller copy of the task replaced it, a skill's
    description changed or LLM analysis described the task again. `config` holds the predictor's feature
    options and `labels` what LLM analysis said about each task.
    """
    have = have or {}
    return [
        (task_id, name, text)
        for task_id, name, text in _texts(tasks, chosen, limit, config, labels)
        if text.strip() and not _made_from(have.get(name), task_id, text)
    ]


def analysis_plan(tasks: list[Task], config: dict, labels: dict, limit: int | None = None) -> list:
    """The tasks LLM analysis still has to read, newest first: the ones it never read, or read with another model."""
    from . import llm

    how = config.get("llm") or {}
    if "llm" not in (config.get("features") or []) or how.get("provider") not in llm.PROVIDERS:
        return []
    fields = llm.parts(features.options(config, "llm"))
    return [
        llm.Ask(task.id, task.prompt, task.context, task.session)
        for task in learnable(tasks, limit)
        if llm.needs(labels.get(task.id), how, fields, llm.read_of(task.prompt, task.context))
    ]


def run_analysis(
    predictor: Predictor, asks: list, on_progress: Callable[[int, int], None] | None = None, budget=None
) -> tuple[int, str | None]:
    """Have the AI model read the tasks. Each request's answers are saved as they arrive, so a stopped run resumes.

    Returns how many tasks it answered, and why a request failed, when one did. A task it wouldn't answer is
    noted with how many times it was tried, so it's skipped after llm.MAX_TRIES. With an llm.Budget, no request
    starts that may take the spending past it.
    """
    from . import llm

    fields = llm.parts(features.options(predictor.config, "llm"))
    answered = 0

    def save(found: dict, done: int) -> None:
        nonlocal answered
        answered += sum(1 for labels in found.values() if not labels.get("failed"))
        # Merged with what the same AI model told any predictor about the task, as the store holds it then.
        predictor.put_llm_labels(found)
        if on_progress:
            on_progress(done, len(asks))

    if on_progress and asks:
        on_progress(0, len(asks))  # the bar starts with the step, so it can tell how fast it goes
    try:
        limited = {"budget": budget} if budget is not None else {}
        problem = llm.analyze(asks, predictor.config["llm"], fields, on_batch=save, **limited).problem
    finally:
        predictor.compact_features(answers=True)  # each batch was saved on its own
    return answered, problem  # counted only once every answer is in


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

    A task's vector counts only when it was made from the task's text as it is now. One made from another text,
    or for a text the task no longer has, stays in the feature store unused.
    """
    chosen = chosen_features(predictor)
    have = {name: predictor.embedded(name) for name in chosen if features.is_text(name)}
    tasks, limit = predictor.tasks(), predictor.config.get("max_tasks")
    labels = predictor.llm_labels() if "llm" in chosen else None
    items = embedding_items(tasks, chosen, have, limit, predictor.config, labels)
    return price(items, predictor.embedding_model, key)


@dataclass
class Refresh:
    """What a refresh does with what a scan found, worked out before anything is spent or saved."""

    found: Scan
    tasks: list[Task]
    """Every task the predictor has once the scan is saved."""
    embed: EmbedPlan
    """The texts to embed, priced. The descriptions LLM analysis writes aren't in it, as they don't exist yet."""
    asks: list
    """The tasks LLM analysis reads."""
    described: int
    """How many descriptions LLM analysis writes. Each is embedded once it's written."""
    model: dict | None
    """What the saved model says about itself, or None without one."""
    behind: str | None
    """Why the model is trained again even with nothing new: unfinished when training didn't finish last time,
    outdated when a version before the previous release made it, previous when the previous release did, damaged
    when a file of it is missing or can't be read."""
    stored: float = 0.0
    """About how many bytes the feature store holds for its tasks already: their vectors and LLM answers. Its data
    counts them, as they're its data too, outside its dataset."""

    @property
    def up_to_date(self) -> bool:
        """Whether there's nothing to do: nothing new or changed, nothing to embed or analyze, and a model that's
        current."""
        found = self.found
        changed = found.new_tasks or found.updated_tasks or found.removed
        return self.model is not None and not (changed or self.embed.items or self.asks or self.behind)


def plan_refresh(predictor: Predictor, found: Scan, key: str | None) -> Refresh:
    """What a refresh of the predictor does with what `found` holds, and what each step costs. Nothing is saved."""
    meta = model_meta(predictor) if model_ready(predictor) else None
    behind = None
    if predictor.config.get("needs_training"):
        behind = "unfinished"
    elif meta is not None and outdated(predictor, meta):
        behind = "outdated"
    elif meta is not None and from_previous_version(predictor, meta):
        behind = "previous"
    elif meta is not None and not loads(predictor):
        behind = "damaged"
    chosen = chosen_features(predictor)
    # A saved task that grew replaces its old copy. It may now be learned from, such as one that finished. One its
    # records no longer hold is taken out as the scan is saved, so nothing is planned, priced or analyzed for it.
    updated = {task.id: task for task in found.updated_tasks}
    gone = set(found.removed)
    everything = [updated.get(task.id, task) for task in predictor.tasks() if task.id not in gone] + found.new_tasks
    # The new tasks count too: the feature store may have their vectors already, from another predictor.
    ids = {task.id for task in everything}
    have = {name: predictor.embedded(name, ids) for name in chosen if features.is_text(name)}
    labels = predictor.llm_labels() if "llm" in chosen else None
    limit = predictor.config.get("max_tasks")
    pending = embedding_items(everything, chosen, have, limit, predictor.config, labels)
    asks = analysis_plan(everything, predictor.config, labels or {}, limit)
    described = len(asks) if asks and "description" in features.options(predictor.config, "llm") else 0
    from .estimate import DISK_PER_ANSWER, DISK_PER_VECTOR

    stored = DISK_PER_VECTOR * sum(len(vectors) for vectors in have.values()) + DISK_PER_ANSWER * len(labels or {})
    embed = price(pending, predictor.embedding_model, key)
    return Refresh(found, everything, embed, asks, described, meta, behind, stored)


def newest_that_fit(plan: EmbedPlan, tokens: int, more_per_task: int = 0) -> int:
    """How many of the newest tasks fit in a number of tokens, with all their texts.

    A text that several tasks share is paid for once, by the newest of them. A text the cache holds is free.
    `more_per_task` adds tokens for a text still to come, such as the description LLM analysis writes.
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
        per_task[task_id] = per_task.get(task_id, more_per_task) + count
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
    vector is saved with the hash of its text, so a task whose text changes later gets a new one. The feature
    store keeps them for every predictor.
    """
    try:
        _embed_plan(predictor, plan, key, on_progress, on_wait)
    finally:
        predictor.compact_features(fields=[name for _, name, _ in plan.items])  # each request was saved on its own


def _embed_plan(predictor: Predictor, plan: EmbedPlan, key: str, on_progress, on_wait) -> None:
    """Embed the plan's texts and save each task's vectors, as run_embeddings says."""
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
    if on_progress and texts:
        on_progress(0, len(texts))  # the bar starts with the step, so it can tell how fast it goes
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
        from . import ui

        kept = "Keeping the ones saved before." if (predictor.dir / "benchmarks.json").is_file() else ""
        return f"Couldn't download the model benchmarks ({ui.clean(error)}). {kept}".strip()
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


def feature_rows(predictor: Predictor, tasks: list[Task], known: list[Task] | None = None) -> list:
    """Each task's features, as the model reads them.

    What came before each task, in its project and, with the conversation feature, in its conversation, comes
    from `known`: by default every task the predictor has. A task's vector counts only when it was made from
    the task's text as it is now. One made from an older text, such as before a fuller copy of the task
    replaced it, describes another task, so it's left out.
    """
    return _rows_and_texts(predictor, tasks, known)[0]


def _rows_and_texts(
    predictor: Predictor, tasks: list[Task], known: list[Task] | None = None
) -> tuple[list, list[dict[str, str]]]:
    """Each task's features, as feature_rows gives them, and for each task, the hash of the text that each of its
    vectors was made from, by text feature."""
    from .model import Features

    chosen = chosen_features(predictor)
    conversation = features.built_in("conversation", chosen)
    names = [name for name in chosen if features.is_text(name)]
    ids = {task.id for task in tasks}
    stored = {name: predictor.get_embeddings(name, ids) for name in names}
    labels = predictor.llm_labels() if "llm" in chosen else None
    table = benchmark_table(predictor)
    past = Past(predictor.tasks() if known is None else known)
    rows, texts = [], []
    for task in tasks:
        info = task_info(task, predictor.config, labels)
        vectors, made_from = {}, {}
        for name in names:
            made = stored[name].get(task.id)  # by the hash of each text the task had
            digest = embeddings.text_hash(features.text(name, info)) if made else ""
            vector = made.get(digest) if made else None
            if vector is not None:
                vectors[name] = vector
                made_from[name] = digest
        texts.append(made_from)
        before = past.numbers(
            task.project,
            task.session,
            task.started_at,
            task.origin,
            task.context_tokens,
            task.agent,
            task.prompt,
            conversation=conversation,
        )
        rows.append(
            Features(
                model=task.model,
                effort=task.effort,
                agent=task.agent,
                origin=task.origin,
                prompt_chars=len(task.prompt),
                vectors=vectors,
                numbers={**features.numbers(chosen, info, table, predictor.config), **before},
                started_at=task.started_at,
            )
        )
    return rows, texts


@dataclass
class FeatureUse:
    """One of a predictor's features, as the features command lists it."""

    name: str
    """The feature's name, such as context, or prompt, task, similar or history for the ones always used."""
    title: str
    says: str
    stored: list = field(default_factory=list)
    """The feature store's features it reads, as feature_store.Feature. Empty when training works it out."""
    covered: int | None = None
    """How many of the tasks the predictor learns from have a value for it. None when it wasn't counted."""
    known: bool = True
    """False for a feature the settings list that this version doesn't know. It's skipped."""


_CONVERSATION_NUMBERS = frozenset(
    {
        "session_tasks_before",
        "previous_seconds",
        "previous_tokens",
        "previous_direct",
        "previous_usd",
        "minutes_since_previous",
        "conversation_tokens",
        "conversation_tokens_program",
    }
)
"""The numbers history.Past adds with the conversation feature."""


def _owner(number: str, own: set[str]) -> str:
    """The feature a number belongs to: one of yours in `own`, or the built-in feature that adds it."""
    if number in own:
        return number
    if number in features.shape("") or number == "effort_level":
        return "task"
    if number.startswith("project_"):
        return "history"
    if number in _CONVERSATION_NUMBERS:
        return "conversation"
    if number in benchmarks.FACTS:
        return "benchmarks"
    if number in features.time_numbers(features.TREND_FROM, ["trend", "clock"]):
        return "time"
    if number.startswith("llm_"):
        return "llm"
    return number


def feature_uses(predictor: Predictor, count: bool = True) -> tuple[int | None, list[FeatureUse]]:
    """Each of the predictor's features, and how many of the tasks it learns from have a value for it.

    It counts from the rows training reads, so it says what training sees: a text feature has a value when
    the task has its vector, made from its text as it is now, and a number feature when any of its numbers
    does. The first value is how many tasks it learns from. Without `count`, nothing is counted, and both are
    None. Load the predictor's plugins first, or its own features count as unknown.
    """
    chosen = chosen_features(predictor)
    menu = {group.key: group for group in features.MENU}
    total = rows = None
    if count:
        tasks = learnable(predictor.tasks(), predictor.config.get("max_tasks"))
        rows = feature_rows(predictor, tasks)
        total = len(rows)
    own = {name for name in features.available() if name not in features.BUILT_IN and not features.is_text(name)}
    owners: dict[str, str] = {}
    with_vectors: dict[str, int] = {}
    with_numbers: dict[str, int] = {}
    any_vector = read = 0
    for row in rows or []:
        for name in row.vectors:
            with_vectors[name] = with_vectors.get(name, 0) + 1
        found = set()
        for number in row.numbers:
            if number not in owners:
                owners[number] = _owner(number, own)
            found.add(owners[number])
        for owner in found:
            with_numbers[owner] = with_numbers.get(owner, 0) + 1
        any_vector += bool(row.vectors)
        # LLM analysis has a value for a task when its answer gave a number or its description has a vector.
        read += "llm" in found or "llm" in row.vectors

    def counted(found: int) -> int | None:
        return found if count else None

    prompt_says = "what the prompt means, read by Voyage"
    task_says = "how the prompt is laid out, the AI model, the effort level and who started it"
    uses = [
        FeatureUse("prompt", "Prompt", prompt_says, [predictor.embedding_feature("prompt")]),
        FeatureUse("task", "Task", task_says),
        FeatureUse("similar", menu["similar"].title, menu["similar"].says),
        FeatureUse("history", menu["history"].title, menu["history"].says),
    ]
    uses[0].covered = counted(with_vectors.get("prompt", 0))
    uses[1].covered = counted(with_numbers.get("task", 0))
    uses[2].covered = counted(any_vector)
    uses[3].covered = counted(with_numbers.get("history", 0))
    for name in chosen:
        if name == features.REQUIRED:
            continue
        group = menu.get(name)
        title, says = (group.title, group.says) if group else (name, features.available()[name])
        use = FeatureUse(name, title, says)
        if name == "llm":
            answers = predictor.llm_feature()
            use.stored = [answers] if answers is not None else []
            if "description" in features.options(predictor.config, "llm"):
                use.stored.append(predictor.embedding_feature("llm"))
            use.covered = counted(read)
        elif features.is_text(name):
            use.stored = [predictor.embedding_feature(name)]
            use.covered = counted(with_vectors.get(name, 0))
        else:
            use.covered = counted(with_numbers.get(name, 0))
        uses.append(use)
    for name in unknown_features(predictor):
        uses.append(FeatureUse(name, name, "a feature this version doesn't know, so it's skipped", known=False))
    return total, uses


@dataclass
class Remake:
    """What making a deleted stored feature again takes. The next refresh of each predictor that uses it makes it."""

    tasks: int = 0
    """How many tasks get it again: the ones each predictor learns from that have its text, or that LLM analysis
    reads. A task several predictors share counts once."""
    analyses: int = 0
    """For LLM answers: the most times LLM analysis reads the tasks when the predictors refresh one at a time, which
    is what it costs then, at most. A refresh has a task read when the answer saved so far lacks a part it asks for,
    such as the kind of task or a description. Predictors that ask for the same parts share one reading of each
    task. Like every estimate here, it counts each reading as answered: a task the AI model won't answer is tried
    again later."""
    analyses_least: int = 0
    """For LLM answers: the fewest times LLM analysis reads the tasks when the predictors refresh one at a time.
    It's fewer than the most when a predictor that asks for more parts can refresh first, so that the others find
    their parts already saved."""
    texts: int = 0
    """For embeddings: how many different texts are embedded again."""
    cached: int = 0
    """For embeddings: how many of those texts the cache of vectors holds. They come back for free."""
    tokens: int = 0
    """For embeddings: the tokens of the other texts, which go to Voyage again."""
    usd: float | None = 0.0
    """What it costs at list price: Voyage's for embeddings, the AI model's for LLM answers. None when the price
    isn't known, and for LLM answers through a subscription, which has no extra bill."""
    exact: bool = True
    """Whether the tokens were counted with Voyage's own tokenizer. Otherwise they're an estimate."""


def remake(feature, predictors: list[Predictor]) -> Remake:
    """What making a stored feature again takes, as the next refresh of each of `predictors` would make it. They
    are the predictors that use it, with their plugins loaded. It reads and changes nothing in the feature store."""
    from . import feature_store, llm
    from .store import llm_answers

    if feature.kind == feature_store.LLM:
        # The sets of parts the predictors ask for about each task. How many readings a task takes depends on which
        # of them refreshes first, so it's counted both ways: the fewest and the most.
        asked: dict[str, set[frozenset[str]]] = {}
        for predictor in predictors:
            parts = frozenset(llm.parts(features.options(predictor.config, "llm")))
            for ask in analysis_plan(predictor.tasks(), predictor.config, {}, predictor.config.get("max_tasks")):
                asked.setdefault(ask.id, set()).add(parts)
        readings = [_readings(frozenset(sets)) for sets in asked.values()]
        least, most = sum(low for low, _ in readings), sum(high for _, high in readings)
        provider, model = feature.made_with.get("provider"), str(feature.made_with.get("model"))
        billed = provider in llm.PROVIDERS and llm.PROVIDERS[provider].billed
        return Remake(
            tasks=len(asked),
            analyses=most,
            analyses_least=least,
            usd=llm.cost(most, model) if billed else None,
        )
    field, model = feature.made_with.get("field"), str(feature.made_with.get("model"))
    items: list[tuple[str, str, str]] = []
    for predictor in predictors:
        # The descriptions LLM analysis wrote, as the store keeps them. It's only read, so nothing moves in.
        labels = llm_answers(predictor.config) if field == "llm" else None
        limit = predictor.config.get("max_tasks")
        items += embedding_items(predictor.tasks(), [field], None, limit, predictor.config, labels)
    plan = price(items, model, None)
    texts = len({text for _, _, text in items})
    return Remake(
        tasks=len({task_id for task_id, _, _ in items}),
        texts=texts,
        cached=texts - len(plan.texts),
        tokens=plan.cost.tokens,
        usd=plan.cost.list_usd if model in embeddings.MODELS else None,
        exact=plan.cost.exact,
    )


@cache
def _readings(sets: frozenset[frozenset[str]], saved: frozenset[str] | None = None) -> tuple[int, int]:
    """The fewest and the most times LLM analysis reads a task, when predictors asking for these sets of parts
    each refresh once, one at a time, in any order, and every reading is answered. `saved` holds the parts of the
    task's answer saved so far, None when there's no answer. A refresh has the task read when there's no answer or
    it lacks a part the refresh asks for, and saves the parts it asked for with the others."""
    after = [_readings(sets, (saved or frozenset()) | parts) for parts in sets if saved is None or not parts <= saved]
    if not after:
        return 0, 0
    return 1 + min(low for low, _ in after), 1 + max(high for _, high in after)


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


def train(predictor: Predictor, on_progress: Callable[[int, int], None] | None = None):
    """Train the predictor's model on its whole dataset and save it, with the prepared dataset: what went into it.

    The built-in model tells `on_progress` how many of the models it fits are done, and how many there are. A
    model type from a plugin trains as it always has, without saying.
    """
    load_plugins(predictor)
    tasks = predictor.tasks()
    if predictor.config.get("max_tasks"):
        # The same newest tasks that were embedded, oldest first.
        tasks = learnable(tasks, predictor.config["max_tasks"])[::-1]
    model_settings = settings(predictor)
    kind = model_settings.get("type", "xgboost")
    trainer, _ = model_types.get(kind)
    rows, texts = _rows_and_texts(predictor, tasks)
    if kind == "xgboost" and on_progress is not None:
        model = trainer(tasks, rows, model_settings, on_progress=on_progress)
    else:
        model = trainer(tasks, rows, model_settings)
    if isinstance(getattr(model, "meta", None), dict):
        model.meta["features_without_values"] = features_without_values(predictor, rows)
    prepared = prepared_dataset(predictor, tasks, rows, texts)
    from . import files

    with _model_lock(predictor):
        # A save that was stopped halfway leaves its folder behind. No save runs while this lock is held.
        for unfinished in predictor.dir.glob(".model-*"):
            shutil.rmtree(unfinished, ignore_errors=True)
        model.save(predictor.model_dir)
        files.write_bytes(predictor.prepared_path, prepared)  # in one step, with the model it went into
    predictor.config["refreshed_at"] = time.time()
    predictor.config["needs_training"] = False
    predictor.save_config()
    return model


PREPARED_FORMAT = 1
"""The prepared dataset's format, in the metadata of its file."""


def prepared_dataset(predictor: Predictor, tasks: list[Task], rows: list, texts: list[dict[str, str]]) -> bytes:
    """What went into a training, as a Parquet file: one row for each task the model learned from, oldest first.

    `rows` are the tasks' features, as training read them, and `texts` the hashes of the texts their vectors were
    made from, as _rows_and_texts gives both. Each row has the task's id, when it started, its AI model, effort
    level, agent and who started it, each number feature, for each text feature the hash of the text whose
    vector the model read, and what the task took: seconds, tokens, dollars and whether it was a direct answer.
    A number or a text the task has none of is empty. It never holds a prompt or any other text, nor a vector.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    model_settings = settings(predictor)
    learned = _learned_from(tasks, rows, model_settings)
    found = sorted(
        ((task, row, made) for task, row, made in zip(tasks, rows, texts) if task.id in learned),
        key=lambda one: (one[0].started_at, one[0].id),
    )
    chosen = chosen_features(predictor)
    numbers = list(dict.fromkeys(name for _, row, _ in found for name in row.numbers))
    columns = {
        "task_id": pa.array([task.id for task, _, _ in found], pa.string()),
        "started_at": pa.array([task.started_at for task, _, _ in found], pa.float64()),
        **{
            name: pa.array([str(getattr(row, name)) for _, row, _ in found], pa.string())
            for name in ("model", "effort", "agent", "origin")
        },
        "number.prompt_chars": pa.array([row.prompt_chars for _, row, _ in found], pa.int64()),
        **{
            f"number.{name}": pa.array([row.numbers.get(name) for _, row, _ in found], pa.float64()) for name in numbers
        },
        **{
            f"text_hash.{name}": pa.array([made.get(name) for _, _, made in found], pa.string())
            for name in chosen
            if features.is_text(name)
        },
        "seconds": pa.array([task.seconds for task, _, _ in found], pa.float64()),
        "tokens": pa.array([task.tokens for task, _, _ in found], pa.int64()),
        "usd": pa.array([task.usd for task, _, _ in found], pa.float64()),
        "direct": pa.array([bool(task.direct) for task, _, _ in found], pa.bool_()),
    }
    meta = {"format": PREPARED_FORMAT, "features": chosen, "model_type": model_settings.get("type", "xgboost")}
    table = pa.table(columns).replace_schema_metadata({b"yardstick": json.dumps(meta)})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue().to_pybytes()


def _learned_from(tasks: list[Task], rows: list, model_settings: dict) -> set[str]:
    """The ids of the tasks a model learned from: the ones the built-in model picks, the number its "tasks" counts.
    A model type from a plugin picks its own, so for one, every task it got that can be learned from."""
    from . import model

    if model_settings.get("type", "xgboost") == "xgboost":
        return {task.id for task in model.learned_from(tasks, rows, model_settings)}
    return {task.id for task in model.usable(tasks)}


def outdated(predictor: Predictor, meta: dict | None = None) -> bool:
    """Whether the saved model was made by a version before the previous release, which read other features. It
    gives no forecasts until it's trained again.

    `meta` is what the saved model says about itself, when it's read already. Only the built-in model is checked.
    """
    saved = _saved_format(predictor, meta)
    return saved is not None and saved < model_types.PREVIOUS_FORMAT


def from_previous_version(predictor: Predictor, meta: dict | None = None) -> bool:
    """Whether the saved model was made by the previous release. It still forecasts, as that release did, and
    training it again brings this version's improvements. `meta` is as outdated takes it."""
    saved = _saved_format(predictor, meta)
    return saved is not None and model_types.PREVIOUS_FORMAT <= saved < model_types.BUILT_IN_FORMAT


def _saved_format(predictor: Predictor, meta: dict | None) -> int | None:
    """The format the built-in model was saved in, 0 when it doesn't say. None without a built-in model."""
    if settings(predictor).get("type", "xgboost") != "xgboost":
        return None
    meta = model_meta(predictor) if meta is None else meta
    if meta is None:
        return None
    saved = meta.get("format")
    return saved if isinstance(saved, int) and not isinstance(saved, bool) else 0


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


def loads(predictor: Predictor) -> bool:
    """Whether the predictor's model loads, every file of it. False when one is missing or can't be read."""
    try:
        load_model(predictor)
    except (OSError, ValueError, KeyError, EOFError):
        return False
    return True


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
