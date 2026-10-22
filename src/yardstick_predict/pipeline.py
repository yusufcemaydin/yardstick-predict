"""The steps behind setup and refresh: read the records, add new tasks, embed them and train."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import benchmarks, embeddings, features, model_types
from .archives import SETTLE_SECONDS, Task, codex_families, keep_most_complete, read_archive, record_files
from .store import Predictor


@dataclass
class Scan:
    """What a look through the archives found."""

    new_tasks: list[Task]
    files: dict[str, tuple[int, float]] = field(default_factory=dict)
    per_archive: dict[str, int] = field(default_factory=dict)
    updated_tasks: list[Task] = field(default_factory=list)
    """Tasks already stored that now have more model calls, such as a parent whose helper finished later."""


def scan(predictor: Predictor, now: float | None = None) -> Scan:
    """Find tasks the dataset doesn't have yet, and stored tasks that grew. Only changed record files are read."""
    now = time.time() if now is None else now
    known_files = predictor.known_files()
    known_calls = predictor.known_calls()
    result = Scan(new_tasks=[])
    # A task can be in more than one archive, such as in a backup. Only its most complete copy counts.
    copies: dict[str, Task] = {}
    for archive in predictor.config["archives"]:
        current = record_files(archive)
        changed = {path for path, state in current.items() if known_files.get(path) != state}
        if archive["format"] == "claude-code":
            # A helper agent's record lives in a subfolder. When it changes, read its session again.
            for path in list(changed):
                parent = Path(path).parent
                if parent.name == "subagents":
                    main = str(parent.parent) + ".jsonl"
                    if main in current:
                        changed.add(main)
        if archive["format"] == "codex" and changed:
            # A Codex helper thread adds to a task in the main thread at the top of its family.
            # When any thread in a family changes, read the whole family again.
            family = codex_families(current)
            touched = {family[path] for path in changed}
            changed.update(path for path in current if family[path] in touched)
        for task in read_archive(archive, only=changed, now=now) if changed else []:
            keep_most_complete(copies, task)
        result.per_archive[archive["name"]] = 0
        # A file written to very recently may end with a task that's still running. Read it again next time.
        result.files.update({p: current[p] for p in changed if now - current[p][1] >= SETTLE_SECONDS})
    for task in copies.values():
        if task.id not in known_calls:
            result.new_tasks.append(task)
            result.per_archive[task.archive] = result.per_archive.get(task.archive, 0) + 1
        elif task.calls > known_calls[task.id]:
            result.updated_tasks.append(task)
    return result


def save_scan(predictor: Predictor, found: Scan) -> int:
    # The model is out of date until training succeeds, even if this run stops before then.
    predictor.config["needs_training"] = True
    predictor.save_config()
    added = predictor.add_tasks(found.new_tasks)
    predictor.replace_tasks(found.updated_tasks)
    predictor.remember_files(found.files)
    return added


def read_all(archives: list[dict], now: float | None = None) -> Scan:
    """Read every task in the archives, for a new predictor."""
    now = time.time() if now is None else now
    result = Scan(new_tasks=[])
    # A task can be in more than one archive, such as in a backup. Only its most complete copy counts.
    copies: dict[str, Task] = {}
    for archive in archives:
        current = record_files(archive)
        for task in read_archive(archive, now=now):
            keep_most_complete(copies, task)
        result.per_archive[archive["name"]] = 0
        result.files.update({p: s for p, s in current.items() if now - s[1] >= SETTLE_SECONDS})
    result.new_tasks = list(copies.values())
    for task in result.new_tasks:
        result.per_archive[task.archive] = result.per_archive.get(task.archive, 0) + 1
    return result


@dataclass
class EmbedPlan:
    """The texts still to embed, and what embedding them costs."""

    items: list[tuple[str, str, str]]
    counts: list[int]
    cost: embeddings.Cost

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
    model_types.load_plugins(predictor.config.get("plugins") or [])


def learnable(tasks: list[Task], limit: int | None = None) -> list[Task]:
    """The tasks the model learns from, newest first. Stopped tasks and tasks left waiting are left out.

    With `limit`, only the newest that many. Embedding and training both pick tasks here, so they agree.
    """
    kept = sorted((t for t in tasks if not t.stopped and not t.idle), key=lambda t: (t.started_at, t.id), reverse=True)
    return kept[: int(limit)] if limit else kept


def embedding_items(
    tasks: list[Task], chosen: list[str], have: dict[str, set[str]] | None = None, limit: int | None = None
) -> list[tuple[str, str, str]]:
    """Each task's texts to embed, one for each chosen embedded feature, newest tasks first.

    Only tasks the model learns from are embedded, so stopped tasks and tasks left waiting are skipped.
    A task with no text for a feature, such as no context, has nothing to embed for it.
    """
    have = have or {}
    fields = [name for name in chosen if features.is_text(name)]
    items = []
    for task in learnable(tasks, limit):
        info = features.TaskInfo.of(task)
        for name in fields:
            if task.id in have.get(name, set()):
                continue
            text = features.text(name, info)
            if text.strip():
                items.append((task.id, name, text))
    return items


def price(items: list[tuple[str, str, str]], model: str, key: str | None) -> EmbedPlan:
    """Count the items' tokens with the model's own tokenizer and price them."""
    if not items:
        return EmbedPlan([], [], embeddings.Cost(0, True, 0.0, None))
    counts, exact = embeddings.count_tokens([text for _, _, text in items], model)
    return EmbedPlan(items, counts, embeddings.cost(sum(counts), exact, model, key))


def plan_embeddings(predictor: Predictor, key: str | None) -> EmbedPlan:
    """What the predictor still has to embed, and what it costs."""
    chosen = chosen_features(predictor)
    have = {name: predictor.embedded(name) for name in chosen if features.is_text(name)}
    items = embedding_items(predictor.tasks(), chosen, have, predictor.config.get("max_tasks"))
    return price(items, predictor.embedding_model, key)


def newest_that_fit(plan: EmbedPlan, tokens: int) -> int:
    """How many of the newest tasks fit in a number of tokens, with all their texts."""
    # A task counts only when all its texts fit. The plan lists the newest tasks first, and so does this.
    per_task: dict[str, int] = {}
    for (task_id, _, _), count in zip(plan.items, plan.counts):
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
    """Embed the plan's texts. Each request's results are saved at once, so a stopped run resumes."""
    texts = [text for _, _, text in plan.items]
    done = 0
    for positions, vectors in embeddings.embed(texts, predictor.embedding_model, key, plan.counts, on_wait):
        by_field: dict[str, dict] = {}
        for position, vector in zip(positions, vectors):
            task_id, name, _ = plan.items[position]
            by_field.setdefault(name, {})[task_id] = vector
        for name, found in by_field.items():
            predictor.put_embeddings(name, found)
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
    (predictor.dir / "benchmarks.json").write_text(json.dumps(table) + "\n")
    return None


def benchmark_table(predictor: Predictor) -> dict:
    try:
        return json.loads((predictor.dir / "benchmarks.json").read_text())
    except (OSError, ValueError):
        return {}


def feature_rows(predictor: Predictor, tasks: list[Task]) -> list:
    """Each task's features, as the model reads them."""
    from .model import Features

    chosen = chosen_features(predictor)
    stored = {name: predictor.get_embeddings(name) for name in chosen if features.is_text(name)}
    table = benchmark_table(predictor)
    rows = []
    for task in tasks:
        info = features.TaskInfo.of(task)
        rows.append(
            Features(
                model=task.model,
                effort=task.effort,
                agent=task.agent,
                origin=task.origin,
                prompt_chars=len(task.prompt),
                vectors={name: found[task.id] for name, found in stored.items() if task.id in found},
                numbers=features.numbers(chosen, info, table),
            )
        )
    return rows


def train(predictor: Predictor):
    """Train the predictor's model on its whole dataset and save it."""
    load_plugins(predictor)
    tasks = predictor.tasks()
    if predictor.config.get("max_tasks"):
        # The same newest tasks that were embedded, oldest first.
        tasks = learnable(tasks, predictor.config["max_tasks"])[::-1]
    model_settings = settings(predictor)
    trainer, _ = model_types.get(model_settings.get("type", "xgboost"))
    model = trainer(tasks, feature_rows(predictor, tasks), model_settings)
    model.save(predictor.model_dir)
    predictor.config["refreshed_at"] = time.time()
    predictor.config["needs_training"] = False
    predictor.save_config()
    return model


def load_model(predictor: Predictor):
    load_plugins(predictor)
    _, loader = model_types.get(settings(predictor).get("type", "xgboost"))
    return loader(predictor.model_dir)
