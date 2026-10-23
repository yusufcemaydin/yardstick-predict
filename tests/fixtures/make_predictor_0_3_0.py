"""Make predictor-0.3.0: a small predictor saved by yardstick-predict 0.3.0's own code, and what it forecast.

It runs with 0.3.0's code, never this version's. README.md in this folder shows how. It reads nothing of
yours: its tasks are made up, a stand-in gives each text a small vector, and another stands in for LLM
analysis. Everything it writes goes to a new temporary folder first. Only the predictor, without its lock
files, is copied here, with forecasts-0.3.0.json: requests, and what 0.3.0 answered to them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PREDICTOR = HERE / "predictor-0.3.0"
FORECASTS = HERE / "forecasts-0.3.0.json"
NAME = "default"
START = 1_780_000_000.0
DIMENSIONS = 16
"""Each vector's size. Voyage's have 256 numbers. Fewer keep the files small, and 0.3.0 takes any size."""
WORDS = (
    "fix add test refactor page login api cache docs build deploy schema query parser cli model "
    "upload search report export import config theme email billing invoice chart table form"
).split()
MODELS = (
    # The AI model, its agent, its effort levels and how many tasks it ran. The last has too few to be tested.
    ("claude-opus-5-5", "claude-code", ("high", "max"), 100),
    ("gpt-6.1-sol", "codex", ("xhigh",), 80),
    ("claude-haiku-5-5", "claude-code", ("low",), 12),
)
PROJECTS = ("shop", "blog", "tools")
KINDS = ("feature", "bug_fix", "refactor", "small_change")
SEED = 20
"""Picked so that 0.3.0 trains three versions of the model, and the requests below reach each of them."""
CLEAR = 1e-4
"""How much closer, at least, each forecast's similar tasks must be than the next ones in line. Processors
round the last digits of a sum differently, so a nearer tie could pick other similar tasks on another computer."""


def vector(text: str) -> list[float]:
    """A stand-in for Voyage: similar texts get similar vectors.

    Each word points its own way, and texts that share words point the same way. The whole text adds a little
    of its own, so two texts with the same words in another order get different vectors.
    """
    found = [0.0] * DIMENSIONS
    for part, weight in [(word, 1.0) for word in text.lower().split()] + [(text, 0.3)]:
        digest = hashlib.sha256(part.encode()).digest()
        for i in range(DIMENSIONS):
            found[i] += weight * (digest[i] / 255 - 0.5)
    norm = math.sqrt(sum(v * v for v in found)) or 1.0
    return [round(v / norm, 6) for v in found]


def made_up(Task) -> tuple[list, dict[str, dict]]:
    """Tasks in conversations of a few each, and what LLM analysis says about each one.

    Claude Opus's time follows the work LLM analysis foresees and how hard its conversation's work is. GPT's
    follows the prompt's length and, less, how hard its conversation's work is. What LLM analysis says about
    GPT's tasks is noise. Claude Haiku has too few tasks to be tested.
    """
    rng = random.Random(SEED)
    left = {name: count for name, _, _, count in MODELS}
    tasks, said = [], {}
    clock, number, sessions = START, 0, 0
    while any(left.values()):
        name, agent, efforts, _ = rng.choice([m for m in MODELS if left[m[0]]])
        session, project = f"session-{sessions:03d}", rng.choice(PROJECTS)
        sessions += 1
        hard = rng.gauss(0, 0.7)  # how hard this conversation's work is
        previous = ""
        for position in range(min(rng.randint(3, 6), left[name])):
            words = rng.randint(2, 30)
            prompt = " ".join(rng.choice(WORDS) for _ in range(words))
            direct = words < 8 and rng.random() < 0.8
            minutes = math.exp(rng.gauss(math.log(2 + words / 4), 0.5)) * (0.2 if direct else 1.0)
            if name.startswith("claude-opus"):
                seconds = math.exp(math.log(60 * minutes) + 0.6 * hard + rng.gauss(0, 0.15))
                tokens = int(30_000 * (1 + minutes) * math.exp(0.4 * hard + rng.gauss(0, 0.15)))
                foreseen = minutes * math.exp(rng.gauss(0, 0.1))
            else:
                pace = 2.0 if name.startswith("gpt") else 1.5
                seconds = math.exp(math.log(pace * (5 if direct else 30 + 4 * words)) + 0.3 * hard + rng.gauss(0, 0.25))
                size = 20_000 if direct else 40_000 * (1 + words / 10)
                tokens = int(math.exp(math.log(size) + 0.3 * hard + rng.gauss(0, 0.25)))
                foreseen = math.exp(rng.gauss(1.0, 0.8))
            started = clock
            task = Task(
                id=f"task-{number:04d}",
                archive=agent,
                agent=agent,
                agent_version="2.9.0" if agent == "claude-code" else "0.160.0",
                session=session,
                project=project,
                origin="program" if rng.random() < 0.2 else "person",
                started_at=started,
                model=name,
                effort=efforts[position % len(efforts)],
                prompt=prompt,
                context=f"{previous}\nDone." if previous else "",
                context_tokens=18_000 + 6_000 * position,
                seconds=round(seconds, 3),
                input_tokens=tokens // 10,
                cache_write_tokens=tokens // 20,
                cache_read_tokens=tokens - tokens // 10 - tokens // 20 - 300,
                output_tokens=300,
                usd=round(tokens * 2 / 1e6, 6),
                calls=1 if direct else 2 + words // 8,
                direct=direct,
                stopped=False,
                idle=False,
                finished_at=round(started + seconds + rng.uniform(0, 20), 3),
            )
            tasks.append(task)
            # LLM analysis reads Claude Opus's tasks well, and says nothing useful about the others'.
            knows = name.startswith("claude-opus")
            seems_direct = direct if knows else rng.random() < 0.3
            mentioned = prompt.split() if knows else [rng.choice(WORDS), rng.choice(WORDS)]
            if rng.random() < 0.08:
                said[task.id] = {"failed": True}  # LLM analysis couldn't read this one
            else:
                said[task.id] = {
                    "kind": "question" if seems_direct else rng.choice(KINDS),
                    "files": float(0 if seems_direct else 1 + (words if knows else rng.randint(2, 30)) // 10),
                    "steps": float(1 if seems_direct else 1 + (words if knows else rng.randint(2, 30)) // 6),
                    "runs_code": not seems_direct,
                    "looks_up": rng.random() < 0.2,
                    "continues": position > 0,
                    "clarity": rng.choice(("clear", "some_guessing")),
                    "minutes": round(foreseen, 2),
                    "description": f"Change the {mentioned[0]} and the {mentioned[-1]}, then check the "
                    f"{rng.choice(WORDS)} and the {rng.choice(WORDS)}.",
                }
            previous, number, left[name] = prompt, number + 1, left[name] - 1
            clock = task.finished_at + rng.uniform(30, 300)
        clock += rng.uniform(600, 7200)
    return tasks, said


def answer_of(kind: str, files: int, steps: int, minutes: float, description: str) -> dict:
    """What LLM analysis says about a forecast request's task."""
    return {
        "kind": kind,
        "files": float(files),
        "steps": float(steps),
        "runs_code": True,
        "looks_up": False,
        "continues": False,
        "clarity": "clear",
        "minutes": minutes,
        "description": description,
    }


def watch_similar_tasks(model) -> list[float]:
    """Keep, for every forecast, how much closer its similar tasks are than the next ones in line: its closest
    task, its closest ten and its closest hundred, the ones the model summarizes."""
    import numpy as np

    gaps: list[float] = []
    real = model.similar_tasks

    def similar_tasks(pool, results, queries, has, earlier_only, *args, **kwargs):
        if not earlier_only:
            asked = queries[queries.any(axis=1)].astype(float)
            similarity = -np.sort(-(asked @ pool.astype(float).T), axis=1)
            for rank in (0, model.CLOSE - 1, model.NEIGHBORS - 1):
                if rank + 1 < similarity.shape[1]:
                    gaps.extend((similarity[:, rank] - similarity[:, rank + 1]).tolist())
        return real(pool, results, queries, has, earlier_only, *args, **kwargs)

    model.similar_tasks = similar_tasks
    return gaps


def request(tasks: list) -> dict:
    """Forecast requests that reach every way 0.3.0 forecast: each AI model's version, the version without LLM
    analysis for a task it couldn't read, a conversation it knows and a new one, and the reasons for no forecast."""
    last = next(t for t in reversed(tasks) if t.model == "claude-opus-5-5")
    every = ["claude-opus-5-5", "gpt-6.1-sol", "claude-haiku-5-5", "brand-new-model"]
    known = {"project": last.project, "session": last.session, "context": f"{last.prompt}\nDone."}
    return {
        "version": 1,
        "distribution": True,
        "tasks": [
            {"id": "read", "prompt": "add a search page with tests and docs", "models": every, **known},
            {"id": "unread", "prompt": "fix the billing api cache", "models": every, **known, "origin": "person"},
            {
                "id": "new-conversation",
                "prompt": "refactor the parser",
                "models": ["claude-opus-5-5", "gpt-6.1-sol"],
                "origin": "program",
                "project": "blog",
                "conversation_tokens": 30_000,
            },
            {"id": "effort", "prompt": "export the report table", "models": ["claude-opus-5-5"], "effort": "max"},
            {"id": "unknown-effort", "prompt": "fix the login form", "models": ["claude-opus-5-5"], "effort": "ultra"},
        ],
    }


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="yardstick-0.3.0-"))
    try:
        return make(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make(work: Path) -> int:
    """Make the predictor and its forecasts in `work`, then copy them here."""
    # Everything 0.3.0 reads or writes stays in the new folder, so nothing of yours is read or changed.
    os.environ.update(
        HOME=str(work), YARDSTICK_HOME=str(work / "yardstick"), PYTHON_KEYRING_BACKEND="keyring.backends.fail.Keyring"
    )
    for name in ("VOYAGE_API_KEY", "VOYAGE_API_BASE", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(name, None)
    import yardstick_predict
    from yardstick_predict import embeddings, forecast, llm, menu, model, model_types, pipeline
    from yardstick_predict.archives import READER_VERSION, Task
    from yardstick_predict.store import Predictor, set_default

    if yardstick_predict.__version__ != "0.3.0" or model_types.BUILT_IN_FORMAT != 3:
        sys.exit(f"Run this with yardstick-predict 0.3.0, not {yardstick_predict.__version__}, as README.md shows.")
    tasks, said = made_up(Task)

    # The predictor, as `create --features context,llm --llm-provider codex --test-windows 1` makes it.
    picked = menu.from_flags("context,llm", "codex", None)
    archives = [
        {"name": "claude-code", "format": "claude-code", "path": "/records/claude-code"},
        {"name": "codex", "format": "codex", "path": "/records/codex"},
    ]
    embedding = {"model": "voyage-4", "dimensions": DIMENSIONS}
    settings = model.settings_for({"test_windows": 1})
    predictor = Predictor.create(NAME, archives, embedding, **picked.config(), model=settings, plugins=[])
    set_default(NAME)
    pipeline.save_scan(predictor, pipeline.Scan(new_tasks=tasks, reader=READER_VERSION))

    # LLM analysis, with a stand-in for the AI model. A task it couldn't read is asked again, as a refresh does.
    def analyze(asks, how, fields, on_batch=None):
        by = f"{how['provider']}:{how['model']}"
        found = {ask.id: {**said[ask.id], "by": by, "fields": fields} for ask in asks}
        if on_batch is not None:
            on_batch(found, len(asks))
        return llm.Analysis(found)

    llm.analyze = analyze
    for _ in range(llm.MAX_TRIES):
        asks = pipeline.analysis_plan(predictor.tasks(), predictor.config, predictor.llm_labels())
        if asks:
            pipeline.run_analysis(predictor, asks)

    # Embedding, with a stand-in for Voyage.
    def embed(texts, model_name, key, counts=None, on_wait=None, patience=None):
        yield list(range(len(texts))), [vector(text) for text in texts]

    embeddings.count_tokens = lambda texts, model_name: ([len(t.split()) + 1 for t in texts], True)
    embeddings.embed = embed
    pipeline.run_embeddings(predictor, pipeline.plan_embeddings(predictor, None), "stand-in")

    trained = pipeline.train(predictor)
    meta = trained.meta
    if not meta.get("versions") or not meta.get("without_llm"):
        sys.exit("0.3.0 trained a single version this time. Change the made-up tasks until it trains several.")
    print(f"Trained {', '.join(meta['versions'])}. Each AI model uses: {json.dumps(meta['uses'])}")
    print(f"Without LLM analysis: {json.dumps(meta['without_llm'])}")

    # Forecasts at a fixed moment, with the same stand-ins: the vectors and LLM analysis's answers are kept.
    asked = request(tasks)
    now = round(max(t.finished_at for t in tasks) + 2 * 3600)
    vectors: dict[str, list[float]] = {}
    labels = {
        "read": answer_of("feature", 4, 5, 25.0, "Add a search page, with tests and docs."),
        "new-conversation": answer_of("refactor", 3, 4, 12.0, "Split the parser into smaller parts."),
        "effort": answer_of("small_change", 1, 2, 4.0, "Add an export button to the report table."),
        "unknown-effort": answer_of("bug_fix", 2, 3, 6.0, "Fix the checks on the login form."),
    }

    def cached(model_name, texts):
        vectors.update({text: vector(text) for text in texts})
        return {text: vectors[text] for text in texts}

    embeddings.cached = cached
    llm.analyze_requests = lambda predictor, tasks: {t.id: labels[t.id] for t in tasks if t.id in labels}
    forecast.time = SimpleNamespace(time=lambda: now)
    gaps = watch_similar_tasks(model)
    parsed, distribution, _ = forecast.parse(asked)
    again = Predictor(NAME)
    answered = forecast.answer(again, parsed, distribution)
    if min(gaps) < CLEAR:
        sys.exit(f"A forecast's similar tasks are only {min(gaps):.1e} ahead of the next. Change the made-up tasks.")
    print(f"Each forecast's similar tasks are at least {min(gaps):.1e} ahead of the next ones in line.")

    for opened in (predictor, again):
        opened.close()
    shutil.rmtree(PREDICTOR, ignore_errors=True)
    shutil.copytree(predictor.dir, PREDICTOR, ignore=shutil.ignore_patterns("*.lock"))
    kept = {
        "made_by": f"yardstick-predict {yardstick_predict.__version__}",
        "now": now,
        "request": asked,
        "labels": labels,
        "vectors": vectors,
        "answer": answered,
    }
    FORECASTS.write_text(json.dumps(kept, indent=1) + "\n")
    size = sum(path.stat().st_size for path in PREDICTOR.rglob("*") if path.is_file())
    print(f"Saved {PREDICTOR.name}: {size / 1e6:.2f} MB, and {FORECASTS.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
