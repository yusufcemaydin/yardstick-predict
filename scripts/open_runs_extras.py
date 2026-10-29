"""Make the Voyage vectors and LLM answers that come with Open Runs, so nobody pays for them again.

Run it from the repository's root folder, after build_open_runs.py:

    uv run python scripts/open_runs_extras.py

It reads each different Open Runs prompt once. LLM analysis reads them with the AI models named below, and Voyage
embeds the prompts and the descriptions LLM analysis wrote, with every Voyage 4 model. It needs your Voyage key, and
a way to pay for each AI model: the Codex command signed in, for codex-default. It writes vectors.sqlite and
answers.json next to the runs.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from array import array
from pathlib import Path

from yardstick_predict import archives, embeddings, features, llm

FOLDER = Path("src/yardstick_predict/open_runs")
READERS = [{"provider": "codex", "model": "codex-default"}]
"""The AI models whose answers come with the tool, and how they're paid for."""


def main() -> None:
    tasks = archives.read_archive(archives.OPEN_RUNS)
    prompts = sorted({task.prompt for task in tasks})
    print(f"{len(tasks):,} runs, {len(prompts)} prompts", file=sys.stderr)
    fields = llm.parts(["kind", "description"])
    sets, texts = [], {features.text("prompt", features.TaskInfo.of(task)) for task in tasks}
    for how in READERS:
        asks = [llm.Ask(f"p{n}", prompt) for n, prompt in enumerate(prompts)]
        found = llm.analyze(asks, how, fields, lambda done, count: print(f"  read {count}", file=sys.stderr))
        answers = {}
        for ask in asks:
            labels = found.answers.get(ask.id) or {}
            if not labels.get("failed") and all(name in labels for name in fields):
                answers[llm.read_of(ask.prompt, "")] = {name: labels[name] for name in fields}
                texts.add(str(labels["description"])[: archives.TEXT_CHARS])
        print(f"{how['provider']}:{how['model']} answered {len(answers)} of {len(asks)}", found.problem or "")
        sets.append({"by": f"{how['provider']}:{how['model']}", "fields": fields, "answers": answers})
    (FOLDER / "answers.json").write_text(json.dumps({"sets": sets}, indent=1, sort_keys=True) + "\n")

    key, _ = embeddings.find_key()
    if not key:
        sys.exit("No Voyage key found.")
    texts = sorted(text for text in texts if text.strip())
    path = FOLDER / "vectors.sqlite"
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE vectors "
        "(model TEXT, dimensions INTEGER, hash TEXT, vector BLOB, PRIMARY KEY (model, dimensions, hash))"
    )
    for model in embeddings.MODELS:
        for positions, vectors in embeddings.embed(texts, model, key):
            db.executemany(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
                [
                    (model, embeddings.DIMENSIONS, embeddings.text_hash(texts[i]), array("f", vector).tobytes())
                    for i, vector in zip(positions, vectors)
                ],
            )
        print(f"{model}: {len(texts)} texts", file=sys.stderr)
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"vectors.sqlite {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
