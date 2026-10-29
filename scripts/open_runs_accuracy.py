"""How well a model made from Open Runs forecasts tasks it has never seen.

Run it from the repository's root folder, after build_open_runs.py and open_runs_extras.py:

    uv run python scripts/open_runs_accuracy.py [CREATE OPTIONS]

Options after the script's name go to `yardstick-predict create`, such as
`--features conversation,context,llm --llm-provider codex --llm-model codex-default`.

It makes a model from Open Runs alone in a folder of its own, so nothing of yours is read. Then it splits the
different prompts into five groups. For each group, it trains on the runs of the other four and forecasts the
runs of this one. So every forecast is of a prompt the model never saw, as with a new task of yours. It scores
the forecasts as the tool's own test does: against the simple rule, which gives every task the same forecast,
from the AI model's past spread. It prints the results as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

FOLDS = 5


def prompt_share(tasks, value) -> float:
    """How much of the spread of a value, on a log scale, the prompt explains within each AI model. 1 means runs
    of one prompt on one AI model always take the same, so knowing the prompt is all a forecast needs."""
    import math

    total = within = 0.0
    for name in {task.model for task in tasks}:
        logs = {}
        for task in tasks:
            if task.model == name and value(task):
                logs.setdefault(task.prompt, []).append(math.log(value(task)))
        every = [v for group in logs.values() for v in group]
        mean = sum(every) / len(every)
        total += sum((v - mean) ** 2 for v in every)
        within += sum((v - sum(group) / len(group)) ** 2 for group in logs.values() for v in group)
    return round(1 - within / total, 3)


def main() -> None:
    home = tempfile.mkdtemp(prefix="open-runs-")
    os.environ["YARDSTICK_HOME"] = home
    made = subprocess.run(
        [sys.executable, "-m", "yardstick_predict", "create", "open-runs", "--archive", "open-runs", "--express"]
        + sys.argv[1:],
        env={**os.environ, "YARDSTICK_HOME": home},
        capture_output=True,
        text=True,
    )
    if made.returncode:
        sys.exit(made.stdout + made.stderr)

    from yardstick_predict import model as M
    from yardstick_predict import pipeline
    from yardstick_predict.store import Predictor

    predictor = Predictor("open-runs")
    settings = M.settings_for(predictor.config["model"])
    tasks = predictor.tasks()
    rows, _ = pipeline._rows_and_texts(predictor, tasks)
    tasks, rows = M._eligible(tasks, rows, settings)

    def fold(prompt: str) -> int:
        return int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % FOLDS

    scored, learned = [], []
    for k in range(FOLDS):
        train = [i for i, task in enumerate(tasks) if fold(task.prompt) != k]
        test = [i for i, task in enumerate(tasks) if fold(task.prompt) == k]
        trial = M.fit([tasks[i] for i in train], [rows[i] for i in train], settings)
        scored.append(
            M._score_window(trial, [tasks[i] for i in train], [tasks[i] for i in test], [rows[i] for i in test])
        )
        learned.append(len(train))
        print(f"group {k + 1} of {FOLDS}: {len(test)} runs", file=sys.stderr)
    seen = sorted({task.model for task in tasks})
    unseen = M._verdicts(scored, trained_on=min(learned), tested_on=len(tasks), settings=settings, seen=seen)
    counts = Counter(task.model for task in tasks)
    pooled = {}
    for head in settings["heads"]:
        found = [result[head] for result in unseen["models"].values() if (result.get(head) or {}).get("score")]
        tested = sum(result["tasks_tested"] for result in found)
        pooled[head] = {
            name: round(sum(result[name] * result["tasks_tested"] for result in found) / tested, 4)
            for name in ("better_than_simple_rule", "in_50_range", "in_90_range")
        } | {"tasks_tested": tested}
    report = {
        "options": sys.argv[1:],
        "pooled": pooled,
        "served": sorted(name for name, result in unseen["models"].items() if result.get("served")),
        "explained_by_the_prompt": {
            "seconds": prompt_share(tasks, lambda task: task.seconds),
            "tokens": prompt_share(tasks, lambda task: task.tokens),
        },
        "runs": len(tasks),
        "prompts": len({task.prompt for task in tasks}),
        "ai_models": len(seen),
        "new_prompts": {
            name: {**found, "runs": counts[name]}
            for name, found in sorted(unseen["models"].items(), key=lambda m: -counts[m[0]])
        },
        "own_test": (pipeline.model_meta(predictor) or {}).get("report"),
    }
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
