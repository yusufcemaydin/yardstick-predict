"""Build Open Runs: public coding-agent runs, packed in the simple record format and shipped with the tool.

Run it from the repository's root folder:

    uv run --with duckdb python scripts/build_open_runs.py --cache /tmp/open-runs

It downloads three public collections, keeps each run's prompt, AI model, agent, time, tokens and cost, and writes
src/yardstick_predict/open_runs/runs.jsonl.xz. Downloads are kept in the cache folder, so a second build is quick.

1. Terminal-Bench 2.0 runs gathered by Yoonho Lee (Apache 2.0): yoonholee/terminalbench-trajectories.
2. The Terminal-Bench 2.0 leaderboard's runs (Apache 2.0): harborframework/terminal-bench-2-leaderboard.
3. SWE-bench Lite runs by Claude Code and Pi (CC BY 4.0): melissapan/swe-bench-lite-agent-traces-v14.

The Terminal-Bench prompts are the tasks' own instructions (Apache 2.0): harborframework/terminal-bench-2.0 and 2.1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

HF = "https://huggingface.co/datasets"
OUT = Path("src/yardstick_predict/open_runs/runs.jsonl.xz")
RUNS_PER_GROUP = 2
"""At most this many runs of one prompt by one AI model and agent. More of the same adds little."""
TEXT_CHARS = 24_000

# The same shapes the privacy test looks for. A prompt is cleaned, never dropped, since these are public texts.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HOME = re.compile(r"(/Users/|/home/|C:\\Users\\)[A-Za-z0-9._-]+")
_KEYS = re.compile(
    r"sk-ant-[A-Za-z0-9_-]+|sk-proj-[A-Za-z0-9_-]+|sk-[A-Za-z0-9]{20,}|pa-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]+"
    r"|github_pat_[A-Za-z0-9_]+|AKIA[A-Z0-9]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


def clean(text: str) -> str:
    text = _KEYS.sub("[hidden]", text)
    text = _EMAIL.sub("[email]", text)
    text = _HOME.sub(lambda m: m.group(1) + "me", text)
    return text.strip()[:TEXT_CHARS]


def model_name(raw: str) -> str:
    """The AI model's own name, as agents write it: no provider before a slash, none after an @ or a colon.
    Claude models are named with dashes, as Claude Code writes them, such as claude-opus-4-6."""
    name = raw.split("@")[0].split("/")[-1].split(":")[0].strip().lower()
    return name.replace(".", "-") if name.startswith("claude-") else name


NOT_MODELS = {"multiple", "unknown", ""}
"""Names that don't say which AI model did the work, such as an agent that used several."""


def iso(moment: float) -> str:
    return datetime.fromtimestamp(moment, UTC).isoformat().replace("+00:00", "Z")


def when(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value / 1000 if value > 1e11 else float(value)  # milliseconds, or seconds, since 1970
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def fetch(url: str, cache: Path, headers: dict | None = None) -> bytes:
    """A file from the web, kept in the cache folder."""
    path = cache / "files" / hashlib.sha256((url + json.dumps(headers or {})).encode()).hexdigest()
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers=headers or {})
    for attempt in range(30):
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                data = reply.read()
            break
        except urllib.error.HTTPError as error:
            if error.code == 416 and headers:  # a part asked of a file shorter than it: take the whole file
                return fetch(url, cache)
            if error.code != 429 or attempt == 29:
                raise
            time.sleep(float(error.headers.get("Retry-After") or 30))  # too many requests: wait as asked
        except OSError:
            if attempt >= 3:
                raise
            time.sleep(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def instructions(cache: Path, version: str, names: set[str]) -> dict[str, str]:
    repo = {"2.0": "harborframework/terminal-bench-2.0", "2.1": "harborframework/terminal-bench-2.1"}[version]
    found = {}

    def one(name: str):
        try:
            text = fetch(f"{HF}/{repo}/resolve/main/{name}/instruction.md", cache).decode("utf-8", "replace")
        except OSError:
            return
        if text.strip() and not text.lstrip().startswith("{"):
            found[name] = clean(text)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, sorted(names)))
    return found


def terminal_bench(cache: Path) -> list[dict]:
    import duckdb

    print("Terminal-Bench runs", file=sys.stderr)
    # Two whole files, each in one request. Reading them over the web in small parts would use up the limit.
    folder = cache / "terminalbench"
    folder.mkdir(exist_ok=True)
    for n in range(2):
        path = folder / f"train-{n:05d}-of-00002.parquet"
        if not path.exists():
            url = f"{HF}/yoonholee/terminalbench-trajectories/resolve/main/data/{path.name}"
            with urllib.request.urlopen(url, timeout=600) as reply, open(f"{path}.part", "wb") as out:
                while chunk := reply.read(1 << 20):
                    out.write(chunk)
            Path(f"{path}.part").rename(path)
    rows = duckdb.sql(
        "select task_name, agent, model, duration_seconds, input_tokens, output_tokens, cache_tokens, cost_cents, "
        f"coalesce(nullif(trial_id, ''), trial_name), started_at from read_parquet('{folder}/*.parquet')"
    ).fetchall()
    prompts = instructions(cache, "2.0", {row[0] for row in rows})
    # Some runs don't say when they started. They get the earliest start of the rest: Terminal-Bench 2.0's runs
    # all come from the months after it came out. A run must never look newer than it is.
    earliest = min(start for start in (when(row[-1]) for row in rows) if start)
    runs = []
    for task, agent, model, seconds, input_tokens, output_tokens, cached, cents, trial, started in rows:
        if task not in prompts or not seconds or seconds <= 0 or not output_tokens or not trial:
            continue
        cached = int(cached or 0)
        start = when(started) or earliest
        runs.append(
            {
                "id": "tb-" + hashlib.sha256(f"{model}|{agent}|{trial}".encode()).hexdigest()[:24],
                "prompt": prompts[task],
                "model": model_name(model),
                "agent": agent,
                "started_at": iso(start),
                "seconds": round(float(seconds), 3),
                "input_tokens": max(int(input_tokens or 0) - cached, 0),
                "cache_read_tokens": cached,
                "output_tokens": int(output_tokens),
                "usd": round(cents / 100, 6) if cents else None,
                "group": f"terminal-bench/{task}",
            }
        )
    return runs


def leaderboard(cache: Path) -> list[dict]:
    """The leaderboard's runs. Each run's result is a small file of its own, tens of thousands in all, so they come
    in one git download of just those files. One web request each would hit Hugging Face's request limits."""
    print("Terminal-Bench leaderboard runs", file=sys.stderr)
    folder = cache / "leaderboard"
    if not folder.exists():
        url = f"{HF}/harborframework/terminal-bench-2-leaderboard"
        quiet = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", url, str(folder)],
            check=True,
            env=quiet,
        )
        subprocess.run(["git", "-C", str(folder), "sparse-checkout", "init", "--no-cone"], check=True)
        (folder / ".git/info/sparse-checkout").write_text("/submissions/terminal-bench/2.0/*/*/*/result.json\n")
        subprocess.run(["git", "-C", str(folder), "checkout", "-q", "main"], check=True, env=quiet)
    results = []
    for path in sorted(folder.glob("submissions/terminal-bench/2.0/*/*/*/result.json")):
        try:
            found = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(found, dict):
            results.append(found)
    print(f"  {len(results)} run files", file=sys.stderr)
    names = {"2.0": set(), "2.1": set()}
    for r in results:
        names["2.1" if "2-1" in str(r.get("source")) else "2.0"].add(str(r.get("task_name", "")).split("/")[-1])
    prompts = {version: instructions(cache, version, found) for version, found in names.items()}
    runs = []
    for r in results:
        task = str(r.get("task_name", "")).split("/")[-1]
        version = "2.1" if "2-1" in str(r.get("source")) else "2.0"
        prompt = prompts[version].get(task) or prompts["2.0"].get(task)
        execution = r.get("agent_execution") or {}
        start, end = when(execution.get("started_at")), when(execution.get("finished_at"))
        result = r.get("agent_result") or {}
        info = (r.get("agent_info") or {}).get("model_info") or {}
        if not prompt or not start or not end or end <= start or not info.get("name"):
            continue
        cached = int(result.get("n_cache_tokens") or 0)
        output = int(result.get("n_output_tokens") or 0)
        run = {
            "id": f"tbl-{r.get('id')}",
            "prompt": prompt,
            "model": model_name(info["name"]),
            "agent": str((r.get("agent_info") or {}).get("name") or "unknown"),
            "started_at": iso(start),
            "seconds": round(end - start, 3),
            "usd": round(result["cost_usd"], 6) if result.get("cost_usd") else None,
            "stopped": bool(r.get("exception_info")),
            "group": f"terminal-bench/{task}",
        }
        if output:
            run |= {
                "input_tokens": max(int(result.get("n_input_tokens") or 0) - cached, 0),
                "cache_read_tokens": cached,
                "output_tokens": output,
            }
        else:
            total = (result.get("metadata") or {}).get("totalTokens")
            if not isinstance(total, int) or total <= 0:
                continue
            run["tokens"] = total
        runs.append(run)
    return runs


def _jsonl(data: bytes) -> list[dict]:
    lines = []
    for raw in data.decode("utf-8", "replace").splitlines():
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, dict):
            lines.append(value)
    return lines


def swe_bench_lite(cache: Path) -> list[dict]:
    print("SWE-bench Lite runs", file=sys.stderr)
    base = f"{HF}/melissapan/swe-bench-lite-agent-traces-v14/resolve/main"
    index = _jsonl(fetch(f"{base}/data/traces.jsonl", cache))
    index = [row for row in index if row.get("harness") in ("claude-code", "pi")]

    def head(row: dict) -> list[dict]:
        return _jsonl(fetch(f"{base}/{row['trace_path']}", cache, {"Range": "bytes=0-400000"}))

    def tail(row: dict) -> list[dict]:
        return _jsonl(fetch(f"{base}/{row['trace_path']}", cache, {"Range": "bytes=-20000"}))

    prompts = {}
    for row in index:
        if row["harness"] == "pi" and row["task_id"] not in prompts:
            for line in head(row):
                message = line.get("message") or {}
                if line.get("type") == "message_start" and message.get("role") == "user":
                    text = "".join(p.get("text", "") for p in message.get("content", []) if isinstance(p, dict))
                    prompts[row["task_id"]] = clean(text)
                    break

    def timing(row: dict) -> tuple[float, float, bool] | None:
        last = tail(row)
        if row["harness"] == "claude-code":
            result = next((line for line in reversed(last) if line.get("type") == "result"), None)
            moments = [when(line.get("timestamp")) for line in last if line.get("timestamp")]
            moments = [m for m in moments if m]
            if not result or not result.get("duration_ms") or not moments:
                return None
            seconds = result["duration_ms"] / 1000
            stopped = bool(result.get("is_error")) or result.get("subtype") != "success"
            return max(moments) - seconds, seconds, stopped
        first = next((when(line.get("timestamp")) for line in head(row) if line.get("type") == "session"), None)
        moments = [when(line.get("timestamp")) for line in last if line.get("timestamp")]
        moments += [when((line.get("message") or {}).get("timestamp")) for line in last]
        moments = [m for m in moments if m]
        if not first or not moments or max(moments) <= first:
            return None
        return first, max(moments) - first, False

    with ThreadPoolExecutor(8) as pool:
        timings = list(pool.map(timing, index))
    runs = []
    for row, found in zip(index, timings):
        if not found or row["task_id"] not in prompts:
            continue
        start, seconds, stopped = found
        runs.append(
            {
                "id": f"swe-{row['attempt_id']}",
                "prompt": prompts[row["task_id"]],
                "model": model_name(row["model_id"]),
                "agent": row["harness"],
                "started_at": iso(start),
                "seconds": round(seconds, 3),
                "input_tokens": int(row.get("input_tokens") or 0),
                "cache_write_tokens": int(row.get("cache_creation_tokens") or 0),
                "cache_read_tokens": int(row.get("cache_read_tokens") or 0),
                "output_tokens": int(row.get("output_tokens") or 0),
                "usd": row.get("reported_cost_usd"),
                "stopped": stopped,
                "group": f"swe-bench/{row['task_id']}",
            }
        )
    return runs


def keep(runs: list[dict]) -> list[dict]:
    """At most RUNS_PER_GROUP runs of each prompt, AI model and agent, picked the same way every build."""
    groups = defaultdict(list)
    seen = set()
    for run in runs:
        if run["id"] in seen or run["model"] in NOT_MODELS:
            continue
        seen.add(run["id"])
        groups[(run["prompt"], run["model"], run["agent"])].append(run)
    kept = []
    for group in groups.values():
        group.sort(key=lambda run: hashlib.sha256(run["id"].encode()).hexdigest())
        kept += group[:RUNS_PER_GROUP]
    kept.sort(key=lambda run: (run["group"], run["model"], run["agent"], run["id"]))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="a folder for the downloads")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    runs = keep(terminal_bench(args.cache) + leaderboard(args.cache) + swe_bench_lite(args.cache))
    lines = []
    for run in runs:
        run = {key: value for key, value in run.items() if value not in (None, False, "")}
        lines.append(json.dumps(run, ensure_ascii=False, sort_keys=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(lzma.compress(("\n".join(lines) + "\n").encode(), preset=9 | lzma.PRESET_EXTREME))
    prompts = {run["prompt"] for run in runs}
    models = {run["model"] for run in runs}
    print(f"{len(runs)} runs, {len(prompts)} prompts, {len(models)} AI models, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
