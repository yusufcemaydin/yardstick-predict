"""LLM analysis: an AI model reads each task before it starts and describes the work.

It reads only what a forecast can know: the prompt and the end of the conversation before it. It reports
the kind of task and its expected size, and with the description option, a few sentences on the work,
which are embedded like the prompt. It never sees how a task really went.

It's paid for one of four ways: an Anthropic or an OpenAI API key, billed per use, or a Claude or ChatGPT
subscription, through the Claude Code or Codex command on this computer. A subscription has no extra bill,
but its calls count toward the plan's usage limits. Those calls run with no tools and save no session, so
they never show up in the records the predictor learns from.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from . import prices

KINDS = (
    "question",
    "explanation",
    "small_change",
    "feature",
    "bug_fix",
    "refactor",
    "investigation",
    "review",
    "design",
    "operations",
    "writing",
    "other",
)
CLARITY = ("clear", "some_guessing", "vague")
PROMPT_CHARS = 6_000
"""How much of a prompt the AI model reads, from its start."""
CONTEXT_CHARS = 1_500
"""How much of the conversation before the prompt it reads, from its end."""
TOKENS_PER_TASK = (1_000, 200)
"""About how many tokens one task's analysis reads and writes, for the cost shown before it runs."""
DESCRIPTION_TOKENS = 150
"""About how many tokens a description takes to embed, at most, for fitting a budget before they're written."""
INSIDE = "YARDSTICK_PREDICT_LLM_CALL"
"""Set for the Claude Code and Codex commands this module starts, so yardstick's own hooks stay out of them."""
DEFAULT_CODEX = "codex-default"
INSTALL = (
    "uv tool install --reinstall 'yardstick-predict[llm] @ git+https://github.com/yusufcemaydin/yardstick-predict'"
)
"""How to add the providers' libraries to a yardstick-predict installed with uv."""
"""Codex's own default model, whichever the plan gives."""


@dataclass(frozen=True)
class Provider:
    key: str
    title: str
    says: str
    models: tuple[str, ...]
    billed: bool
    """Paid per use, with the cost shown first. Otherwise a subscription pays."""
    batch: int
    """How many tasks go in one request."""
    parallel: int
    """How many requests run at once."""


PROVIDERS = {
    "anthropic": Provider(
        "anthropic",
        "Anthropic API key",
        "pay per use. The cost is shown before anything is spent.",
        ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
        True,
        8,
        4,
    ),
    "openai": Provider(
        "openai",
        "OpenAI API key",
        "pay per use. The cost is shown before anything is spent.",
        ("gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna"),
        True,
        8,
        4,
    ),
    "claude": Provider(
        "claude",
        "Claude subscription",
        "through Claude Code. No extra bill, but it counts toward your plan's usage limits.",
        ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
        False,
        20,
        2,
    ),
    "codex": Provider(
        "codex",
        "ChatGPT subscription",
        "through Codex. No extra bill, but it counts toward your plan's usage limits.",
        (DEFAULT_CODEX,),
        False,
        20,
        2,
    ),
}
INSTRUCTIONS = """\
You read coding tasks before an AI coding agent starts them, and describe the work each one needs.
For each task you get the person's prompt and the end of the conversation before it, if there was one.
Judge only from that text. Don't run anything and don't look anything up.

For each task, by its id, report:
{fields}
Answer every task you were given, once."""
FIELDS = {
    "kind": "kind: the kind of task",
    "files": "files: how many files the agent will likely read or change",
    "steps": "steps: how many distinct steps the work takes",
    "runs_code": "runs_code: whether the agent must run code, tests or builds",
    "looks_up": "looks_up: whether it must look things up on the web or in documentation",
    "continues": "continues: whether it continues work from the conversation before it",
    "clarity": "clarity: how clear the request is",
    "minutes": "minutes: your best guess of the agent's working minutes",
    "description": "description: two or three plain sentences on what the agent must do",
}
KIND_FIELDS = ("kind", "files", "steps", "runs_code", "looks_up", "continues", "clarity", "minutes")


MAX_TRIES = 2
"""A task the AI model wouldn't answer this many times is skipped from then on, until another model reads it."""


class LLMError(Exception):
    """LLM analysis couldn't run, with the reason in plain words.

    `lasting` marks a problem with the tasks themselves, such as a refusal, which trying again may not fix.
    The others, such as no key or no network, go away once fixed.
    """

    def __init__(self, message: str, lasting: bool = False) -> None:
        super().__init__(message)
        self.lasting = lasting


@dataclass(frozen=True)
class Ask:
    """One task for the AI model to read."""

    id: str
    prompt: str
    context: str = ""
    conversation: str = ""
    """The conversation the task is in. Two tasks of one conversation never share a request: the later one's
    context would show the AI model how the earlier one ended."""


@dataclass
class Analysis:
    """What a run of LLM analysis found."""

    answers: dict[str, dict]
    """By task id: what the AI model said, or a note that it didn't answer, for a lasting problem."""
    problem: str | None = None
    """Why a request failed, when one did."""


def access(provider: str | None, model: str | None) -> dict:
    """How LLM analysis runs, from a program's flags. Raises ValueError for a way to pay that doesn't exist."""
    key = (provider or "anthropic").strip().lower()
    key = {"chatgpt": "codex", "claude-code": "claude"}.get(key, key)
    if key not in PROVIDERS:
        raise ValueError(f"Pay for LLM analysis with one of: {', '.join(PROVIDERS)}.")
    return {"provider": key, "model": (model or "").strip() or PROVIDERS[key].models[0]}


def title(model: str) -> str:
    """A model's name for people: claude-opus-5-5 is Claude Opus 5.5."""
    if model == DEFAULT_CODEX:
        return "Codex's own model"
    words, digits = [], []
    for part in model.split("-"):
        if part.isdigit():
            digits.append(part)
            continue
        if digits:
            words.append(".".join(digits))
            digits = []
        words.append("GPT" if part == "gpt" else part.capitalize())
    if digits:
        words.append(".".join(digits))
    return " ".join(words)


def cost_per_thousand(model: str) -> float | None:
    """About what analyzing 1,000 tasks costs at the model's list price. None when its price isn't known."""
    return cost(1000, model)


def cost(tasks: int, model: str) -> float | None:
    rates = prices.price(model)
    if rates is None:
        return None
    reads, writes = TOKENS_PER_TASK
    return tasks * (reads * rates["input"] + writes * rates["output"]) / 1e6


def parts(options: list[str]) -> list[str]:
    """The fields to ask for, from the feature's options: the kind and size, the description, or both."""
    fields = list(KIND_FIELDS) if "kind" in options else []
    return fields + (["description"] if "description" in options else [])


def schema(fields: list[str]) -> dict:
    """The answer's shape: one entry per task, with just the fields asked for."""
    shapes = {
        "kind": {"type": "string", "enum": list(KINDS)},
        "files": {"type": "integer"},
        "steps": {"type": "integer"},
        "runs_code": {"type": "boolean"},
        "looks_up": {"type": "boolean"},
        "continues": {"type": "boolean"},
        "clarity": {"type": "string", "enum": list(CLARITY)},
        "minutes": {"type": "number"},
        "description": {"type": "string"},
    }
    task = {
        "type": "object",
        "properties": {"id": {"type": "string"}, **{name: shapes[name] for name in fields}},
        "required": ["id", *fields],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"tasks": {"type": "array", "items": task}},
        "required": ["tasks"],
        "additionalProperties": False,
    }


def _message(asks: list[Ask]) -> str:
    blocks = []
    for ask in asks:
        block = [f"Task {ask.id}", "Prompt:", "<<<", ask.prompt[:PROMPT_CHARS], ">>>"]
        if ask.context.strip():
            block += ["The end of the conversation before it:", "<<<", ask.context[-CONTEXT_CHARS:], ">>>"]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def _clean(found: dict, fields: list[str]) -> dict | None:
    """One task's answer with every field checked. None when a field is missing or makes no sense."""
    labels: dict = {}
    for name in fields:
        value = found.get(name)
        if name in ("kind", "clarity"):
            if value not in (KINDS if name == "kind" else CLARITY):
                return None
        elif name in ("files", "steps", "minutes"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
            value = max(float(value), 0.0)
        elif name == "description":
            if not isinstance(value, str) or not value.strip():
                return None
            value = value.strip()
        elif not isinstance(value, bool):
            return None
        labels[name] = value
    return labels


def batches(asks: list[Ask], size: int) -> list[list[Ask]]:
    """The tasks in requests of up to `size`, no two of them from the same conversation."""
    found: list[tuple[list[Ask], set[str]]] = []
    for ask in asks:
        for batch, conversations in found:
            if len(batch) < size and (not ask.conversation or ask.conversation not in conversations):
                batch.append(ask)
                conversations.add(ask.conversation)
                break
        else:
            found.append(([ask], {ask.conversation}))
    return [batch for batch, _ in found]


def analyze(
    asks: list[Ask],
    how: dict,
    fields: list[str],
    on_batch: Callable[[dict[str, dict], int], None] | None = None,
) -> Analysis:
    """Ask the AI model about each task, with every answer saying who gave it.

    A task it refused, skipped or answered with something that makes no sense gets a note saying so. A request
    that failed for another reason, such as no network, leaves its tasks out, so a later run tries again.
    `on_batch` gets each request's answers as they arrive, and how many tasks are done, so a long run can save
    as it goes. Only a few requests run or wait at a time, so stopping, such as with Ctrl+C, stops it quickly.
    """
    provider = PROVIDERS[how["provider"]]
    call = {"anthropic": _anthropic, "openai": _openai, "claude": _claude_code, "codex": _codex}[provider.key]
    system = INSTRUCTIONS.format(fields="\n".join(f"- {FIELDS[name]}" for name in fields))
    shape = schema(fields)
    by = f"{provider.key}:{how['model']}"
    waiting = iter(batches(asks, provider.batch))
    answers: dict[str, dict] = {}
    problems: list[str] = []
    done = 0

    def run(batch: list[Ask]) -> tuple[dict[str, dict], str | None]:
        unanswered = {"by": by, "fields": fields, "failed": True}
        try:
            answer = call(how["model"], system, _message(batch), shape)
        except LLMError as error:
            if not error.lasting:
                raise
            return {ask.id: dict(unanswered) for ask in batch}, str(error)
        wanted, found = {ask.id for ask in batch}, {}
        for entry in answer.get("tasks") or [] if isinstance(answer, dict) else []:
            if isinstance(entry, dict) and entry.get("id") in wanted and entry["id"] not in found:
                clean = _clean(entry, fields)
                if clean is not None:
                    found[entry["id"]] = {**clean, "by": by, "fields": fields}
        for ask in batch:
            found.setdefault(ask.id, dict(unanswered))  # skipped, or answered with something that makes no sense
        return found, None

    pool = ThreadPoolExecutor(max_workers=provider.parallel)
    running: dict = {}

    def start_next() -> None:
        batch = next(waiting, None)
        if batch is not None:
            running[pool.submit(run, batch)] = batch

    try:
        for _ in range(provider.parallel):
            start_next()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                batch = running.pop(future)
                try:
                    found, problem = future.result()
                except LLMError as error:
                    found, problem = {}, str(error)
                if problem:
                    problems.append(problem)
                done += len(batch)
                answers.update(found)
                if on_batch is not None:
                    on_batch(found, done)
                start_next()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return Analysis(answers, problems[0] if problems else None)


def needs(labels: dict | None, how: dict, fields: list[str]) -> bool:
    """Whether a task needs analyzing: no answer yet, one from another model or without every field, or a
    refusal that hasn't been tried MAX_TRIES times."""
    if not labels:
        return True
    if labels.get("by") != f"{how['provider']}:{how['model']}" or not set(fields) <= set(labels.get("fields") or []):
        return True
    return bool(labels.get("failed")) and labels.get("tries", 1) < MAX_TRIES


def numbers(labels: dict) -> dict[str, float]:
    """The number features of what the AI model said about a task. Nothing when it said nothing."""
    if not labels:
        return {}
    found: dict[str, float] = {}
    if labels.get("kind") in KINDS:
        found.update({f"llm_kind_{kind}": float(labels["kind"] == kind) for kind in KINDS})
    for key in ("files", "steps", "minutes"):
        value = labels.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            found[f"llm_{key}"] = math.log1p(max(float(value), 0.0))
    for key in ("runs_code", "looks_up", "continues"):
        if isinstance(labels.get(key), bool):
            found[f"llm_{key}"] = float(labels[key])
    if labels.get("clarity") in CLARITY:
        found["llm_clarity"] = float(CLARITY.index(labels["clarity"]))
    return found


def analyze_requests(predictor, tasks) -> dict[str, dict]:
    """What the AI model says about each forecast request's task, by the task's id.

    A forecast still comes when LLM analysis can't run, such as with no network. It's just less informed.
    """
    from . import features

    how = predictor.config.get("llm") or {}
    if how.get("provider") not in PROVIDERS:
        return {}
    fields = parts(features.options(predictor.config, "llm"))
    # One prompt forecast on several models or effort levels is read once.
    same: dict[tuple[str, str], list[str]] = {}
    for task in tasks:
        same.setdefault((task.prompt, task.context), []).append(task.id)
    asks = [Ask(f"t{n}", prompt, context, f"t{n}") for n, (prompt, context) in enumerate(same)]
    found = analyze(asks, {"provider": how["provider"], "model": how.get("model") or ""}, fields).answers
    return {task_id: found[ask.id] for ask, ids in zip(asks, same.values()) if ask.id in found for task_id in ids}


# The four ways to call an AI model ------------------------------------------------------------------------


def _anthropic(model: str, system: str, text: str, shape: dict) -> dict:
    try:
        import anthropic
    except ImportError:
        raise LLMError(
            f"Paying with an Anthropic API key needs the anthropic package. Add it with: {INSTALL}"
        ) from None
    request = {
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [{"role": "user", "content": text}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": shape}},
    }
    try:
        client = anthropic.Anthropic()
        if model.startswith("claude-haiku"):
            response = client.messages.create(**request)  # Haiku has no fallback model to retry a decline on
        else:
            # A request its safety checks decline is retried on the model Anthropic recommends for it.
            response = client.beta.messages.create(
                **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
    except anthropic.AuthenticationError:
        raise LLMError("Anthropic didn't accept the API key. Set ANTHROPIC_API_KEY to a working key.") from None
    except anthropic.APIStatusError as error:
        raise LLMError(f"Anthropic's API said: {error.message}") from None
    except anthropic.APIConnectionError:
        raise LLMError("Couldn't reach Anthropic's API. Check the network connection.") from None
    except TypeError as error:
        if "authentication" not in str(error).lower():
            raise
        raise LLMError("No Anthropic API key was found. Set ANTHROPIC_API_KEY to your key.") from None
    if response.stop_reason == "refusal":
        raise LLMError("The AI model declined to describe these tasks.", lasting=True)
    text = next((block.text for block in response.content if block.type == "text"), "")
    return _json(text)


def _openai(model: str, system: str, text: str, shape: dict) -> dict:
    try:
        import openai
    except ImportError:
        raise LLMError(f"Paying with an OpenAI API key needs the openai package. Add it with: {INSTALL}") from None
    try:
        response = openai.OpenAI().responses.create(
            model=model,
            instructions=system,
            input=text,
            text={"format": {"type": "json_schema", "name": "task_analysis", "schema": shape, "strict": True}},
        )
    except openai.AuthenticationError:
        raise LLMError("OpenAI didn't accept the API key. Set OPENAI_API_KEY to a working key.") from None
    except openai.APIStatusError as error:
        raise LLMError(f"OpenAI's API said: {error.message}") from None
    except openai.APIConnectionError:
        raise LLMError("Couldn't reach OpenAI's API. Check the network connection.") from None
    except openai.OpenAIError as error:  # the library couldn't start, such as with no key
        if "credentials" in str(error).lower() or "api_key" in str(error).lower():
            raise LLMError("No OpenAI API key was found. Set OPENAI_API_KEY to your key.") from None
        raise LLMError(f"OpenAI's library said: {str(error)[:200]}") from None
    return _json(response.output_text)


def _claude_code(model: str, system: str, text: str, shape: dict) -> dict:
    command = shutil.which("claude")
    if command is None:
        raise LLMError("Paying with a Claude subscription needs Claude Code. Install it, then sign in.")
    # No tools, no saved session, and none of your settings, so none of your hooks run inside it.
    reply = _run(
        [
            command,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(shape),
            "--model",
            model,
            "--tools",
            "",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--system-prompt",
            system,
        ],
        text,
        "Claude Code",
    )
    try:
        result = json.loads(reply)
    except ValueError:
        raise LLMError("Claude Code's answer wasn't JSON.") from None
    if isinstance(result, dict) and result.get("is_error"):
        raise LLMError(f"Claude Code said: {str(result.get('result') or 'an error')[:200]}")
    structured = result.get("structured_output") if isinstance(result, dict) else None
    if isinstance(structured, dict):
        return structured
    return _json(str(result.get("result") or "") if isinstance(result, dict) else "")


def _codex(model: str, system: str, text: str, shape: dict) -> dict:
    command = shutil.which("codex")
    if command is None:
        raise LLMError("Paying with a ChatGPT subscription needs Codex. Install it, then sign in.")
    with tempfile.TemporaryDirectory() as folder:
        schema_path, answer_path = Path(folder) / "schema.json", Path(folder) / "answer.json"
        schema_path.write_text(json.dumps(shape))
        # It runs in an empty folder, can't change anything and saves no session. It skips your Codex settings,
        # so none of your tool servers or notify scripts run. Your sign-in still works.
        arguments = [command, "exec", "--ephemeral", "--ignore-user-config", "--sandbox", "read-only"]
        arguments.append("--skip-git-repo-check")
        arguments += ["--output-schema", str(schema_path), "-o", str(answer_path)]
        if model and model != DEFAULT_CODEX:
            arguments += ["-m", model]
        _run([*arguments, "-"], f"{system}\n\n{text}", "Codex", cwd=folder)
        try:
            return _json(answer_path.read_text())
        except OSError:
            raise LLMError("Codex gave no answer.") from None


def _run(arguments: list[str], text: str, name: str, cwd: str | None = None) -> str:
    try:
        done = subprocess.run(
            arguments,
            input=text,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=cwd,
            env={**os.environ, INSIDE: "1"},
        )
    except subprocess.TimeoutExpired:
        raise LLMError(f"{name} took longer than 15 minutes to answer.") from None
    except OSError as error:
        raise LLMError(f"Couldn't start {name}: {error}") from None
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        raise LLMError(f"{name} stopped with an error: {detail[-1][:200] if detail else 'no details'}")
    return done.stdout


def _json(text: str) -> dict:
    try:
        found = json.loads(text)
    except ValueError:
        raise LLMError("The AI model's answer wasn't the JSON it was asked for.", lasting=True) from None
    if not isinstance(found, dict):
        raise LLMError("The AI model's answer wasn't the JSON it was asked for.", lasting=True)
    return found
