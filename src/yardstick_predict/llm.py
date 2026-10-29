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

import hashlib
import itertools
import json
import math
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import prices, privacy

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
"""Fields the AI model wouldn't answer about a task this many times aren't asked for again, until another model
reads it. Each set of fields asked for, such as the kind and size or the description, counts its own tries."""


class LLMError(Exception):
    """LLM analysis couldn't run, with the reason in plain words.

    `lasting` marks a problem with the tasks themselves, such as a refusal, which trying again may not fix.
    `stops` marks one that every request would meet the same way, such as a command that isn't signed in: no more
    requests are tried, and the tasks are left for a later run, as for the others. The others, such as no key or
    no network, go away once fixed.
    """

    def __init__(self, message: str, lasting: bool = False, stops: bool = False) -> None:
        from . import ui

        # Often in another service's or another command's own words, which are shown, but never codes a terminal
        # acts on.
        super().__init__(ui.clean(message))
        self.lasting = lasting
        self.stops = stops


SIGNED_OUT = {
    "Claude Code": (
        "Claude Code isn't signed in here. Run claude in a terminal and type /login, then try again. If you "
        "started this from inside a Claude Code session, run it from your own terminal instead."
    ),
    "Codex": "Codex isn't signed in. Run codex login in a terminal, then try again.",
}
"""What to do when the command a subscription runs through isn't signed in, as when its sign-in expired."""
SIGN_IN_WORDS = ("authenticat", "oauth", "log in", "login", "not logged in", "expired")
"""Words in a Claude Code or Codex error that mean it isn't signed in, such as "Failed to authenticate: OAuth
session expired"."""


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


class Budget:
    """What LLM analysis paid with an API key may spend under a spending limit, in US dollars at list price.

    A request starts only when the most it can cost fits in what's left, with the most the requests still running
    can cost set aside. The most a request can cost counts each byte of its text as a token, since no token is
    shorter than a byte, and its answer at the most tokens it may hold. So the spending never goes past the limit,
    however long a task's text or its answer is. When a whole request doesn't fit, one with fewer of its tasks may.
    """

    def __init__(self, usd: float) -> None:
        self.usd = usd
        self.spent = 0.0
        self.held = 0.0
        """Set aside for the requests still running: the most each can cost."""
        self.full = False
        """Whether a request didn't start because the most it can cost didn't fit."""

    def fits(self, most: float) -> bool:
        """Whether a request that can cost at most this much may start now."""
        return self.spent + self.held + most <= self.usd + 1e-12

    def take(self, most: float) -> None:
        self.held += most

    def paid(self, held: float, usd: float | None) -> None:
        """A request came back. It cost `usd`, or the most it could, when what it cost isn't known."""
        self.held -= held
        self.spent += held if usd is None else usd


ANSWER_TOKENS = 2_000
"""Under a spending limit, the most tokens an answer may hold for each task in its request, thinking included."""
MOST_ANSWER_TOKENS = 16_000
"""The most tokens a whole answer may hold."""
FRAMING_TOKENS = 200
"""Room for what a provider wraps around a request's text, such as its roles and the answer's shape."""


def answer_tokens(tasks: int) -> int:
    """The most tokens the answer to a request for this many tasks may hold, under a spending limit."""
    return min(MOST_ANSWER_TOKENS, ANSWER_TOKENS * tasks)


def spent(found: dict, model: str) -> float | None:
    """What a request really cost, from the tokens the answer says it used, at the AI model's list price. None when
    the answer doesn't say, or the price isn't known."""
    usage = found.pop("_usage", None) if isinstance(found, dict) else None
    if not isinstance(usage, dict):
        return None
    price = None
    for name in dict.fromkeys([str(usage.get("model") or model), model]):  # a fallback model, or the one asked for
        price = prices.call_cost(
            name,
            int(usage.get("input") or 0),
            int(usage.get("cache_write") or 0),
            0,
            int(usage.get("cache_read") or 0),
            int(usage.get("output") or 0),
        )
        if price is not None:
            break
    return price


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
    """What the AI model reads of each task. Keys and passwords are hidden before the text is cut to length."""
    blocks = []
    for ask in asks:
        prompt, context = privacy.hide_secrets(ask.prompt), privacy.hide_secrets(ask.context)
        block = [f"Task {ask.id}", "Prompt:", "<<<", prompt[:PROMPT_CHARS], ">>>"]
        if context.strip():
            block += ["The end of the conversation before it:", "<<<", context[-CONTEXT_CHARS:], ">>>"]
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
    budget: Budget | None = None,
) -> Analysis:
    """Ask the AI model about each task, with every answer saying who gave it.

    A task it refused, skipped or answered with something that makes no sense gets a note saying so. A request
    that failed for another reason, such as no network, leaves its tasks out, so a later run tries again. One that
    every request would meet, such as Claude Code or Codex not signed in, stops it: the requests already sent
    finish, and no more are sent. `on_batch` gets each request's answers as they arrive, and how many tasks are
    done, so a long run can save as it goes. Only a few requests run or wait at a time, so stopping, such as with
    Ctrl+C, stops it quickly. With a `budget`, no request starts that may take the spending past it, and the tasks
    it would have held are left for a later run.
    """
    provider = PROVIDERS[how["provider"]]
    call = {"anthropic": _anthropic, "openai": _openai, "claude": _claude_code, "codex": _codex}[provider.key]
    system = INSTRUCTIONS.format(fields="\n".join(f"- {FIELDS[name]}" for name in fields))
    shape = schema(fields)
    by = f"{provider.key}:{how['model']}"
    waiting = deque(batches(asks, provider.batch))
    answers: dict[str, dict] = {}
    problems: list[str] = []
    stopped: list[str] = []  # why it sends no more requests, once a problem every request would meet came
    done = 0

    def run(batch: list[Ask]) -> tuple[dict[str, dict], str | None, float | None]:
        def unanswered(ask: Ask) -> dict:
            return {"by": by, "fields": fields, "failed": True, "read": read_of(ask.prompt, ask.context)}

        # Under a budget, an answer can't run longer than was set aside, and no other AI model is asked instead.
        limits = {"most_tokens": answer_tokens(len(batch)), "fallback": False} if budget is not None else {}
        try:
            answer = call(how["model"], system, _message(batch), shape, **limits)
        except LLMError as error:
            if not error.lasting:
                raise
            return {ask.id: unanswered(ask) for ask in batch}, str(error), None
        usd = spent(answer, how["model"])
        wanted, found = {ask.id: ask for ask in batch}, {}
        for entry in answer.get("tasks") or [] if isinstance(answer, dict) else []:
            if isinstance(entry, dict) and entry.get("id") in wanted and entry["id"] not in found:
                clean = _clean(entry, fields)
                if clean is not None:
                    ask = wanted[entry["id"]]
                    found[entry["id"]] = {**clean, "by": by, "fields": fields, "read": read_of(ask.prompt, ask.context)}
        for ask in batch:
            found.setdefault(ask.id, unanswered(ask))  # skipped, or answered with something that makes no sense
        return found, None, usd

    def most(batch: list[Ask]) -> float | None:
        """The most a request for these tasks can cost. None when the AI model's price isn't known."""
        text = system + _message(batch) + json.dumps(shape)
        tokens = len(text.encode("utf-8", "surrogatepass")) + FRAMING_TOKENS
        return prices.call_cost(how["model"], tokens, 0, 0, 0, answer_tokens(len(batch)))

    results: queue.Queue = queue.Queue()
    running: dict[int, tuple[list[Ask], float]] = {}
    numbers = itertools.count()

    def work(number: int, batch: list[Ask]) -> None:
        try:
            results.put((number, run(batch), None))
        except BaseException as error:  # said where the results are read
            results.put((number, None, error))

    def next_batch() -> tuple[list[Ask], float] | None:
        """The next request to send and the most it can cost, or None when there's none to send now."""
        if stopped or not waiting or (budget is not None and budget.full):
            return None
        batch = waiting.popleft()
        if budget is None:
            return batch, 0.0
        for size in range(len(batch), 0, -1):  # the whole request, else one with fewer of its tasks
            cost = most(batch[:size])
            if cost is not None and budget.fits(cost):
                if size < len(batch):
                    waiting.appendleft(batch[size:])
                return batch[:size], cost
        waiting.appendleft(batch)
        if not running:  # nothing set aside will come free: it can't fit
            budget.full = True
        return None

    def start_next() -> bool:
        found = next_batch()
        if found is None:
            return False
        batch, held = found
        if budget is not None:
            budget.take(held)
        number = next(numbers)
        running[number] = (batch, held)
        # Python doesn't wait for this thread before it exits, so Ctrl+C never waits for a request to come back.
        threading.Thread(target=work, args=(number, batch), daemon=True).start()
        return True

    while len(running) < provider.parallel and start_next():
        pass
    while running:
        try:
            number, outcome, error = results.get(timeout=0.5)  # a wait that Ctrl+C stops, on Windows too
        except queue.Empty:
            continue
        batch, held = running.pop(number)
        usd = None
        if error is None:
            found, problem, usd = outcome
        elif isinstance(error, LLMError):
            found, problem = {}, str(error)
            if error.stops:
                stopped.append(problem)
        else:
            raise error
        if budget is not None:
            budget.paid(held, usd)
        if problem:
            problems.append(problem)
        done += len(batch)
        answers.update(found)
        if on_batch is not None:
            on_batch(found, done)
        while len(running) < provider.parallel and start_next():
            pass
    if budget is not None and budget.full and not stopped:
        from .spending import dollars

        left = sum(len(batch) for batch in waiting)
        stopped.append(
            f"LLM analysis stopped at your spending limit, after about {dollars(budget.spent)} at list price. "
            f"{left:,} task{' is' if left == 1 else 's are'} left for a later run."
        )
    return Analysis(answers, (stopped or problems or [None])[0])


def read_of(prompt: str, context: str) -> str:
    """A short fingerprint of the text the AI model reads of a task, as it's sent: the start of its prompt and the
    end of the conversation before it. An answer made from another text is read again."""
    sent = f"{prompt[:PROMPT_CHARS]}\x00{context[-CONTEXT_CHARS:] if context.strip() else ''}"
    return hashlib.sha256(sent.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def needs(labels: dict | None, how: dict, fields: list[str], read: str | None = None) -> bool:
    """Whether a task needs analyzing: no answer yet, one from another model or without every field, one made from
    another text than `read` says the task has now, or a refusal of these fields that hasn't been tried MAX_TRIES
    times. A refusal of other fields doesn't count. An answer saved before answers said what they read is taken as
    made from the text the task has, so nothing is read again, and paid for again, on an upgrade."""
    if not labels:
        return True
    if labels.get("by") != f"{how['provider']}:{how['model']}" or not set(fields) <= set(labels.get("fields") or []):
        return True
    if read is not None and labels.get("read") not in (None, read):
        return True  # the task's text changed since it was read
    if all(name in labels for name in fields):
        return False  # every field asked for is answered, as when it refused only another predictor's
    return bool(labels.get("failed")) and unanswered(labels).get(_parts(fields), 0) < MAX_TRIES


_COUNTS = ("failed", "tries", "unanswered")
"""What an answer keeps about the times the AI model didn't answer. The rest is what it said, and who said it."""


def _parts(fields) -> str:
    """The name a set of fields is counted under: their names in alphabetical order, joined by commas."""
    return ",".join(sorted(fields))


def unanswered(labels: dict) -> dict[str, int]:
    """How many times the AI model didn't answer each set of fields asked about a task, by the set's name.

    Each set of fields asked for, such as the kind and size or the description, has its own count, so a refusal
    of one never uses up the tries of another. An answer saved before they were counted this way has one count,
    `tries`, for the fields it names.
    """
    found = labels.get("unanswered")
    if isinstance(found, dict):
        return {
            name: count
            for name, count in found.items()
            if isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool) and count > 0
        }
    if not labels.get("failed"):
        return {}
    tries = labels.get("tries", 1)
    tries = tries if isinstance(tries, int) and not isinstance(tries, bool) and tries > 0 else 1
    return {_parts(labels.get("fields") or []): tries}


def _counted(merged: dict, counts: dict[str, int]) -> dict:
    """An answer with the counts of the sets of fields it still lacks. A set answered since needs none."""
    left = {name: count for name, count in counts.items() if any(f not in merged for f in name.split(",") if f)}
    if left:
        merged.update(failed=True, unanswered=left)
    return merged


def merge(before: dict | None, labels: dict) -> dict:
    """A task's new answer, with what the same AI model said about the task before, in fields not asked again.

    Predictors that ask the same AI model share its answers, through the feature store, even when they ask for
    different fields. So one that asks for the description keeps the kind and size another one asked for, and
    neither asks again for what the other did. A refusal keeps the answers given before, and counts as one
    more try of the fields it was asked for.
    """
    before = before or {}
    if before.get("by") != labels.get("by"):
        before = {}  # another AI model's answer is replaced whole
    elif before.get("read") and labels.get("read") and before["read"] != labels["read"]:
        before = {}  # so is an answer about another text the task had
    kept = {name: value for name, value in before.items() if name in FIELDS}
    asked = labels.get("fields") or []
    fields = list(dict.fromkeys([*(before.get("fields") or []), *asked]))
    counts = unanswered(before)
    if labels.get("failed"):
        counts[_parts(asked)] = counts.get(_parts(asked), 0) + 1
    merged = {**kept, **{name: value for name, value in labels.items() if name not in _COUNTS}}
    if fields or "fields" in labels:
        merged["fields"] = fields
    return _counted(merged, counts)


def combine(stored: dict | None, other: dict) -> dict:
    """Two saved answers about one task as one, as when an older version's answer moves into the feature store.

    Answers by the same AI model keep the fields of both. For a field both have, the stored answer's stays, and an
    answer is kept over a refusal. Each set of fields keeps the higher of its two counts of tries, so combining
    the same answers again changes nothing. Answers by different AI models aren't mixed: the stored one stays,
    unless it's a refusal and the other isn't.
    """
    if not stored:
        return other
    if stored.get("by") != other.get("by"):
        return other if stored.get("failed") and not other.get("failed") else stored
    combined = {name: value for name, value in {**other, **stored}.items() if name not in _COUNTS}
    fields = list(dict.fromkeys([*(stored.get("fields") or []), *(other.get("fields") or [])]))
    if fields or "fields" in stored or "fields" in other:
        combined["fields"] = fields
    counts = unanswered(stored)
    for name, count in unanswered(other).items():
        counts[name] = max(counts.get(name, 0), count)
    return _counted(combined, counts)


def only(labels: dict, fields: list[str]) -> dict:
    """What the AI model said about a task, with only the fields a predictor asks for, and who said it."""
    return {name: value for name, value in labels.items() if name in fields or name not in FIELDS}


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


def _anthropic(
    model: str, system: str, text: str, shape: dict, most_tokens: int = MOST_ANSWER_TOKENS, fallback: bool = True
) -> dict:
    """Ask Anthropic's API. `most_tokens` is the longest the answer may be. Without `fallback`, a request its safety
    checks decline isn't tried again on another AI model, as under a spending limit."""
    import anthropic

    request = {
        "model": model,
        "max_tokens": most_tokens,
        "system": system,
        "messages": [{"role": "user", "content": text}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": shape}},
    }
    try:
        client = anthropic.Anthropic()
        if model.startswith("claude-haiku") or not fallback:
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
    return _with_usage(_json(text), response, model)


def _openai(
    model: str, system: str, text: str, shape: dict, most_tokens: int | None = None, fallback: bool = True
) -> dict:
    """Ask OpenAI's API. `most_tokens`, when given, is the longest the answer may be, its reasoning included."""
    import openai

    limited = {"max_output_tokens": most_tokens} if most_tokens is not None else {}
    try:
        response = openai.OpenAI().responses.create(
            model=model,
            instructions=system,
            input=text,
            text={"format": {"type": "json_schema", "name": "task_analysis", "schema": shape, "strict": True}},
            **limited,
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
    return _with_usage(_json(response.output_text), response, model)


def _with_usage(found: dict, response, model: str) -> dict:
    """The answer, with the tokens the request used and the AI model that answered, so a spending limit can count
    what it really cost. They're taken out again before the answer is read."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return found
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0  # OpenAI's, inside input
    found["_usage"] = {
        "model": getattr(response, "model", None) or model,
        "input": max((getattr(usage, "input_tokens", 0) or 0) - cached, 0),
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": (getattr(usage, "cache_read_input_tokens", 0) or 0) + cached,
        "output": getattr(usage, "output_tokens", 0) or 0,
    }
    return found


def _claude_code(model: str, system: str, text: str, shape: dict, **limits) -> dict:
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
        said = str(result.get("result") or "an error")
        if _signed_out([said]):
            raise LLMError(SIGNED_OUT["Claude Code"], stops=True)
        raise LLMError(f"Claude Code said: {' '.join(said.split())[:200]}")
    structured = result.get("structured_output") if isinstance(result, dict) else None
    if isinstance(structured, dict):
        return structured
    return _json(str(result.get("result") or "") if isinstance(result, dict) else "")


def _codex(model: str, system: str, text: str, shape: dict, **limits) -> dict:
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
        raise _failed(name, done.stdout or "", done.stderr or "", text)
    return done.stdout


def _failed(name: str, stdout: str, stderr: str, sent: str) -> LLMError:
    """Why Claude Code or Codex stopped with an error, in plain words.

    Claude Code prints a JSON object even then, and its "result" says why. Otherwise the reason is in the last
    line the command printed. A line of the text it was sent, which a command may print back, never counts. When
    the command isn't signed in, as when its sign-in expired, the error says how to sign in and stops LLM
    analysis: every other request would fail the same way.
    """
    said = _result(stdout)
    lines = [line.strip() for line in f"{said or stdout}\n{stderr}".splitlines()]
    theirs = [line for line in lines if line and line not in sent]
    if _signed_out(theirs):
        return LLMError(SIGNED_OUT[name], stops=True)
    if said:
        detail = " ".join(said.split())
    else:
        detail = (theirs or [line for line in lines if line] or [""])[-1]
    return LLMError(f"{name} stopped with an error: {detail[:200] or 'no details'}")


def _result(stdout: str) -> str:
    """What the JSON object Claude Code prints says, as its "result". Empty when it printed none."""
    for text in (stdout, *reversed(stdout.strip().splitlines())):
        try:
            found = json.loads(text)
        except ValueError:
            continue
        if isinstance(found, dict) and isinstance(found.get("result"), str):
            return found["result"]
    return ""


def _signed_out(lines: list[str]) -> bool:
    """Whether a command's error says it isn't signed in."""
    said = "\n".join(lines).lower()
    return any(word in said for word in SIGN_IN_WORDS)


def _json(text: str) -> dict:
    try:
        found = json.loads(text)
    except ValueError:
        raise LLMError("The AI model's answer wasn't the JSON it was asked for.", lasting=True) from None
    if not isinstance(found, dict):
        raise LLMError("The AI model's answer wasn't the JSON it was asked for.", lasting=True)
    return found
