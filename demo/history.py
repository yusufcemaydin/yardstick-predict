"""A made-up history of coding-agent runs, for the demo.

It writes Claude Code and Codex records into a home folder made for the demo, in the formats those agents use.
Every prompt, time and token count is invented. Together they read like eight weeks of work on three small
projects with two AI models: claude-opus-5-5 at max effort in Claude Code and gpt-6.1-sol at xhigh in Codex.

A task's size depends on its kind, such as a question or a new feature. That gives a model something to learn
from the prompts. The task the demo forecasts is left out of the history.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from builders import ClaudeSession, CodexSession, usage  # noqa: E402

DEMO_TASK = "Add a login page with tests"
"""The task the demo forecasts. It's never in the history."""

PROJECTS = ("storefront", "billing-api", "mobile-app")
PAGES = ("login", "signup", "settings", "profile", "checkout", "search", "billing", "admin", "invite", "orders")
MODULES = (
    "auth module",
    "billing service",
    "cart",
    "API client",
    "parser",
    "cache layer",
    "router",
    "email sender",
    "date helpers",
    "user model",
    "upload handler",
    "rate limiter",
    "job queue",
    "search index",
)
FEATURES = (
    "dark mode toggle",
    "password reset flow",
    "CSV export",
    "pagination",
    "search bar",
    "health check endpoint",
    "retry with backoff",
    "audit log",
    "invite link",
    "two-factor login",
    "webhook handler",
    "image resizer",
)

KINDS = {
    "question": (
        "Explain how the {module} works",
        "What does the {module} do when the input is empty?",
        "Why does the {page} page load slowly?",
        "Where is the {module} configured?",
        "Is the {module} covered by tests?",
    ),
    "small": (
        "Fix the typo in the {page} page heading",
        "Rename the {module} to something clearer",
        "Remove the unused imports in the {module}",
        "Update the README section on the {module}",
        "Add a comment that explains the retries in the {module}",
        "Bump the dependencies and check that the build still works",
    ),
    "fix": (
        "Fix the failing test in the {module}",
        "The {page} page crashes when the form is empty. Fix it",
        "Fix the flaky test in the {module}",
        "Fix the off-by-one error in the {module}",
        "The {module} times out under load. Find out why and fix it",
    ),
    "feature": (
        "Add a {page} page",
        "Add a {page} page with tests",
        "Add a {feature} to the {page} page",
        "Add a {feature} with tests",
        "Add a {feature} to the {module} and cover it with tests",
    ),
    "big": (
        "Refactor the {module} into smaller modules",
        "Migrate the {module} to the new API, with tests",
        "Rewrite the {module} and add tests for every case",
        "Move the {module} into a package of its own",
    ),
}
SHARES = {"question": 0.12, "small": 0.22, "fix": 0.24, "feature": 0.28, "big": 0.14}
"""How often each kind of task comes up."""
CALLS = {"question": 1, "small": 5, "fix": 13, "feature": 26, "big": 52}
"""How many times a typical task of each kind calls claude-opus-5-5. A question gets a direct answer."""


def prompt(rng: random.Random) -> tuple[str, str]:
    """A new prompt and its kind. The demo's own task never comes up."""
    while True:
        kind = rng.choices(list(SHARES), weights=list(SHARES.values()))[0]
        text = rng.choice(KINDS[kind]).format(
            page=rng.choice(PAGES), module=rng.choice(MODULES), feature=rng.choice(FEATURES)
        )
        if text.lower() != DEMO_TASK.lower():
            return text, kind


def calls(rng: random.Random, kind: str, text: str, share: float = 1.0) -> int:
    """How many times the task calls the AI model. Tests add work. `share` scales it for an AI model that calls less."""
    if kind == "question":
        return 1
    typical = CALLS[kind] * (1.35 if "test" in text.lower() else 1.0) * share
    return max(2, round(typical * math.exp(rng.gauss(0, 0.33))))


def claude_tasks(home: Path, rng: random.Random, start: float, end: float, count: int) -> None:
    """Claude Code sessions on claude-opus-5-5 at max effort, in ~/.claude/projects."""
    root = home / ".claude" / "projects"
    made = session = 0
    while made < count:
        clock = rng.uniform(start, end)
        builder = ClaudeSession(root, f"opus-{session:03d}", clock, project=rng.choice(PROJECTS), version="2.9.4")
        for _ in range(min(rng.randint(1, 5), count - made)):
            text, kind = prompt(rng)
            line = builder.prompt(text)
            context = rng.randint(21_000, 27_000)  # the system prompt, the tools and the project's notes
            steps = calls(rng, kind, text)
            for step in range(steps):
                last = step == steps - 1
                grown = 0 if step == 0 else rng.randint(1_200, 3_600)  # what the last tool result added
                context += grown
                spent = usage(
                    new_input=rng.randint(4, 60),
                    write=grown or rng.randint(800, 2_400),
                    read=context,
                    output=rng.randint(500, 1_500) if kind == "question" else rng.randint(150, 1_800),
                )
                seconds = math.exp(math.log(10.5) + rng.gauss(0, 0.4)) * (2.2 if kind == "question" else 1.0)
                tool = None if last else rng.choice(("Read", "Edit", "Bash", "Grep", "Edit", "Bash"))
                tool_id = builder.reply(seconds, model="claude-opus-5-5", effort="max", tool=tool, calls_usage=spent)
                if tool_id:
                    ran_tests = tool == "Bash" and "test" in text.lower() and rng.random() < 0.4
                    waited = rng.uniform(8, 35) if ran_tests else rng.uniform(0.2, 3.5)
                    builder.tool_result(tool_id, waited, line["promptId"])
            made += 1
            builder.clock += rng.uniform(90, 900)  # the person reads the answer, then asks again
        builder.save(mtime=builder.clock)
        session += 1


def codex_tasks(home: Path, rng: random.Random, start: float, end: float, count: int) -> None:
    """Codex sessions on gpt-6.1-sol at xhigh effort, in ~/.codex/sessions."""
    root = home / ".codex" / "sessions"
    made = session = 0
    while made < count:
        clock = rng.uniform(start, end)
        name = f"sol-{session:03d}"
        builder = CodexSession(root, name, clock, originator="codex_cli_rs")
        builder.lines[0]["payload"]["cwd"] = f"/work/{rng.choice(PROJECTS)}"
        for turn in range(min(rng.randint(1, 4), count - made)):
            text, kind = prompt(rng)
            steps = calls(rng, kind, text, share=0.75)
            context, spent = rng.randint(18_000, 24_000), []
            for step in range(steps):
                context += 0 if step == 0 else rng.randint(1_800, 4_200)
                new = rng.randint(1_500, 4_500)
                output = rng.randint(400, 1_400) if kind == "question" else rng.randint(700, 3_000)
                spent.append((new, context, output))
            seconds = sum(math.exp(math.log(21.0) + rng.gauss(0, 0.45)) for _ in range(steps))
            builder.turn(
                f"{name}-turn-{turn}",
                text,
                spent,
                seconds,
                model="gpt-6.1-sol",
                effort="xhigh",
                tools=kind != "question",
            )
            made += 1
            builder.clock += rng.uniform(90, 900)
        builder.save(mtime=builder.clock)
        session += 1


def write(home: Path, seed: int = 7, now: float | None = None) -> None:
    """Write the whole made-up history into `home`, ending a day before `now`."""
    rng = random.Random(seed)
    end = (now or time.time()) - 86_400
    start = end - 56 * 86_400
    claude_tasks(home, rng, start, end, count=260)
    codex_tasks(home, rng, start, end, count=200)


if __name__ == "__main__":
    write(Path(sys.argv[1]))
