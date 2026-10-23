"""What came before a task: its project's recent tasks and its conversation so far.

A task's outcome is known only once it really finished, so each task only sees tasks that had finished
when it started. These numbers come from the records alone: they cost nothing and send nothing anywhere.

The project's numbers are always used. The conversation's come with the conversation feature, which a
predictor can leave out. They tell the most about a person's sessions.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .archives import Task

PROJECT_TASKS = 5
"""How many of the project's most recent finished tasks the project's numbers describe."""


def finished_at(task: Task) -> float:
    """When a task really finished.

    A task stored by an older version doesn't know, so its start plus its seconds stands in.
    """
    return task.finished_at if task.finished_at is not None else task.started_at + task.seconds


@dataclass
class _Timeline:
    """Tasks in the order they started, to find the ones before a moment quickly."""

    tasks: list[Task] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)

    def before(self, moment: float) -> list[Task]:
        return self.tasks[: bisect.bisect_left(self.starts, moment)]


class Past:
    """The known tasks, by project and by conversation, to describe what came before any task."""

    def __init__(self, known: list[Task]) -> None:
        from .model import usable

        self.projects: dict[str, _Timeline] = defaultdict(_Timeline)
        self.sessions: dict[str, _Timeline] = defaultdict(_Timeline)
        # A project's typical outcomes come from tasks the model could learn from. A conversation's last task
        # counts even when it was stopped: what it did so far still says something about the next one.
        for task in usable(known):
            if task.project:
                self.projects[task.project].tasks.append(task)
        for task in sorted(known, key=lambda t: (t.started_at, t.id)):
            if task.session:
                self.sessions[task.session].tasks.append(task)
        for timeline in [*self.projects.values(), *self.sessions.values()]:
            timeline.starts = [t.started_at for t in timeline.tasks]
        # How big a new conversation is at its first call, by agent: its instructions and tools, before any talk.
        firsts: dict[str, list[int]] = defaultdict(list)
        for timeline in self.sessions.values():
            first = timeline.tasks[0]
            if first.context_tokens > 0:
                firsts[first.agent].append(first.context_tokens)
        self.new_sizes = {agent: int(np.median(sizes)) for agent, sizes in firsts.items()}
        every = [size for sizes in firsts.values() for size in sizes]
        self.new_size = int(np.median(every)) if every else 0

    def numbers(
        self,
        project: str,
        session: str,
        at: float,
        origin: str,
        conversation_tokens: int,
        agent: str = "",
        prompt: str = "",
        conversation: bool = True,
    ) -> dict[str, float]:
        """What the project showed before a task that starts `at`, and with `conversation`, what its conversation
        showed and how big it was.

        A conversation of unknown size, such as a new one, gets the size a new conversation with this agent
        usually has at its first call, plus the prompt.
        """
        found = self.project_numbers(project, at)
        if not conversation:
            return found
        if conversation_tokens <= 0:
            start = self.new_sizes.get(agent, self.new_size)
            conversation_tokens = start + len(prompt) // 4 if start else 0
        return {**found, **self.session_numbers(session, at), **size_numbers(conversation_tokens, origin)}

    def project_numbers(self, project: str, at: float) -> dict[str, float]:
        """The project's last few finished tasks: their typical outcomes, how many came before, and how long ago."""
        # A task with no known project counts as the first of a new one, in training and in a forecast alike.
        before = self.projects[project].before(at) if project in self.projects else []
        done = [t for t in before if finished_at(t) <= at]
        found = {"project_tasks_before": math.log1p(len(done))}
        if not done:
            return found
        last = done[-PROJECT_TASKS:]
        costs = [math.log(t.usd) for t in last if t.usd]
        found.update(
            project_seconds=float(np.median([math.log(t.seconds) for t in last])),
            project_tokens=float(np.median([math.log(t.tokens) for t in last])),
            project_direct=float(np.mean([t.direct for t in last])),
            project_days_since=math.log1p(max(at - max(finished_at(t) for t in done), 0.0) / 86400),
        )
        if costs:
            found["project_usd"] = float(np.median(costs))
        return found

    def session_numbers(self, session: str, at: float) -> dict[str, float]:
        """The conversation so far: how many tasks came before, and what the last finished one took."""
        # A conversation not in the records yet, or one a request doesn't name, counts as new: as in training,
        # where a conversation's first task has none before it.
        before = self.sessions[session].before(at) if session in self.sessions else []
        found = {"session_tasks_before": math.log1p(len(before))}
        done = [t for t in before if finished_at(t) <= at]
        if done:
            last = done[-1]
            found.update(
                previous_seconds=math.log(max(last.seconds, 0.5)),
                previous_tokens=math.log(max(last.tokens, 1)),
                previous_direct=float(last.direct),
                minutes_since_previous=math.log1p(max(at - finished_at(last), 0.0) / 60),
            )
            if last.usd:
                found["previous_usd"] = math.log(last.usd)
        return found


def size_numbers(conversation_tokens: int, origin: str) -> dict[str, float]:
    """How big the conversation was when the task started, in tokens.

    A person's conversation and a program's one-off run grow differently, so each has its own number.
    """
    if not conversation_tokens or conversation_tokens <= 0:
        return {}
    key = "conversation_tokens_program" if origin == "program" else "conversation_tokens"
    return {key: math.log1p(conversation_tokens)}
