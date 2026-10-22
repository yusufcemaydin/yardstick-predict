"""Write small Claude Code and Codex records in the formats those agents use, for tests."""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path


def iso(moment: float) -> str:
    return datetime.fromtimestamp(moment, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def usage(new_input=100, write=0, read=20_000, output=200, write_1h=0, speed="standard") -> dict:
    return {
        "input_tokens": new_input,
        "cache_creation_input_tokens": write + write_1h,
        "cache_read_input_tokens": read,
        "output_tokens": output,
        "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": write_1h},
        "speed": speed,
    }


class ClaudeSession:
    """Builds one Claude Code session file, turn by turn."""

    def __init__(self, root: Path, session: str, start: float, project: str = "demo", version: str = "2.9.0"):
        self.folder = root / f"-work-{project}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / f"{session}.jsonl"
        self.session, self.project, self.version = session, project, version
        self.clock = start
        self.lines: list[dict] = []
        self.count = 0

    def _id(self, kind: str) -> str:
        self.count += 1
        return f"{kind}-{self.session}-{self.count}"

    def prompt(self, text: str, origin: str | None = "human", prompt_id: str | None = None, uuid: str | None = None):
        line = {
            "type": "user",
            "uuid": uuid or self._id("u"),
            "promptId": prompt_id or self._id("p"),
            "sessionId": self.session,
            "timestamp": iso(self.clock),
            "cwd": f"/work/{self.project}",
            "version": self.version,
            "isSidechain": False,
            "message": {"role": "user", "content": text},
        }
        if origin is not None:
            line["turnOrigin"] = origin
        self.lines.append(line)
        return line

    def reply(
        self,
        seconds: float,
        model="claude-opus-5-5",
        effort="high",
        tool=None,
        text="Done.",
        calls_usage=None,
        repeat=1,
    ):
        """One model call. `repeat` writes the same reply over several lines, as Claude Code does."""
        self.clock += seconds
        message_id, request_id = self._id("msg"), self._id("req")
        tool_id = self._id("tool") if tool else None
        for n in range(repeat):
            content = (
                [{"type": "tool_use", "id": tool_id, "name": tool, "input": {}}]
                if tool and n == 0
                else [{"type": "text", "text": text}]
            )
            self.lines.append(
                {
                    "type": "assistant",
                    "uuid": self._id("a"),
                    "requestId": request_id,
                    "sessionId": self.session,
                    "timestamp": iso(self.clock),
                    "effort": effort,
                    "isSidechain": False,
                    "message": {
                        "id": message_id,
                        "model": model,
                        "role": "assistant",
                        "content": content,
                        "usage": calls_usage or usage(),
                    },
                }
            )
        return tool_id

    def tool_result(self, tool_id: str, seconds: float, prompt_id: str, source: str = "config"):
        self.clock += seconds
        self.lines.append(
            {
                "type": "user",
                "uuid": self._id("r"),
                "promptId": prompt_id,
                "sessionId": self.session,
                "timestamp": iso(self.clock),
                "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}],
                },
                "permissionDecision": {"decision": "accept", "source": source},
            }
        )

    def synthetic(self, seconds: float, error: str | None = "rate_limit", sidechain: bool = False):
        """A reply Claude Code writes itself, with no model call: a failed call such as a server error or a
        usage limit, or with `error=None`, the placeholder it writes when a message needs no answer."""
        self.clock += seconds
        line = {
            "type": "assistant",
            "uuid": self._id("y"),
            "sessionId": self.session,
            "timestamp": iso(self.clock),
            "isSidechain": sidechain,
            "isApiErrorMessage": error is not None,
            "message": {
                "id": self._id("ymsg"),
                "model": "<synthetic>",
                "role": "assistant",
                "content": [{"type": "text", "text": "API Error: 429" if error else "No response requested."}],
                "usage": usage(new_input=0, read=0, output=0),
            },
        }
        if error is not None:
            line["error"] = error
        self.lines.append(line)

    def interrupt(self, seconds: float = 1):
        self.clock += seconds
        self.lines.append(
            {
                "type": "user",
                "uuid": self._id("i"),
                "sessionId": self.session,
                "timestamp": iso(self.clock),
                "isSidechain": False,
                "message": {"role": "user", "content": [{"type": "text", "text": "[Request interrupted by user]"}]},
            }
        )

    def bookkeeping(self, seconds: float):
        """A line the app writes long after the agent finished, such as a title."""
        self.clock += seconds
        self.lines.append({"type": "custom-title", "timestamp": iso(self.clock), "sessionId": self.session})

    def side_line(self, seconds: float, kind: str):
        """A line that isn't the agent's own reply or tool result: a helper's message, a progress update or a hook."""
        self.clock += seconds
        line = {"uuid": self._id("x"), "sessionId": self.session, "timestamp": iso(self.clock), "isSidechain": False}
        if kind == "helper":
            # Older versions of Claude Code wrote a helper's messages into the main record.
            message = {"role": "assistant", "content": [{"type": "text", "text": "Still looking."}]}
            line.update(type="assistant", isSidechain=True, message=message)
        elif kind == "progress":
            line.update(type="progress", data={"type": "agent_progress", "agentId": "a1"}, toolUseID="agent_a1")
        elif kind == "hook":
            line.update(type="attachment", attachment={"type": "hook_success", "hookEvent": "Notification"})
        else:
            raise ValueError(kind)
        self.lines.append(line)

    def helper(self, prompt_id: str, model="claude-haiku-5-5", calls=1, call_usage=None):
        folder = self.folder / self.session / "subagents"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"agent-{self._id('h')}.jsonl", "w") as handle:
            for _ in range(calls):
                handle.write(
                    json.dumps(
                        {
                            "type": "assistant",
                            "promptId": prompt_id,
                            "isSidechain": True,
                            "requestId": self._id("hreq"),
                            "timestamp": iso(self.clock),
                            "message": {
                                "id": self._id("hmsg"),
                                "model": model,
                                "role": "assistant",
                                "content": [],
                                "usage": call_usage or usage(new_input=10, read=5_000, output=50),
                            },
                        }
                    )
                    + "\n"
                )

    def save(self, mtime: float | None = None):
        with open(self.path, "w") as handle:
            for line in self.lines:
                handle.write(json.dumps(line) + "\n")
        if mtime is not None:
            import os

            os.utime(self.path, (mtime, mtime))
        return self.path


class CodexSession:
    """Builds one Codex session file. `name` names the file, for a thread spread over several files."""

    def __init__(
        self,
        root: Path,
        session: str,
        start: float,
        originator="codex_exec",
        parent: str | None = None,
        name: str | None = None,
    ):
        day = datetime.fromtimestamp(start, tz=timezone.utc)
        folder = root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"rollout-{name or session}.jsonl"
        self.clock = start
        self.total = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
        }
        source = {"subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": 1}}} if parent else "exec"
        self.lines = [
            {
                "timestamp": iso(start),
                "type": "session_meta",
                "payload": {
                    "id": session,
                    "cwd": "/work/demo",
                    "cli_version": "0.160.0",
                    "originator": originator,
                    "source": source,
                },
            }
        ]

    def add(self, kind, payload):
        self.lines.append({"timestamp": iso(self.clock), "type": kind, "payload": payload})

    def turn(
        self,
        turn_id: str,
        prompt: str,
        calls: list[tuple[int, int, int]],
        seconds: float,
        model="gpt-6.1-sol",
        effort="xhigh",
        tools=True,
        aborted=False,
        root: str | None = None,
    ):
        """One turn. `root` is the main thread's turn behind it, which newer versions of Codex write down."""
        started = {"type": "task_started", "turn_id": turn_id, "started_at": int(self.clock)}
        if root is not None:
            started["root_turn_id"] = root
        self.add("event_msg", started)
        self.add(
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /work/demo\n\nBe careful."}],
            },
        )
        self.add("turn_context", {"model": model, "effort": effort, "turn_id": turn_id})
        self.add(
            "response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}
        )
        for new_input, cached, output in calls:
            if tools:
                self.add("response_item", {"type": "function_call", "name": "shell", "arguments": "{}", "call_id": "c"})
            self.clock += seconds / max(len(calls), 1)
            last = {
                "input_tokens": new_input + cached,
                "cached_input_tokens": cached,
                "cache_write_input_tokens": 0,
                "output_tokens": output,
                "reasoning_output_tokens": 0,
                "total_tokens": new_input + cached + output,
            }
            for key in self.total:
                self.total[key] += last[key]
            self.add(
                "event_msg",
                {"type": "token_count", "info": {"total_token_usage": dict(self.total), "last_token_usage": last}},
            )
            # Codex sometimes repeats a token count with nothing new. It must not count twice.
            self.add(
                "event_msg",
                {"type": "token_count", "info": {"total_token_usage": dict(self.total), "last_token_usage": last}},
            )
        self.add(
            "response_item",
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Fixed."}]},
        )
        finished = int(self.clock)
        if aborted:
            self.add(
                "event_msg",
                {"type": "turn_aborted", "turn_id": turn_id, "reason": "interrupted", "completed_at": finished},
            )
        else:
            self.add(
                "event_msg",
                {
                    "type": "task_complete",
                    "turn_id": turn_id,
                    "duration_ms": int(seconds * 1000),
                    "completed_at": finished,
                },
            )

    def save(self, mtime: float | None = None):
        with open(self.path, "w") as handle:
            for line in self.lines:
                handle.write(json.dumps(line) + "\n")
        if mtime is not None:
            import os

            os.utime(self.path, (mtime, mtime))
        return self.path


WORDS = "fix add test refactor page login api cache docs build deploy schema query parser cli model".split()


def history(
    root: Path,
    tasks: int,
    start: float,
    models=("claude-opus-5-5",),
    seed: int = 0,
    sessions_of: int = 5,
    efforts=("high",),
):
    """Many Claude Code tasks whose time and tokens depend on the prompt, so a model has something to learn."""
    rng = random.Random(seed)
    made = 0
    session = 0
    clock = start
    while made < tasks:
        builder = ClaudeSession(root, f"s{seed}-{session}", clock)
        for _ in range(min(sessions_of, tasks - made)):
            model = models[made % len(models)]
            effort = efforts[(made // len(models)) % len(efforts)]
            size = rng.randint(1, 60)
            prompt = " ".join(rng.choice(WORDS) for _ in range(size))
            line = builder.prompt(prompt)
            quick = size < 8
            steps = 1 if quick else 1 + int(size / 6)
            pace = 2.0 if model.startswith("claude-opus") else 6.0
            for step in range(steps):
                seconds = math.exp(math.log(pace) + rng.gauss(0, 0.3)) * (1 + size / 20)
                last = step == steps - 1
                tool = builder.reply(
                    seconds,
                    model=model,
                    effort=effort,
                    tool=None if last else "Bash",
                    calls_usage=usage(read=20_000 + 3_000 * step, output=100 + 10 * size),
                )
                if tool:
                    builder.tool_result(tool, 0.5, line["promptId"])
            made += 1
            builder.clock += 60
        clock = builder.clock + 3600
        builder.save(mtime=start)
        session += 1
    return made
