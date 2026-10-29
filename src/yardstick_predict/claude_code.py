"""Show forecasts in Claude Code's status line.

When you send a prompt, a hook starts a forecast in the background and returns at once, so the prompt
isn't held up. Claude Code's status line then shows the forecast for the model you're on, and counts
up while the agent works. When the agent stops, it shows how long the agent worked on the task, by the
rules training uses.

Claude Code runs these itself:

    python -P -m yardstick_predict.claude_code hook       on each prompt, and each time the agent stops
    python -P -m yardstick_predict.claude_code status     to fill in the status line

Claude Code runs them in the session's folder. -P keeps Python from loading code from there, so a file
such as json.py in your project never runs in their place. They load only what they need, so the status
line stays quick. `yardstick-predict claude-code` adds them to Claude Code's settings, and removes them.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MARK = "yardstick_predict.claude_code"
"""Every command this module adds to Claude Code's settings contains this, so it can find them again."""
KEEP_DAYS = 7
TAIL_BYTES = 256_000
STOPS = ("Stop", "StopFailure")
"""Claude Code sends Stop when the agent finishes, and StopFailure when a request to the model fails."""
FIND_SECONDS = 3.0
"""How long a forecast waits for its prompt to show up in the session's record. Claude Code writes it a
moment after the hook runs."""
KEY_SECONDS = 10.0
"""How long installing waits for the password store. A locked one can wait for you to unlock it."""
QUIET_SECONDS = 5.0
"""How long, at most, working out the agent's time waits for the record to catch up after the agent stops."""
EARLIER = 3
"""How many earlier tasks a session keeps, for a prompt that turns out not to start a task."""
OTHERS = 20
"""How many prompts that started no task a session remembers, so the stops after them don't count."""
TASK_BYTES = 32_000_000
"""The most of a record that working out one task's time reads. Real tasks take far less."""
TASK_KEYS = ("prompt_at", "prompt_id", "offset", "found", "done_at", "took", "took_for")
"""What a session's state says about its task, rather than about the session."""
HEADLESS = "sdk"
"""How Claude Code's name for a run with no status line begins: sdk-cli for claude -p, sdk-ts and sdk-py
for the Agent SDKs."""
REPLACED = "status-lines.json"
"""The status lines that adding this one replaced, by settings file, so removing it can put them back."""


# Files -------------------------------------------------------------------------------------------------


def folder() -> Path:
    """Where the forecasts for each Claude Code session are kept, inside the private yardstick folder."""
    home = _home()
    found = home / "claude-code"
    found.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(home, 0o700)
    return found


def _home(home: str | None = None) -> Path:
    return Path(home or os.environ.get("YARDSTICK_HOME") or "~/.yardstick").expanduser()


def _session_file(session: str, ending: str) -> Path:
    """One of a session's files. Each name starts with the session's id, cut down to safe letters."""
    safe = "".join(c for c in session if c.isalnum() or c in "-_")[:80] or "session"
    return folder() / f"{safe}{ending}"


def _state_path(session: str) -> Path:
    """Where a session's task is kept: when its prompt was sent, when the agent last stopped, and more."""
    return _session_file(session, ".json")


def _prompt_path(session: str, prompt_at: float) -> Path:
    """Where the hook leaves a prompt for its forecast. Each prompt has its own file, named by when it was sent."""
    return _session_file(session, f".{float(prompt_at)!r}.prompt")


def _forecast_path(session: str, prompt_at: float) -> Path:
    """Where a prompt's forecast is saved. Each prompt has its own, so one forecast can never replace another."""
    return _session_file(session, f".{float(prompt_at)!r}.forecast.json")


def _lock_path(session: str, name: str) -> Path:
    return _session_file(session, f".{name}.lock")


def _read(path: Path) -> dict:
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return found if isinstance(found, dict) else {}


def _write(path: Path, data: dict) -> None:
    from .files import write_text  # here, so the status line never loads what only saving needs

    write_text(path, json.dumps(data))


def _update(session: str, change) -> dict:
    """Change a session's state in one step, so the hook and the work in the background never undo each other.

    `change` gets the state and gives the new one, or None to leave it as it is. Returns the state.
    """
    from . import locks

    path = _state_path(session)
    with locks.held(_lock_path(session, "state")):
        state = _read(path)
        new = change(dict(state))
        if new is None or new == state:
            return state
        _write(path, new)
        return new


def _clean(value: object) -> str:
    """Text that any file can hold. Half of an emoji, which no file can hold alone, becomes the � sign."""
    if not isinstance(value, str):
        return ""
    return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _number(value: object) -> float | None:
    """A finite number, or None. JSON's true and false don't count."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _size(path: str | None) -> int | None:
    try:
        return os.path.getsize(path) if path else None
    except (OSError, ValueError):
        return None


def _event() -> object:
    """The JSON Claude Code sends on standard input, or None when it isn't JSON."""
    try:
        stream = getattr(sys.stdin, "buffer", None)
        text = stream.read().decode("utf-8", "replace") if stream is not None else sys.stdin.read()
        return json.loads(text)
    except Exception:
        return None


# The hook ----------------------------------------------------------------------------------------------


def hook() -> int:
    """Note when a prompt is sent and when the agent stops, and start the forecast.

    It prints nothing, since Claude Code would hand that to the agent. And it never fails, whatever it's
    sent: a hook that fails gets in your way on every prompt.
    """
    from .llm import INSIDE

    if os.environ.get(INSIDE):
        return 0  # Claude Code started by LLM analysis, to read tasks: forecasting it would start another
    try:
        event = _event()
        if isinstance(event, dict):
            _handle(event)
    except Exception:
        pass
    return 0


def _handle(event: dict) -> None:
    session = _clean(event.get("session_id"))
    if not session:
        return
    if event.get("hook_event_name") == "UserPromptSubmit":
        _prompted(session, event)
    elif event.get("hook_event_name") in STOPS:
        _stopped(session, event)


def _prompted(session: str, event: dict) -> None:
    """A prompt was sent. It starts a task, unless training would say it doesn't.

    A background job reporting back starts none: its task goes on. Whether any other prompt starts a task,
    and who sent it, its line in the record says. The forecast in the background looks it up, and puts the
    task before back if this one starts none.
    """
    prompt = _clean(event.get("prompt"))
    if not prompt.strip():
        return
    now = time.time()
    if _reports_back(prompt):
        _update(session, _going_on)
        return
    transcript = _record(event)
    lines = _tail(transcript)
    model, effort = _model_and_effort(lines)
    headless = _headless(lines)
    # The prompt's id finds its line in the record. Where the record ended says where to look for it.
    sent = {"prompt_id": _clean(event.get("prompt_id")) or None, "transcript": transcript, "offset": _size(transcript)}
    _update(session, lambda state: _new_task(state, now, sent, model, effort, headless))
    _tidy()
    if headless:
        return  # no status line shows a forecast there, so none is made
    # The project the session works in, as training names a task's project.
    _write(_prompt_path(session, now), {"prompt": prompt, **sent, "cwd": _clean(event.get("cwd"))})
    _start_worker(session)


def _reports_back(prompt: str) -> bool:
    """Whether a prompt is Claude Code's report that a background job finished, by training's rule for its text."""
    if not prompt.lstrip().startswith("<task-notification>"):
        return False
    from .archives import _notices

    return any(status for _, _, status in _notices(prompt))


def _going_on(state: dict) -> dict | None:
    """The state with its task going on: the clock runs again if it had stopped."""
    if _number(state.get("prompt_at")) is None:
        return None
    return {key: value for key, value in state.items() if key != "done_at"}


def _new_task(state: dict, now: float, sent: dict, model: str, effort: str, headless: bool) -> dict:
    """The state for a new task. It keeps the tasks before it, in case this prompt turns out not to start one."""
    earlier = _earlier(state)
    if _number(state.get("prompt_at")) is not None:
        earlier = [_task_of(state), *earlier]
    new = {"prompt_at": now, **sent, "model": model, "effort": effort, "earlier": earlier[:EARLIER]}
    if _others(state):
        new["others"] = _others(state)
    if headless:
        new["headless"] = True
    return new


def _task_of(state: dict) -> dict:
    return {key: state[key] for key in TASK_KEYS if key in state}


def _earlier(state: dict) -> list[dict]:
    earlier = state.get("earlier")
    return [entry for entry in earlier if isinstance(entry, dict)] if isinstance(earlier, list) else []


def _others(state: dict) -> list[str]:
    """The ids of the session's recent prompts that started no task."""
    others = state.get("others")
    return [name for name in others if isinstance(name, str)] if isinstance(others, list) else []


def _stopped(session: str, event: dict) -> None:
    """The agent stopped. The latest stop counts, as another hook can keep the agent working after one.

    A stop after a prompt that started no task, such as a notice from Claude Code, says nothing about the task.
    """
    now = time.time()
    transcript = _record(event)
    prompt_id = _clean(event.get("prompt_id"))

    def stop(state: dict) -> dict | None:
        if _number(state.get("prompt_at")) is None or (prompt_id and prompt_id in _others(state)):
            return None
        return {**state, "done_at": now, "transcript": state.get("transcript") or transcript}

    state = _update(session, stop)
    if _number(state.get("prompt_at")) is not None and not state.get("headless") and not _headless([]):
        _start_worker(session)  # to work out how long the agent worked, once the record catches up


def _tail(transcript: str | None, size: int = TAIL_BYTES) -> list[str]:
    """The last lines of a session's record."""
    if not transcript:
        return []
    try:
        with open(transcript, "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(file.tell() - size, 0))
            return file.read().decode("utf-8", "replace").splitlines()
    except (OSError, ValueError):
        return []


def last_model_and_effort(transcript: str | None) -> tuple[str, str]:
    """The model and effort level of the session's latest real reply, from the end of its record."""
    return _model_and_effort(_tail(transcript))


def _model_and_effort(lines: list[str]) -> tuple[str, str]:
    for line in reversed(lines):
        if '"assistant"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if _main_reply(event):
            return _clean(event["message"]["model"]), _clean(str(event.get("effort") or ""))
    return "", ""


def _main_reply(event: object) -> bool:
    """Whether a line is a real reply from the main agent.

    A helper's reply (a sidechain) is another agent's. An error notice, which Claude Code writes as a reply
    from "<synthetic>", isn't a reply from any model. Neither says what you're running.
    """
    if not isinstance(event, dict) or event.get("type") != "assistant" or event.get("isSidechain"):
        return False
    message = event.get("message")
    model = message.get("model") if isinstance(message, dict) else None
    return isinstance(model, str) and bool(model) and model != "<synthetic>"


def _headless(lines: list[str]) -> bool:
    """Whether the session runs with no status line to show a forecast, as claude -p and SDK runs do.

    Claude Code names how it was started in CLAUDE_CODE_ENTRYPOINT, which its hooks inherit, and in each
    line of the session's record.
    """
    entry = os.environ.get("CLAUDE_CODE_ENTRYPOINT") or _entrypoint(lines)
    return entry.startswith(HEADLESS)


def _entrypoint(lines: list[str]) -> str:
    for line in reversed(lines):
        if '"entrypoint"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("entrypoint"), str):
            return event["entrypoint"]
    return ""


def _start_worker(session: str) -> None:
    """Start the session's work in the background, unless it's running. Then it picks up what's new itself."""
    from . import locks

    with locks.held(_lock_path(session, "worker"), wait=False) as free:
        if not free:
            return
    env = dict(os.environ)
    openmp = _bundled_openmp()
    if openmp:
        fallback = env.get("DYLD_FALLBACK_LIBRARY_PATH") or f"{Path.home()}/lib:/usr/local/lib:/usr/lib"
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{openmp}:{fallback}"
    options: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        options["creationflags"] = 0x00000008 | 0x00000200  # run on its own, apart from the hook
    else:
        options["start_new_session"] = True
    # -P, and a folder of its own to run in, keep the session's folder out of it.
    command = [sys.executable, "-P", "-m", MARK, "work", session]
    subprocess.Popen(command, env=env, cwd=str(folder()), close_fds=True, **options)


def _bundled_openmp() -> str | None:
    """On a Mac, the folder of scikit-learn's OpenMP, which XGBoost needs. Found without loading scikit-learn."""
    if sys.platform != "darwin":
        return None
    import importlib.util

    spec = importlib.util.find_spec("sklearn")
    if spec is None or not spec.origin:
        return None
    found = Path(spec.origin).parent / ".dylibs"
    return str(found) if (found / "libomp.dylib").is_file() else None


def _tidy() -> None:
    """Delete what's no longer needed, once a day at most, so most prompts only look at the date it last ran.

    That's sessions untouched for a week, prompts whose forecast never started, and copies that a process
    left when it stopped halfway.
    """
    root = folder()
    now = time.time()
    mark = root / ".tidied"
    try:
        if now - mark.stat().st_mtime < 86400:
            return
    except OSError:
        pass
    try:
        mark.touch()
        paths = list(root.iterdir())
    except OSError:
        return
    week, hour = now - KEEP_DAYS * 86400, now - 3600
    changed = {}  # when each session's state last changed: at its latest prompt or stop
    for path in paths:
        if path.suffix == ".json" and "." not in path.stem:
            changed[path.stem] = _mtime(path)
    for path in paths:
        made = _mtime(path)
        if path.name in (mark.name, REPLACED) or made is None:
            continue
        last = changed.get(path.name.split(".", 1)[0]) or made
        leftover = path.name.endswith((".prompt", ".task.jsonl", ".partial"))
        if last < week or (leftover and made < hour):
            try:
                path.unlink()
            except OSError:
                pass


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# The work in the background ----------------------------------------------------------------------------


def work(session: str) -> int:
    """Forecast the session's newest prompt, and work out how long the agent worked once it stops.

    At most one of these runs for a session at a time, so a password store that keeps one waiting never
    piles up more. A prompt sent while it runs is picked up before it ends, and a newer prompt takes the
    place of older ones whose forecast hasn't started.
    """
    from . import locks

    for _ in range(100):  # each round is quick, so even work that keeps failing ends soon
        with locks.held(_lock_path(session, "worker"), wait=False) as mine:
            if not mine:
                return 0  # another one is running, and it picks up what's new
            while _next_job(session):
                pass
        # A prompt or a stop that came just as it finished saw it running, so it looks once more.
        if not _has_work(session):
            return 0
    return 0


def _next_job(session: str) -> bool:
    """Do the next piece of work. False when there's none, or when it failed."""
    pending = _pending(session)
    try:
        if pending:
            prompt_at, path = pending[-1]
            try:
                _take(session, prompt_at, path, [older for _, older in pending[:-1]])
            except Exception:
                path.unlink(missing_ok=True)
            return True
        return _settle(session)
    except Exception:
        return False


def _has_work(session: str) -> bool:
    return bool(_pending(session)) or _settle_key(_read(_state_path(session))) is not None


def _pending(session: str) -> list[tuple[float, Path]]:
    """The session's prompts whose forecast hasn't started, oldest first."""
    stem = _state_path(session).stem
    found = []
    for path in folder().glob(f"{stem}.*.prompt"):
        prompt_at = _number_in(path.name, stem, ".prompt")
        if prompt_at is not None:
            found.append((prompt_at, path))
    return sorted(found)


def _number_in(name: str, stem: str, ending: str) -> float | None:
    try:
        return _number(float(name[len(stem) + 1 : -len(ending)]))
    except ValueError:
        return None


def _take(session: str, prompt_at: float, path: Path, older: list[Path]) -> None:
    """Look at the newest prompt with no forecast yet, and forecast it if training would count it as a task."""
    sent = _read(path)
    path.unlink(missing_ok=True)  # taken, so a prompt that breaks the forecast is never tried over and over
    prompt = _clean(sent.get("prompt"))
    prompt_id = _clean(sent.get("prompt_id")) or None
    transcript = _str(sent.get("transcript")) or None
    if not prompt.strip():
        return

    def newer() -> bool:
        return any(at > prompt_at for at, _ in _pending(session))

    verdict = sent.get("verdict")
    if not isinstance(verdict, dict):
        verdict = _verdict(prompt, prompt_id, transcript, prompt_at, sent.get("offset"), newer)
    if verdict is None:
        _write(path, sent)  # a newer prompt came while this one waited for its line, and goes first
        return
    if verdict.get("origin") not in ("person", "program"):
        _update(session, lambda state: _not_a_task(state, prompt_at, prompt_id))
        return
    if newer():
        # A newer prompt came meanwhile. It goes first. This one waits, in case that one starts no task.
        _write(path, {**sent, "verdict": verdict})
        return
    for stale in older:
        stale.unlink(missing_ok=True)  # their forecasts never started, and this prompt takes their place
    _update(session, lambda state: _found(state, prompt_at, verdict))
    words = _clean(verdict.get("prompt")) or prompt
    line = {"until": _str(verdict.get("uuid")) or None, "start": verdict.get("start"), "cwd": _str(sent.get("cwd"))}
    _forecast(session, prompt_at, words, verdict["origin"], transcript=transcript, **line)
    _calibrate(transcript)


def _calibrate(transcript: str | None) -> None:
    """Work out again how much of the weekly limit a dollar uses, once the forecast is saved, so it never waits.

    The session's record sits in the folder of each project's records, which holds every session's.
    """
    try:
        from . import weekly

        records = Path(transcript).parent.parent if transcript else Path("~/.claude/projects").expanduser()
        weekly.calibrate(records)
    except Exception:
        pass  # until a later prompt works it out, the brackets show the forecast's time alone


def _verdict(prompt: str, prompt_id: str | None, transcript: str | None, sent_at: float, offset, give_up):
    """Who sent a prompt, by training's rules for its line in the session's record.

    The origin is "person" or "program", or None when training doesn't count the prompt as a task. Claude
    Code writes the line a moment after the hook runs, so this waits for it a little. When it doesn't
    come, the prompt's text decides. The prompt is then its text as the record holds it, as training reads it.
    Returns None when it gives up waiting, as `give_up` says to.
    """
    from .archives import _claude_origin, _claude_turn_start

    found = _wait_for_prompt_line(transcript, prompt_id, prompt, sent_at, offset, give_up)
    if found is False:
        return None
    if found is None:
        return {"origin": _claude_origin({}, prompt), "prompt": prompt}
    line, start = found
    text = _claude_turn_start(line)
    origin = None if text is None else _claude_origin(line, text)
    return {"origin": origin, "prompt": text or prompt, "uuid": _str(line.get("uuid")) or None, "start": start}


def _not_a_task(state: dict, prompt_at: float, prompt_id: str | None) -> dict:
    """The state after a prompt that starts no task, such as a notice from Claude Code.

    The task before it is back as it was, with its clock where it was. The prompt's id is kept, so the stop
    after its turn doesn't count.
    """
    earlier, others = _earlier(state), _others(state)
    if prompt_id and prompt_id not in others:
        others = [*others, prompt_id][-OTHERS:]
    if _number(state.get("prompt_at")) != prompt_at:
        # A newer prompt came since. Forget this one, so it's never brought back.
        kept = [entry for entry in earlier if _number(entry.get("prompt_at")) != prompt_at]
        return {**state, "earlier": kept, "others": others}
    kept = {key: value for key, value in state.items() if key not in TASK_KEYS}
    if not earlier:
        return {**kept, "earlier": [], "others": others}  # no task before it: there's none to show
    return {**kept, **earlier[0], "earlier": earlier[1:], "others": others}


def _found(state: dict, prompt_at: float, verdict: dict) -> dict:
    """The state, with where a task's prompt is in the record. Once it's sure to be a task, the tasks before it
    can't come back."""
    found = {key: verdict[key] for key in ("uuid", "start") if verdict.get(key) is not None}
    if _number(state.get("prompt_at")) == prompt_at:
        return {**state, "found": found, "earlier": []}
    earlier = []
    for entry in _earlier(state):
        if _number(entry.get("prompt_at")) == prompt_at:
            earlier.append({**entry, "found": found})
            break
        earlier.append(entry)
    return {**state, "earlier": earlier}


def _forecast(
    session: str,
    prompt_at: float,
    prompt: str,
    origin: str,
    until: str | None = None,
    transcript=None,
    start: int | None = None,
    cwd: str = "",
) -> None:
    """Forecast the prompt on every AI model and effort level the predictor covers, and save it.

    Each prompt's forecast has its own file, and the status line shows the current task's, so a slower,
    older forecast never takes a newer one's place. `until` is the id of the prompt's line in the record,
    and `start` the byte it starts at, when the line was found.
    """
    found: dict = {"prompt_at": prompt_at}
    predictor = None
    try:
        from . import cli
        from .forecast import answer, load, parse
        from .pipeline import recent_tasks

        predictor = cli.request_predictor(None)
        model = load(predictor)  # once: its meta picks the AI models and effort levels, and it forecasts them
        meta = getattr(model, "meta", None)
        pairs = cli.effort_pairs(predictor, cli.forecast_models(predictor, meta), None, meta)
        # The prompt came after the conversation so far, and the skills known by then, as training sees each task.
        context, skills = before_prompt(transcript, prompt, prompt_at, until)
        size = conversation_size(transcript, prompt_at, start)
        task = {
            "prompt": prompt,
            "origin": origin,
            "agent": "claude-code",
            "context": context,
            "skills": skills,
            "session": session,
            # As in training, the size at the first call: the conversation so far plus the new prompt. A new
            # conversation gets 0, and the forecast fills in a new conversation's usual size.
            "conversation_tokens": size + len(prompt) // 4 if size else 0,
        }
        if cwd:
            task["project"] = Path(cwd).name
        tasks = [
            {"id": f"t{number}", **task, "models": [name], **({"effort": effort} if effort else {})}
            for number, (name, effort) in enumerate(pairs)
        ]
        parsed, _, _ = parse({"version": 1, "tasks": tasks})
        # The project's tasks since the last refresh tell the most about this one. They're read, not saved.
        # They only add detail, so a forecast still comes when they can't be read.
        recent = []
        if isinstance(transcript, str) and transcript:
            try:
                recent = recent_tasks(predictor, transcript)
            except Exception:
                recent = []
        # The forecast asks the password store for the Voyage key only when a text isn't in the cache, and
        # gives a locked store only a few seconds.
        answered = answer(predictor, parsed, model=model, recent=recent)
        found.update(predictor=predictor.name, forecasts=answered["forecasts"])
    except Exception as error:
        found["error"] = _no_forecast(error, predictor)
    _write(_forecast_path(session, prompt_at), found)
    _forget_old_forecasts(session)


def _no_forecast(error: Exception, predictor) -> str:
    """Why a prompt got no forecast, in a few plain words for the status line.

    A predictor deleted while it was forecast says so, whatever broke as its files went. With several predictors
    and none the default, as after the default one was deleted, it says how to choose one.
    """
    if getattr(error, "code", None) == "no_default_predictor":
        return "no default model. Choose one with: yardstick-predict default NAME"
    folder = getattr(predictor, "dir", None)
    if isinstance(folder, Path) and not (folder / "predictor.json").is_file():
        return f"{predictor.name} was deleted"
    from . import ui

    # Often in another service's words, such as Voyage's. The status line shows them, but never a code that acts.
    return ui.clean(getattr(error, "message", None) or str(error) or type(error).__name__)[:160]


def _forget_old_forecasts(session: str) -> None:
    """Delete the forecasts a session no longer needs. Its task's stays, and so do those of the tasks before
    it, which may yet come back."""
    state = _read(_state_path(session))
    keep = {_number(state.get("prompt_at")), *(_number(entry.get("prompt_at")) for entry in _earlier(state))}
    stem = _state_path(session).stem
    for path in folder().glob(f"{stem}.*.forecast.json"):
        if _number_in(path.name, stem, ".forecast.json") not in keep:
            path.unlink(missing_ok=True)


def stored_key(seconds: float) -> tuple[str | None, str]:
    """The Voyage key in your computer's password store, looked up in at most `seconds`.

    The second value says how the store answered: "ok", "no_store" when there's no password store to ask,
    "error", or "no_answer" when it didn't answer in time, as a locked macOS Keychain waits for you.
    """
    import threading

    from .embeddings import KEY_SERVICE, KEY_USER

    found: list[tuple[str | None, str]] = []

    def look() -> None:
        try:
            import keyring
            from keyring.errors import NoKeyringError
        except Exception:
            found.append((None, "no_store"))
            return
        try:
            found.append(((keyring.get_password(KEY_SERVICE, KEY_USER) or "").strip() or None, "ok"))
        except NoKeyringError:
            found.append((None, "no_store"))
        except Exception:
            found.append((None, "error"))

    thread = threading.Thread(target=look, daemon=True)  # a store that never answers can't keep it from ending
    thread.start()
    thread.join(seconds)
    return found[0] if found else (None, "no_answer")


def _settle(session: str) -> bool:
    """Work out how long the agent worked on the session's task once it stopped, and keep it with the task.

    Returns whether there was a time to work out. The record can lag behind the stop, so it gets a moment
    to catch up first.
    """
    state = _read(_state_path(session))
    key = _settle_key(state)
    if key is None:
        return False

    def same(now: dict) -> bool:
        return [_number(now.get("prompt_at")), _number(now.get("done_at"))] == key

    seconds = None
    if _wait_for_record(state["transcript"], lambda: same(_read(_state_path(session))) and not _pending(session)):
        try:
            seconds = agent_seconds(session, state)
        except Exception:
            seconds = None  # then the line shows the time on the clock
    _update(session, lambda now: {**now, "took": seconds, "took_for": key} if same(now) else None)
    return True


def _settle_key(state: dict) -> list[float] | None:
    """When the session's task began and its agent last stopped, if the agent's time for that is still to work out."""
    started, stopped = _number(state.get("prompt_at")), _number(state.get("done_at"))
    if started is None or stopped is None or state.get("headless") or not _str(state.get("transcript")):
        return None
    key = [started, stopped]
    return None if state.get("took_for") == key else key


def _wait_for_record(transcript: str, unchanged) -> bool:
    """Wait for the record to hold still for a moment, a few seconds at most. False when the task changed meanwhile."""
    deadline = time.monotonic() + QUIET_SECONDS
    size = _size(transcript)
    while True:
        time.sleep(0.5)
        if not unchanged():
            return False
        now = _size(transcript)
        if now == size or time.monotonic() >= deadline:
            return True
        size = now


def agent_seconds(session: str, state: dict) -> float | None:
    """How long the agent worked on the session's task, by the rules training uses, from the session's record.

    Time it spent waiting for you, to answer a question, approve a plan or allow a tool, is left out, and
    a background job's report back counts toward the task. The task's own part of the record is copied out
    and read the way training reads a session, so a long session costs no more than its last task. None
    when the record doesn't hold the task.
    """
    from .archives import _timestamp, read_records

    transcript = _str(state.get("transcript"))
    prompt_at = _number(state.get("prompt_at"))
    found = state.get("found") if isinstance(state.get("found"), dict) else {}
    start, uuid = found.get("start"), _str(found.get("uuid")) or None
    if not transcript or prompt_at is None:
        return None
    if not isinstance(start, int) or isinstance(start, bool):
        # Its forecast didn't find the prompt's line in time. By now it's there, near where the record ended
        # when the prompt was sent.
        located = _prompt_line(transcript, _str(state.get("prompt_id")) or None, None, prompt_at, state.get("offset"))
        if located is None:
            return None
        start, uuid = located[1], _str(located[0].get("uuid")) or None
    with open(transcript, "rb") as record:
        record.seek(start)
        part = record.read(TASK_BYTES + 1)
    if len(part) > TASK_BYTES:
        return None
    first = _json_line(part.split(b"\n", 1)[0])
    if first is None or (uuid is not None and first.get("uuid") != uuid):
        return None  # the record changed since the prompt was found in it
    copy = _session_file(session, f".{prompt_at!r}.task.jsonl")
    try:
        copy.write_bytes(part)
        archive = {"name": "claude-code", "format": "claude-code", "path": str(copy.parent)}
        # With no end in sight, the copy's last task counts as finished.
        tasks = read_records(archive, {str(copy): (len(part), _mtime(copy) or 0.0)}, now=math.inf).tasks
    finally:
        copy.unlink(missing_ok=True)
    began = _timestamp(first.get("timestamp"))
    for task in tasks:
        if (task.id == f"claude-code:{uuid}") if uuid else (task.started_at == began):
            return task.seconds
    return None


def _json_line(raw: bytes) -> dict | None:
    try:
        found = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, RecursionError):
        return None
    return found if isinstance(found, dict) else None


def _wait_for_prompt_line(transcript, prompt_id, prompt, sent_at, offset, give_up):
    """The prompt's own line in the session's record, waiting a few seconds at most for it to be written.

    None when it doesn't come. False when it gives up waiting, as `give_up` says to.
    """
    if not transcript:
        return None
    deadline = time.monotonic() + FIND_SECONDS
    while True:
        found = _prompt_line(transcript, prompt_id, prompt, sent_at, offset)
        if found is not None or time.monotonic() >= deadline:
            return found
        if give_up():
            return False
        time.sleep(0.2)


def _prompt_line(transcript, prompt_id, prompt, sent_at: float, offset) -> tuple[dict, int] | None:
    """A prompt's own line in the session's record, and the byte it starts at. None when it isn't there.

    With the prompt's id, it's the first line from you that carries the id. Without, it's a line from you
    with the same text, written within a minute of the prompt. Lines with a tool's result don't count.
    """
    from .archives import _claude_text, _timestamp

    size = _size(transcript)
    if size is None or not (prompt_id or prompt):
        return None
    near = offset if isinstance(offset, int) and not isinstance(offset, bool) else size
    needle = prompt_id.encode("utf-8", "replace") if prompt_id else None
    best = None
    try:
        for start, raw in _raw_lines(transcript, max(min(near, size) - TAIL_BYTES, 0)):
            if b'"user"' not in raw or (needle is not None and needle not in raw):
                continue
            line = _json_line(raw)
            message = line.get("message") if line else None
            if line is None or line.get("type") != "user" or line.get("isSidechain") or not isinstance(message, dict):
                continue
            text = _claude_text(message)
            if text is None:
                continue
            if prompt_id:
                if line.get("promptId") == prompt_id:
                    return line, start
            elif prompt is not None and text.strip() == prompt.strip():
                moment = _timestamp(line.get("timestamp"))
                if moment is not None and abs(moment - sent_at) <= 60:
                    best = line, start
    except OSError:
        return None
    return best


def _raw_lines(transcript: str, start: int):
    """Each whole line of a record from byte `start` on, with the byte it begins at."""
    with open(transcript, "rb") as file:
        file.seek(max(start - 1, 0))
        position = file.tell()
        if start > 0:
            position += len(file.readline())  # the rest of a line that began before `start`
        while raw := file.readline():
            yield position, raw
            position += len(raw)


def conversation_size(transcript: object, sent_at: float, start: int | None = None) -> int:
    """How big the conversation was when a prompt was sent, in tokens: all the last reply read and wrote.

    Training reads the size from a task's first call, which also holds the new prompt. A prompt is small
    next to a conversation, so the two agree closely. 0 for a new conversation. Only the end of the record
    before the prompt's line, which starts at byte `start` when it was found, is read: the last reply is there.
    """
    from .archives import _claude_call, _timestamp

    if not isinstance(transcript, str) or not transcript:
        return 0
    end = start if isinstance(start, int) and not isinstance(start, bool) else _size(transcript)
    if end is None:
        return 0
    size = 0
    try:
        for at, raw in _raw_lines(transcript, max(end - TAIL_BYTES, 0)):
            if at >= end:
                break
            if b'"assistant"' not in raw:
                continue
            line = _json_line(raw)
            if line is None or line.get("type") != "assistant" or line.get("isSidechain"):
                continue
            moment = _timestamp(line.get("timestamp"))
            if moment is not None and moment >= sent_at:
                break
            found = _claude_call(line)
            if found:
                call = found[1]
                size = call.new_input + call.cache_write_5m + call.cache_write_1h + call.cache_read + call.output
    except OSError:
        return 0
    return size


def conversation_before(transcript: object, prompt: str, sent_at: float, until: str | None = None) -> str:
    """The end of the session's conversation before a prompt, built the way training builds a task's context."""
    return before_prompt(transcript, prompt, sent_at, until)[0]


def before_prompt(transcript: object, prompt: str, sent_at: float, until: str | None = None) -> tuple[str, list[str]]:
    """The session's conversation before a prompt, and the skills known by then, as training builds them for a task.

    The skills are those the prompt invokes, such as a typed /command, and those every line before it invoked,
    used or loaded, as training counts them. Both leave a helper's lines out: they're another agent's. With
    `until`, the id of the prompt's own line in the record, that's everything before the line. Without, only
    lines written before the prompt was sent count. When the record already holds the prompt
    itself, it's left out, as training leaves each task's own prompt out of what came before it. Each line
    counts once, as in training, though Claude Code sometimes writes the history out again. One pass reads both.
    """
    from .archives import _claude_line, _History, _lines, _timestamp, prompt_skills

    skills = prompt_skills(prompt)
    if not isinstance(transcript, str) or not transcript:
        return "", sorted(skills)
    history = _History()
    known: set[str] = set()
    latest = None  # the latest prompt
    before_latest, known_before_latest = "", set()  # the context and skills the latest prompt had
    answered = False  # whether the agent answered the latest prompt
    seen: set[str] = set()  # training counts each line once, as a record can repeat its lines
    try:
        for line in _lines(Path(transcript)):
            if until is not None and line.get("uuid") == until:
                return history.context(), sorted(known | skills)
            moment = _timestamp(line.get("timestamp"))
            if until is None and moment is not None and moment >= sent_at:
                break
            uuid = line.get("uuid")
            if isinstance(uuid, str):
                if uuid in seen:
                    continue
                seen.add(uuid)
            slim = _claude_line(line)
            if slim.start is not None:
                before_latest, known_before_latest, latest, answered = history.context(), set(known), slim.start, False
                if slim.origin is not None:
                    history.add("User", slim.start)
                known |= slim.skills
            elif latest is not None and not slim.sidechain:
                known |= slim.skills  # as in training, every line of a turn adds the skills it used or loaded
                if slim.kind == "assistant" and slim.call is not None:
                    answered = True
                    for words in slim.texts:
                        history.add("Assistant", words)
    except OSError:
        return "", sorted(skills)
    if latest is not None and latest.strip() == prompt.strip() and not answered:
        return before_latest, sorted(known_before_latest | skills)  # the record already holds this prompt
    return history.context(), sorted(known | skills)


# The status line ---------------------------------------------------------------------------------------


def status() -> int:
    """Print the status line. It prints one whatever Claude Code sends, and never fails."""
    line = _dim("yardstick")
    try:
        event = _event()
        event = event if isinstance(event, dict) else {}
        model, effort = _named(event.get("model"), "id"), _named(event.get("effort"), "level")
        from . import weekly

        week = weekly.note(event)
        line = status_line(_clean(event.get("session_id")), model, time.time(), effort, _record(event), week)
    except Exception:
        pass
    if old_command():
        line += " " + _dim("· update it: yardstick-predict claude-code install")
    _say(line)
    return 0


def old_command(argv: list[str] | None = None, safe: bool | None = None) -> bool:
    """Whether Claude Code runs a command an older version installed, which starts Python without -P.

    Such a command lets Python look in the session's folder first for the code it imports. Installing the
    commands again replaces it.
    """
    argv = list(getattr(sys, "orig_argv", [])) if argv is None else argv
    safe = bool(sys.flags.safe_path) if safe is None else safe
    return not safe and MARK in argv


def _say(text: str) -> None:
    """Print a line in UTF-8, which Claude Code reads, whatever this computer's own encoding is."""
    try:
        stream = getattr(sys.stdout, "buffer", None)
        if stream is None:
            print(text)
            return
        sys.stdout.flush()
        stream.write(text.encode("utf-8", "replace") + b"\n")
        stream.flush()
    except (OSError, ValueError):
        pass  # no one is reading


def _named(value: object, key: str) -> str:
    """The model's id or the effort level, sent as an object such as {"id": ...} or as plain text."""
    if isinstance(value, dict):
        value = value.get(key)
    return value if isinstance(value, str) else ""


def _record(event: dict) -> str | None:
    """The path of the session's record, when Claude Code sent one."""
    path = event.get("transcript_path")
    return path if isinstance(path, str) and path else None


def status_line(
    session: str, model: str, now: float, effort: str = "", transcript: str | None = None, week: dict | None = None
) -> str:
    """One short line: the forecast for the model you're on, and how long the task has taken so far.

    `week` is the weekly limit's reading, which Claude Code sends on a Claude subscription. With it, the line
    shows the task's share of the weekly limit and the forecast's share in place of dollars.

    Anything it can't make sense of, such as a damaged file, gives the plain label rather than an error.
    """
    try:
        return _status_line(session, model, now, effort, transcript, week)
    except Exception:
        return _dim("yardstick")


def _status_line(session: str, model: str, now: float, effort: str, transcript: str | None, week: dict | None) -> str:
    label = _dim("yardstick")
    if not session:
        return label
    state = _read(_state_path(session))
    started = _number(state.get("prompt_at"))
    if started is None:
        return label
    record = transcript or _str(state.get("transcript")) or None
    replied, interrupted, waited = _since(record, started, set(_others(state)))
    stopped = _number(state.get("done_at"))
    if stopped is not None and replied is not None and replied > stopped:
        stopped = None  # the agent went on after that stop, as when another hook kept it working
    done = stopped if stopped is not None else interrupted
    worked = _number(state.get("took"))
    if stopped is not None and worked is not None and state.get("took_for") == [started, stopped]:
        seconds = worked  # from the record, by the rules training uses
    else:
        seconds = (now if done is None else done) - started - waited
    took = _clock(seconds)
    spent = ""  # on a Claude subscription, the task's share of the weekly limit so far
    per_usd = None
    if week is not None:
        from . import weekly

        per_usd = weekly.rate()
        so_far = weekly.task_spent(session, state, record) if per_usd is not None else None
        if so_far is not None and per_usd is not None:
            spent = f" {_dim('·')} {weekly.share(so_far, per_usd)}"
    found = _read(_forecast_path(session, started))
    if found.get("prompt_at") != started:
        return f"{label} {_dim('·')} took {took}{spent}" if done is not None else f"{label} {_dim('· forecasting…')}"
    if found.get("error"):
        from . import ui

        return f"{label} {_dim('· no forecast: ' + ui.clean(found['error']))}"
    effort = effort or _str(state.get("effort"))
    name = model.split("[")[0] or _str(state.get("model"))
    pick = _pick(found.get("forecasts"), name, effort)
    expected = _number(pick.get("seconds")) if pick else None
    if expected is None:
        missing = _dim(f"· no forecast for {(name or 'this AI model').strip()} yet")
        return f"{label} {_dim('·')} took {took}{spent} {missing}" if done is not None else f"{label} {missing}"
    usd = _number(pick.get("usd")) if pick else None
    if week is not None:
        # On a subscription, the forecast in brackets is its time and its share of the weekly limit.
        part = f" {_dim('·')} ~{weekly.share(usd, per_usd)}" if usd is not None and per_usd is not None else ""
        forecast = f" {_dim('(')}~{_minutes(expected)}{part}{_dim(')')}"
        return f"{label} {_dim('·')} {'took ' if done is not None else ''}{took}{spent}{forecast}"
    cost = f" {_dim('·')} ~${usd:.2f}" if usd is not None else ""
    if done is not None:
        return f"{label} {_dim('·')} took {took} {_dim('· forecast')} ~{_minutes(expected)}{cost}"
    return f"{label} {_dim('·')} {took} {_dim('of')} ~{_minutes(expected)}{cost}"


def _since(transcript: str | None, since: float, others: set[str]) -> tuple[float | None, float | None, float]:
    """What the end of the session's record says about the main agent's task since `since`.

    When it last replied. When you stopped it with Esc, which sends no hook, if it didn't reply after.
    And how long it waited for you in waits that are over: a question, a plan to approve or a tool to
    allow, by training's rules. A helper's lines say nothing about the main agent, an error notice isn't a
    reply, and the turns of prompts that started no task, the ids in `others`, aren't the task's.
    """
    lines = _tail(transcript)
    if not lines:
        return None, None, 0.0
    from .archives import _claude_line, _covered, _person_wait, _timestamp

    replied = interrupted = latest = None
    ours = True  # whether the lines so far belong to the task's turns
    tools: dict[str, tuple[str, float]] = {}
    waits: list[tuple[float, float]] = []
    for raw in lines:
        if '"assistant"' not in raw and '"user"' not in raw:
            continue
        try:
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            moment = _timestamp(event.get("timestamp"))
            latest = latest if moment is None else moment  # a line with no time is as late as the one before
            if latest is None or latest < since or event.get("isSidechain"):
                continue
            if event.get("type") == "user" and isinstance(event.get("promptId"), str):
                ours = event["promptId"] not in others
            if not ours:
                continue
            if event.get("type") == "assistant" and _main_reply(event):
                replied, interrupted = latest, None
            elif event.get("type") == "user" and _text(event).lstrip().startswith("[Request interrupted"):
                interrupted = latest
            if '"tool_use"' in raw or '"tool_result"' in raw:
                slim = _claude_line(event)
                for tool_id, name in slim.tools if slim.moment is not None else ():
                    if tool_id:
                        tools[tool_id] = (name, slim.moment)
                wait = _person_wait(slim, tools) if slim.kind == "user" else None
                if wait is not None:
                    waits.append(wait)
        except Exception:
            continue  # a line it can't read says nothing
    return replied, interrupted, _covered(waits)


def interrupted_at(transcript: str | None, since: float) -> float | None:
    """When you stopped the agent with Esc, which sends no hook, from the end of the session's record."""
    return _since(transcript, since, set())[1]


def _text(event: dict) -> str:
    """What a person or the agent wrote in a line. A tool's output doesn't count, whatever it says."""
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    for part in content if isinstance(content, list) else []:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            return part["text"]
    return ""


def _pick(forecasts: object, model: str, effort: str) -> dict | None:
    forecasts = [f for f in forecasts if isinstance(f, dict)] if isinstance(forecasts, list) else []
    mine = [f for f in forecasts if f.get("model") == model]
    if not mine:  # Claude Code may name a model with a date after it, or without the date its records have
        mine = [f for f in forecasts if _same_model(str(f.get("model") or ""), model)]
    return next((f for f in mine if f.get("effort") == effort), mine[0] if mine else None)


def _same_model(one: str, other: str) -> bool:
    """Whether two names are one model, as claude-opus-4-5[1m] and claude-opus-4-5-20251101 are.

    A [..] suffix doesn't count, and neither does a date at the end when only one name has it. Two
    different dates are two versions of a model, so they don't match.
    """
    (name, date), (other_name, other_date) = _name_and_date(one), _name_and_date(other)
    return name == other_name and (not date or not other_date or date == other_date)


def _name_and_date(model: str) -> tuple[str, str]:
    name = model.split("[")[0].strip()
    found = re.search(r"-(\d{8})$", name)
    return (name[: found.start()], found.group(1)) if found else (name, "")


def _clock(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def _minutes(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _dim(text: str) -> str:
    return "" if not text else (text if os.environ.get("NO_COLOR") else f"\033[2m{text}\033[22m")


# Settings ----------------------------------------------------------------------------------------------


def settings_path(here: bool) -> Path:
    """Claude Code's settings for every session, or only for sessions in this folder."""
    return Path.cwd() / ".claude" / "settings.local.json" if here else Path.home() / ".claude" / "settings.json"


def commands(python: str, home: str | None = None) -> tuple[str, str]:
    """The hook's command and the status line's.

    -P keeps Python from loading code from the session's folder, where Claude Code runs them. With `home`,
    they use that yardstick folder, whatever Claude Code's own environment says.
    """
    run = f"{_quoted(python)} -P -m {MARK}" + (f" --home {_quoted(home)}" if home else "")
    return f"{run} hook", f"{run} status"


WINDOWS_SHELLS_READ = '$`"%!'
"""What cmd or Git Bash read inside double quotes, as more than text: a variable, a command, the end of the quotes."""


def unsafe_path(path: str, windows: bool | None = None) -> str | None:
    """Why a path can't go into Claude Code's commands, in plain words, or None when it can.

    On Windows, Claude Code runs them in cmd or in Git Bash, and no one way of quoting keeps every character as
    it is in both. Inside double quotes cmd still reads % and !, and Git Bash reads $ and `. A path with any of
    those, or with a double quote, could become another command, so it's refused. Elsewhere every path is fine.
    """
    if not (os.name == "nt" if windows is None else windows):
        return None
    found = [char for char in WINDOWS_SHELLS_READ if char in path]
    if not found:
        return None
    listed = found[0] if len(found) == 1 else ", ".join(found[:-1]) + " and " + found[-1]
    return (
        f"The path {path} has {listed} in it. Windows' command shells read that as more than text, so it can't go "
        "into Claude Code's commands. Install yardstick-predict in a folder without it."
    )


def _quoted(text: str, windows: bool | None = None) -> str:
    """A path as one word for the shell that runs Claude Code's commands.

    On Windows that may be cmd or Git Bash. Double quotes keep spaces in both. Forward slashes mean the
    same as backslashes there, and neither shell reads them as an escape, as bash does a backslash before
    a quote. Elsewhere it's a POSIX shell, and single quotes keep every character as it is.
    """
    if os.name == "nt" if windows is None else windows:
        return '"' + text.replace("\\", "/") + '"'
    import shlex

    return shlex.quote(text)


def added(settings: dict, python: str, home: str | None = None) -> dict:
    """The settings with the hook on each prompt and each stop, and the status line. Nothing else changes.

    Hooks in a shape Claude Code doesn't read, such as null or text, are left as they are, without ours.
    Commands this module added before, such as ones without -P, give way to the new ones.
    """
    hook_command, status_command = commands(python, home)
    result = json.loads(json.dumps(settings))
    if not isinstance(result, dict):
        return result
    hooks = result.setdefault("hooks", {})
    if isinstance(hooks, dict):
        for event in ("UserPromptSubmit", *STOPS):
            groups = hooks.setdefault(event, [])
            if isinstance(groups, list):
                hooks[event] = [*_without_ours(groups), {"hooks": [{"type": "command", "command": hook_command}]}]
    # Without a timer the line redraws only when the agent writes a message. Once a second, the clock ticks.
    result["statusLine"] = {"type": "command", "command": status_command, "padding": 0, "refreshInterval": 1}
    return result


def removed(settings: dict) -> dict:
    """The settings without anything this module added. Everything else stays as it is, whatever its shape."""
    result = json.loads(json.dumps(settings))
    if not isinstance(result, dict):
        return result
    hooks = result.get("hooks")
    if isinstance(hooks, dict):
        for event, groups in list(hooks.items()):
            if isinstance(groups, list) and (kept := _without_ours(groups)) != groups:
                if kept:
                    hooks[event] = kept
                else:
                    del hooks[event]  # it held only ours
        if not hooks and settings["hooks"]:
            del result["hooks"]  # it held only ours
    if _ours(result.get("statusLine")):
        del result["statusLine"]
    return result


def _without_ours(groups: list) -> list:
    """Hook groups without this module's commands. A group that held only ours goes too."""
    kept = []
    for group in groups:
        entries = group.get("hooks") if isinstance(group, dict) else None
        if not isinstance(entries, list) or not any(_ours(entry) for entry in entries):
            kept.append(group)
        elif rest := [entry for entry in entries if not _ours(entry)]:
            kept.append({**group, "hooks": rest})
    return kept


def _ours(entry: object) -> bool:
    return isinstance(entry, dict) and MARK in str(entry.get("command") or "")


def other_status_line(settings: dict) -> str | None:
    """The command of a status line someone else set up, which adding this one would replace."""
    line = settings.get("statusLine") if isinstance(settings, dict) else None
    command = line.get("command") if isinstance(line, dict) else line
    return command if isinstance(command, str) and command and MARK not in command else None


def installed_home(settings: object) -> str | None:
    """The yardstick folder that this module's commands in the settings name, if they name one."""
    import shlex

    for command in _our_commands(settings):
        try:
            words = shlex.split(command)
        except ValueError:
            continue
        if "--home" in words[:-1]:
            return words[words.index("--home") + 1]
    return None


def _our_commands(settings: object) -> list[str]:
    if not isinstance(settings, dict):
        return []
    found = [str(settings["statusLine"]["command"])] if _ours(settings.get("statusLine")) else []
    hooks = settings.get("hooks")
    for groups in hooks.values() if isinstance(hooks, dict) else []:
        for group in groups if isinstance(groups, list) else []:
            entries = group.get("hooks") if isinstance(group, dict) else None
            found += [str(entry["command"]) for entry in entries if _ours(entry)] if isinstance(entries, list) else []
    return found


def read_settings(path: Path) -> object:
    """Claude Code's settings in a file, or None when there's no file. Raises ValueError when it isn't JSON."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except RecursionError:
        raise ValueError("nested too deeply") from None


def in_use(path: Path) -> bool:
    """Whether the settings in a file still hold this module's commands."""
    try:
        settings = read_settings(path)
    except (OSError, ValueError):
        return False
    return isinstance(settings, dict) and removed(settings) != settings


def save_settings(path: Path, settings: dict) -> Path | None:
    """Save Claude Code's settings in one step, keeping the file's permissions. Returns where a copy of the
    file as it was is kept, or None when there was no file.

    A settings file that links to another file, as one in a dotfiles folder can, stays a link.
    """
    from .files import write_text

    target = path.resolve() if path.is_symlink() else path
    backup = None
    if target.is_file():
        backup = path.with_name(path.name + ".before-yardstick")
        write_text(backup, target.read_text(encoding="utf-8"))
    try:
        text = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
        text.encode("utf-8")
    except UnicodeEncodeError:  # half of an emoji, written as an escape in the file, stays an escape
        text = json.dumps(settings, indent=2) + "\n"
    write_text(target, text)
    return backup


def remember_status_line(settings: dict, path: Path, home: str | None) -> None:
    """Keep the status line that adding this one replaces, so removing it puts that one back."""
    line = settings.get("statusLine")
    if _ours(line):
        return  # ours is set already, so the one it replaced is kept from before
    kept = _read(_home(home) / "claude-code" / REPLACED)
    if line is None:
        kept.pop(str(path), None)
    else:
        kept[str(path)] = line
    _keep_replaced(kept, home)


def put_back_status_line(before: dict, after: dict, path: Path, home: str | None) -> tuple[dict, object]:
    """The settings with the status line that adding this one replaced, if ours was still the one set.

    Returns the settings, and the status line it put back, or None.
    """
    kept = _read(_home(home) / "claude-code" / REPLACED)
    if str(path) not in kept:
        return after, None
    line = kept.pop(str(path))
    _keep_replaced(kept, home)
    if not _ours(before.get("statusLine")) or "statusLine" in after:
        return after, None  # you set another status line since, and it stays
    return {**after, "statusLine": line}, line


def _keep_replaced(kept: dict, home: str | None) -> None:
    path = _home(home) / "claude-code" / REPLACED
    if kept:
        from .files import write_text

        write_text(path, json.dumps(kept, indent=2, ensure_ascii=False) + "\n")
        if os.name == "posix":
            os.chmod(_home(home), 0o700)
    else:
        path.unlink(missing_ok=True)


def clean(home: str | None = None) -> None:
    """Delete what the hook kept for each session: prompts, forecasts and clocks. Never fails."""
    root = _home(home) / "claude-code"
    try:
        paths = list(root.iterdir())
    except OSError:
        return
    for path in paths:
        if path.name != REPLACED:
            try:
                path.unlink()
            except OSError:
                pass
    try:
        root.rmdir()  # only once nothing is left in it
    except OSError:
        pass


def main(argv: list[str]) -> int:
    if argv[:1] == ["--home"] and len(argv) > 1:
        os.environ["YARDSTICK_HOME"] = argv[1]  # the folder named when the commands were added
        argv = argv[2:]
    if argv[:1] == ["hook"]:
        return hook()
    if argv[:1] == ["status"]:
        return status()
    if argv[:1] == ["work"] and len(argv) == 2:
        return work(argv[1])
    if argv[:1] == ["forecast"] and len(argv) == 3:
        return work(argv[1])  # what an earlier version's hook started
    print("Use: python -P -m yardstick_predict.claude_code [--home FOLDER] hook|status", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
