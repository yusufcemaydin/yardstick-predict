"""Show forecasts in Claude Code's status line.

When you send a prompt, a hook starts a forecast in the background and returns at once, so the prompt
isn't held up. Claude Code's status line then shows the forecast for the model you're on, and counts
up while the agent works. When the agent stops, it shows how long the task really took.

Claude Code runs these itself:

    python -m yardstick_predict.claude_code hook       on each prompt, and each time the agent stops
    python -m yardstick_predict.claude_code status     to fill in the status line

They load only what they need, so the status line stays quick. `yardstick-predict claude-code`
adds them to Claude Code's settings, and removes them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MARK = "yardstick_predict.claude_code"
"""Every command this module adds to Claude Code's settings contains this, so it can find them again."""
KEEP_DAYS = 7
TAIL_BYTES = 256_000
STOPS = ("Stop", "StopFailure")
"""Claude Code sends Stop when the agent finishes, and StopFailure when a request to the model fails."""


def folder() -> Path:
    """Where the forecasts for each Claude Code session are kept, inside the private yardstick folder."""
    home = Path(os.environ.get("YARDSTICK_HOME") or "~/.yardstick").expanduser()
    found = home / "claude-code"
    found.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(home, 0o700)
    return found


def _paths(session: str) -> tuple[Path, Path]:
    safe = "".join(c for c in session if c.isalnum() or c in "-_")[:80] or "session"
    base = folder() / safe
    return base.with_suffix(".json"), base.with_suffix(".forecast.json")


def _prompt_path(session: str, prompt_at: float) -> Path:
    """Where the hook leaves a prompt for its forecast. Each prompt has its own file, named by when it was sent."""
    state_path, _ = _paths(session)
    return state_path.with_name(f"{state_path.stem}.{prompt_at!r}.prompt")


def _read(path: Path) -> dict:
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _write(path: Path, data: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(data))
    os.replace(partial, path)


# The hook ----------------------------------------------------------------------------------------------


def hook() -> int:
    """Start a forecast when a prompt is sent, and note the time when the agent stops. Prints nothing."""
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    session = str(event.get("session_id") or "")
    if not session:
        return 0
    state_path, _ = _paths(session)
    if event.get("hook_event_name") == "UserPromptSubmit":
        prompt = str(event.get("prompt") or "")
        if not prompt.strip():
            return 0
        model, effort = last_model_and_effort(event.get("transcript_path"))
        prompt_at = time.time()
        _write(state_path, {"prompt_at": prompt_at, "model": model, "effort": effort})
        _prompt_path(session, prompt_at).write_text(prompt)
        _start_forecast(session, prompt_at)
        _tidy()
    elif event.get("hook_event_name") in STOPS:
        state = _read(state_path)
        if state.get("prompt_at") and not state.get("done_at"):
            state["done_at"] = time.time()
            _write(state_path, state)
    return 0


def _tail(transcript: str | None, size: int = TAIL_BYTES) -> list[str]:
    """The last lines of a session's record."""
    if not transcript:
        return []
    try:
        with open(transcript, "rb") as file:
            file.seek(0, os.SEEK_END)
            file.seek(max(file.tell() - size, 0))
            return file.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def last_model_and_effort(transcript: str | None) -> tuple[str, str]:
    """The model and effort level of the session's latest reply, from the end of its record."""
    for line in reversed(_tail(transcript)):
        if '"assistant"' not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "assistant":
            return str((event.get("message") or {}).get("model") or ""), str(event.get("effort") or "")
    return "", ""


def _start_forecast(session: str, prompt_at: float) -> None:
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
    command = [sys.executable, "-m", MARK, "forecast", session, repr(prompt_at)]
    subprocess.Popen(command, env=env, close_fds=True, **options)


def _bundled_openmp() -> str | None:
    """On a Mac, the folder of scikit-learn's OpenMP, which XGBoost needs. Found without loading scikit-learn."""
    if sys.platform != "darwin":
        return None
    spec = importlib.util.find_spec("sklearn")
    if spec is None or not spec.origin:
        return None
    found = Path(spec.origin).parent / ".dylibs"
    return str(found) if (found / "libomp.dylib").is_file() else None


def _tidy() -> None:
    """Forget sessions untouched for a week."""
    cutoff = time.time() - KEEP_DAYS * 86400
    for path in folder().iterdir():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


# The forecast, in the background -----------------------------------------------------------------------


def forecast(session: str, prompt_at: float) -> int:
    """Forecast the prompt sent at prompt_at on every AI model and effort level the predictor covers, and save it.

    A newer prompt may come while this runs. Its forecast is the one to show, so this one is saved only
    if its prompt is still the session's latest when it's done.
    """
    state_path, forecast_path = _paths(session)
    prompt_path = _prompt_path(session, prompt_at)
    try:
        prompt = prompt_path.read_text()
    except OSError:
        return 0
    prompt_path.unlink(missing_ok=True)
    found: dict = {"prompt_at": prompt_at}
    try:
        from . import cli
        from .forecast import answer, parse

        predictor = cli.request_predictor(None)
        pairs = cli.effort_pairs(predictor, cli.forecast_models(predictor), None)
        tasks = [
            {"id": f"t{number}", "prompt": prompt, "models": [model], **({"effort": effort} if effort else {})}
            for number, (model, effort) in enumerate(pairs)
        ]
        parsed, _, _ = parse({"version": 1, "tasks": tasks})
        found.update(predictor=predictor.name, forecasts=answer(predictor, parsed)["forecasts"])
    except Exception as error:
        found["error"] = (getattr(error, "message", None) or str(error) or type(error).__name__)[:160]
    if _read(state_path).get("prompt_at") == prompt_at:
        _write(forecast_path, found)
    return 0


# The status line ---------------------------------------------------------------------------------------


def status() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        event = {}
    model = str((event.get("model") or {}).get("id") or "")
    effort = str((event.get("effort") or {}).get("level") or "")
    session = str(event.get("session_id") or "")
    print(status_line(session, model, time.time(), effort, event.get("transcript_path")))
    return 0


def status_line(session: str, model: str, now: float, effort: str = "", transcript: str | None = None) -> str:
    """One short line: the forecast for the model you're on, and how long the task has taken so far."""
    label = _dim("yardstick")
    if not session:
        return label
    state_path, forecast_path = _paths(session)
    state = _read(state_path)
    started = state.get("prompt_at")
    if not started:
        return label
    done = state.get("done_at") or interrupted_at(transcript, started)
    took = _clock((done or now) - started)
    found = _read(forecast_path)
    if found.get("prompt_at") != started:
        return f"{label} {_dim('·')} took {took}" if done else f"{label} {_dim('· forecasting…')}"
    if found.get("error"):
        return f"{label} {_dim('· no forecast: ' + found['error'])}"
    effort = effort or state.get("effort", "")
    pick = _pick(found.get("forecasts") or [], model.split("[")[0] or state.get("model", ""), effort)
    if pick is None or pick.get("seconds") is None:
        name = (model.split("[")[0] or state.get("model") or "this model").strip()
        missing = _dim(f"· no forecast for {name} yet")
        return f"{label} {_dim('·')} took {took} {missing}" if done else f"{label} {missing}"
    expected = f"~{_minutes(pick['seconds'])}"
    cost = f" {_dim('·')} ~${pick['usd']:.2f}" if pick.get("usd") is not None else ""
    if done:
        return f"{label} {_dim('·')} took {took} {_dim('· forecast')} {expected}{cost}"
    return f"{label} {_dim('·')} {took} {_dim('of')} {expected}{cost}"


def interrupted_at(transcript: str | None, since: float) -> float | None:
    """When you stopped the agent with Esc, which sends no hook, from the end of the session's record."""
    for line in reversed(_tail(transcript, 64_000)):
        if '"assistant"' not in line and "[Request interrupted" not in line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "assistant":
            return None  # the agent replied after any stop
        if event.get("type") == "user" and _text(event).lstrip().startswith("[Request interrupted"):
            try:
                stamp = datetime.fromisoformat(str(event.get("timestamp") or "").replace("Z", "+00:00"))
            except ValueError:
                return None
            return stamp.timestamp() if stamp.timestamp() >= since else None
    return None


def _text(event: dict) -> str:
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    for part in content or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
        if isinstance(part, dict) and isinstance(part.get("content"), str):
            return part["content"]
    return ""


def _pick(forecasts: list[dict], model: str, effort: str) -> dict | None:
    mine = [f for f in forecasts if f.get("model") == model]
    if not mine:  # Claude Code may name a model with a date after it
        mine = [f for f in forecasts if model.startswith(str(f.get("model") or "~"))]
    return next((f for f in mine if f.get("effort") == effort), mine[0] if mine else None)


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


def commands(python: str) -> tuple[str, str]:
    run = f'"{python}" -m {MARK}'
    return f"{run} hook", f"{run} status"


def added(settings: dict, python: str) -> dict:
    """The settings with the hook on each prompt and each stop, and the status line. Nothing else changes."""
    hook_command, status_command = commands(python)
    result = json.loads(json.dumps(settings))
    hooks = result.setdefault("hooks", {})
    for event in ("UserPromptSubmit", *STOPS):
        groups = [g for g in hooks.get(event, []) if not _ours(g)]
        groups.append({"hooks": [{"type": "command", "command": hook_command}]})
        hooks[event] = groups
    # Without a timer the line redraws only when the agent writes a message. Once a second, the clock ticks.
    result["statusLine"] = {"type": "command", "command": status_command, "padding": 0, "refreshInterval": 1}
    return result


def removed(settings: dict) -> dict:
    """The settings without anything this module added."""
    result = json.loads(json.dumps(settings))
    hooks = result.get("hooks") or {}
    for event in list(hooks):
        kept = [g for g in hooks[event] if not _ours(g)]
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    if "hooks" in result and not result["hooks"]:
        del result["hooks"]
    if MARK in str((result.get("statusLine") or {}).get("command", "")):
        del result["statusLine"]
    return result


def _ours(group: dict) -> bool:
    return any(MARK in str(entry.get("command", "")) for entry in group.get("hooks", []))


def other_status_line(settings: dict) -> str | None:
    """The command of a status line someone else set up, which adding this one would replace."""
    command = str((settings.get("statusLine") or {}).get("command", ""))
    return command if command and MARK not in command else None


def main(argv: list[str]) -> int:
    if argv[:1] == ["hook"]:
        return hook()
    if argv[:1] == ["status"]:
        return status()
    if argv[:1] == ["forecast"] and len(argv) == 3:
        return forecast(argv[1], float(argv[2]))
    print("Use: python -m yardstick_predict.claude_code hook|status", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
