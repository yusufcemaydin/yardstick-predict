"""Whether a newer yardstick-predict is out, checked on PyPI at most once a day.

The home screen asks. What PyPI said last is kept in the yardstick folder, so the screen shows it at once. A
new look at PyPI runs in the background once the last one is a day old, so the screen never waits for it,
and what it finds shows from the next screen on. It sends PyPI nothing but the request for the package's
public page. Set YARDSTICK_NO_UPDATE_CHECK to turn it off.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time

from .. import __version__

URL = "https://pypi.org/pypi/yardstick-predict/json"
CHECK_EVERY = 24 * 3600
"""Seconds between two looks at PyPI."""
OFF = "YARDSTICK_NO_UPDATE_CHECK"
UPGRADE = "uv tool upgrade yardstick-predict"

_looking = False
"""Whether this session has started a look at PyPI already."""


def notice() -> str | None:
    """The line saying a newer version is out, from what PyPI said last, or None. It starts a new look at PyPI in
    the background when the last one is a day old."""
    global _looking
    if os.environ.get(OFF):
        return None
    saved = _saved()
    if not _looking and time.time() - _number(saved.get("at")) >= CHECK_EVERY:
        _looking = True
        threading.Thread(target=check, daemon=True).start()
    latest = saved.get("latest")
    if isinstance(latest, str) and newer(latest, __version__):
        return f"yardstick-predict {latest} is out. You have {__version__}. Update with: {UPGRADE}"
    return None


def check() -> None:
    """Ask PyPI for the newest release and keep the answer. With no answer, what was known stays. The next look
    waits a day either way."""
    from .. import files, store

    saved = _saved()
    try:
        latest = latest_release()
    except Exception:  # offline, PyPI busy, or an answer it can't read: try again in a day
        latest = saved.get("latest")
    try:
        store.private_home()
        files.write_text(_path(), json.dumps({"at": time.time(), "latest": latest}) + "\n")
    except OSError:
        pass


def latest_release() -> str:
    """The newest release on PyPI, such as 0.5.0. PyPI names the newest one that isn't a test release."""
    import urllib.request

    request = urllib.request.Request(URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as answer:
        found = json.loads(answer.read().decode())["info"]["version"]
    if not isinstance(found, str) or not found.strip():
        raise ValueError("PyPI gave no version")
    return found.strip()


def newer(latest: str, current: str) -> bool:
    """Whether a release is newer than this version, by its numbers: 0.10.0 is newer than 0.9.1."""
    return _numbers(latest) > _numbers(current)


def _numbers(version: str) -> tuple[int, ...]:
    found = re.match(r"\d+(\.\d+)*", version.strip())
    return tuple(int(part) for part in found.group(0).split(".")) if found else ()


def _path():
    from .. import store

    return store.home() / "update-check.json"


def _saved() -> dict:
    try:
        found = json.loads(_path().read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _number(value) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
