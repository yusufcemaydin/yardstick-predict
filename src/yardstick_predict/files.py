"""Saving a file so that no one ever sees half of it.

A save writes the new text to a file beside the old one, then swaps it in with one rename. A process
that stops halfway, or another process reading in the middle of a save, finds the whole old file or the
whole new one, never an empty or cut-off file.

It uses only the standard library, so the status line can use it and stay quick.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def write_text(path: Path | str, text: str) -> None:
    """Replace a file's text in one step. A file that exists keeps its permissions. A new one is yours alone."""
    _write(path, text, "w")


def write_bytes(path: Path | str, data: bytes) -> None:
    """Replace a file's contents in one step, as write_text does for text."""
    _write(path, data, "wb")


def _write(path: Path | str, contents: str | bytes, how: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o600
    handle, partial = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(handle, how, encoding="utf-8" if how == "w" else None) as file:
            file.write(contents)
            file.flush()
            os.fsync(file.fileno())
        if os.name == "posix":
            os.chmod(partial, mode)
        _replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


def _replace(source: str, target: Path) -> None:
    """Rename over the target. Windows refuses while another process has the target open, so it tries again."""
    for attempt in range(100):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 99:
                raise
            time.sleep(0.02)
