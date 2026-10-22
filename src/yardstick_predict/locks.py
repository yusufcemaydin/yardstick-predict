"""File locks that keep two yardstick processes from changing the same thing at once.

They use only the standard library: fcntl on macOS and Linux, msvcrt on Windows. Closing the file lets a
lock go, so a process that crashes never leaves one behind.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def held(path: Path, wait: bool = True) -> Iterator[bool]:
    """Hold the lock at `path` while the block runs, and yield True.

    With `wait`, it waits its turn. Without, it yields False at once when another process holds the lock.
    """
    lock = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(lock, msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if not wait:
                        yield False
                        return
                    if error.errno != errno.EDEADLOCK:  # Windows stops trying after ten seconds. Keep waiting.
                        raise
            try:
                yield True
            finally:
                msvcrt.locking(lock, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError:
                yield False
                return
            yield True
    finally:
        os.close(lock)
