"""Forecast how long a coding-agent task will take and what it will cost, before it runs."""

import os
import sys

__version__ = "0.4.0"


def _keep_the_current_folder_out(argv: list[str] | None = None) -> None:
    """When Python runs one of this package's modules with -m, keep the folder it started in off the import path.

    Python's -m puts that folder first, so a file there, such as a project's own json.py, would be imported in
    place of Python's own. The commands this package installs start Python with -P, which keeps the folder
    out. Commands installed by older versions don't, so this takes the folder out before the package imports
    anything else. A file that Python itself loads before this package can still get in, so installing the
    commands again is the real fix.
    """
    if sys.flags.safe_path:
        return
    argv = list(getattr(sys, "orig_argv", [])) if argv is None else argv
    try:
        module = argv[argv.index("-m", 1) + 1]
    except (ValueError, IndexError):
        return
    if module.split(".")[0] != __name__:
        return
    try:
        here = os.getcwd()
    except OSError:
        here = None
    if sys.path and sys.path[0] in ("", here):
        del sys.path[0]


_keep_the_current_folder_out()
