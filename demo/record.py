"""Record the demo clips in the README: the real yardstick-predict forecasting a task from a made-up history.

It writes the history into a home folder made for the demo, starts the stand-in for Voyage that the tests use and
trains a model with `yardstick-predict create --express`. Then VHS types the keys in each tape and films the
terminal. Nothing reads your own records, your keys or your password store. Nothing is sent over the network.

You need VHS, a program that films a terminal: https://github.com/charmbracelet/vhs. Run this from the
repository's folder:

    uv run python demo/record.py

It writes a GIF for the README and an MP4 for posts of each clip: demo/demo for the home screen and demo/cli for
the command line.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tests"))

import history  # noqa: E402  (demo/history.py, next to this file)
from conftest import KEY, FakeVoyage  # noqa: E402

TAPES = ("demo.tape", "cli.tape")


def environment(folder: Path, voyage: str) -> dict[str, str]:
    """What the demo's programs see: the made-up home, the stand-in for Voyage and a key that's never real.

    It starts empty. So no key or setting of your own gets in. The update check and the password store are off.
    Voyage's tokenizer, which counts tokens, isn't downloaded either. Setup's counts are estimates.
    """
    programs = Path(sys.executable).parent  # where uv put this checkout's yardstick-predict
    return {
        "PATH": os.pathsep.join([str(programs), os.environ.get("PATH", "")]),
        "HOME": str(folder / "home"),
        "USERPROFILE": str(folder / "home"),
        "YARDSTICK_HOME": str(folder / "yardstick"),
        "TMPDIR": tempfile.gettempdir(),
        "LANG": "en_US.UTF-8",
        "VOYAGE_API_KEY": KEY,
        "VOYAGE_API_BASE": voyage,
        "YARDSTICK_NO_UPDATE_CHECK": "1",
        "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
        "HF_HUB_OFFLINE": "1",
    }


@contextmanager
def stand_in_for_voyage() -> Iterator[str]:
    """The stand-in for Voyage, running on this computer until the block ends. Gives its address."""
    voyage = FakeVoyage()
    threading.Thread(target=voyage.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        yield voyage.base
    finally:
        voyage.shutdown()


def made_up_model(folder: Path, voyage: str) -> dict[str, str]:
    """Write the made-up history into `folder` and train a model on it. Returns the environment that uses them."""
    env = environment(folder, voyage)
    history.write(Path(env["HOME"]))
    made = subprocess.run(["yardstick-predict", "create", "--express"], env=env, capture_output=True, text=True)
    if made.returncode:
        sys.exit(f"Training failed.\n{made.stdout}\n{made.stderr}")
    return env


def main() -> int:
    vhs = shutil.which("vhs")
    if vhs is None:
        print("VHS isn't installed. It's at https://github.com/charmbracelet/vhs", file=sys.stderr)
        return 1
    with stand_in_for_voyage() as voyage, tempfile.TemporaryDirectory(prefix="yardstick-demo-") as folder:
        print("Training a model on the made-up history...")
        env = made_up_model(Path(folder), voyage)
        for tape in TAPES:
            print(f"Filming {tape}...")
            subprocess.run([vhs, str(HERE / tape)], env=env, cwd=HERE.parent, check=True)
    print("Wrote a GIF and an MP4 of each clip in demo/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
