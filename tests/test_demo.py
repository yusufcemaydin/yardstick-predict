"""The clips in the README run the real program on made-up past runs. demo/README.md says how they're made."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from yardstick_predict.archives import DEFAULT_ARCHIVES
from yardstick_predict.pipeline import learnable, read_all

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

import history  # noqa: E402  (demo/history.py)

START = 1_780_000_000.0
CLIP = re.compile(r"!\[[^\]]+\]\(https://raw\.githubusercontent\.com/yusufcemaydin/yardstick-predict/main/([^)]+)\)")
"""A clip in the README. Its address is in full. So it shows on the PyPI page too."""
CLIPS = {"demo/demo.gif", "demo/cli.gif", "demo/claude-code.gif"}


def test_the_readme_shows_each_clip_and_says_its_numbers_are_an_example():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    clips = list(CLIP.finditer(readme))
    assert sorted(clip.group(1) for clip in clips) == sorted(CLIPS)
    for clip in clips:
        assert (ROOT / clip.group(1)).stat().st_size < 3_000_000  # small enough to load quickly in a page
        said = readme[clip.end() :].strip().split("\n\n")[0]
        assert "made-up past runs" in said and "an example" in said, clip.group(1)


def test_the_clips_made_up_runs_have_enough_tasks_on_two_ai_models(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    history.write(tmp_path, now=START)
    tasks = learnable(read_all([dict(archive) for archive in DEFAULT_ARCHIVES], now=START).new_tasks)
    counts = Counter((task.model, task.effort) for task in tasks)
    assert counts == {("claude-opus-5-5", "max"): 260, ("gpt-6.1-sol", "xhigh"): 200}
    assert history.DEMO_TASK.lower() not in {task.prompt.lower() for task in tasks}  # the clips forecast a new task
