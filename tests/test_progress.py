"""Progress bars: how long is left, and a bar for every long step of setup and refresh."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest
from builders import CodexSession, history
from test_model import ONE_WINDOW, rows_for, synthetic

from yardstick_predict import model as M
from yardstick_predict import pipeline, ui
from yardstick_predict.store import Predictor

START = 1_780_000_000.0
LATER = START + 90 * 86400


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def terminal(monkeypatch):
    """Output styled as in a terminal, with a fresh bar and a clock the test moves."""
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    clock = Clock()
    monkeypatch.setattr(ui, "_now", clock)
    monkeypatch.setattr(ui, "_bar", None)
    monkeypatch.setattr(ui, "_transient", False)
    return clock


def drawn(capsys) -> str:
    """The bar as it was drawn last, without its color codes."""
    out = capsys.readouterr().out
    return re.sub(r"\033\[[0-9;]*[mK]", "", out.split("\r")[-1]).strip()


def test_the_time_left_is_said_in_calm_words():
    assert ui.time_left(0) == ui.time_left(59) == "under a minute left"
    assert ui.time_left(60) == "about 1 min left"
    assert ui.time_left(185) == "about 3 min left"
    assert ui.time_left(89 * 60) == "about 89 min left"
    assert ui.time_left(90 * 60) == "about 1.5 h left"
    assert ui.time_left(4 * 3600) == "about 4 h left"


def test_a_bar_says_how_long_is_left_once_there_is_enough_to_go_on(capsys, terminal):
    ui.progress(0, 1000, "texts")
    assert drawn(capsys).endswith(" 0 / 1,000 texts")
    terminal.now += 10
    ui.progress(20, 1000, "texts")  # 2% done: too little to go on
    assert drawn(capsys).endswith(" 20 / 1,000 texts")
    terminal.now += 5
    ui.progress(30, 1000, "texts")  # 3% after 15 seconds, at 2 a second
    assert drawn(capsys).endswith(" 30 / 1,000 texts · about 8 min left")
    terminal.now += 1
    ui.progress(1000, 1000, "texts")
    assert drawn(capsys).endswith(" 1,000 / 1,000 texts")  # nothing left to say once it's done

    ui.progress(0, 50, "tasks")  # another step starts its own bar
    terminal.now += 4
    ui.progress(25, 50, "tasks")  # half done, but only 4 seconds in
    assert "left" not in drawn(capsys)


def test_one_quick_or_slow_batch_doesnt_make_the_time_left_jump(capsys, terminal):
    ui.progress(0, 1000, "texts")
    for done in range(20, 420, 20):  # 2 a second for 200 seconds
        terminal.now += 10
        ui.progress(done, 1000, "texts")
    assert drawn(capsys).endswith("about 5 min left")
    terminal.now += 0.5
    ui.progress(430, 1000, "texts")  # 30 more at once: at that pace, all would be done in 10 seconds
    assert drawn(capsys).endswith("about 4 min left")
    terminal.now += 30
    ui.progress(432, 1000, "texts")  # then half a minute for 2: at that pace, over two hours
    assert drawn(capsys).endswith("about 7 min left")


def test_a_bar_of_bytes_shows_sizes(capsys, terminal):
    ui.progress(0, 2_500_000_000, "bytes")
    assert drawn(capsys).endswith(" 0 KB / 2.5 GB")
    terminal.now += 1
    ui.progress(1_200_000_000, 2_500_000_000, "bytes")
    assert drawn(capsys).endswith(" 1.2 GB / 2.5 GB")
    assert [ui.size(n) for n in (0, 800_000, 3_400_000, 120_000_000)] == ["0 KB", "800 KB", "3.4 MB", "120 MB"]
    assert [ui.size(n) for n in (999_600, 9_960_000, 999_600_000)] == ["1.0 MB", "10 MB", "1.0 GB"]  # never 1000 KB


def test_reading_many_small_files_draws_the_bar_a_thousand_times_at_most():
    seen: list[int] = []
    read = pipeline._Bytes(100_000_000, lambda done, total: seen.append(done))
    read.reach(0)
    for _ in range(100_000):
        read.read(1000)
    assert len(seen) == 1001 and seen[0] == 0 and seen[-1] == 100_000_000


def test_every_batch_is_drawn_even_when_several_arrive_at_once(capsys, terminal):
    ui.progress(0, 32, "tasks")
    for done in (8, 16, 24):  # requests sent together finish together
        ui.progress(done, 32, "tasks")
    assert drawn(capsys).endswith(" 24 / 32 tasks")


def test_a_bar_shows_only_in_a_terminal(capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    ui.progress(0, 10, "texts")
    ui.progress(5, 10, "texts")
    assert capsys.readouterr().out == ""  # output to a file or a program stays as it was


def test_reading_records_reports_the_bytes_read(tmp_path):
    claude = tmp_path / "claude"
    history(claude, tasks=30, start=START)
    codex = tmp_path / "codex"
    for n in range(3):
        session = CodexSession(codex, f"thread-{n}", START + n * 3600)
        session.turn(f"turn-{n}", "fix the parser", [(100, 1000, 50)], 60)
        session.save(mtime=START + n * 3600)
    archives = [
        {"name": "claude-code", "format": "claude-code", "path": str(claude)},
        {"name": "codex", "format": "codex", "path": str(codex)},
    ]
    size = sum(path.stat().st_size for folder in (claude, codex) for path in folder.rglob("*.jsonl"))
    seen: list[tuple[int, int]] = []
    found = pipeline.read_all(archives, now=LATER, on_progress=lambda done, total: seen.append((done, total)))
    assert len(found.new_tasks) == 33
    assert seen[0] == (0, size) and seen[-1] == (size, size)
    assert len(seen) > 4 and all(a[0] <= b[0] for a, b in zip(seen, seen[1:]))  # file by file, never back

    predictor = Predictor.create("default", archives, {"model": "voyage-4", "dimensions": 256})
    pipeline.save_scan(predictor, found)
    seen.clear()
    pipeline.scan(predictor, now=LATER, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == []  # nothing changed, so nothing is read
    session = CodexSession(codex, "thread-new", START + 86400)
    session.turn("turn-new", "add a test", [(100, 1000, 50)], 60)
    session.save(mtime=START + 86400)
    found = pipeline.scan(predictor, now=LATER, on_progress=lambda done, total: seen.append((done, total)))
    assert len(found.new_tasks) == 1
    assert seen[0] == (0, session.path.stat().st_size) and seen[-1][0] == seen[-1][1]  # only the new file


def test_training_reports_each_model_it_fits(monkeypatch):
    tasks = synthetic(200)
    fitted, real_fit = [], M.fit
    monkeypatch.setattr(M, "fit", lambda *args: fitted.append(len(args[0])) or real_fit(*args))
    seen: list[tuple[int, int]] = []
    M.train(tasks, rows_for(tasks), ONE_WINDOW, on_progress=lambda done, total: seen.append((done, total)))
    total = len(M.fit_sizes(tasks, ONE_WINDOW))
    assert fitted == M.fit_sizes(tasks, ONE_WINDOW)  # each trial of the test, then the final model
    assert seen == [(done, total) for done in range(total + 1)]

    # With LLM analysis, each is fitted again without it.
    rows = [replace(r, numbers={"llm_minutes": float(np.log(t.seconds))}) for t, r in zip(tasks, rows_for(tasks))]
    seen.clear()
    M.train(tasks, rows, ONE_WINDOW, on_progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (0, 2 * total) and seen[-1] == (2 * total, 2 * total) and len(seen) == 2 * total + 1


def test_trials_shared_by_windows_are_counted_once():
    tasks = synthetic(400)
    sizes = M.fit_sizes(tasks, {})
    # Three windows for each of the two AI models. Their tasks take turns, so no two open with the same tasks.
    assert len(sizes) == 7 and sizes[-1] == 400 and sizes == sorted(sizes)
    same = [replace(t, model="claude-opus-5-5") for t in tasks]
    assert len(M.fit_sizes(same, {})) == 4  # one AI model: three windows, then the final model
    assert M.fit_sizes(tasks[:50], {}) == [50]  # too few tasks to test: only the final model


def test_a_terminal_without_fancy_characters_gets_plain_ones(terminal, monkeypatch):
    import io

    shown = io.TextIOWrapper(io.BytesIO(), encoding="ascii", errors="strict")
    monkeypatch.setattr("sys.stdout", shown)
    ui.progress(0, 1000, "texts")
    terminal.now += 15
    ui.progress(30, 1000, "texts")  # once the time left shows
    ui.out(ui.join(["one", "two"]))
    shown.flush()
    text = re.sub(r"\033\[[0-9;]*[mK]", "", shown.buffer.getvalue().decode("ascii"))
    assert "30 / 1,000 texts - about 8 min left" in text and "one - two" in text


@pytest.mark.parametrize("width", [45, 80])
def test_a_bar_and_a_working_line_fit_the_terminal_and_keep_what_matters(capsys, terminal, monkeypatch, width):
    monkeypatch.setenv("COLUMNS", str(width))
    ui.working("Comparing forecasts with every feature, then without each one")
    shown = drawn(capsys)
    assert ui.width(shown) <= width - 3 and shown.startswith("… Comparing forecasts")
    assert shown.endswith("without each one" if width == 80 else "…")  # cut short only where it wouldn't fit
    ui.progress(0, 1000, "tasks")
    terminal.now += 15
    ui.progress(30, 1000, "tasks")  # once the time left shows
    shown = drawn(capsys)
    assert ui.width(shown) <= width - 3 and shown.startswith("Comparing") and "30 / 1,000 tasks" in shown
    # At 80 columns there's room for the time left too. At 45, it makes way for the step's words.
    assert ("about 8 min left" in shown) == (width == 80)
    ui.out("done")
    ui.progress(2_000_000, 6_200_000_000, "bytes")  # a bar with nothing in front grows to its full length
    assert ui.width(drawn(capsys)) <= width - 3


def test_a_bar_in_a_very_narrow_terminal_keeps_the_counts(capsys, terminal, monkeypatch):
    monkeypatch.setenv("COLUMNS", "20")
    ui.working("Reading your records")
    capsys.readouterr()
    ui.progress(5, 1000, "tasks")
    assert drawn(capsys) == "5 / 1,000 tasks"


def test_wide_characters_count_twice_and_are_cut_by_columns():
    assert ui.width("修复登录页面") == 12 and ui.width("café") == 4 and ui.width("café") == 4
    assert ui.cut("修复登录页面的错误", 9) == "修复登录…" and ui.width(ui.cut("修复登录页面的错误", 9)) == 9
    assert ui.cut("a long task", 6) == "a lon…" and ui.cut("fits", 4) == "fits"


def test_a_cut_fits_where_the_mark_is_three_dots(monkeypatch):
    import io

    monkeypatch.setattr("sys.stdout", io.TextIOWrapper(io.BytesIO(), encoding="ascii"))
    assert ui.cut("abcdefghij", 6) == "abc..." and ui.cut("abcdefghij", 2) == "ab"
