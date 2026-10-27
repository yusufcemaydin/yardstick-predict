"""Your predictors: the list, each feature's coverage in status, the prepared dataset, and deleting a predictor or a
stored feature."""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from builders import ClaudeSession, flat, history
from conftest import KEY
from test_cli import ONE_WINDOW, QuickModel, fake_analysis, quick, quick_predictor, quick_train  # noqa: F401
from test_feature_store import old_predictor
from test_setup_problems import no_xgboost  # noqa: F401 (a fixture: XGBoost can't load)

from yardstick_predict import claude_code as cc
from yardstick_predict import cli as command
from yardstick_predict import embeddings, feature_store, features, llm, locks, model_types, pipeline, store
from yardstick_predict import forecast as forecasting
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor, default_name, predictor_names

START = 1_780_000_000.0
EMBEDDING = {"model": "voyage-4", "dimensions": 256}
CONTEXT = "embedding.context.voyage-4.256"


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    """Plain text, as in a file: no colors and no marks."""
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture
def quickly(tmp_path, voyage, monkeypatch) -> list[str]:
    """The flags for a quick setup: 40 Claude Code tasks on one AI model, and a model type that trains at once.
    A Voyage key and the stand-in Voyage server come with it."""
    records = tmp_path / "records"
    history(records, tasks=40, start=START)
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    load = lambda folder: QuickModel(json.loads((folder / "model.json").read_text()))  # noqa: E731
    monkeypatch.setitem(model_types._TYPES, "quick", (quick_train, load))
    return ["--model-type", "quick", "--archive", f"claude-code={records}"]


def run(capsys, *args: str) -> tuple[int, str]:
    code = main(list(args))
    return code, capsys.readouterr().out


def answering(monkeypatch, *replies: str) -> list[str]:
    """A person at a terminal, who gives these replies in turn. Returns every question asked."""
    asked, waiting = [], list(replies)

    def answer(question: str = "") -> str:
        asked.append(question.strip())
        return waiting.pop(0)

    monkeypatch.setattr(command, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", answer)
    return asked


def everything(*folders: Path) -> dict[str, bytes]:
    """Every file under the folders, with what it holds, to see what changed."""
    return {str(path): path.read_bytes() for root in folders for path in sorted(root.rglob("*")) if path.is_file()}


def records_of(flags: list[str]) -> Path:
    """The folder of records a quick setup reads, from its flags."""
    return Path(next(flag for flag in flags if flag.startswith("claude-code=")).split("=", 1)[1])


# The list -------------------------------------------------------------------------------------------------


def test_list_shows_one_line_for_each_predictor_and_agrees_with_status(capsys, monkeypatch, quickly):
    assert main(["create", "--yes", *quickly, "--features", "context,time"]) == 0
    Predictor.create("work", [], EMBEDDING, features=["prompt"])  # not trained yet
    broken = store.predictor_folder("broken")
    broken.mkdir()
    (broken / "predictor.json").write_text("{")
    capsys.readouterr()

    monkeypatch.setenv("COLUMNS", "400")  # room for the table, and for the paths of the test's own folders
    code, out = run(capsys, "list")
    assert code == 1  # a predictor it shows can't be used
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert re.search(r"features +learned from +trained +forecasts for", out)
    assert re.search(r"main \(default\) +Context, Time +40 tasks +just now +claude-opus-5-5$", lines["main"])
    assert re.search(r"work +only the ones always on +- +not yet +none yet$", lines["work"])
    assert "can't be used, see below" in lines["broken"]
    assert "Model 'broken' can't be used" in out and "yardstick-predict delete broken" in out

    # Where the table won't fit, each predictor gets a few lines of its own.
    for width in (80, 45):
        monkeypatch.setenv("COLUMNS", str(width))
        code, out = run(capsys, "list")
        shown = out[: out.index("Model 'broken'")].splitlines()
        assert max(len(line) for line in shown) < width and "learned from" not in shown[0]
        assert "  main (default)" in shown and "  work" in shown and "  broken" in shown
        assert "Context, Time, learned from 40 tasks, trained just now" in flat(out)
        assert "    forecasts for claude-opus-5-5" in shown and "    forecasts for none yet" in shown
        assert "    only the ones always on, not trained yet" in shown
    monkeypatch.setenv("COLUMNS", "80")

    code, out = run(capsys, "list", "--json")
    listed = {entry["name"]: entry for entry in json.loads(out)["predictors"]}
    assert listed["main"]["features"] == ["prompt", "context", "time"] and listed["main"]["default"]
    assert listed["main"]["learned_from"] == 40 and listed["main"]["forecasts_for"] == ["claude-opus-5-5"]
    assert listed["work"] == {
        "name": "work",
        "default": False,
        "features": ["prompt"],
        "learned_from": None,
        "trained_at": None,
        "forecasts_for": [],
    }
    assert "error" in listed["broken"]
    # Status shows the same, in detail.
    code, out = run(capsys, "status", "--json")
    shown = {entry["name"]: entry for entry in json.loads(out)["predictors"]}
    for name in ("main", "work"):
        for fact in ("learned_from", "trained_at", "forecasts_for"):
            assert shown[name][fact] == listed[name][fact], (name, fact)
    code, out = run(capsys, "status", "--predictor", "main")
    assert "Trained on 40 tasks." in out and re.search(r"claude-opus-5-5 +forecasts on", out)


@pytest.mark.usefixtures("quick")
def test_list_says_when_theres_nothing_to_list_and_what_needs_training_again(capsys):
    code, out = run(capsys, "list")
    assert code == 0 and "There's no model yet." in out
    assert json.loads(run(capsys, "list", "--json")[1]) == {"version": 1, "predictors": []}
    predictor = quick_predictor()
    predictor.config["model"] = {}  # the built-in model, so its format counts
    predictor.save_config()
    path = predictor.model_dir / "model.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "format": 1}))  # made long before the last release
    code, out = run(capsys, "list")
    assert "none until it's trained again" in out
    assert json.loads(run(capsys, "list", "--json")[1])["predictors"][0]["forecasts_for"] == []


# Coverage in status ----------------------------------------------------------------------------------------


def test_status_shows_the_coverage_the_features_listing_counts(capsys, quickly):
    assert main(["create", "--yes", *quickly, "--features", "context,time"]) == 0
    capsys.readouterr()
    code, out = run(capsys, "features", "--json")
    listed = json.loads(out)["predictors"][0]
    shares = {use["title"]: use["coverage"] for use in listed["features"]}
    assert shares["Time"] == 1.0 and 0 < shares["Context"] < 1

    code, out = run(capsys, "status")
    assert code == 0
    # The line is broken to fit the terminal, so its parts are read back together.
    lines = out.splitlines()
    start = next(number for number, line in enumerate(lines) if line.strip().startswith("Coverage"))
    end = next(
        number
        for number in range(start + 1, len(lines))
        if not lines[number].startswith("  ") or lines[number].strip().startswith(("Features", "Trained", "Its "))
    )
    line = " ".join(part.strip() for part in lines[start:end])
    expected = ", ".join(f"{title} {share:.0%}" for title, share in shares.items())
    assert line == f"Coverage of the 40 tasks it can learn from: {expected}"

    code, out = run(capsys, "status", "--json")
    found = json.loads(out)
    entry = found["predictors"][0]
    assert entry["coverage"] == {"tasks": 40, "features": listed["features"]} and found["notes"] == []


@pytest.mark.usefixtures("quick")
def test_status_says_why_it_cant_count_coverage_and_shows_the_rest(capsys):
    quick_predictor(plugins=["a_plugin_that_isnt_installed"])
    code, out = run(capsys, "status")
    assert code == 0 and "Trained on 40 tasks." in out
    assert "Its coverage can't be counted until its plugin loads." in out and "Coverage of" not in out
    assert json.loads(run(capsys, "status", "--json")[1])["predictors"][0]["coverage"] is None

    old = old_predictor("older")  # an older version made it, so its vectors move into the feature store first
    with locks.held(old.dir / "refresh.lock"):
        code, out = run(capsys, "status", "--predictor", "older")
    assert "while another command sets it up or refreshes it" in flat(out)
    assert not Predictor("older").config.get("feature_store")  # nothing moved while it was busy


def test_a_feature_that_fails_never_breaks_status(capsys, monkeypatch):
    from test_embeddings import _task

    monkeypatch.setitem(features._CUSTOM, "broken", ("number", lambda info: 1 / 0))  # a plugin's, with a bug
    predictor = Predictor.create("default", [], EMBEDDING, features=["prompt", "broken"])
    predictor.add_tasks([_task(n, f"task {n}") for n in range(2)])
    code, out = run(capsys, "status")
    assert code == 0 and "Its coverage can't be counted: ZeroDivisionError: division by zero" in out


@pytest.mark.usefixtures("no_xgboost")
def test_status_without_xgboost_shows_everything_but_coverage(capsys):
    import sys

    Predictor.create("default", [], EMBEDDING)
    code, out = run(capsys, "status")
    assert code == 0 and "Coverage isn't shown: it's counted from what training reads" in out
    assert "yardstick_predict.model" not in sys.modules
    assert json.loads(run(capsys, "status", "--json")[1])["predictors"][0]["coverage"] is None


# The prepared dataset --------------------------------------------------------------------------------------


@pytest.fixture
def ready(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    return voyage


def test_each_training_saves_what_went_into_it_without_any_text(capsys, tmp_path, ready):
    records = tmp_path / "records"
    history(records, tasks=40, start=START)
    assert main(["create", "--yes", "--archive", f"claude-code={records}", *ONE_WINDOW]) == 0
    capsys.readouterr()
    predictor = Predictor("main")
    meta = json.loads((predictor.model_dir / "model.json").read_text())
    table = pq.read_table(predictor.prepared_path)
    assert table.num_rows == meta["tasks"] == 40  # one row for each task the model learned from
    names = table.column_names
    assert names[:6] == ["task_id", "started_at", "model", "effort", "agent", "origin"]
    assert names[-4:] == ["seconds", "tokens", "usd", "direct"]
    assert {"number.prompt_chars", "number.prompt_lines", "number.session_tasks_before"} <= set(names)
    assert [name for name in names if name.startswith("text_hash.")] == ["text_hash.prompt", "text_hash.context"]
    meta_of_file = json.loads(table.schema.metadata[b"yardstick"])
    assert meta_of_file["features"] == ["prompt", "conversation", "context"] and meta_of_file["format"] == 1

    rows = {row["task_id"]: row for row in table.to_pylist()}
    tasks = {task.id: task for task in predictor.tasks()}
    assert set(rows) == set(tasks)
    started = [row["started_at"] for row in table.to_pylist()]
    assert started == sorted(started)  # oldest first, as training reads them
    for task_id, row in rows.items():
        task = tasks[task_id]
        info = features.TaskInfo.of(task)
        assert row["text_hash.prompt"] == embeddings.text_hash(features.text("prompt", info))
        context = features.text("context", info)
        assert row["text_hash.context"] == (embeddings.text_hash(context) if context.strip() else None)
        assert (row["seconds"], row["tokens"], row["usd"], row["direct"]) == (
            task.seconds,
            task.tokens,
            task.usd,
            task.direct,
        )
        assert (row["model"], row["effort"], row["number.prompt_chars"]) == (task.model, task.effort, len(task.prompt))
    # It never holds a prompt, a conversation or any other text.
    texts = {text for task in tasks.values() for text in (task.prompt, task.context, task.project, task.session)}
    values = {value for row in rows.values() for value in row.values() if isinstance(value, str)}
    assert not (values & texts)

    # Status says how big it is, and a later training replaces it in one step.
    code, out = run(capsys, "status", "--json")
    assert json.loads(out)["predictors"][0]["prepared_bytes"] == predictor.prepared_path.stat().st_size
    assert re.search(r"prepared dataset \d+ KB", run(capsys, "status")[1])
    session = ClaudeSession(records, "later", START + 90 * 86400)
    for n in range(3):
        session.prompt(f"a later task {n}")
        session.reply(5)
        session.clock += 60
    session.save(mtime=START)
    assert main(["refresh", "--yes"]) == 0
    assert pq.read_table(predictor.prepared_path).num_rows == 43
    assert not list(predictor.dir.glob(".*.partial"))


def test_a_model_type_from_a_plugin_gets_a_prepared_dataset_too(capsys, quickly):
    assert main(["create", "--yes", *quickly, "--features", "time"]) == 0
    table = pq.read_table(Predictor("main").prepared_path)
    assert table.num_rows == 40 and {"number.days_since_2020", "number.hour_of_day"} <= set(table.column_names)
    assert json.loads(table.schema.metadata[b"yardstick"])["model_type"] == "quick"


# Deleting a predictor ------------------------------------------------------------------------------------


def test_deleting_a_predictor_asks_first_and_changes_nothing_else(capsys, monkeypatch, quickly, wide):
    for name in ("default", "work"):
        assert main(["create", name, "--yes", *quickly, "--features", "context"]) == 0
    capsys.readouterr()
    home, records = store.home(), records_of(quickly)
    work = store.predictor_folder("work")
    before = everything(home, records)

    asked = answering(monkeypatch, "")  # Enter says no
    code, out = run(capsys, "delete", "work")
    assert code == 0 and asked == ["Delete work for good? [y/N]"]
    assert f"This deletes its folder, {work}, and everything in it:" in out
    for part in (
        "its settings",
        "its dataset of 40 tasks, with their prompts (",
        "its prepared dataset, what went into its last training (",
        "what it learned (",
        "Your records and the feature store stay, and so do your other models.",
    ):
        assert part in out, part
    assert "Nothing was deleted." in out and everything(home, records) == before

    answering(monkeypatch, "y")
    code, out = run(capsys, "delete", "work")
    assert code == 0 and "Deleted work." in out
    assert not work.exists() and predictor_names() == ["default"]
    # Its folder is all that went: the records, the feature store, the cache and the other predictor stay.
    assert everything(home, records) == {path: kept for path, kept in before.items() if not path.startswith(str(work))}
    assert "work is your only predictor" not in out and default_name() == "default"


@pytest.mark.usefixtures("quick")
def test_a_program_deletes_a_predictor_with_yes_and_reads_json(capsys):
    for name in ("default", "work", "home"):
        quick_predictor(name)
    store.set_default("work")
    code, out = run(capsys, "delete", "work")
    assert code == 2 and "Add --yes to delete it. Nothing was deleted." in out
    code, out = run(capsys, "delete", "work", "--json")
    assert code == 2 and json.loads(out)["error"]["code"] == "needs_yes"
    assert "work" in predictor_names()

    code, out = run(capsys, "delete", "work", "--yes", "--json")
    assert code == 0
    found = json.loads(out)
    deleted = found["deleted"]
    assert deleted["name"] == "work" and deleted["folder"] == str(store.predictor_folder("work"))
    assert deleted["tasks"] == 0 and deleted["bytes"] > 0
    assert found["default"] == "default"  # the one called default is used now
    assert "work" not in store.settings().values()  # the default setting that named it is gone

    code, out = run(capsys, "delete", "work", "--json")
    assert code == 1 and json.loads(out)["error"]["code"] == "unknown_predictor"
    assert main(["delete", "nobody", "--yes"]) == 1
    assert "There's no model called 'nobody'. Your models: default, home." in capsys.readouterr().err


def test_deleting_the_default_says_what_commands_use_now(capsys):
    for name in ("first", "second", "third"):
        Predictor.create(name, [], EMBEDDING)
    store.set_default("second")
    code, out = run(capsys, "delete", "second", "--yes")
    assert code == 0 and "It's your default model." in out
    assert "It was your default model, so there's no default now." in out
    assert "Choose one with: yardstick-predict default NAME" in out and default_name() is None
    code, out = run(capsys, "delete", "first", "--yes")
    assert "third is your only model now, so commands use it when they name none." in out
    assert default_name() == "third"
    code, out = run(capsys, "delete", "third", "--yes")
    assert "There's no model now. Run yardstick-predict to create one." in out and predictor_names() == []


def test_a_predictor_being_refreshed_isnt_deleted_and_a_forecast_reading_its_model_finishes_first(capsys):
    predictor = Predictor.create("default", [], EMBEDDING)
    predictor.model_dir.mkdir()
    (predictor.model_dir / "model.json").write_text("{}")
    with locks.held(predictor.dir / "refresh.lock"):  # another command is refreshing it
        assert main(["delete", "default", "--yes"]) == 1
        assert "being set up or refreshed by another command" in capsys.readouterr().err
        assert main(["delete", "default", "--yes", "--json"]) == 1
        assert json.loads(capsys.readouterr().out)["error"]["code"] == "busy"
    assert (predictor.model_dir / "model.json").is_file()

    reading, seen = threading.Event(), []

    def forecast() -> None:
        with locks.held(predictor.dir / "model.lock", shared=True):  # as a forecast holds it while it reads
            reading.set()
            time.sleep(0.3)
            seen.append((predictor.model_dir / "model.json").is_file())

    thread = threading.Thread(target=forecast)
    thread.start()
    reading.wait()
    started = time.monotonic()
    assert main(["delete", "default", "--yes"]) == 0
    thread.join()
    assert time.monotonic() - started >= 0.2 and seen == [True]  # the model stayed until the forecast was done
    assert not predictor.dir.exists()


def can_link(folder: Path) -> bool:
    """Whether this computer lets the tests make a link to a folder. Windows needs a setting for it."""
    probe = folder / "probe-link"
    try:
        probe.symlink_to(folder, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    probe.unlink()
    return True


@pytest.mark.usefixtures("quick")
def test_deleting_a_predictor_that_is_a_link_deletes_only_the_link(capsys, tmp_path, wide):
    if not can_link(tmp_path):
        pytest.skip("This computer doesn't let the tests make links.")
    from test_embeddings import _task

    default = quick_predictor("default")
    default.add_tasks([_task(n, f"task {n}") for n in range(3)])
    default.close()
    work = store.predictor_folder("work")
    work.symlink_to(default.dir, target_is_directory=True)  # another name for the same predictor
    assert predictor_names() == ["default", "work"]
    store.set_default("work")
    before = everything(default.dir)

    code, out = run(capsys, "delete", "work")
    assert code == 2 and f"work is a link to {default.dir.resolve()}, not a folder of its own." in out
    assert "This deletes only the link. What it links to stays, with everything in it." in out
    assert "its dataset" not in out and everything(default.dir) == before
    code, out = run(capsys, "delete", "work", "--yes")
    assert code == 0 and f"Deleted the link work. {default.dir.resolve()} stays as it was." in out
    assert not work.is_symlink() and not work.exists() and predictor_names() == ["default"]
    assert everything(default.dir) == before  # nothing changed where the link led, not even a lock file
    assert default_name() == "default" and "work" not in store.settings().values()

    work.symlink_to(default.dir, target_is_directory=True)
    code, out = run(capsys, "delete", "work", "--yes", "--json")
    deleted = json.loads(out)["deleted"]
    assert code == 0 and deleted["link_to"] == str(default.dir.resolve())
    assert (deleted["tasks"], deleted["bytes"]) == (0, 0)
    assert not work.is_symlink() and everything(default.dir) == before
    assert Predictor("default").count_tasks() == 3


def test_deleting_a_predictor_never_follows_a_link_out_of_its_folder(capsys, tmp_path):
    if not can_link(tmp_path):
        pytest.skip("This computer doesn't let the tests make links.")
    outside = tmp_path / "outside"
    (outside / "deeper").mkdir(parents=True)
    (outside / "notes.txt").write_text("mine")
    (outside / "deeper" / "more.txt").write_text("mine too")
    other = Predictor.create("other", [], EMBEDDING)
    other.model_dir.mkdir()
    (other.model_dir / "model.json").write_text("{}")
    work = Predictor.create("work", [], EMBEDDING)
    (work.dir / "model").symlink_to(other.model_dir, target_is_directory=True)
    (work.dir / "notes").symlink_to(outside, target_is_directory=True)
    (work.dir / "notes.txt").symlink_to(outside / "notes.txt")
    (work.dir / "extra").mkdir()
    (work.dir / "extra" / "deeper").symlink_to(outside / "deeper", target_is_directory=True)
    before = everything(outside, other.dir)
    assert main(["delete", "work", "--yes"]) == 0
    assert not work.dir.exists() and everything(outside, other.dir) == before


def asks_for_lock(
    monkeypatch, thread: list[threading.Thread], picked, ready: threading.Event, refused: list[bool]
) -> None:
    """Set `ready` as the thread in `thread` asks for a lock that `picked` picks out by its path and whether it's
    shared. Just before, while the command that started the thread still waits for `ready`, it tries the lock
    without waiting, and adds to `refused` whether another command held it then."""
    from contextlib import contextmanager

    real = locks.held

    @contextmanager
    def held(path, wait=True, shared=False):
        if not (thread and threading.current_thread() is thread[0] and picked(Path(path), shared)):
            with real(path, wait, shared) as mine:
                yield mine
            return
        with real(path, False, shared) as mine:
            refused.append(not mine)
            ready.set()
            if mine or not wait:
                yield mine
                return
        with real(path, wait, shared) as mine:
            yield mine

    monkeypatch.setattr(locks, "held", held)


def test_a_setup_of_the_same_name_as_a_predictor_being_deleted_keeps_all_it_makes(capsys, monkeypatch, quickly):
    import shutil

    assert main(["create", "--yes", *quickly]) == 0
    folder = store.predictor_folder("main")
    setup: list[threading.Thread] = []
    codes: list[int] = []
    ready = threading.Event()
    refused: list[bool] = []
    asks_for_lock(monkeypatch, setup, lambda path, shared: path.parent == store.home() / "locks", ready, refused)
    real_rmtree = shutil.rmtree

    def setting_up() -> None:
        try:
            codes.append(main(["create", "--yes", *quickly]))
        finally:
            ready.set()

    def rmtree(path, *args, **kwargs):
        if Path(path) == folder and not setup:
            # A setup of a new predictor with the same name starts just as the old one's folder is about to go.
            # Unless something keeps it waiting, it makes the new predictor first.
            setup.append(threading.Thread(target=setting_up))
            setup[0].start()
            ready.wait(60)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    assert main(["delete", "main", "--yes"]) == 0
    setup[0].join(120)
    assert not setup[0].is_alive() and codes == [0]
    assert refused[:1] == [True]  # the delete held the name's lock as the setup first asked for it
    predictor = Predictor("main")
    assert predictor.count_tasks() == 40 and (predictor.model_dir / "model.json").is_file()
    assert predictor.prepared_path.is_file() and default_name() == "main"


def test_each_name_has_a_lock_of_its_own_with_a_short_file_name():
    for name in ("work", "é" * 200, "x" * 300):  # longer than a file's name can be
        with store.name_lock(name) as mine, store.name_lock(name, wait=False) as again:
            assert mine and not again
        with store.name_lock(name), store.name_lock(f"{name}2", wait=False) as other:
            assert other
    # Names a folder can't tell apart, where letter case doesn't count, share one lock.
    with store.name_lock("Work"), store.name_lock("work", wait=False) as other:
        assert not other
    files = list((store.home() / "locks").iterdir())
    assert len(files) == 6 and all(len(path.name) == len(f"predictor-{'0' * 32}.lock") for path in files)


def test_a_predictor_renamed_by_hand_or_beside_an_old_lock_folder_can_be_deleted(capsys):
    work = Predictor.create("work", [], EMBEDDING)
    work.close()
    (work.dir.parent / ".work.lock").mkdir()  # where an older version kept the name's lock, now a folder
    hidden = Predictor.create("hidden", [], EMBEDDING)
    hidden.close()
    hidden.dir.rename(hidden.dir.parent / ".locks")  # a name a lock folder might have
    names = ["work", ".locks"]
    if os.name != "nt":  # Windows' paths can't be this long unless long paths are turned on
        renamed = Predictor.create("renamed", [], EMBEDDING)
        renamed.close()
        name = "a" * 250  # a folder's name can be 255 bytes long
        renamed.dir.rename(renamed.dir.parent / name)
        names.append(name)
    assert predictor_names() == sorted(names)
    with store.name_lock("other"):
        for name in names:
            code, out = run(capsys, "delete", name, "--yes")
            assert code == 0 and not store.predictor_folder(name).exists()
        with store.name_lock("other", wait=False) as mine:
            assert not mine  # another name's lock stays where it was, held
    assert predictor_names() == [] and (work.dir.parent / ".work.lock").is_dir()


def test_a_broken_predictor_can_be_deleted(capsys):
    predictor = Predictor.create("default", [], EMBEDDING)
    predictor.add_tasks([])
    predictor.close()
    (predictor.dir / "predictor.json").write_text("")  # an older version stopped halfway through saving it
    assert main(["status"]) == 1
    assert "or delete the model with yardstick-predict delete default" in flat(capsys.readouterr().out)
    code, out = run(capsys, "delete", "default", "--yes")
    assert code == 0 and "Deleted default." in out and not predictor.dir.exists()


def test_a_refresh_that_started_as_its_predictor_was_deleted_never_saves_it_again(capsys, monkeypatch):
    predictor = Predictor.create("default", [], EMBEDDING)
    real = command.pick_predictor

    def found_then_deleted(name):
        picked = real(name)
        assert main(["delete", "default", "--yes"]) == 0  # deleted just as the refresh found it
        return picked

    monkeypatch.setattr(command, "pick_predictor", found_then_deleted)
    assert main(["refresh", "--yes"]) == 1
    assert "it was deleted" in capsys.readouterr().err
    assert not (predictor.dir / "predictor.json").exists() and predictor_names() == []


def test_the_status_line_says_plainly_when_its_predictor_is_gone(monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(
        cc.subprocess, "Popen", lambda command, **options: commands.append(command[command.index(cc.MARK) + 1 :])
    )

    def forecast(prompt: str) -> str:
        """Send a prompt, run its forecast as the background job would, and read the status line."""
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**event, "prompt": prompt})))
        assert cc.hook() == 0 and cc.main(commands[-1]) == 0
        started = json.loads(cc._state_path("s1").read_text())["prompt_at"]
        return cc.status_line("s1", "claude-opus-5-5", started + 1)

    event = {"session_id": "s1", "hook_event_name": "UserPromptSubmit"}
    for name in ("default", "work"):
        Predictor.create(name, [], EMBEDDING)

    def deleted_while_read(predictor):
        assert main(["delete", predictor.name, "--yes"]) == 0
        raise FileNotFoundError(str(predictor.dir / "model.lock"))  # what reading its files then meets

    real = forecasting.load
    monkeypatch.setattr(forecasting, "load", deleted_while_read)
    assert forecast("Add a login page") == "yardstick · no forecast: default was deleted"
    monkeypatch.setattr(forecasting, "load", real)
    # The one left is used without a default. It isn't trained yet.
    line = (
        "yardstick · no forecast: Model 'work' isn't trained yet. Train it with: yardstick-predict refresh "
        "--predictor work"
    )
    assert forecast("Fix the parser") == line
    Predictor.create("home", [], EMBEDDING)
    line = "yardstick · no forecast: no default model. Choose one with: yardstick-predict default NAME"
    assert forecast("Write the docs") == line
    for name in ("work", "home"):
        assert main(["delete", name, "--yes"]) == 0
    line = "yardstick · no forecast: There's no model yet. Run yardstick-predict to create one."
    assert forecast("Add a test") == line


# Deleting a stored feature --------------------------------------------------------------------------------


def test_deleting_a_stored_feature_says_who_uses_it_and_what_making_it_again_costs(capsys, monkeypatch, quickly):
    for name in ("default", "work"):
        assert main(["create", name, "--yes", *quickly, "--features", "context"]) == 0
    capsys.readouterr()
    home, records = store.home(), records_of(quickly)
    context = feature_store.folder() / CONTEXT
    with_context = len(Predictor("default").embedded("context"))
    before = everything(home, records)

    asked = answering(monkeypatch, "n")
    code, out = run(capsys, "features", "delete", CONTEXT)
    out = flat(out)  # the lines broken to fit the terminal, joined
    assert code == 0 and asked == ["Delete it for good? [y/N]"]
    assert "Delete a stored feature" in out and "context embeddings by voyage-4, 256 numbers" in out
    assert f"Used by default and work. The next refresh of each makes it again, for {with_context} tasks." in out
    assert re.search(r"The cache of vectors holds all \d+ of their texts, so it costs nothing\.", out)
    assert "Until then, those models forecast as they do now: a forecast never reads the feature store." in out
    assert "Your records and your models stay, and so does the rest of the feature store." in out
    assert "Nothing was deleted." in out and everything(home, records) == before

    # Without the cache, the texts would go to Voyage again, and it says what that costs.
    (home / "cache" / "embeddings.sqlite").unlink()
    answering(monkeypatch, "y")
    code, out = run(capsys, "features", "delete", CONTEXT)
    out = flat(out)  # the lines broken to fit the terminal, joined
    assert re.search(r"The cache of vectors holds none of their \d+ texts, so they go to Voyage: [\d,]+ tokens, ", out)
    assert "A refresh shows what it costs and asks before it spends anything, unless you give it --yes." in out
    assert code == 0 and "Deleted context embeddings by voyage-4, 256 numbers." in out
    assert not context.exists() and CONTEXT not in feature_store.index()
    # Only that feature's folder and its line in the index changed. The cache of vectors, emptied above, was opened
    # again to look for the texts.
    after = everything(home, records)
    changed = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
    changed = {path for path in changed if not path.startswith((str(context), str(home / "cache")))}
    assert {Path(path).name for path in changed} == {"index.json"}
    assert not Predictor("work").embedded("context") and len(Predictor("work").embedded("prompt")) == 40


def test_a_program_deletes_a_stored_feature_with_yes_and_reads_json(capsys, monkeypatch, quickly):
    monkeypatch.setattr(llm, "analyze", fake_analysis([]))
    assert main(["create", "--yes", *quickly, "--features", "llm:kind", "--llm-provider", "codex"]) == 0
    capsys.readouterr()
    answers = "llm.codex.codex-default"
    answering(monkeypatch, "n")
    code, out = run(capsys, "features", "delete", answers)
    assert "LLM answers by Codex's own model, through your ChatGPT subscription" in out
    assert "Used by main. Its next refresh makes it again, for 40 tasks." in out
    assert "LLM analysis: 40 tasks with Codex's own model through your ChatGPT subscription" in flat(out)
    monkeypatch.setattr(command, "interactive", lambda: False)

    code, out = run(capsys, "features", "delete", answers)
    assert code == 2 and "Add --yes to delete it. Nothing was deleted." in out
    code, out = run(capsys, "features", "delete", answers, "--json")
    assert code == 2 and json.loads(out)["error"]["code"] == "needs_yes"
    with feature_store.writing():  # as a refresh, which may be saving to it, holds it
        code, out = run(capsys, "features", "delete", answers, "--yes", "--json")
        assert code == 1 and json.loads(out)["error"]["code"] == "busy"
    assert answers in feature_store.index()

    code, out = run(capsys, "features", "delete", answers, "--yes", "--json")
    assert code == 0
    found = json.loads(out)
    assert found["deleted"]["id"] == answers and found["deleted"]["used_by"] == ["main"]
    assert found["deleted"]["tasks"] == 40 and found["notes"] == []
    assert found["remake"]["tasks"] == 40 and found["remake"]["usd"] is None  # a subscription pays for it
    assert answers not in feature_store.index() and Predictor("main").llm_labels() == {}
    for flags, error in (((), "invalid_request"), ((answers,), "unknown_feature"), (("nope",), "unknown_feature")):
        code, out = run(capsys, "features", "delete", *flags, "--yes", "--json")
        assert code == 1 and json.loads(out)["error"]["code"] == error


def test_a_setup_that_starts_as_a_stored_feature_is_deleted_keeps_what_it_saves(capsys, monkeypatch, quickly):
    monkeypatch.setattr(llm, "analyze", fake_analysis([]))
    assert main(["create", "--yes", *quickly, "--features", "llm:kind", "--llm-provider", "codex"]) == 0
    answers = "llm.codex.codex-default"
    # The new predictor asks the same AI model for a description, which nothing asked for before.
    flags = [*quickly, "--features", "llm:description", "--llm-provider", "codex"]
    setup: list[threading.Thread] = []
    codes: list[int] = []
    ready = threading.Event()
    refused: list[bool] = []
    asks_for_lock(
        monkeypatch, setup, lambda path, shared: shared and path.parent == feature_store.folder(), ready, refused
    )
    real_save, real_delete = feature_store.save_answers, feature_store.delete

    def save_answers(*args, **kwargs):
        found = real_save(*args, **kwargs)
        if setup and threading.current_thread() is setup[0]:
            ready.set()  # the setup saved what LLM analysis said
        return found

    def setting_up() -> None:
        try:
            codes.append(main(["create", "work", "--yes", *flags]))
        finally:
            ready.set()

    def delete(feature_id: str) -> bool:
        # A new predictor's setup starts just as the feature is deleted. Unless something keeps it waiting, it
        # pays for LLM analysis and saves the answers first.
        setup.append(threading.Thread(target=setting_up))
        setup[0].start()
        ready.wait(60)
        return real_delete(feature_id)

    monkeypatch.setattr(feature_store, "save_answers", save_answers)
    monkeypatch.setattr(feature_store, "delete", delete)
    assert main(["features", "delete", answers, "--yes"]) == 0
    setup[0].join(120)
    assert not setup[0].is_alive() and codes == [0]
    assert refused[:1] == [True]  # the delete held the lock of the store's users as the setup first asked for it
    labels = Predictor("work").llm_labels()
    assert len(labels) == 40 and all(answer.get("description") for answer in labels.values())


def test_a_stored_feature_isnt_deleted_while_a_command_that_saves_to_the_store_runs(capsys, monkeypatch, quickly):
    monkeypatch.setattr(llm, "analyze", fake_analysis([]))
    assert main(["create", "--yes", *quickly, "--features", "llm:kind", "--llm-provider", "codex"]) == 0
    capsys.readouterr()
    answers = "llm.codex.codex-default"
    real_refresh = command._refresh
    seen: list[tuple[int, str]] = []

    def refresh(args, predictor, answer):
        # While the refresh runs, a program tries to delete a stored feature it may be saving to.
        seen.append(run(capsys, "features", "delete", answers, "--yes", "--json"))
        return real_refresh(args, predictor, answer)

    monkeypatch.setattr(command, "_refresh", refresh)
    assert main(["refresh", "--yes"]) == 0
    code, out = seen[0]
    error = json.loads(out)["error"]
    assert code == 1 and error["code"] == "busy" and "Try again when it's done." in error["message"]
    assert answers in feature_store.index() and len(Predictor("main").llm_labels()) == 40


def test_an_older_predictor_moves_into_the_store_only_while_no_stored_feature_is_deleted(capsys):
    old_predictor("older")  # an older version made it, so its vectors and LLM answers move into the store first
    with feature_store.deleting() as alone:  # a features delete is running
        assert alone
        code, out = run(capsys, "check", "older", "--json")
        error = json.loads(out)["error"]
        assert code == 1 and error == {"code": "busy", "message": command.STORE_BUSY}
        code, out = run(capsys, "features", "--json")
        assert code == 1 and json.loads(out)["predictors"][0]["error"] == command.STORE_BUSY
        code, out = run(capsys, "status")
        assert "Its coverage can't be counted while another command is using the feature store." in flat(out)
        assert not Predictor("older").config.get("feature_store")  # nothing moved in meanwhile
    code, out = run(capsys, "features", "--json")
    assert code == 0 and Predictor("older").config.get("feature_store")


def test_making_llm_answers_again_counts_an_analysis_for_each_set_of_parts_asked_for(capsys, monkeypatch):
    from test_embeddings import _task

    how = {"provider": "anthropic", "model": "claude-opus-5-5"}
    tasks = [_task(n, f"task {n}") for n in range(6)]
    made = {}
    for name, parts in (("kind", ["kind"]), ("described", ["description"]), ("alike", ["kind"])):
        made[name] = Predictor.create(name, [], EMBEDDING, features=["prompt", "llm"], options={"llm": parts}, llm=how)
        made[name].add_tasks(tasks)
    answers = feature_store.answers(how["provider"], how["model"])
    # One asks for the kind of task, the other for a description, so each has the tasks they share read again.
    found = pipeline.remake(answers, [made["kind"], made["described"]])
    assert found.usd == pytest.approx(llm.cost(12, how["model"])) and (found.tasks, found.analyses) == (6, 12)
    # Two that ask for the same parts share one analysis of each task.
    found = pipeline.remake(answers, [made["kind"], made["alike"]])
    assert found.usd == pytest.approx(llm.cost(6, how["model"])) and (found.tasks, found.analyses) == (6, 6)

    feature_store.save_answers(answers, {"t0": {"kind": "feature", "by": "anthropic:claude-opus-5-5"}})
    answering(monkeypatch, "n")
    code, out = run(capsys, "features", "delete", answers.id)
    out = flat(out)  # the lines broken to fit the terminal, joined
    assert "Used by alike, described and kind. The next refresh of each makes it again, for 6 tasks." in out
    assert "They ask for different parts of the analysis, so their tasks are read once for each." in out
    assert "LLM analysis: 6 tasks, read 12 times in all, with Claude Opus 5.5, about $0.10 at list price." in out
    monkeypatch.setattr(command, "interactive", lambda: False)
    code, out = run(capsys, "features", "delete", answers.id, "--yes", "--json")
    remade = json.loads(out)["remake"]
    assert code == 0 and (remade["tasks"], remade["analyses"]) == (6, 12)
    assert remade["usd"] == pytest.approx(llm.cost(12, how["model"]))


def test_making_llm_answers_again_counts_the_fewest_and_the_most_readings(capsys, monkeypatch):
    from test_embeddings import _task

    how = {"provider": "anthropic", "model": "claude-opus-5-5"}
    tasks = [_task(n, f"task {n}") for n in range(6)]
    made = {}
    for name, parts in (("kind", ["kind"]), ("described", ["description"]), ("both", ["kind", "description"])):
        made[name] = Predictor.create(name, [], EMBEDDING, features=["prompt", "llm"], options={"llm": parts}, llm=how)
        made[name].add_tasks(tasks)
    answers = feature_store.answers(how["provider"], how["model"])
    # When the one asking for both parts refreshes first, the other finds its parts saved. When it refreshes
    # second, it has each task read again, for the part the first didn't ask for.
    found = pipeline.remake(answers, [made["kind"], made["both"]])
    assert (found.tasks, found.analyses_least, found.analyses) == (6, 6, 12)
    # With all three, the last to refresh always finds its parts saved.
    found = pipeline.remake(answers, list(made.values()))
    assert (found.tasks, found.analyses_least, found.analyses) == (6, 6, 12)
    assert found.usd == pytest.approx(llm.cost(12, how["model"]))
    found = pipeline.remake(answers, [made["kind"], made["described"]])
    assert (found.analyses_least, found.analyses) == (12, 12)  # whichever goes first, the other needs its own

    feature_store.save_answers(answers, {"t0": {"kind": "feature", "by": "anthropic:claude-opus-5-5"}})
    answering(monkeypatch, "n")
    code, out = run(capsys, "features", "delete", answers.id)
    out = flat(out)  # the lines broken to fit the terminal, joined
    assert "so a task can be read more than once: from 6 to 12 readings in all when they refresh one at a time" in out
    assert "depending on which goes first. Below is the most." in out and "read once for each" not in out
    assert "LLM analysis: 6 tasks, read 12 times in all, with Claude Opus 5.5, about $0.10 at list price." in out
    monkeypatch.setattr(command, "interactive", lambda: False)
    code, out = run(capsys, "features", "delete", answers.id, "--yes", "--json")
    remade = json.loads(out)["remake"]
    assert code == 0 and (remade["tasks"], remade["analyses_least"], remade["analyses"]) == (6, 6, 12)


def test_a_stored_feature_no_predictor_uses_goes_without_any_cost(capsys, monkeypatch):
    answers = feature_store.answers("claude", "claude-opus-5-5")
    feature_store.save_answers(answers, {"t1": {"kind": "feature"}})  # made for a predictor deleted since
    code, out = run(capsys, "features")
    assert f"id {answers.id}" in out and "Delete one with: yardstick-predict features delete ID" in out
    code, out = run(capsys, "features", "delete", answers.id, "--yes")
    assert code == 0 and "No model uses it, so nothing makes it again." in out
    assert answers.id not in feature_store.index()


def test_the_features_listing_keeps_its_flags(capsys, quickly):
    assert main(["create", "--yes", *quickly]) == 0
    capsys.readouterr()
    code, out = run(capsys, "features", "--predictor", "main", "--json")
    assert code == 0 and [entry["name"] for entry in json.loads(out)["predictors"]] == ["main"]
    code, out = run(capsys, "features")
    assert code == 0 and "id embedding.prompt.voyage-4.256" in out
    assert main(["features", "remove", "x"]) == 2  # only delete goes there
    assert "invalid choice: 'remove'" in capsys.readouterr().err


def test_what_making_a_feature_again_costs_counts_a_shared_text_once():
    predictor = Predictor.create("default", [], EMBEDDING, features=["prompt", "context"])
    other = Predictor.create("other", [], EMBEDDING, features=["prompt"])
    from test_embeddings import _task

    tasks = [_task(n, "continue" if n % 2 else f"task {n}", context="we talked") for n in range(6)]
    for one in (predictor, other):
        one.add_tasks(tasks)
    prompts = feature_store.embedding("prompt", "voyage-4", 256)
    found = pipeline.remake(prompts, [predictor, other])
    assert (found.tasks, found.texts, found.cached) == (6, 4, 0)  # "continue" goes once
    embeddings.remember("voyage-4", {"continue": [0.1] * 256})
    found = pipeline.remake(prompts, [predictor, other])
    assert found.cached == 1 and found.tokens == sum(len(f"task {n}".split()) + 1 for n in (0, 2, 4))
    assert pipeline.remake(feature_store.embedding("context", "voyage-4", 256), [predictor]).texts == 1


def test_a_predictor_whose_settings_cant_be_read_isnt_renamed(capsys):
    quick_predictor("default")
    work = quick_predictor("work")
    store.set_default("work")
    (work.dir / "predictor.json").write_text("{not json")
    assert main(["rename", "work", "fixed"]) == 1
    assert "It wasn't renamed." in capsys.readouterr().err
    assert predictor_names() == ["default", "work"] and store.settings()["default"] == "work"  # nothing moved


@pytest.mark.usefixtures("quick")
def test_renaming_a_link_leaves_the_folder_it_leads_to_as_it_was(capsys, tmp_path):
    if not can_link(tmp_path):
        pytest.skip("This computer doesn't let the tests make links.")
    default = quick_predictor("default")
    default.close()
    store.predictor_folder("work").symlink_to(default.dir, target_is_directory=True)
    before = (default.dir / "predictor.json").read_text()
    assert main(["rename", "work", "home"]) == 0
    assert store.predictor_folder("home").is_symlink() and predictor_names() == ["default", "home"]
    assert (default.dir / "predictor.json").read_text() == before  # its name is still default


def test_settings_changed_at_the_same_time_all_stay():
    import threading

    names = [f"note_{number}" for number in range(24)]
    threads = [threading.Thread(target=store.save_setting, args=(name, True)) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(store.settings().get(name) is True for name in names)  # no change was lost to another
    store.forget_setting("note_0")
    assert "note_0" not in store.settings() and store.settings().get("note_1") is True
