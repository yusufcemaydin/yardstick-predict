"""Your predictors: the list, each feature's coverage in status, the prepared dataset, and deleting a predictor or a
stored feature."""

from __future__ import annotations

import io
import json
import re
import threading
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from builders import ClaudeSession, history
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


def test_list_shows_one_line_for_each_predictor_and_agrees_with_status(capsys, quickly):
    assert main(["create", "--yes", *quickly, "--features", "context,time"]) == 0
    Predictor.create("work", [], EMBEDDING, features=["prompt"])  # not trained yet
    broken = store.predictor_folder("broken")
    broken.mkdir()
    (broken / "predictor.json").write_text("{")
    capsys.readouterr()

    code, out = run(capsys, "list")
    assert code == 1  # a predictor it shows can't be used
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert re.search(r"features +learned from +trained +forecasts for", out)
    assert re.search(r"default \(default\) +Context, Time +40 tasks +just now +claude-opus-5-5$", lines["default"])
    assert re.search(r"work +only the ones always on +- +not yet +none yet$", lines["work"])
    assert "can't be used, see below" in lines["broken"]
    assert "Predictor 'broken' can't be used" in out and "yardstick-predict delete broken" in out

    code, out = run(capsys, "list", "--json")
    listed = {entry["name"]: entry for entry in json.loads(out)["predictors"]}
    assert listed["default"]["features"] == ["prompt", "context", "time"] and listed["default"]["default"]
    assert listed["default"]["learned_from"] == 40 and listed["default"]["forecasts_for"] == ["claude-opus-5-5"]
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
    for name in ("default", "work"):
        for fact in ("learned_from", "trained_at", "forecasts_for"):
            assert shown[name][fact] == listed[name][fact], (name, fact)
    code, out = run(capsys, "status", "--predictor", "default")
    assert "Trained on 40 tasks." in out and re.search(r"claude-opus-5-5 +forecasts on", out)


@pytest.mark.usefixtures("quick")
def test_list_says_when_theres_nothing_to_list_and_what_needs_training_again(capsys):
    code, out = run(capsys, "list")
    assert code == 0 and "There's no predictor yet." in out
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
    line = next(line for line in out.splitlines() if line.strip().startswith("Coverage"))
    expected = ", ".join(f"{title} {share:.0%}" for title, share in shares.items())
    assert line.strip() == f"Coverage of the 40 tasks it can learn from: {expected}"

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
    assert "while another command sets it up or refreshes it" in out
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
    predictor = Predictor("default")
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
    table = pq.read_table(Predictor("default").prepared_path)
    assert table.num_rows == 40 and {"number.days_since_2020", "number.hour_of_day"} <= set(table.column_names)
    assert json.loads(table.schema.metadata[b"yardstick"])["model_type"] == "quick"


# Deleting a predictor ------------------------------------------------------------------------------------


def test_deleting_a_predictor_asks_first_and_changes_nothing_else(capsys, monkeypatch, quickly):
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
        "its trained model (",
        "Your records and the feature store stay, and so do your other predictors.",
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
    assert "There's no predictor called 'nobody'. Your predictors: default, home." in capsys.readouterr().err


def test_deleting_the_default_says_what_commands_use_now(capsys):
    for name in ("first", "second", "third"):
        Predictor.create(name, [], EMBEDDING)
    store.set_default("second")
    code, out = run(capsys, "delete", "second", "--yes")
    assert code == 0 and "It's your default predictor." in out
    assert "It was your default predictor, so there's no default now." in out
    assert "Choose one with: yardstick-predict default NAME" in out and default_name() is None
    code, out = run(capsys, "delete", "first", "--yes")
    assert "third is your only predictor now, so commands use it when they name none." in out
    assert default_name() == "third"
    code, out = run(capsys, "delete", "third", "--yes")
    assert "There's no predictor now. Run yardstick-predict to create one." in out and predictor_names() == []


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


def test_a_broken_predictor_can_be_deleted(capsys):
    predictor = Predictor.create("default", [], EMBEDDING)
    predictor.add_tasks([])
    predictor.close()
    (predictor.dir / "predictor.json").write_text("")  # an older version stopped halfway through saving it
    assert main(["status"]) == 1
    assert "or delete the predictor with yardstick-predict delete default" in capsys.readouterr().out
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
    # The one left is used without a default. It has no model yet.
    line = "yardstick · no forecast: Predictor 'work' has no model yet. Run yardstick-predict."
    assert forecast("Fix the parser") == line
    Predictor.create("home", [], EMBEDDING)
    line = "yardstick · no forecast: no default predictor. Choose one with: yardstick-predict default NAME"
    assert forecast("Write the docs") == line
    for name in ("work", "home"):
        assert main(["delete", name, "--yes"]) == 0
    line = "yardstick · no forecast: There's no predictor yet. Run yardstick-predict to create one."
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
    assert code == 0 and asked == ["Delete it for good? [y/N]"]
    assert "Delete a stored feature" in out and "context embeddings by voyage-4, 256 numbers" in out
    assert f"Used by default and work. The next refresh of each makes it again, for {with_context} tasks." in out
    assert re.search(r"The cache of vectors holds all \d+ of their texts, so it costs nothing\.", out)
    assert "Until then, their models forecast as they do now: a forecast never reads the feature store." in out
    assert "Your records and your predictors stay, and so does the rest of the feature store." in out
    assert "Nothing was deleted." in out and everything(home, records) == before

    # Without the cache, the texts would go to Voyage again, and it says what that costs.
    (home / "cache" / "embeddings.sqlite").unlink()
    answering(monkeypatch, "y")
    code, out = run(capsys, "features", "delete", CONTEXT)
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
    assert "Used by default. Its next refresh makes it again, for 40 tasks." in out
    assert "LLM analysis: 40 tasks with Codex's own model through your chatgpt subscription" in out
    monkeypatch.setattr(command, "interactive", lambda: False)

    code, out = run(capsys, "features", "delete", answers)
    assert code == 2 and "Add --yes to delete it. Nothing was deleted." in out
    code, out = run(capsys, "features", "delete", answers, "--json")
    assert code == 2 and json.loads(out)["error"]["code"] == "needs_yes"
    with locks.held(store.predictor_folder("default") / "refresh.lock"):  # a refresh may be saving to it
        code, out = run(capsys, "features", "delete", answers, "--yes", "--json")
        assert code == 1 and json.loads(out)["error"]["code"] == "busy"
    assert answers in feature_store.index()

    code, out = run(capsys, "features", "delete", answers, "--yes", "--json")
    assert code == 0
    found = json.loads(out)
    assert found["deleted"]["id"] == answers and found["deleted"]["used_by"] == ["default"]
    assert found["deleted"]["tasks"] == 40 and found["notes"] == []
    assert found["remake"]["tasks"] == 40 and found["remake"]["usd"] is None  # a subscription pays for it
    assert answers not in feature_store.index() and Predictor("default").llm_labels() == {}
    for flags, error in (((), "invalid_request"), ((answers,), "unknown_feature"), (("nope",), "unknown_feature")):
        code, out = run(capsys, "features", "delete", *flags, "--yes", "--json")
        assert code == 1 and json.loads(out)["error"]["code"] == error


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
    assert "Used by alike, described and kind. The next refresh of each makes it again, for 6 tasks." in out
    assert "They ask for different parts of the analysis, so their tasks are read once for each." in out
    assert "LLM analysis: 6 tasks, read 12 times in all, with Claude Opus 5.5, about $0.10 at list price." in out
    monkeypatch.setattr(command, "interactive", lambda: False)
    code, out = run(capsys, "features", "delete", answers.id, "--yes", "--json")
    remade = json.loads(out)["remake"]
    assert code == 0 and (remade["tasks"], remade["analyses"]) == (6, 12)
    assert remade["usd"] == pytest.approx(llm.cost(12, how["model"]))


def test_a_stored_feature_no_predictor_uses_goes_without_any_cost(capsys, monkeypatch):
    answers = feature_store.answers("claude", "claude-opus-5-5")
    feature_store.save_answers(answers, {"t1": {"kind": "feature"}})  # made for a predictor deleted since
    code, out = run(capsys, "features")
    assert f"id {answers.id}" in out and "Delete one with: yardstick-predict features delete ID" in out
    code, out = run(capsys, "features", "delete", answers.id, "--yes")
    assert code == 0 and "No predictor uses it, so nothing makes it again." in out
    assert answers.id not in feature_store.index()


def test_the_features_listing_keeps_its_flags(capsys, quickly):
    assert main(["create", "--yes", *quickly]) == 0
    capsys.readouterr()
    code, out = run(capsys, "features", "--predictor", "default", "--json")
    assert code == 0 and [entry["name"] for entry in json.loads(out)["predictors"]] == ["default"]
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
