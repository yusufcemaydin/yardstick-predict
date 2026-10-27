from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
from builders import ClaudeSession, CodexSession, history
from conftest import KEY

from yardstick_predict import benchmarks, cli, embeddings, forecast, llm, model_types, pipeline
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor, default_name, predictor_names

START = 1_780_000_000.0
SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"
ONE_WINDOW = ("--test-windows", "1")
"""Test each AI model in one window, not three, so training is quicker. For tests that don't check the test itself."""


@pytest.fixture
def records(tmp_path):
    """180 Claude Code tasks on two AI models at two effort levels: enough for each model's own test."""
    root = tmp_path / "claude-projects"
    history(root, tasks=180, start=START, models=("claude-opus-5-5", "claude-sonnet-5-5"), efforts=("high", "max"))
    return root


@pytest.fixture
def ready(voyage, monkeypatch):
    """A Voyage key and the stand-in Voyage server."""
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    return voyage


def run(capsys, *args, stdin: str | None = None, monkeypatch=None) -> tuple[int, str]:
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = main(list(args))
    return code, capsys.readouterr().out


def test_express_takes_every_default_in_one_line(capsys, records, ready, monkeypatch):
    code, out = run(capsys, "create", "--express", "--archive", f"claude-code={records}")
    assert code == 0, out
    assert "180 tasks" in out
    assert re.search(r"claude-opus-5-5\s+forecasts on", out)
    assert "Forecasts are on for claude-opus-5-5" in out and 'yardstick-predict predict "Add a login page' in out
    assert default_name() == "default"  # the first predictor is the default
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt", "conversation", "context"]
    assert predictor.config["embedding"]["model"] == "voyage-4"
    assert predictor.config["model"]["type"] == "xgboost"
    assert len(predictor.embedded("prompt")) == 180

    code, out = run(
        capsys, "predict", "--model", "claude-opus-5-5", "--model", "gpt-unknown", "--prompt", "fix the api"
    )
    assert code == 0
    result = json.loads(out)
    jsonschema.validate(result, json.loads((SCHEMAS / "forecast-response.v1.json").read_text()))
    by_model = {f["model"]: f for f in result["forecasts"]}
    assert by_model["claude-opus-5-5"]["seconds"] > 0
    assert by_model["gpt-unknown"]["reason"] == "unknown_model" and "effort" not in by_model["gpt-unknown"]
    # A model is forecast at every effort level it often ran at.
    assert {f["effort"] for f in result["forecasts"] if f["model"] == "claude-opus-5-5"} == {"high", "max"}
    assert {f["task"] for f in result["forecasts"]} == {"task"}  # one prompt is one task

    # In a terminal, a task in quotes gets a table, on every model the predictor forecasts.
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    sent = len(ready.requests)
    code, out = run(capsys, "predict", "fix the api")
    row = r"\s+\d+ (s|min|h)\s+[\d.]+[kM]?\s+\$"
    assert code == 0 and re.search(r"claude-opus-5-5\s+high" + row, out) and re.search(r"\n\s+max" + row, out)
    # One prompt at several effort levels is embedded once. The JSON forecast above already paid for it, and
    # the cache of vectors kept it, so this one sends it no more.
    texts = [text for body in ready.requests for text in body["input"]]
    assert texts.count("fix the api") == 1 and len(ready.requests) == sent
    assert "predictor default, updated just now" in out and "{" not in out
    monkeypatch.setattr(cli, "terminal_output", lambda: False)

    code, out = run(capsys, "status", "--json")
    status = json.loads(out)["predictors"][0]
    assert status["tasks"] == 180 and status["features"] == ["prompt", "conversation", "context"]
    assert status["report"]["windows"] == 6  # three for each of the two AI models


def test_adding_a_folder_is_optional_only_once_some_records_are_chosen(tmp_path, monkeypatch, capsys):
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    copied = tmp_path / "copied-runs"
    copied.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows finds the home folder here, not in HOME

    def answer(*replies: str) -> list[str]:
        asked, queue = [], list(replies)

        def fake_input(question: str = "") -> str:
            asked.append(question.strip())
            return queue.pop(0)

        monkeypatch.setattr("builtins.input", fake_input)
        return asked

    asked = answer("y", "")
    assert [a["name"] for a in cli.choose_archives(True, [])] == ["claude-code"]
    assert "records in their default location (~/.claude/projects)?" in asked[0]
    assert asked[1].startswith("(Optional) Add a folder")
    assert "No codex records in their default location (~/.codex/sessions)." in capsys.readouterr().out

    asked = answer("n", str(copied), "codex", "")
    assert [a["path"] for a in cli.choose_archives(True, [])] == [str(copied)]
    assert asked[1].startswith("Add a folder")
    assert asked[3].startswith("(Optional) Add a folder")


def test_the_cost_is_said_plainly_and_free_tokens_count_only_on_a_new_account():
    def said(free_left: int | None, exact: bool = True) -> str:
        return " ".join(cli.cost_lines(embeddings.Cost(2_000_000, exact, 0.06, free_left), "voyage-4", 2_500))

    plain = "Embedding 2,500 tasks with voyage-4 uses 2,000,000 tokens. It costs $0.12 at Voyage's list price."
    assert said(None) == plain and said(0) == plain
    covered = said(200_000_000)
    assert "Your free tokens cover it, so it costs nothing. 198.0 million free tokens will be left." in covered
    assert "Your free tokens cover 1.5 million of them. The rest costs $0.03." in said(1_500_000)
    assert "uses about 2,000,000 tokens" in said(None, exact=False) and "an estimate" in said(None, exact=False)


def test_a_task_counts_toward_the_free_tokens_only_when_all_its_texts_fit():
    from yardstick_predict.pipeline import EmbedPlan, newest_that_fit

    items = [("newest", "prompt", "a"), ("newest", "context", "b"), ("older", "prompt", "c"), ("older", "context", "d")]
    plan = EmbedPlan(items, [10, 10, 10, 10], embeddings.Cost(40, True, 0.06, 30), texts=["a", "b", "c", "d"])
    assert newest_that_fit(plan, 30) == 1  # the older task's prompt fits, but its context doesn't
    assert newest_that_fit(plan, 40) == 2
    assert newest_that_fit(plan, 19) == 0


def test_pressing_enter_on_the_free_tokens_question_assumes_no_credits(capsys, ready, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda question="": "")
    assert cli.get_key("voyage-4", can_ask=True) == KEY
    assert "y   yes, or I'm not sure - report costs assuming no credits" in capsys.readouterr().out
    assert embeddings.account(KEY)["free_tokens"] == "unknown"


def test_a_program_must_agree_to_the_cost(capsys, records, ready):
    code, out = run(capsys, "create", "--archive", f"claude-code={records}")
    assert code == 2
    assert "list price" in out and "--yes" in out
    assert predictor_names() == []  # nothing is created before the yes
    assert len(ready.requests) == 1  # only the tiny key check was sent

    code, out = run(capsys, "create", "--archive", f"claude-code={records}", "--yes", *ONE_WINDOW)
    assert code == 0, out
    stored = b"".join(p.read_bytes() for p in Predictor("default").dir.parent.parent.rglob("*") if p.is_file())
    assert KEY.encode() not in stored
    assert sum(embeddings.account(KEY)["used"].values()) > 0


def test_features_and_advanced_settings_are_chosen_with_flags(capsys, records, ready, monkeypatch):
    monkeypatch.setattr(
        benchmarks,
        "download",
        lambda timeout=30: {
            "claude-opus-5-5": {"benchmark_price_input": 0.7},
            "claude-sonnet-5-5": {"benchmark_price_input": 0.48},
        },
    )
    code, out = run(
        capsys,
        "create",
        "--yes",
        "--archive",
        f"claude-code={records}",
        "--features",
        "benchmarks",
        "--heads",
        "seconds,usd",
        "--shape",
        "weibull",
        "--test-windows",
        "2",
    )
    assert code == 0, out
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt", "benchmarks"]
    assert predictor.config["model"]["heads"] == ["seconds", "usd"]
    assert predictor.config["model"]["family"] == "weibull"
    assert (predictor.dir / "benchmarks.json").is_file()
    assert not predictor.embedded("context")

    code, out = run(capsys, "predict", "--model", "claude-opus-5-5", "--prompt", "fix the api", "--distribution")
    forecast = json.loads(out)["forecasts"][0]
    assert forecast["tokens"] is None and forecast["seconds"] > 0 and forecast["usd"] > 0
    assert forecast["distribution"]["shape"] == "two_part_weibull"


def test_setup_stops_before_spending_when_too_few_tasks_can_be_learned_from(tmp_path, capsys, ready):
    few = tmp_path / "few"
    history(few, tasks=20, start=START)
    assert main(["create", "--yes", "--archive", f"claude-code={few}"]) == 1
    assert "Only 20 of those tasks can be learned from" in capsys.readouterr().err
    assert ready.requests == [] and predictor_names() == []  # no key check, no embedding, no predictor
    assert main(["create", "--yes", "--max-tasks", "10", "--archive", f"claude-code={few}"]) == 1
    assert "--max-tasks 30 or more" in capsys.readouterr().err


def test_archives_with_the_same_name_get_their_own_names():
    named = cli.unique_names([{"name": "sessions"}, {"name": "sessions"}, {"name": "codex"}, {"name": "sessions"}])
    assert [a["name"] for a in named] == ["sessions", "sessions-2", "codex", "sessions-3"]


def test_forecasting_with_an_untrained_predictor_says_so(capsys):
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert main(["predict", "fix the login page"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "not_trained"


def test_a_typed_prompt_is_forecast_as_a_persons(capsys, monkeypatch):
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    asked = []

    def answer(predictor, tasks, distribution, model=None, rough=False):
        asked.extend(tasks)
        return {"forecasts": []}

    monkeypatch.setattr(forecast, "answer", answer)
    monkeypatch.setattr(forecast, "load", lambda predictor: SimpleNamespace(meta={}))
    assert main(["predict", "fix the login page", "--model", "m", "--effort", "high"]) == 0
    request = {"version": 1, "tasks": [{"id": "a", "prompt": "fix the login page", "models": ["m"]}]}
    run(capsys, "predict", stdin=json.dumps(request), monkeypatch=monkeypatch)
    assert [task.origin for task in asked] == ["person", "program"]  # a JSON request says who for itself


def test_an_unknown_model_type_stops_setup_before_anything_is_spent(capsys, records, ready):
    assert main(["create", "--yes", "--model-type", "nope", "--archive", f"claude-code={records}"]) == 1
    error = capsys.readouterr().err
    assert "Unknown model type 'nope'" in error and "Nothing was read or spent" in error
    assert ready.requests == [] and predictor_names() == []


def test_setup_never_offers_fewer_free_tasks_than_training_needs(capsys, records, ready, monkeypatch):
    embeddings.set_free_tokens(KEY, "new")
    embeddings.record_usage(KEY, "voyage-4", embeddings.FREE_TOKENS - 300)  # room for only a few tasks
    asked = []

    def fake_input(question=""):
        asked.append(question.strip())
        return "n" if "Embed them" in question else ""

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(cli, "interactive", lambda: True)
    code = main(["create", "--archive", f"claude-code={records}", "--features", "", "--embedding", "voyage-4"])
    assert code == 0 and "Your free tokens cover only" in capsys.readouterr().out
    assert not any("fit in your free tokens" in question for question in asked)
    assert predictor_names() == []  # it stopped at the question, as told


def test_only_one_refresh_of_a_predictor_runs_at_a_time(capsys):
    from yardstick_predict import locks

    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    with locks.held(predictor.dir / "refresh.lock"):  # another command is refreshing it
        assert main(["refresh", "--yes"]) == 1
    assert "being set up or refreshed by another command" in capsys.readouterr().err


def test_a_forecast_waits_for_a_model_being_swapped_in():
    import threading
    import time

    from yardstick_predict import locks
    from yardstick_predict.pipeline import model_ready

    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert not model_ready(predictor)  # no model, and none on its way: no waiting
    (predictor.dir / ".model-left-behind").mkdir()  # a save stopped halfway long ago doesn't make it wait
    started = time.time()
    assert not model_ready(predictor) and time.time() - started < 0.1
    swapping = threading.Event()

    def swap():
        with locks.held(predictor.dir / "model.lock"):  # a save holds the lock while it swaps the model in
            swapping.set()
            time.sleep(0.2)
            predictor.model_dir.mkdir()
            (predictor.model_dir / "model.json").write_text("{}")

    threading.Thread(target=swap).start()
    swapping.wait()
    started = time.time()
    assert model_ready(predictor) and time.time() - started >= 0.15


def test_bad_choices_get_a_plain_reason(capsys, records, ready):
    assert main(["create", "--yes", "--archive", f"claude-code={records}", "--features", "horoscope"]) == 1
    assert "Unknown feature" in capsys.readouterr().err
    assert main(["create", "--yes", "--archive", f"claude-code={records}", "--shape", "weibull", "--loss", "crps"]) == 1
    assert "crps" in capsys.readouterr().err


def test_a_new_predictor_is_named_first(monkeypatch):
    replies = iter(["", "", "Not A Name", "main", "work"])
    monkeypatch.setattr("builtins.input", lambda question="": next(replies))
    assert cli.ask_name([]) == "main"  # the first predictor is called main unless you pick a name
    assert cli.ask_name(["main"]) == "work"  # an empty, a bad and a taken name are each asked again


def test_commands_use_the_default_predictor_you_choose(capsys):
    for name in ("default", "work"):
        Predictor.create(name, [], {"model": "voyage-4", "dimensions": 256})
    assert default_name() == "default" and cli.pick_predictor(None).name == "default"
    assert main(["default", "work"]) == 0
    assert default_name() == "work" and cli.pick_predictor(None).name == "work"
    assert cli.pick_predictor("default").name == "default"  # a name always wins
    main(["default"])
    assert "Your default predictor is work" in capsys.readouterr().out
    assert main(["default", "nope"]) == 1 and "no predictor called 'nope'" in capsys.readouterr().err

    for name in ("home", "lab"):
        Predictor.create(name, [], {"model": "voyage-4", "dimensions": 256})
    assert default_name() == "work"
    (Predictor("work").dir / "predictor.json").unlink()
    (Predictor("default").dir / "predictor.json").unlink()
    assert default_name() is None  # several left, none chosen
    with pytest.raises(SystemExit, match="no default"):
        cli.pick_predictor(None)


def test_predict_reads_a_json_request_and_never_asks(capsys, records, ready, monkeypatch):
    run(capsys, "create", "--express", "--archive", f"claude-code={records}", *ONE_WINDOW)
    request = {
        "version": 1,
        "distribution": True,
        "tasks": [
            {"id": "a", "prompt": "add login page", "models": ["claude-opus-5-5"], "context": "We use Flask."},
            {"id": "b", "prompt": "refactor the parser and the cli " * 5, "models": ["claude-opus-5-5"]},
        ],
    }
    code, out = run(capsys, "predict", stdin=json.dumps(request), monkeypatch=monkeypatch)
    assert code == 0
    forecasts = json.loads(out)["forecasts"]
    assert [f["task"] for f in forecasts] == ["a", "b"]
    assert forecasts[0]["distribution"]["shape"] == "two_part_lognormal"

    sent = len(ready.requests)
    code, out = run(capsys, "predict", "--model", "never-used-model", "--prompt", "fix the login page")
    assert json.loads(out)["forecasts"][0]["reason"] == "unknown_model"
    assert len(ready.requests) == sent  # nothing to forecast, so nothing was sent to Voyage


def test_predict_errors_come_back_in_the_format(capsys, monkeypatch):
    code, out = run(capsys, "predict", stdin='{"version": 1, "tasks": []}', monkeypatch=monkeypatch)
    assert code == 1 and json.loads(out)["error"]["code"] == "invalid_request"
    code, out = run(capsys, "predict", stdin="not json", monkeypatch=monkeypatch)
    assert code == 1 and json.loads(out)["error"]["code"] == "invalid_request"
    request = '{"version": 1, "tasks": [{"id": "a", "prompt": "x", "models": ["m"]}]}'
    code, out = run(capsys, "predict", stdin=request, monkeypatch=monkeypatch)
    assert code == 1 and json.loads(out)["error"]["code"] == "no_predictor"


def test_refresh_adds_only_new_tasks(capsys, records, ready):
    run(capsys, "create", "--express", "--archive", f"claude-code={records}", *ONE_WINDOW)
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "Nothing new" in out

    session = ClaudeSession(records, "brand-new", START + 90 * 86400)
    for n in range(3):
        session.prompt(f"new task {n}")
        session.reply(5)
    session.save(mtime=START)
    code, out = run(capsys, "refresh")
    assert code == 2 and "--yes" in out
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "Found 3 new tasks" in out
    assert Predictor("default").count_tasks() == 183
    assert len(Predictor("default").embedded("prompt")) == 183


def test_setup_and_refresh_answer_a_program_in_json(capsys, few, ready, quick):
    # Without --yes, setup says what it would do and what it costs, and does nothing.
    code, out = run(capsys, "create", "--json", "--archive", f"claude-code={few}", *quick)
    answer = json.loads(out)  # standard output holds the answer alone: what setup says on the way went elsewhere
    assert code == 2 and answer["done"] is False and answer["predictor"] == "default" and answer["found_tasks"] == 40
    embedding = answer["cost"]["embedding"]
    assert embedding["texts"] > 0 and embedding["model"] == "voyage-4" and answer["cost"]["analysis"] is None
    assert answer["cost"]["usd"] == embedding["usd"] and predictor_names() == []
    code, out = run(capsys, "create", "--yes", "--json", "--archive", f"claude-code={few}", *quick)
    answer = json.loads(out)
    assert code == 0 and answer["done"] and answer["trained"] and answer["default"]
    assert answer["learned_from"] == 40 and answer["forecasts_for"] == ["claude-opus-5-5"]
    assert answer["embedded"] > 0 and answer["notes"] == []
    code, out = run(capsys, "refresh", "--json")
    answer = json.loads(out)
    assert code == 0 and answer["done"] and answer["up_to_date"] and not answer["trained"]
    assert answer["forecasts_for"] == ["claude-opus-5-5"]

    session = ClaudeSession(few, "brand-new", START + 90 * 86400)
    for n in range(3):
        session.prompt(f"new task {n}")
        session.reply(5)
    session.save(mtime=START)
    code, out = run(capsys, "refresh", "--json")
    answer = json.loads(out)
    assert code == 2 and not answer["done"] and answer["new_tasks"] == 3
    planned = answer["cost"]["embedding"]["texts"]  # their prompts, and the ends of the conversation before them
    assert planned >= 3 and Predictor("default").count_tasks() == 40  # nothing saved without --yes
    code, out = run(capsys, "refresh", "--yes", "--json")
    answer = json.loads(out)
    assert code == 0 and answer["done"] and answer["trained"] and answer["new_tasks"] == 3
    assert answer["embedded"] == planned
    assert Predictor("default").count_tasks() == 43

    code, out = run(capsys, "refresh", "--json", "--predictor", "nope")
    assert code == 1 and json.loads(out)["error"]["code"] == "unknown_predictor"
    code, out = run(capsys, "create", "default", "--json", "--yes", "--archive", f"claude-code={few}", *quick)
    assert code == 1 and json.loads(out)["error"]["code"] == "predictor_exists"
    code, out = run(capsys, "create", "--json", "--yes", "--archive", f"claude-code={few}", *quick)
    assert code == 1 and json.loads(out)["error"]["code"] == "invalid_request"  # a second one needs a name


def test_a_spending_limit_stops_before_anything_is_spent(capsys, few, ready, quick):
    setup = ["create", "--yes", "--json", "--archive", f"claude-code={few}", *quick]
    code, out = run(capsys, *setup, "--max-cost", "0")
    error = json.loads(out)["error"]
    assert code == 1 and error["code"] == "over_max_cost"
    assert "more than your limit of $0. Nothing was spent or saved. Raise --max-cost" in error["message"]
    assert predictor_names() == [] and len(ready.requests) == 1  # only the tiny key check was sent
    for bad in ("-1", "nan"):
        code, out = run(capsys, *setup, "--max-cost", bad)
        assert code == 1 and json.loads(out)["error"]["code"] == "invalid_request"
    code, out = run(capsys, *setup, "--max-cost", "1")
    answer = json.loads(out)
    assert code == 0 and answer["done"] and answer["max_cost"] == 1 and 0 < answer["spent"] <= 1

    session = ClaudeSession(few, "brand-new", START + 90 * 86400)
    for n in range(3):
        session.prompt(f"new task {n}")
        session.reply(5)
    session.save(mtime=START)
    sent = len(ready.requests)
    code, out = run(capsys, "refresh", "--yes", "--json", "--max-cost", "0")
    assert code == 1 and json.loads(out)["error"]["code"] == "over_max_cost"
    assert len(ready.requests) == sent and Predictor("default").count_tasks() == 40  # nothing spent or saved
    code, out = run(capsys, "refresh", "--yes", "--max-cost", "0")  # a person gets the same reason
    assert code == 1 and Predictor("default").count_tasks() == 40
    code, out = run(capsys, "refresh", "--yes", "--json", "--max-cost", "1")
    answer = json.loads(out)
    assert code == 0 and answer["trained"] and 0 < answer["spent"] <= 1 and Predictor("default").count_tasks() == 43


def test_llm_analysis_stops_at_the_spending_limit_and_training_goes_on(capsys, few, ready, quick, monkeypatch):
    from test_llm import answer_for, ids_in

    reads, writes = llm.TOKENS_PER_TASK

    def costly(model, system, text, shape, **limits):  # each task really costs twice the price shown
        ids = ids_in(text)
        return {
            **answer_for(ids),
            "_usage": {"model": model, "input": 2 * reads * len(ids), "output": 2 * writes * len(ids)},
        }

    monkeypatch.setattr(llm, "_anthropic", costly)
    per_task = llm.cost(1, "claude-opus-5-5")
    limit = 40 * per_task + 0.01  # the price shown fits, with room for the embedding
    flags = ["--features", "llm:kind", "--llm-provider", "anthropic", "--archive", f"claude-code={few}", *quick]
    code, out = run(capsys, "create", "--yes", "--json", "--max-cost", f"{limit:.4f}", *flags)
    answer = json.loads(out)
    assert code == 0 and answer["trained"] and answer["analyzed"] < 40
    assert answer["spent"] <= limit and any("stopped at your spending limit" in note for note in answer["notes"])
    assert answer["cost"]["analysis"]["usd"] == pytest.approx(40 * per_task)


def test_a_spending_limit_cant_be_kept_without_a_price(capsys, few, ready, quick):
    flags = ["--features", "llm:kind", "--llm-provider", "anthropic", "--llm-model", "claude-not-in-the-list"]
    code, out = run(
        capsys, "create", "--yes", "--json", "--max-cost", "5", "--archive", f"claude-code={few}", *flags, *quick
    )
    error = json.loads(out)["error"]
    assert code == 1 and error["code"] == "cost_unknown" and "claude-not-in-the-list" in error["message"]
    assert predictor_names() == []


def test_a_refresh_that_stopped_during_training_trains_again_next_time(capsys, records, ready, monkeypatch):
    trained: list[int] = []
    stop = {"now": False}

    class Stub:
        def __init__(self, tasks):
            self.meta = {"tasks": len(tasks), "report": {}}

        def save(self, folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "model.json").write_text(json.dumps(self.meta))

    def stub_train(tasks, rows, settings):
        if stop["now"]:
            raise KeyboardInterrupt  # the person pressed Ctrl+C while it trained
        trained.append(len(tasks))
        return Stub(tasks)

    monkeypatch.setitem(model_types._TYPES, "stub", (stub_train, lambda folder: None))
    code, out = run(capsys, "create", "--yes", "--archive", f"claude-code={records}", "--model-type", "stub")
    assert code == 0, out
    assert trained == [180]

    session = ClaudeSession(records, "brand-new", START + 90 * 86400)
    for n in range(3):
        session.prompt(f"new task {n}")
        session.reply(5)
    session.save(mtime=START)
    stop["now"] = True
    code, out = run(capsys, "refresh", "--yes")
    assert code == 130
    assert Predictor("default").count_tasks() == 183  # the new tasks were saved, but the model is the old one

    stop["now"] = False
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0, out
    assert "Nothing new" not in out and "Training didn't finish last time" in out
    assert trained == [180, 183]
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "Nothing new" in out
    assert trained == [180, 183]


def test_a_task_that_finishes_after_it_was_saved_as_stopped_is_embedded(tmp_path, capsys, ready, monkeypatch):
    class Stub:
        meta = {"tasks": 0, "report": {}}

        def save(self, folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "model.json").write_text(json.dumps(self.meta))

    monkeypatch.setitem(model_types._TYPES, "stub", (lambda tasks, rows, settings: Stub(), lambda folder: Stub()))
    runs = tmp_path / "codex-runs"
    session = CodexSession(runs, "c1", START)
    session.turn("t1", "Run the long build", calls=[(1000, 0, 100)], seconds=60)
    finished = session.lines.pop()  # the completion hasn't been written yet
    session.save(mtime=START)
    code, out = run(capsys, "create", "--yes", "--archive", f"codex={runs}", "--features", "", "--model-type", "stub")
    assert code == 0, out
    assert not Predictor("default").embedded("prompt")  # saved as stopped, so not learned from

    session.lines.append(finished)
    session.save(mtime=START + 3600)
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0, out
    assert Predictor("default").embedded("prompt") == {"codex:t1"}


def test_a_failed_embedding_run_keeps_what_it_did_and_resumes(capsys, records, ready, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_TEXTS_PER_REQUEST", 50)
    calls = {"n": 0}
    real_post = embeddings._post

    def flaky(body, key, timeout=120):
        calls["n"] += 1
        if calls["n"] == 4:
            raise embeddings.VoyageError(400, "Your free tokens are used up. Add a payment method.")
        return real_post(body, key, timeout)

    monkeypatch.setattr(embeddings, "_post", flaky)
    assert main(["create", "--yes", "--archive", f"claude-code={records}", "--features", "", *ONE_WINDOW]) == 1
    error = capsys.readouterr().err  # a problem that stops the command goes to the error stream
    assert "payment method" in error and "Run yardstick-predict refresh --predictor default to finish" in error
    predictor = Predictor("default")
    # The two requests that got through held 100 texts. A text several tasks share gives each its vector.
    assert 100 <= len(predictor.embedded("prompt")) < 180
    assert embeddings.account(KEY)["free_tokens"] == "used"

    monkeypatch.setattr(embeddings, "_post", real_post)
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0, out
    assert len(predictor.embedded("prompt")) == 180


# LLM analysis, and a model made by an older version -----------------------------------------------------


def fake_analysis(seen: list):
    def analyze(asks, how, fields, on_batch=None):
        seen.append(len(asks))
        found = {
            ask.id: {
                "kind": "feature",
                "files": 2.0,
                "steps": 2.0,
                "runs_code": True,
                "looks_up": False,
                "continues": False,
                "clarity": "clear",
                "minutes": 5.0,
                "description": f"Make the change the prompt asks for: {ask.prompt[:40]}",
                "by": f"{how['provider']}:{how['model']}",
                "fields": fields,
            }
            for ask in asks
        }
        if on_batch is not None:
            on_batch(found, len(asks))
        return llm.Analysis(found)

    return analyze


def test_llm_analysis_reads_each_task_once_and_its_descriptions_are_embedded(capsys, records, ready, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(seen))
    code, out = run(
        capsys,
        "create",
        "--yes",
        "--archive",
        f"claude-code={records}",
        "--features",
        "context,llm",
        "--llm-provider",
        "codex",
        *ONE_WINDOW,
    )
    assert code == 0, out
    predictor = Predictor("default")
    assert predictor.config["llm"] == {"provider": "codex", "model": llm.DEFAULT_CODEX}
    assert "LLM analysis: 180 tasks with Codex's own model through your chatgpt subscription" in out
    assert "Analyzed 180 tasks." in out
    assert len(predictor.llm_labels()) == 180 and len(predictor.embedded("llm")) == 180

    code, out = run(capsys, "predict", "--model", "claude-opus-5-5", "--prompt", "fix the api")
    assert code == 0 and json.loads(out)["forecasts"][0]["seconds"] > 0
    assert seen[-1] == 1  # the forecast's own task was read too

    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "Nothing new" in out and seen.count(180) == 1  # nothing was read twice


def test_llm_analysis_that_cant_run_says_why_and_training_goes_on(capsys, records, ready, monkeypatch):
    def broken(asks, how, fields, on_batch=None):
        return llm.Analysis({}, "No Anthropic API key was found. Set ANTHROPIC_API_KEY to your key.")

    monkeypatch.setattr(llm, "analyze", broken)
    code, out = run(capsys, "create", "--yes", "--archive", f"claude-code={records}", "--features", "llm", *ONE_WINDOW)
    assert code == 0 and "about $1.44 at list price" in out
    assert "ANTHROPIC_API_KEY" in out and "Training goes on without LLM analysis" in out
    assert Predictor("default").llm_labels() == {}  # nothing was noted, so a refresh tries them all again
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "LLM analysis: 180 tasks" in out


def test_a_model_made_by_a_version_before_the_previous_release_is_trained_again_on_refresh(capsys, records, ready):
    assert run(capsys, "create", "--yes", "--archive", f"claude-code={records}", "--features", "", *ONE_WINDOW)[0] == 0
    predictor = Predictor("default")
    path = predictor.model_dir / "model.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "format": model_types.PREVIOUS_FORMAT - 1}))
    assert pipeline.outdated(predictor) and not pipeline.from_previous_version(predictor)
    code, out = run(capsys, "status")
    assert code == 0 and "gives no forecasts until it's trained again" in out
    # Its settings are from before the conversation was a feature to choose, when every predictor read it.
    predictor.config["format"] = 1
    predictor.save_config()

    code, out = run(capsys, "predict", "--model", "claude-opus-5-5", "--prompt", "fix the api")
    assert code == 1  # a forecast from it would read the wrong features

    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "made by an older version" in out and not pipeline.outdated(predictor)
    assert Predictor("default").config["features"] == ["prompt", "conversation"]
    assert "session_tasks_before" in json.loads(path.read_text())["numbers"]  # so it still reads the conversation


# Quick stand-ins, for tests of the command itself ------------------------------------------------------


class QuickModel:
    """A stand-in for a trained model that trains at once. It forecasts 60 seconds on each AI model that passed."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta

    def served(self, model: str, effort: str | None = None):
        verdict = self.meta["report"]["models"].get(model)
        return (False, "unknown_model") if verdict is None else (verdict["served"], verdict["reason"])

    def forecast(self, rows):
        seconds = {"expected": 60.0, "distribution": None}
        return [{"shape": None, "direct_probability": None, "seconds": seconds, "tokens": None, "usd": None}] * len(
            rows
        )

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(json.dumps(self.meta))


def quick_meta(counts: dict[str, int], passed: bool = True) -> dict:
    verdict = {"served": passed, "reason": None if passed else "not_accurate_enough", "heads": ["seconds"]}
    models = {model: {**verdict, "tasks_tested": count, "failed_checks": []} for model, count in counts.items()}
    return {
        "tasks": sum(counts.values()),
        "report": {"tested_on": sum(counts.values()), "models": models},
        "categories": {"model": list(counts)},
        "defaults": {},
    }


def quick_train(tasks, rows, settings):
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.model] = counts.get(task.model, 0) + 1
    return QuickModel(quick_meta(counts))


@pytest.fixture
def quick(monkeypatch):
    """The quick model type, named on a command line with these flags."""
    load = lambda folder: QuickModel(json.loads((folder / "model.json").read_text()))  # noqa: E731
    monkeypatch.setitem(model_types._TYPES, "quick", (quick_train, load))
    return ["--model-type", "quick"]


@pytest.fixture
def few(tmp_path):
    """40 Claude Code tasks on one AI model, read in a moment."""
    root = tmp_path / "few-records"
    history(root, tasks=40, start=START)
    return root


def quick_predictor(name="default", passed=True, counts=None, plugins=()) -> Predictor:
    """A predictor with a quick model already saved, made without reading any records."""
    predictor = Predictor.create(
        name, [], {"model": "voyage-4", "dimensions": 256}, features=["prompt"], model={"type": "quick"}
    )
    predictor.config["plugins"] = list(plugins)
    predictor.save_config()
    QuickModel(quick_meta(counts or {"claude-opus-5-5": 40}, passed)).save(predictor.model_dir)
    return predictor


def fails_after_the_key_check(monkeypatch):
    """Voyage takes the key, then refuses to embed anything."""
    real_post = embeddings._post
    calls = {"n": 0}

    def flaky(body, key, timeout=120):
        calls["n"] += 1
        if calls["n"] > 1:
            raise embeddings.VoyageError(400, "Your free tokens are used up. Add a payment method.")
        return real_post(body, key, timeout)

    monkeypatch.setattr(embeddings, "_post", flaky)
    return real_post


class FragileModel(QuickModel):
    """A quick model saved in two files, so a test can take one away."""

    def save(self, folder):
        super().save(folder)
        (folder / "trees.json").write_text("{}")


def load_fragile(folder):
    json.loads((folder / "trees.json").read_text())  # FileNotFoundError when it's gone
    return FragileModel(json.loads((folder / "model.json").read_text()))


def test_a_damaged_model_says_so_and_a_refresh_trains_it_again(capsys, monkeypatch, few, ready):
    train = lambda tasks, rows, settings: FragileModel(quick_train(tasks, rows, settings).meta)  # noqa: E731
    monkeypatch.setitem(model_types._TYPES, "fragile", (train, load_fragile))
    assert main(["create", "--express", "--archive", f"claude-code={few}", "--model-type", "fragile"]) == 0
    capsys.readouterr()
    (Predictor("default").model_dir / "trees.json").unlink()
    code, out = run(
        capsys,
        "predict",
        stdin=json.dumps({"version": 1, "tasks": [{"id": "a", "prompt": "fix it", "models": ["claude-opus-5-5"]}]}),
        monkeypatch=monkeypatch,
    )
    error = json.loads(out)["error"]
    assert code == 1 and error["code"] == "not_trained" and "has a damaged model" in error["message"]
    code, out = run(capsys, "refresh", "--yes", "--json")
    answer = json.loads(out)
    assert code == 0 and answer["trained"] and answer["retrain_reason"] == "damaged" and answer["new_tasks"] == 0
    assert (Predictor("default").model_dir / "trees.json").is_file()  # made again


class Terminal(io.StringIO):
    """Standard input as a person's terminal."""

    def isatty(self) -> bool:
        return True


# What a program sees ---------------------------------------------------------------------------------------


def test_predict_never_prints_a_traceback_to_a_program(capsys, monkeypatch, quick, ready):
    predictor = quick_predictor(plugins=["a_plugin_that_isnt_installed"])
    request = json.dumps({"version": 1, "tasks": [{"id": "a", "prompt": "fix it", "models": ["claude-opus-5-5"]}]})
    answers = json.loads((SCHEMAS / "forecast-response.v1.json").read_text())

    def error_code() -> tuple[int, str]:
        code, out = run(capsys, "predict", stdin=request, monkeypatch=monkeypatch)
        found = json.loads(out)  # nothing but the answer on standard output
        jsonschema.validate(found, answers)
        return code, found["error"]["code"]

    assert error_code() == (1, "plugin_failed")
    predictor.config["plugins"] = []
    predictor.save_config()
    monkeypatch.setattr(forecast, "answer", lambda *args, **kwargs: 1 / 0)  # anything unexpected
    assert error_code() == (1, "internal_error")
    (predictor.dir / "predictor.json").write_text("")  # an older version stopped halfway through saving it
    assert error_code() == (1, "broken_predictor")


def test_command_line_errors_reach_a_program_as_json(capsys, monkeypatch):
    assert main(["predict", "--bogus"]) == 1  # standard output isn't a terminal here, as for a program
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "invalid_request" and "--bogus" in error["message"]
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    assert main(["predict", "--bogus"]) == 2  # a person gets the usage line
    assert "usage: yardstick-predict" in capsys.readouterr().err

    monkeypatch.setattr("sys.stdin", Terminal())
    assert main(["predict", "--json"]) == 1  # no task, and no request either
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_request"
    assert main(["status", "--json", "--predictor", "nope"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "unknown_predictor"


def test_an_unexpected_error_is_said_in_a_line_and_its_details_are_kept(capsys, monkeypatch):
    from yardstick_predict import store

    def broken(args):
        raise RuntimeError("boom \x1b]0;a title\x07today")

    monkeypatch.setattr(cli, "cmd_list", broken)
    assert main(["list"]) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err and captured.out == ""  # no page of Python
    said = " ".join(captured.err.split())
    assert "Something went wrong that yardstick-predict didn't expect: RuntimeError: boom today" in said
    kept = store.home() / cli.CRASH_FILE
    assert f"The details are in {kept}. They may hold text from your work" in said
    assert f"Please report it at {cli.ISSUES}" in said
    details = kept.read_text()
    assert "Traceback" in details and "RuntimeError" in details and "Command: list" in details
    assert main(["list", "--json"]) == 1  # a program gets it in JSON
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "internal_error" and error["message"].endswith("RuntimeError: boom today")
    monkeypatch.setenv(cli.DEBUG, "1")
    assert main(["list"]) == 1
    assert "Traceback" in capsys.readouterr().err  # the details where it happens, on request


def test_output_no_one_reads_anymore_ends_quietly(capsys, monkeypatch):
    def piped(args):
        raise BrokenPipeError

    monkeypatch.setattr(cli, "cmd_list", piped)
    assert main(["list"]) == 141
    assert capsys.readouterr().err == ""


def test_predict_reads_a_request_from_a_file(capsys, monkeypatch, tmp_path, quick, ready):
    quick_predictor()
    request = {"version": 1, "tasks": [{"id": "a", "prompt": "fix it", "models": ["claude-opus-5-5"]}]}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(cli, "terminal_output", lambda: True)  # even in a terminal, a request gets JSON back
    monkeypatch.setattr("sys.stdin", Terminal())
    assert main(["predict", "--request", str(path)]) == 0
    forecast = json.loads(capsys.readouterr().out)["forecasts"][0]
    assert forecast["task"] == "a" and forecast["seconds"] == 60.0
    assert main(["predict", "--request", str(tmp_path / "missing.json")]) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "invalid_request" and "Can't read the request file" in error["message"]
    path.write_text("{not json")
    assert main(["predict", "--request", str(path)]) == 1
    assert "isn't valid JSON" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert main(["predict", "fix it", "--request", str(path)]) == 1  # the task in quotes, or a request: not both
    assert "not both" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    assert main(["predict", "--request", "-"]) == 0  # - is standard input
    assert json.loads(capsys.readouterr().out)["forecasts"][0]["task"] == "a"


def test_choosing_a_predictor_has_its_own_error_codes(capsys, monkeypatch):
    request = {"version": 1, "predictor": "work", "tasks": [{"id": "a", "prompt": "fix it", "models": ["m"]}]}
    code, out = run(capsys, "predict", stdin=json.dumps(request), monkeypatch=monkeypatch)
    assert json.loads(out)["error"]["code"] == "no_predictor"  # none exists, though the request names one

    for name in ("home", "work"):
        Predictor.create(name, [], {"model": "voyage-4", "dimensions": 256})
    request.pop("predictor")
    code, out = run(capsys, "predict", stdin=json.dumps(request), monkeypatch=monkeypatch)
    error = json.loads(out)["error"]
    assert code == 1 and error["code"] == "no_default_predictor" and 'request\'s "predictor"' in error["message"]
    code, out = run(capsys, "predict", "fix it", "--json")
    assert "--predictor" in json.loads(out)["error"]["message"]  # a person names one on the command line


def test_a_plain_predict_with_no_model_passed_gives_each_known_model_its_reason(capsys, quick, ready):
    quick_predictor(passed=False, counts={"claude-opus-5-5": 50, "gpt-6.1-sol": 40})
    code, out = run(capsys, "predict", "fix it", "--json")
    forecasts = [(f["model"], f["reason"]) for f in json.loads(out)["forecasts"]]
    assert code == 0 and forecasts == [
        ("claude-opus-5-5", "not_accurate_enough"),
        ("gpt-6.1-sol", "not_accurate_enough"),
    ]
    assert ready.requests == []  # nothing to forecast, so nothing went to Voyage


def test_predict_loads_the_model_once_and_forecasts_a_model_named_twice_once(capsys, monkeypatch, quick, ready):
    from yardstick_predict import pipeline

    quick_predictor()
    calls = {"meta": 0, "load": 0}
    real_meta, real_load = pipeline.model_meta, pipeline.load_model

    def counted(name, real):
        def call(predictor):
            calls[name] += 1
            return real(predictor)

        return call

    monkeypatch.setattr(pipeline, "model_meta", counted("meta", real_meta))
    monkeypatch.setattr(pipeline, "load_model", counted("load", real_load))
    code, out = run(capsys, "predict", "fix it")
    assert code == 0 and calls == {"meta": 0, "load": 1}  # its meta picks the AI models; it isn't read again
    code, out = run(capsys, "predict", "fix it", "--model", "claude-opus-5-5", "--model", "claude-opus-5-5")
    assert len(json.loads(out)["forecasts"]) == 1


def test_status_reads_only_what_the_saved_model_says_about_itself(capsys, monkeypatch, quick):
    quick_predictor(plugins=["a_plugin_that_isnt_installed"])  # status doesn't need the plugins
    monkeypatch.setattr("yardstick_predict.pipeline.load_model", lambda predictor: pytest.fail("loaded the model"))
    assert main(["status"]) == 0
    assert "Trained on 40 tasks" in capsys.readouterr().out
    assert main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["predictors"][0]["report"]["tested_on"] == 40


def test_the_missing_key_error_goes_to_the_error_stream(capsys, few):
    assert main(["create", "--yes", "--archive", f"claude-code={few}"]) == 1
    captured = capsys.readouterr()
    assert "No Voyage key found" in captured.err and "No Voyage key found" not in captured.out
    with pytest.raises(cli.CommandError) as stopped:
        cli.get_key("voyage-4", can_ask=False)
    assert stopped.value.reason == "no_embedding_key"  # a program that asked for JSON gets this code


# Saved state -----------------------------------------------------------------------------------------------


def test_an_empty_predictor_file_gets_a_plain_reason_not_a_traceback(capsys):
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    (predictor.dir / "predictor.json").write_text("")
    assert main(["refresh", "--yes"]) == 1
    error = capsys.readouterr().err
    assert "can't be used" in error and "empty or broken" in error and str(predictor.dir) in error
    assert main(["status", "--json"]) == 1
    entry = json.loads(capsys.readouterr().out)["predictors"][0]
    assert entry["name"] == "default" and "empty or broken" in entry["error"]
    assert main([]) == 0 and "empty or broken" in capsys.readouterr().out  # the list a program gets


def test_a_predictor_saved_before_the_conversation_was_a_feature_keeps_it():
    from yardstick_predict import store

    embedding = {"model": "voyage-4", "dimensions": 256}
    old = Predictor.create("old", [], embedding, features=["prompt", "context", "llm"])
    old.config["format"] = 1  # as every version before saved its settings, when every predictor read it
    old.save_config()
    Predictor.create("new", [], embedding, features=["prompt", "context"])
    again = Predictor("old")
    assert again.config["features"] == ["prompt", "conversation", "context", "llm"]
    assert pipeline.chosen_features(again) == ["prompt", "conversation", "context", "llm"]
    assert Predictor("new").config["features"] == ["prompt", "context"]  # chosen without it, so it stays out
    again.save_config()
    saved = json.loads((again.dir / "predictor.json").read_text())
    assert saved["format"] == store.FORMAT and saved["features"] == again.config["features"]


def test_settings_are_saved_in_one_step(monkeypatch):
    from yardstick_predict import files, pipeline, store

    monkeypatch.setattr(benchmarks, "download", lambda timeout=30: {"m": {"benchmark_price_input": 0.7}})
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256}, features=["benchmarks"])
    store.set_default("default")
    assert pipeline.refresh_benchmarks(predictor) is None

    def stopped(source, target):
        raise OSError("the computer stopped in the middle of a save")

    monkeypatch.setattr(files, "_replace", stopped)
    monkeypatch.setattr(benchmarks, "download", lambda timeout=30: {"m": {"benchmark_price_input": 9.9}})
    predictor.config["refreshed_at"] = 1_780_000_000.0
    for save in (
        predictor.save_config,
        lambda: store.set_default("other"),
        lambda: pipeline.refresh_benchmarks(predictor),
    ):
        with pytest.raises(OSError):
            save()
    # Each file is still the whole of what it was before.
    assert Predictor("default").config["refreshed_at"] is None
    assert json.loads((store.home() / "settings.json").read_text()) == {"default": "default"}
    assert pipeline.benchmark_table(predictor) == {"m": {"benchmark_price_input": 0.7}}


def test_a_touched_record_file_is_read_again_only_once(capsys, monkeypatch, few, ready, quick):
    import os

    from yardstick_predict import pipeline

    assert main(["create", "--yes", *quick, "--features", "", "--archive", f"claude-code={few}"]) == 0
    touched = sorted(few.rglob("*.jsonl"))[0]
    os.utime(touched, (START + 5, START + 5))  # a new time, and nothing new in it
    reads = []
    real = pipeline.read_records

    def read(archive, files, now=None, on_read=None):
        reads.append(len(files))
        return real(archive, files, now, on_read)

    monkeypatch.setattr(pipeline, "read_records", read)
    for _ in range(2):
        assert main(["refresh", "--yes"]) == 0
    assert reads == [1]


def test_the_effort_levels_offered_come_from_the_tasks_the_model_learns_from(tmp_path):
    from collections import Counter

    from yardstick_predict.pipeline import learnable, read_all, save_scan

    session = ClaudeSession(tmp_path, "s1", START)
    for n, effort in enumerate(["max", "max", "high", "high", "high"]):
        session.prompt(f"task {n}")
        session.reply(5, effort=effort)
        session.clock += 60
    session.prompt("stopped at max")
    session.reply(5, effort="max", tool="Bash")
    session.interrupt()
    session.save(mtime=START)
    archive = {"name": "claude-code", "format": "claude-code", "path": str(tmp_path)}
    predictor = Predictor.create("default", [archive], {"model": "voyage-4", "dimensions": 256})
    save_scan(predictor, read_all([archive], now=START + 30 * 86400))

    def counted() -> dict:
        return {(model, effort): n for model, levels in predictor.effort_counts().items() for effort, n in levels}

    assert counted() == Counter((t.model, t.effort) for t in learnable(predictor.tasks()))
    predictor.config["max_tasks"] = 3
    assert counted() == Counter((t.model, t.effort) for t in learnable(predictor.tasks(), 3))
    # A model counts the levels of the tasks it was trained on, and those win.
    meta = {"effort_counts": {"m": [["high", 40], ["max", 5]]}}
    assert cli.effort_pairs(predictor, ["m"], None, meta) == [("m", "high")]
    assert cli.effort_pairs(predictor, ["m"], "max", meta) == [("m", "max")]


def test_on_windows_the_folder_is_made_private_once_and_never_stops_a_command(tmp_path, monkeypatch):
    from yardstick_predict import store

    monkeypatch.setenv("USERNAME", "sam")
    monkeypatch.setenv("USERDOMAIN", "LAPTOP")
    calls = []

    def icacls(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    assert store.private_on_windows(tmp_path, run=icacls)
    assert calls == [["icacls", str(tmp_path), "/inheritance:r", "/grant:r", "LAPTOP\\sam:(OI)(CI)F"]]
    assert store.private_on_windows(tmp_path, run=icacls) and len(calls) == 1  # once is enough

    other = tmp_path / "other"
    other.mkdir()
    assert not store.private_on_windows(other, run=lambda command, **options: SimpleNamespace(returncode=5))

    def missing(command, **options):
        raise FileNotFoundError("icacls")

    assert not store.private_on_windows(other, run=missing) and not (other / ".private").exists()


def test_predictor_names_that_cant_be_folders_are_refused(capsys):
    from yardstick_predict.store import name_problem

    assert name_problem("work") is None
    assert name_problem("work\n") and name_problem("con") and name_problem("lpt1")
    assert main(["create", "nul", "--yes", "--archive", "codex=/nowhere"]) == 1
    assert "Windows keeps the name nul" in capsys.readouterr().err
    with pytest.raises(ValueError):
        Predictor.create("work\n", [], {"model": "voyage-4", "dimensions": 256})
    assert predictor_names() == []


# Setup and refresh -----------------------------------------------------------------------------------------


def test_bad_numbers_are_refused_before_anything_is_read(capsys, ready):
    for flags, words in (
        (["--half-life", "-5"], "--half-life is a number of days: 0 or more"),
        (["--half-life", "nan"], "--half-life must be a number"),
        (["--half-life", "inf"], "--half-life must be a number"),
        (["--dropout", "nan"], "--dropout must be a number"),
    ):
        assert main(["create", "--yes", *flags, "--archive", "codex=/nowhere"]) == 1
        assert words in capsys.readouterr().err
    assert ready.requests == [] and predictor_names() == []


def test_the_help_names_the_default_features(capsys):
    with pytest.raises(SystemExit):
        main(["create", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "Default: conversation and context." in text
    assert "the prompt with conversation and context, voyage-4" in text


def test_yes_in_a_terminal_asks_nothing(capsys, monkeypatch, few, ready, quick):
    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda question="": pytest.fail(f"It asked: {question}"))
    monkeypatch.setattr(cli.getpass, "getpass", lambda question="": pytest.fail("It asked for a key"))
    assert main(["create", "--yes", *quick, "--archive", f"claude-code={few}"]) == 0
    assert Predictor("default").config["features"] == ["prompt", "conversation", "context"]  # the defaults
    session = ClaudeSession(few, "more", START + 90 * 86400)
    session.prompt("one more task")
    session.reply(5)
    session.save(mtime=START)
    assert main(["refresh", "--yes"]) == 0
    assert "Found 1 new tasks" in capsys.readouterr().out


def test_after_a_failed_setup_the_advice_names_the_predictor(capsys, monkeypatch, few, ready, quick):
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    fails_after_the_key_check(monkeypatch)
    assert main(["create", "work", "--yes", *quick, "--features", "", "--archive", f"claude-code={few}"]) == 1
    assert "Run yardstick-predict refresh --predictor work to finish" in capsys.readouterr().err
    assert main(["create", "work", "--yes", "--archive", f"claude-code={few}"]) == 1
    assert "Refresh it with: yardstick-predict refresh --predictor work" in capsys.readouterr().err


def test_the_first_predictor_is_the_default_even_when_its_setup_stops(capsys, monkeypatch, tmp_path, few, ready, quick):
    real_post = fails_after_the_key_check(monkeypatch)
    assert main(["create", "work", "--yes", *quick, "--features", "", "--archive", f"claude-code={few}"]) == 1
    assert default_name() == "work"
    monkeypatch.setattr(embeddings, "_post", real_post)
    assert main(["refresh", "--yes"]) == 0

    # A later predictor you ask to make your default becomes it once it's trained, not while it's half made.
    monkeypatch.setattr(cli, "interactive", lambda: True)
    replies = {"Make it your default": "y", "Embed them": "y"}
    monkeypatch.setattr(
        "builtins.input", lambda question="": next((a for k, a in replies.items() if k in question), "")
    )
    fails_after_the_key_check(monkeypatch)
    other = tmp_path / "other-records"  # other prompts, so they aren't in the cache of vectors already
    history(other, tasks=40, start=START, seed=1)
    flags = ["--embedding", "voyage-4", "--features", "", "--archive", f"claude-code={other}"]
    assert main(["create", "lab", *quick, *flags]) == 1
    assert default_name() == "work"
    monkeypatch.setattr(embeddings, "_post", real_post)
    assert main(["refresh", "--predictor", "lab", "--yes"]) == 0
    assert default_name() == "lab" and "default_when_ready" not in Predictor("lab").config


def test_setup_holds_its_lock_before_the_predictor_exists(monkeypatch, few, ready, quick):
    from yardstick_predict import locks, store

    real_create = Predictor.create.__func__
    seen = {}

    def create(cls, name, archives, embedding, **settings):
        with locks.held(store.predictor_folder(name) / "refresh.lock", wait=False) as free:
            seen["a refresh could start"] = free
        predictor = real_create(cls, name, archives, embedding, **settings)
        seen["features"] = json.loads((predictor.dir / "predictor.json").read_text()).get("features")
        return predictor

    monkeypatch.setattr(Predictor, "create", classmethod(create))
    assert main(["create", "--yes", *quick, "--features", "", "--archive", f"claude-code={few}"]) == 0
    assert seen == {"a refresh could start": False, "features": ["prompt"]}  # and it finds the whole config


def test_setup_and_refresh_count_each_text_once(capsys, monkeypatch, few, ready, quick):
    counted = []

    def count_tokens(texts, model):
        counted.append(model)
        return [len(text.split()) + 1 for text in texts], True

    monkeypatch.setattr(embeddings, "count_tokens", count_tokens)
    monkeypatch.setattr(cli, "interactive", lambda: True)
    replies = {"Which Voyage model": "1", "Embed them": "y"}
    monkeypatch.setattr(
        "builtins.input", lambda question="": next((a for k, a in replies.items() if k in question), "")
    )
    assert main(["create", "work", *quick, "--features", "", "--archive", f"claude-code={few}"]) == 0
    assert sorted(counted) == sorted(embeddings.MODELS)  # each Voyage model once, for the menu, then reused

    counted.clear()
    session = ClaudeSession(few, "more", START + 90 * 86400)
    session.prompt("one more task")
    session.reply(5)
    session.save(mtime=START)
    monkeypatch.setattr(cli, "interactive", lambda: False)
    assert main(["refresh", "--yes"]) == 0
    assert counted == ["voyage-4"]


def test_a_plugin_in_the_current_folder_loads_from_any_folder_later(capsys, monkeypatch, tmp_path, few, ready, quick):
    import sys

    from yardstick_predict import features

    project = tmp_path / "project"
    project.mkdir()
    (project / "my_features_for_tests.py").write_text(
        "from yardstick_predict.features import register_feature\n"
        "register_feature('words_for_tests', 'number', lambda info: len(info.prompt.split()))\n"
    )
    monkeypatch.setattr(features, "_CUSTOM", {})
    monkeypatch.chdir(project)
    try:
        flags = ["--plugin", "my_features_for_tests", "--features", "words_for_tests"]
        assert main(["create", "--yes", *quick, *flags, "--archive", f"claude-code={few}"]) == 0
        saved = (project / "my_features_for_tests.py").resolve()
        assert Predictor("default").config["plugins"] == [str(saved)]

        # Later, from another folder, as the status line's forecasts run.
        monkeypatch.chdir(tmp_path)
        del sys.modules["my_features_for_tests"]
        monkeypatch.setattr(features, "_CUSTOM", {})
        assert main(["refresh", "--yes"]) == 0
        assert "words_for_tests" in features.available()
    finally:
        sys.modules.pop("my_features_for_tests", None)

    assert model_types.find_plugin("project/my_features_for_tests.py") == str(saved)  # a path works too
    assert model_types.find_plugin("json") == "json"  # a module Python can import keeps its name
    (project / "json.py").write_text("")
    with pytest.raises(model_types.PluginError, match="already has a module called json"):
        model_types.load_plugins([model_types.find_plugin("project/json.py")])
    with pytest.raises(model_types.PluginError, match="no file"):
        model_types.find_plugin("missing.py")


def test_update_all_says_a_predictor_is_busy_and_goes_on(capsys, monkeypatch):
    from yardstick_predict import locks
    from yardstick_predict.screens import keys, update

    for name in ("busy", "free"):
        Predictor.create(name, [], {"model": "voyage-4", "dimensions": 256})
    checked = []
    monkeypatch.setattr(update, "update", lambda work: checked.extend(job.predictor.name for job in work) or 0)
    pressed = ["2", "1", keys.ESC]  # Update, Update all, then quit
    monkeypatch.setattr(keys, "reader", lambda: lambda: pressed.pop(0))
    monkeypatch.setattr(cli, "interactive", lambda: True)
    with locks.held(Predictor("busy").dir / "refresh.lock"):
        assert main([]) == 0
    assert checked == ["free"]
    assert "busy is being set up or updated by another command" in capsys.readouterr().out


def test_ctrl_d_at_a_question_stops_cleanly(capsys, monkeypatch):
    def end_of_input(question=""):
        raise EOFError

    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", end_of_input)
    assert main([]) == 130
    captured = capsys.readouterr()
    assert "Stopped." in captured.err and "Stopped." not in captured.out and "Traceback" not in captured.err


def test_a_model_saved_by_a_newer_version_asks_for_an_upgrade(capsys, monkeypatch):
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.model_dir.mkdir()
    (predictor.model_dir / "model.json").write_text(json.dumps({"format": 99, "report": {}}))
    code, out = run(capsys, "predict", "--model", "m", "--prompt", "fix it")
    error = json.loads(out)["error"]
    assert code == 1 and error["code"] == "model_too_new" and "uv tool upgrade yardstick-predict" in error["message"]
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    assert main(["predict", "fix it", "--model", "m"]) == 1  # a person gets the same words, without JSON
    assert "uv tool upgrade yardstick-predict" in capsys.readouterr().err


def test_a_damaged_record_of_voyage_usage_is_told_once_where_cost_is_shown(capsys, few, ready, quick):
    from yardstick_predict import store

    assert main(["create", "--yes", *quick, "--features", "", "--archive", f"claude-code={few}"]) == 0
    (store.home() / "voyage-usage.json").write_text("{not json")
    told = []
    for n in range(2):
        session = ClaudeSession(few, f"more-{n}", START + (90 + n) * 86400)
        session.prompt(f"another new task {n}")
        session.reply(5)
        session.save(mtime=START)
        assert main(["refresh", "--yes"]) == 0
        told.append("Your record of Voyage usage was damaged" in capsys.readouterr().out)
    assert told == [True, False]
    assert list(store.home().glob("voyage-usage.damaged-*.json"))  # where the note says it is


def test_a_rate_limit_shows_its_reason_with_the_advice():
    alone = "You have exceeded your TPM rate limit. One text alone is more than this account allows in a minute."
    said = cli.explain(embeddings.VoyageError(429, alone))
    assert alone in said and "add a payment method to raise your limits" in said
    assert "used up for now." in cli.explain(
        embeddings.VoyageError(429, "this account's tokens a minute are used up for now")
    )
    plain = "Voyage kept limiting requests. Try again later, or add a payment method to raise your limits."
    assert cli.explain(embeddings.VoyageError(429, "")) == plain


def test_a_refresh_stops_using_the_vector_of_a_text_that_became_empty_and_pays_nothing(capsys, few, ready, quick):
    assert main(["create", "--yes", *quick, "--archive", f"claude-code={few}"]) == 0
    predictor = Predictor("default")
    emptied = sorted(predictor.embedded("context"))[0]
    predictor.db.execute("UPDATE tasks SET context = '' WHERE id = ?", (emptied,))
    predictor.db.commit()
    predictor.config["needs_training"] = True  # so the refresh trains again
    predictor.save_config()
    capsys.readouterr()
    sent = len(ready.requests)
    assert main(["refresh", "--yes"]) == 0
    assert len(ready.requests) == sent and " uses " not in capsys.readouterr().out  # no cost line, nothing paid
    # Its vector stays in the feature store, for a predictor whose task still has that text. Training here
    # finds it made from another text, so it leaves it out.
    predictor = Predictor("default")
    task = next(task for task in predictor.tasks() if task.id == emptied)
    assert "context" not in pipeline.feature_rows(predictor, [task])[0].vectors


def test_a_refresh_takes_out_tasks_their_records_no_longer_hold(capsys, tmp_path, ready, quick):
    root = tmp_path / "records"
    session = ClaudeSession(root, "s1", START)
    for n in range(3):
        session.prompt(f"task {n}")
        session.reply(5)
        session.clock += 60
    session.save(mtime=START)
    assert main(["create", "--yes", *quick, "--features", "", "--archive", f"claude-code={root}"]) == 0
    assert Predictor("default").count_tasks() == 3
    session.lines = session.lines[:-2]  # the last task is gone from the record
    session.save(mtime=START + 60)
    capsys.readouterr()
    assert main(["refresh", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Nothing new" not in out and "taken out" in out
    assert Predictor("default").count_tasks() == 2


def test_a_predictor_read_by_an_older_reader_reads_its_records_again_once(capsys, tmp_path, ready, quick):
    from yardstick_predict.archives import READER_VERSION

    root = tmp_path / "records"
    session = ClaudeSession(root, "s1", START)
    for n in range(3):
        session.prompt(f"task {n}")
        session.reply(5)
        session.clock += 60
    session.save(mtime=START)
    assert main(["create", "--yes", *quick, "--features", "", "--archive", f"claude-code={root}"]) == 0
    predictor = Predictor("default")
    predictor.config["reader"] = READER_VERSION - 1  # as an older version left it
    predictor.save_config()
    capsys.readouterr()
    assert main(["refresh", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "read them all again" in out and "grew" not in out
    assert Predictor("default").config["reader"] == READER_VERSION
    assert main(["refresh", "--yes"]) == 0
    assert "read them all again" not in capsys.readouterr().out  # only once


def test_the_report_says_how_many_tasks_of_its_own_each_ai_model_needs(capsys):
    cli.print_report({"report": {"tested_on": 0, "min_tasks_to_test": 60, "models": {}}})
    assert "Each AI model needs 60 tasks of its own" in " ".join(capsys.readouterr().out.split())
    few = {"served": False, "reason": "not_enough_data", "tasks_tested": 0, "tasks": 12, "failed_checks": []}
    cli.print_report({"report": {"tested_on": 40, "min_tasks_to_test": 60, "models": {"claude-opus-5-5": few}}})
    out = capsys.readouterr().out
    squashed = " ".join(out.split())
    assert "How close its forecasts came on each AI model's newest tasks, 40 in all" in squashed
    assert "not enough data, 12 tasks, needs 60" in squashed or "not enough data · 12 tasks, needs 60" in squashed
    assert "test" not in out
