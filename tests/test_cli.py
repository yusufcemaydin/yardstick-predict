from __future__ import annotations

import io
import json
import re
from pathlib import Path

import jsonschema
import pytest
from builders import ClaudeSession, CodexSession, history
from conftest import KEY

from yardstick_predict import benchmarks, cli, embeddings, forecast, llm, model_types, pipeline
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor, default_name, predictor_names

START = 1_780_000_000.0
SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


@pytest.fixture
def records(tmp_path):
    root = tmp_path / "claude-projects"
    history(root, tasks=420, start=START, models=("claude-opus-5-5", "claude-sonnet-5-5"), efforts=("high", "max"))
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
    assert "420 tasks" in out
    assert re.search(r"claude-opus-5-5\s+forecasts on", out)
    assert "Forecasts are on for claude-opus-5-5" in out and 'yardstick-predict predict "Add a login page' in out
    assert default_name() == "default"  # the first predictor is the default
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt", "context"]
    assert predictor.config["embedding"]["model"] == "voyage-4"
    assert predictor.config["model"]["type"] == "xgboost"
    assert len(predictor.embedded("prompt")) == 420

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
    row = r"\s+\d+ (s|min|h)\s+[\d.]+[kM]? tokens\s+\$"
    assert code == 0 and re.search(r"claude-opus-5-5\s+high" + row, out) and re.search(r"\n\s+max" + row, out)
    texts = [text for body in ready.requests[sent:] for text in body["input"]]
    assert texts.count("fix the api") == 1  # one prompt at several effort levels is embedded once
    assert "predictor default, updated just now" in out and "{" not in out
    monkeypatch.setattr(cli, "terminal_output", lambda: False)

    code, out = run(capsys, "status", "--json")
    status = json.loads(out)["predictors"][0]
    assert status["tasks"] == 420 and status["features"] == ["prompt", "context"]
    assert status["report"]["windows"] == 3


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
    plan = EmbedPlan(items, [10, 10, 10, 10], embeddings.Cost(40, True, 0.06, 30))
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

    code, out = run(capsys, "create", "--archive", f"claude-code={records}", "--yes")
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

    def answer(predictor, tasks, distribution):
        asked.extend(tasks)
        return {"forecasts": []}

    monkeypatch.setattr(forecast, "answer", answer)
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

    from yardstick_predict.pipeline import model_ready

    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert not model_ready(predictor)  # no model, and none on its way: no waiting
    (predictor.dir / ".model-new").mkdir()  # a save has moved the old model aside

    def arrive():
        time.sleep(0.2)
        predictor.model_dir.mkdir()
        (predictor.model_dir / "model.json").write_text("{}")

    threading.Thread(target=arrive).start()
    started = time.time()
    assert model_ready(predictor) and time.time() - started >= 0.15


def test_bad_choices_get_a_plain_reason(capsys, records, ready):
    assert main(["create", "--yes", "--archive", f"claude-code={records}", "--features", "horoscope"]) == 1
    assert "Unknown feature" in capsys.readouterr().err
    assert main(["create", "--yes", "--archive", f"claude-code={records}", "--shape", "weibull", "--loss", "crps"]) == 1
    assert "crps" in capsys.readouterr().err


def test_a_new_predictor_is_named_first(monkeypatch):
    replies = iter(["", "Not A Name", "default", "work"])
    monkeypatch.setattr("builtins.input", lambda question="": next(replies))
    assert cli.ask_name([]) == "default"  # the first predictor is called default unless you pick a name
    assert cli.ask_name(["default"]) == "work"  # an empty, a bad and a taken name are each asked again


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
    run(capsys, "create", "--express", "--archive", f"claude-code={records}")
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
    run(capsys, "create", "--express", "--archive", f"claude-code={records}")
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
    assert Predictor("default").count_tasks() == 423
    assert len(Predictor("default").embedded("prompt")) == 423


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
    assert trained == [420]

    session = ClaudeSession(records, "brand-new", START + 90 * 86400)
    for n in range(3):
        session.prompt(f"new task {n}")
        session.reply(5)
    session.save(mtime=START)
    stop["now"] = True
    code, out = run(capsys, "refresh", "--yes")
    assert code == 130
    assert Predictor("default").count_tasks() == 423  # the new tasks were saved, but the model is the old one

    stop["now"] = False
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0, out
    assert "Nothing new" not in out and "Training didn't finish last time" in out
    assert trained == [420, 423]
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "Nothing new" in out
    assert trained == [420, 423]


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
    monkeypatch.setattr(embeddings, "MAX_TEXTS_PER_REQUEST", 100)
    calls = {"n": 0}
    real_post = embeddings._post

    def flaky(body, key, timeout=120):
        calls["n"] += 1
        if calls["n"] == 4:
            raise embeddings.VoyageError(400, "Your free tokens are used up. Add a payment method.")
        return real_post(body, key, timeout)

    monkeypatch.setattr(embeddings, "_post", flaky)
    code, out = run(capsys, "create", "--yes", "--archive", f"claude-code={records}", "--features", "")
    assert code == 1
    assert "payment method" in out and "refresh" in out
    predictor = Predictor("default")
    assert len(predictor.embedded("prompt")) == 200
    assert embeddings.account(KEY)["free_tokens"] == "used"

    monkeypatch.setattr(embeddings, "_post", real_post)
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0, out
    assert len(predictor.embedded("prompt")) == 420


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
    )
    assert code == 0, out
    predictor = Predictor("default")
    assert predictor.config["llm"] == {"provider": "codex", "model": llm.DEFAULT_CODEX}
    assert "LLM analysis: 420 tasks with Codex's own model through your chatgpt subscription" in out
    assert len(predictor.llm_labels()) == 420 and len(predictor.embedded("llm")) == 420

    code, out = run(capsys, "predict", "--model", "claude-opus-5-5", "--prompt", "fix the api")
    assert code == 0 and json.loads(out)["forecasts"][0]["seconds"] > 0
    assert seen[-1] == 1  # the forecast's own task was read too

    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "Nothing new" in out and seen.count(420) == 1  # nothing was read twice


def test_llm_analysis_that_cant_run_says_why_and_training_goes_on(capsys, records, ready, monkeypatch):
    def broken(asks, how, fields, on_batch=None):
        return llm.Analysis({}, "No Anthropic API key was found. Set ANTHROPIC_API_KEY to your key.")

    monkeypatch.setattr(llm, "analyze", broken)
    code, out = run(capsys, "create", "--yes", "--archive", f"claude-code={records}", "--features", "llm")
    assert code == 0 and "about $3.36 at list price" in out
    assert "ANTHROPIC_API_KEY" in out and "Training goes on without LLM analysis" in out
    assert Predictor("default").llm_labels() == {}  # nothing was noted, so a refresh tries them all again
    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "LLM analysis: 420 tasks" in out


def test_a_model_made_by_an_older_version_is_trained_again_on_refresh(capsys, records, ready):
    assert run(capsys, "create", "--yes", "--archive", f"claude-code={records}", "--features", "")[0] == 0
    predictor = Predictor("default")
    path = predictor.model_dir / "model.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "format": 2}))
    assert pipeline.outdated(predictor)

    code, out = run(capsys, "predict", "--model", "claude-opus-5-5", "--prompt", "fix the api")
    assert code == 1  # a forecast from it would read the wrong features

    code, out = run(capsys, "refresh", "--yes")
    assert code == 0 and "made by an older version" in out and not pipeline.outdated(predictor)
