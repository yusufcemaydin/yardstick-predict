from __future__ import annotations

import io
import json
import re
from pathlib import Path

import jsonschema
import pytest
from builders import ClaudeSession, history
from conftest import KEY

from yardstick_predict import benchmarks, cli, embeddings
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor, default_name, predictor_names

START = 1_780_000_000.0
SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


@pytest.fixture
def records(tmp_path):
    root = tmp_path / "claude-projects"
    history(root, tasks=420, start=START, models=("claude-opus-5-5", "claude-sonnet-5-5"))
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
    assert by_model["gpt-unknown"]["reason"] == "unknown_model"

    # In a terminal, a task in quotes gets a table, on every model the predictor forecasts.
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    code, out = run(capsys, "predict", "fix the api")
    assert code == 0 and re.search(r"claude-opus-5-5\s+\d+ (s|min|h)\s+[\d.]+[kM]? tokens\s+\$", out)
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
