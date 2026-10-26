from __future__ import annotations

import json
import os
import re

import pytest
from builders import ClaudeSession, history
from conftest import KEY
from test_cli import QuickModel, fake_analysis, quick_meta, quick_train

from yardstick_predict import __version__, cli, llm, model_types, store, ui
from yardstick_predict.cli import main
from yardstick_predict.screens import home, keys, update
from yardstick_predict.store import Predictor, default_name, predictor_names, set_default

START = 1_780_000_000.0
CODES = re.compile(r"\033\[[0-9;]*[A-Za-z]")


def plain(text: str) -> str:
    """The text as a person reads it, without the codes that move the cursor or color it."""
    return CODES.sub("", text)


def pressing(monkeypatch, *pressed: str) -> list[str]:
    """A person in a terminal that passes single key presses, who presses these keys in turn."""
    waiting = list(pressed)

    def key() -> str:
        if not waiting:
            pytest.fail("It waited for a key no one pressed")
        return waiting.pop(0)

    monkeypatch.setattr(keys, "reader", lambda: key)
    monkeypatch.setattr(cli, "interactive", lambda: True)
    return waiting


def typing(monkeypatch, *replies: str) -> list[str]:
    """What a person types at each question, in turn. Returns every question asked."""
    asked, waiting = [], list(replies)

    def answer(question: str = "") -> str:
        asked.append(plain(question).strip())
        if not waiting:
            pytest.fail(f"It asked a question no one answered: {question}")
        return waiting.pop(0)

    monkeypatch.setattr("builtins.input", answer)
    return asked


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    """Each test is a new session, which has read no records yet."""
    monkeypatch.setattr(update, "_readings", None)


@pytest.fixture
def quick(monkeypatch):
    load = lambda folder: QuickModel(json.loads((folder / "model.json").read_text()))  # noqa: E731
    monkeypatch.setitem(model_types._TYPES, "quick", (quick_train, load))
    return ["--model-type", "quick"]


@pytest.fixture
def ready(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    return voyage


@pytest.fixture
def records(tmp_path, monkeypatch):
    """40 Claude Code tasks in the usual folder, ~/.claude/projects, of a home folder made for the test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    root = tmp_path / ".claude" / "projects"
    history(root, tasks=40, start=START)
    return root


def made(name: str, served=("claude-opus-5-5",)) -> Predictor:
    """A model made without reading any records, with a saved quick model that serves these AI models."""
    predictor = Predictor.create(
        name, [], {"model": "voyage-4", "dimensions": 256}, features=["prompt", "llm"], model={"type": "quick"}
    )
    QuickModel(quick_meta({model: 40 for model in served})).save(predictor.model_dir)
    predictor.config["refreshed_at"] = START
    predictor.save_config()
    return predictor


def new_work(records, prompts=("new task one", "new task two", "new task three")) -> None:
    session = ClaudeSession(records, "brand-new", START + 90 * 86400)
    for prompt in prompts:
        session.prompt(prompt)
        session.reply(5)
    session.save(mtime=START)


# Menus --------------------------------------------------------------------------------------------------------


CHOICES = [keys.Choice("Estimate", "Forecast a task"), keys.Choice("Update", "Learn"), keys.Choice("Quit")]


def test_arrow_keys_move_through_a_menu_and_enter_chooses(capsys, monkeypatch):
    pressing(monkeypatch, keys.DOWN, keys.DOWN, keys.UP, keys.ENTER)
    assert keys.choose(CHOICES) == 1
    out = plain(capsys.readouterr().out)
    assert "› 2. Update" in out and "1. Estimate  Forecast a task" in out
    pressing(monkeypatch, keys.UP, keys.ENTER)  # up from the first choice goes round to the last
    assert keys.choose(CHOICES) == 2


def test_a_number_key_chooses_at_once_and_esc_goes_back(capsys, monkeypatch):
    pressing(monkeypatch, "3")
    assert keys.choose(CHOICES) == 2
    pressing(monkeypatch, "9", keys.ESC)  # a number with no choice does nothing
    assert keys.choose(CHOICES) is None


def test_a_terminal_that_cant_pass_single_keys_gets_a_number_typed_in(capsys, monkeypatch):
    monkeypatch.setattr(keys, "reader", lambda: None)
    asked = typing(monkeypatch, "7", "2")
    assert keys.choose(CHOICES) == 1
    out = plain(capsys.readouterr().out)
    assert "1. Estimate  Forecast a task" in out and "Type a number from 1 to 3." in out
    assert asked == ["Type a number: [1]", "Type a number: [1]"]
    typing(monkeypatch, "")  # Enter takes the first choice
    assert keys.choose(CHOICES) == 0


# The home screen ----------------------------------------------------------------------------------------------


def test_the_home_screen_greets_lists_the_models_and_reads_nothing_heavy(capsys, monkeypatch):
    made("main")
    made("llm")
    set_default("main")
    monkeypatch.setattr(cli, "cmd_refresh", lambda args: pytest.fail("It refreshed"))
    monkeypatch.setattr(update, "update", lambda work: pytest.fail("It updated"))
    pressing(monkeypatch, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert f"yardstick-predict {__version__}" in out
    assert "Forecasts a coding-agent task's time and" in out
    assert re.search(r"main \(default\)\s+updated \d+ days ago", out)
    assert re.search(r"\n│\s+llm\s+updated", out)
    assert "What would you like to do?" in out
    for line in ("1. Estimate", "2. Update", "3. Models", "4. Add New", "5. Quit"):
        assert line in out, line
    assert "Every command: yardstick-predict --help" in out and "esc quit" in out
    assert "╭" in out and "╯" in out


def test_with_no_model_the_home_screen_says_so_and_add_new_comes_first(capsys, monkeypatch):
    pressing(monkeypatch, "5")
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "No model yet" in out and "It needs a Voyage API key." in out
    assert re.search(r"1\. Add New\s+Make your first model", out) and "5. Quit" in out


def test_a_narrow_terminal_gets_the_home_screen_without_a_box(capsys, monkeypatch):
    made("main")
    monkeypatch.setattr(home.shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((40, 24)))
    pressing(monkeypatch, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "main" in out and "╭" not in out and "│" not in out


def test_a_bare_run_outside_a_terminal_prints_one_line_per_model_as_before(capsys, monkeypatch):
    made("main")
    monkeypatch.setattr(keys, "reader", lambda: pytest.fail("It showed a menu"))
    assert main([]) == 0
    assert capsys.readouterr().out == f"main: voyage-4 embeddings, 0 tasks, updated {cli.ago(START)}\n"


def test_estimate_forecasts_a_task_on_the_default_model_then_comes_home(capsys, monkeypatch, quick):
    made("main", served=("claude-opus-5-5", "claude-sonnet-5-5"))
    asked = typing(monkeypatch, "add a dark mode toggle to the settings page")
    monkeypatch.setattr(cli, "show_forecasts", lambda task, result, predictor: ui.out(f"FORECAST {task}"))
    seen = []

    def forecast(args, task):
        seen.append((args.predictor, task))
        return Predictor("main"), {"forecasts": []}

    monkeypatch.setattr(cli, "forecast_request", forecast)
    pressing(monkeypatch, "1", keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "Describe the task, as you'd ask a coding agent." in out
    assert "FORECAST add a dark mode toggle to the settings page" in out
    assert seen == [(None, "add a dark mode toggle to the settings page")] and asked == ["›"]
    assert out.count("What would you like to do?") == 2  # it came home


# Models -------------------------------------------------------------------------------------------------------


def test_the_models_screen_shows_a_model_and_makes_it_the_default(capsys, monkeypatch):
    made("main")
    made("llm", served=("claude-opus-5-5", "claude-fable-5"))
    set_default("main")
    pressing(monkeypatch, "3", "1", "2", keys.ESC, keys.ESC, keys.ESC)  # Models, llm, Make default, then back
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "Forecasts on for claude-opus-5-5 and claude-fable-5" in out
    assert "Learns from: prompt text, LLM analysis" in out
    assert "2. Make default  Use it when you don't pick one" in out and "3. Rename" in out
    assert "llm is now the default. Estimate and Claude Code's status line use it." in out
    assert default_name() == "llm"
    assert "llm (default)" in out.split("llm is now the default")[1]  # the screen shows it at once


def test_rename_keeps_the_default_even_when_it_was_found_only_by_its_name(capsys, monkeypatch):
    made("default")
    made("other")  # with two models and no setting, default is the default only because of its name
    assert store.settings().get("default") is None and default_name() == "default"
    pressing(monkeypatch, "3", "1", "2", keys.ESC, keys.ESC, keys.ESC)  # Models, default, Rename
    asked = typing(monkeypatch, "Bad Name", "other", "main")
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert asked == ["New name for default:"] * 3
    assert "Use 1 to 40 lowercase letters" in out and "There's already a model called other." in out
    assert "Renamed default to main." in out and "main is still the default." in out
    assert predictor_names() == ["main", "other"] and default_name() == "main"
    assert Predictor("main").config["name"] == "main"


def test_the_rename_command_does_the_same_for_scripts(capsys):
    made("work")
    assert main(["rename", "work", "home"]) == 0
    assert "Renamed work to home." in capsys.readouterr().out
    assert predictor_names() == ["home"] and default_name() == "home"
    assert main(["rename", "missing", "x"]) == 1
    made("other")
    assert main(["rename", "home", "other"]) == 1
    assert "There's already a predictor called other." in capsys.readouterr().err


def test_a_model_being_updated_isnt_renamed(capsys):
    from yardstick_predict import locks

    made("work")
    with locks.held(store.predictor_folder("work") / "refresh.lock"):
        assert main(["rename", "work", "home"]) == 1
    assert "being set up or refreshed by another command" in capsys.readouterr().err
    assert predictor_names() == ["work"]


# Update -------------------------------------------------------------------------------------------------------


def test_update_says_where_the_work_is_and_every_cost_before_spending_anything(
    capsys, monkeypatch, quick, ready, records
):
    assert main(["create", "--express", *quick]) == 0
    new_work(records)
    capsys.readouterr()
    sent = len(ready.requests)
    known = Predictor("default").known_files()
    calls = []
    monkeypatch.setattr(update, "apply", lambda job, key: calls.append(job) or 0)
    pressing(monkeypatch, "2", keys.DOWN, keys.ENTER, keys.ESC)  # Update, then Cancel at the question
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "New work since a moment ago" in out
    assert re.search(r"Claude Code\s+~[/\\]\.claude[/\\]projects", out)
    assert re.search(r"1 chat changed(,| ·) \d+ KB to read", out) and "Codex" not in out
    assert "Free. Your chats never leave this computer. Esc stops it." in out
    assert "Found 3 new tasks, all in Claude Code." in out
    assert re.search(r"Embed \d texts with Voyage", out) and re.search(r"(<\$0\.01|\$\d\.\d\d), \d+ tokens", out)
    assert "Train again, under a minute" in out and "Under a minute in all. Its data grows to" in out
    assert "1. Update default" in out and "2. Cancel" in out
    assert "Nothing was spent or saved." in out
    # Cancel spent nothing and saved nothing.
    assert len(ready.requests) == sent and not calls
    assert Predictor("default").count_tasks() == 40 and Predictor("default").known_files() == known


def test_update_goes_ahead_after_one_question_and_reuses_what_it_read(capsys, monkeypatch, quick, ready, records):
    assert main(["create", "--express", *quick]) == 0
    new_work(records)
    capsys.readouterr()
    from yardstick_predict import archives, pipeline

    reads = []
    real = archives.read_records
    counted = lambda *args, **kwargs: reads.append(1) or real(*args, **kwargs)  # noqa: E731
    monkeypatch.setattr(archives, "read_records", counted)
    monkeypatch.setattr(pipeline, "read_records", counted)
    pressing(monkeypatch, "2", "2", "2", keys.ENTER, keys.ESC)  # Update and Cancel, then Update and go ahead
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "read already" in out  # the second Update read nothing again
    assert len(reads) == 1
    assert "default updated, forecasts on for claude-opus-5-5" in out
    assert Predictor("default").count_tasks() == 43 and len(Predictor("default").embedded("prompt")) == 43
    assert "Tested on each AI model" not in out  # no long test report


def test_nothing_new_says_since_when_and_reads_nothing(capsys, monkeypatch, quick, ready, records):
    assert main(["create", "--express", *quick]) == 0
    capsys.readouterr()
    pressing(monkeypatch, "2", keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "Nothing new since a moment ago." in out
    assert "New work" not in out and "to read" not in out


def test_esc_while_reading_stops_and_saves_nothing(capsys, monkeypatch, quick, ready, records):
    assert main(["create", "--express", *quick]) == 0
    new_work(records)
    known = Predictor("default").known_files()
    from contextlib import contextmanager

    class Pressed:
        def stop_if_asked(self):
            raise keys.Stopped

    @contextmanager
    def watching():
        yield Pressed()

    monkeypatch.setattr(keys, "watching", watching)
    pressing(monkeypatch, "2", keys.ESC)
    assert main([]) == 0
    assert "Stopped. Nothing was spent or saved." in plain(capsys.readouterr().out)
    assert Predictor("default").known_files() == known and Predictor("default").count_tasks() == 40
    assert not update.readings()._kept  # nothing half read is kept


def test_the_summary_prices_the_descriptions_llm_analysis_writes(capsys, monkeypatch, quick, ready, records):
    seen: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(seen))
    flags = ["--features", "llm", "--llm-provider", "codex", *quick]
    assert main(["create", "--yes", *flags]) == 0
    analyzed = len(seen)
    new_work(records)
    capsys.readouterr()
    pressing(monkeypatch, "2", keys.DOWN, keys.ENTER, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "Analyze 3 tasks with Codex" in out and "your ChatGPT plan, about 1 request" in out
    # The three prompts, and the three descriptions LLM analysis writes, are priced before the question. The
    # stand-in Voyage counts a token for each word and one more, so each prompt is 4.
    tokens = 3 * 4 + 3 * llm.DESCRIPTION_TOKENS
    assert "Embed 6 texts with Voyage" in out and f"about {tokens:,} tokens" in out
    assert len(seen) == analyzed  # nothing was analyzed before the question


def test_refresh_in_a_terminal_uses_the_summary_and_yes_asks_nothing(capsys, monkeypatch, quick, ready, records):
    assert main(["create", "--express", *quick]) == 0
    new_work(records)
    capsys.readouterr()
    pressing(monkeypatch, keys.DOWN, keys.ENTER)  # Cancel
    assert main(["refresh"]) == 0
    assert "Found 3 new tasks, all in Claude Code." in plain(capsys.readouterr().out)
    monkeypatch.setattr(keys, "reader", lambda: pytest.fail("It showed a menu"))
    monkeypatch.setattr("builtins.input", lambda question="": pytest.fail(f"It asked: {question}"))
    assert main(["refresh", "--yes"]) == 0
    assert "Found 3 new tasks" in capsys.readouterr().out and Predictor("default").count_tasks() == 43


# The progress bar ---------------------------------------------------------------------------------------------


def test_a_progress_bar_keeps_the_label_of_its_step(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    ui.working("Looking for new tasks")
    ui.progress(1_000_000, 6_200_000_000, "bytes")
    out = plain(capsys.readouterr().out)
    assert out.rstrip().endswith("1.0 MB / 6.2 GB") and "Looking for new tasks  " in out.split("\r")[-1]
    ui.out("done")
    ui.progress(2_000_000, 6_200_000_000, "bytes")  # a bar after a line of its own has no label left
    assert "Looking" not in plain(capsys.readouterr().out).split("\r")[-1]


def test_setup_suggests_main_for_the_first_model(monkeypatch):
    asked = typing(monkeypatch, "")
    assert cli.ask_name([]) == "main" and asked == ["Name this predictor: [main]"]
    asked = typing(monkeypatch, "", "work")  # a later one gets no name it could clash with
    assert cli.ask_name(["main"]) == "work" and asked == ["Name this predictor:", "Name this predictor:"]
