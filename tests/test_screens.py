from __future__ import annotations

import io
import json
import math
import os
import re

import pytest
from builders import ClaudeSession, history
from conftest import KEY
from test_cli import QuickModel, fake_analysis, quick_meta, quick_train

from yardstick_predict import __version__, cli, llm, model_types, store, ui
from yardstick_predict.cli import main
from yardstick_predict.screens import home, keys, update, version
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
    assert re.search(r"│\s+Your models\s+│\n│\s+llm\s+updated", out)  # the list says what it is
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
    assert "No models yet" in out and "It needs a Voyage API key." in out and "Your models" not in out
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


def trained(name: str, verdicts: dict[str, bool]) -> Predictor:
    """A model with a saved quick model that knows these AI models, each run 40 times at max. True passed the test."""
    predictor = Predictor.create(
        name, [], {"model": "voyage-4", "dimensions": 256}, features=["prompt"], model={"type": "quick"}
    )
    meta = quick_meta({model: 40 for model in verdicts})
    for model, passed in verdicts.items():
        meta["report"]["models"][model].update(served=passed, reason=None if passed else "not_accurate_enough")
    meta["effort_counts"] = {model: [["max", 40]] for model in verdicts}
    QuickModel(meta).save(predictor.model_dir)
    return predictor


def test_estimate_asks_which_ai_models_then_what_next(capsys, monkeypatch, quick, ready):
    trained("main", {"claude-opus-5-5": True, "gpt-6.1-sol": False})
    asked = typing(monkeypatch, "add a dark mode toggle to the settings page")
    # Estimate, Enter on the first tested AI model, then other AI models for the same task: from the list, Enter on
    # the second, then home and quit.
    pressing(monkeypatch, "1", "1", keys.ENTER, "2", "2", keys.DOWN, keys.ENTER, "3", keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert asked == ["›"]  # the task is asked once, and kept for other AI models
    assert "Which AI models should it forecast?" in out
    assert "1. Tested AI models    claude-opus-5-5: passed the test" in out
    assert "2. Choose from a list  every one you've used, 2 in all" in out
    assert re.search(r"○ claude-opus-5-5\s+max\s+passed the test", out)  # nothing starts ticked
    assert re.search(r"○ gpt-6\.1-sol\s+max\s+⚠ failed the test", out)
    assert "enter forecast this one" in out
    first, second = out.split("Choose other AI models  for the same task", 1)
    table = first.split("AI model")[-1]
    assert re.search(r"claude-opus-5-5\s+max\s+60 s", table) and "gpt-6.1-sol" not in table  # only the one chosen
    assert re.search(r"gpt-6\.1-sol\s+⚠\s+max\s+60 s", second) and "Failed the test: gpt-6.1-sol." in second
    assert "1. Estimate another task" in out and "3. Back to home" in out
    assert out.count("What would you like to do?") == 2  # home only when asked for


def test_enter_picks_the_line_you_are_on_and_space_ticks_several_across_pages(capsys, monkeypatch):
    from yardstick_predict.screens import checklist

    rows = [keys.Choice(f"model {number}") for number in range(8)]
    pressing(monkeypatch, keys.DOWN, keys.ENTER)
    assert checklist.tick(rows) == [1]  # nothing ticked: Enter picks the line it's on
    pressing(monkeypatch, "3")
    assert checklist.tick(rows) == [2]  # a number picks at once, as in every menu
    pressing(monkeypatch, keys.DOWN, keys.SPACE, keys.RIGHT, "1", keys.ENTER)
    assert checklist.tick(rows) == [1, 6]  # once one is ticked, a number ticks too, on any page
    out = plain(capsys.readouterr().out)
    assert "enter forecast this one" in out and "enter forecast the 2 ticked" in out
    assert "Page 1 of 2" in out and re.search(r"Page 2 of 2(,| ·) 2 ticked", out)
    pressing(monkeypatch, keys.ESC)
    assert checklist.tick(rows) is None


def test_a_list_without_single_keys_takes_typed_numbers(capsys, monkeypatch):
    from yardstick_predict.screens import checklist

    monkeypatch.setattr(keys, "reader", lambda: None)
    rows = [keys.Choice(f"model {number}") for number in range(8)]
    typing(monkeypatch, "", "2 n 1")
    assert checklist.tick(rows) == [1, 6]
    assert "Type at least one number." in plain(capsys.readouterr().out)


def test_menus_and_the_forecast_table_fit_an_80_column_terminal(capsys, monkeypatch):
    from yardstick_predict.screens import checklist

    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("FORCE_COLOR", "1")
    long = "a hint long enough to run past the edge of any terminal that is only eighty columns wide"
    pressing(monkeypatch, keys.DOWN, keys.ENTER)
    keys.choose([keys.Choice("Tested AI models", long), keys.Choice("Back")], footer=[long, long])
    pressing(monkeypatch, keys.ENTER)
    checklist.tick([keys.Choice(f"claude-sonnet-4-6  default {n}", long) for n in range(9)], notes=[long])
    made("main")
    forecasts = [
        {
            "model": "claude-sonnet-4-6",
            "effort": "default",
            "seconds": 5000.0,
            "tokens": 123_400_000,
            "usd": 123.45,
            "distribution": curve(math.log(5000), 1.5),
            "reason": "not_accurate_enough",
        },
    ]
    cli.show_forecasts("x" * 120, {"predictor": "main", "forecasts": forecasts, "too_few": [("m" * 30, 3)] * 4})
    lines = plain(capsys.readouterr().out).replace("\r", "\n").split("\n")
    assert max(ui.width(line) for line in lines) <= 79


def test_forecasts_show_six_to_a_page(capsys, monkeypatch, quick, ready):
    trained("main", {f"claude-model-{number}": True for number in range(8)})
    typing(monkeypatch, "fix the parser")
    # Tested AI models, tick all 8 across the two pages, then the next page of forecasts, then home.
    ticks = [keys.SPACE, "2", "3", "4", "5", "6", keys.RIGHT, "1", "2", keys.ENTER]
    pressing(monkeypatch, "1", "1", *ticks, "1", "4", keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    pages = out.split("fix the parser  page ")[1:]
    assert [page[:6] for page in pages] == ["1 of 2", "2 of 2"]
    assert len(re.findall(r"claude-model-\d\s+max", pages[0].split("Under each")[0].split("AI model")[1])) == 6
    assert re.search(r"1\. Next page\s+forecasts 7 on", out) and "1. Previous page" in pages[1]
    assert "1. Tested AI models    8 AI models: passed the test" in out  # too many to name on one line


# A newer version ---------------------------------------------------------------------------------------------


def checking_pypi(monkeypatch, latest):
    """PyPI's newest release, as the check finds it, and a list of each time it asked."""
    asked = []

    def ask():
        asked.append(1)
        if isinstance(latest, Exception):
            raise latest
        return latest

    monkeypatch.delenv(version.OFF, raising=False)
    monkeypatch.setattr(version, "_looking", False)
    monkeypatch.setattr(version, "latest_release", ask)
    monkeypatch.setattr(version.threading, "Thread", Instant)
    return asked


class Instant:
    """A thread that runs its work at once, so a test needn't wait for it."""

    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


def test_the_home_screen_says_when_a_newer_version_is_out(capsys, monkeypatch):
    made("main")
    asked = checking_pypi(monkeypatch, "99.0.0")
    pressing(monkeypatch, keys.ESC)
    assert main([]) == 0  # this look at PyPI runs in the background, so its answer shows from the next screen on
    assert "is out" not in plain(capsys.readouterr().out) and asked == [1]
    monkeypatch.setattr(version, "_looking", False)  # a new session
    pressing(monkeypatch, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    upgrade = "Update with: uv tool upgrade yardstick-predict"
    assert f"yardstick-predict 99.0.0 is out. You have {__version__}. {upgrade}" in out
    assert asked == [1]  # PyPI is asked at most once a day


def test_no_notice_for_the_same_version_or_when_pypi_cant_be_reached(capsys, monkeypatch):
    made("main")
    for latest in (__version__, "0.0.1", OSError("offline")):
        asked = checking_pypi(monkeypatch, latest)
        (store.home() / "update-check.json").unlink(missing_ok=True)
        pressing(monkeypatch, keys.ESC, keys.ESC)
        assert main([]) == 0
        monkeypatch.setattr(version, "_looking", False)
        assert main([]) == 0
        assert "is out" not in plain(capsys.readouterr().out) and asked == [1]


def test_the_check_can_be_turned_off(capsys, monkeypatch):
    made("main")
    asked = checking_pypi(monkeypatch, "99.0.0")
    monkeypatch.setenv(version.OFF, "1")
    pressing(monkeypatch, keys.ESC)
    assert main([]) == 0
    assert not asked and not (store.home() / "update-check.json").exists()


def test_versions_compare_by_their_numbers():
    assert version.newer("0.10.0", "0.9.1") and version.newer("1.0", "0.9.9")
    assert not version.newer("0.4.0", "0.4.0") and not version.newer("0.4.0", "0.5.0.dev0")


def curve(mu: float, sigma: float) -> dict:
    """A forecast's curve with no chance of a direct answer: one bell on the log scale for each of its targets."""
    part = {"mu": mu, "sigma": sigma}
    every = {target: {"direct": part, "work": part} for target in TARGETS}
    return {"shape": "two_part_lognormal", "direct_probability": 0.0, **every}


TARGETS = ("seconds", "tokens", "usd")


def test_the_range_is_where_the_curve_holds_9_results_in_10():
    from yardstick_predict.forecast import likely_range

    low, high = likely_range(curve(math.log(600), 0.5), "seconds")
    assert low == pytest.approx(600 * math.exp(-1.6449 * 0.5), rel=1e-3)
    assert high == pytest.approx(600 * math.exp(1.6449 * 0.5), rel=1e-3)
    assert likely_range(None, "seconds") is None and likely_range({"shape": "x"}, "seconds") is None


def test_the_forecast_table_shows_ranges_and_warns_on_rough_forecasts(capsys):
    made("main")
    forecasts = [
        {
            "model": "claude-opus-5-5",
            "effort": "max",
            "seconds": 960.0,
            "tokens": 23_000_000,
            "usd": 7.42,
            "distribution": curve(math.log(900), 0.4),
            "runs": 2202,
        },
        {"model": "claude-opus-5-5", "effort": "high", "seconds": 300.0, "tokens": 7_700_000, "usd": 3.18, "runs": 11},
        {
            "model": "gpt-6.1-sol",
            "effort": "xhigh",
            "seconds": 660.0,
            "tokens": 3_900_000,
            "usd": 1.16,
            "reason": "not_accurate_enough",
            "runs": 333,
        },
        {
            "model": "gpt-6-astra",
            "effort": "ultra",
            "seconds": 480.0,
            "tokens": 3_200_000,
            "usd": 1.77,
            "reason": "not_enough_data",
            "runs": 28,
        },
    ]
    result = {"predictor": "main", "forecasts": forecasts, "too_few": [("claude-haiku-5-5", 6), ("claude-opus-4-7", 3)]}
    cli.show_forecasts("Create a fun game", result, screen=True)
    out = plain(capsys.readouterr().out)
    assert re.search(r"AI model\s+effort\s+time\s+tokens\s+cost\s*\n", out)
    assert re.search(r"claude-opus-5-5\s+max\s+16 min\s+23\.0M\s+\$7\.42\s*\n", out)  # it passed on many runs
    # Under it, its range: 900 seconds' curve holds 9 results in 10 from about 466 to 1,738 seconds.
    assert re.search(r"\n\s+8 to 29 min\s+\d+ to [\d,.]+[kM]?\s+\$[\d.]+k? to \$[\d.]+k?\s*\n", out)
    assert re.search(r"⚠\s+high\s+5 min\s+7\.7M\s+\$3\.18", out)  # marked, with why under the table
    assert "⚠ Few runs: claude-opus-5-5 at high (11)." in out
    assert "⚠ Failed the test: gpt-6.1-sol." in out and "⚠ Not tested yet: gpt-6-astra." in out
    squashed = " ".join(out.split())  # notes are broken to fit the terminal
    assert "Too few of your tasks to forecast: claude-haiku-5-5 (6) and claude-opus-4-7 (3)." in squashed
    assert "Under each forecast is the range it lands in 9 times out of 10." in out
    assert "Those are a rough guide. Plan with a wider margin there." in out
    assert "Create a fun game" not in out and "model main, updated" in out and "--json" not in out
    cli.show_forecasts("Create a fun game", result)  # the predict command says the task, and names the predictor
    out = plain(capsys.readouterr().out)
    squashed = " ".join(out.split())
    assert "Create a fun game" in out and "predictor main, updated" in out and "add --json for details" in squashed


def test_a_program_never_gets_a_forecast_that_failed_the_test_but_a_person_sees_it_marked(
    capsys, monkeypatch, quick, ready
):
    embedding = {"model": "voyage-4", "dimensions": 256}
    predictor = Predictor.create("main", [], embedding, features=["prompt"], model={"type": "quick"})
    QuickModel(quick_meta({"claude-opus-5-5": 40, "gpt-6.1-sol": 40}, passed=False)).save(predictor.model_dir)
    request = {"version": 1, "tasks": [{"id": "a", "prompt": "fix it", "models": ["gpt-6.1-sol"]}]}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))
    assert main(["predict"]) == 0
    answer = json.loads(capsys.readouterr().out)["forecasts"][0]
    assert answer["seconds"] is None and answer["reason"] == "not_accurate_enough"
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    assert main(["predict", "fix it", "--model", "gpt-6.1-sol"]) == 0
    out = plain(capsys.readouterr().out)
    assert re.search(r"gpt-6\.1-sol\s+⚠\s+60 s", out) and "Failed the test: gpt-6.1-sol." in out


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
