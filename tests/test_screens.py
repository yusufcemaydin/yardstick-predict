from __future__ import annotations

import io
import json
import math
import os
import re
import time

import pytest
from builders import ClaudeSession, flat, history
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

    def key(timeout=None) -> str:
        if not waiting:
            if timeout is not None:
                return ""  # no key came in time, as when a person stops after one digit
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


def test_the_home_screen_names_five_models_at_most_the_default_first(capsys, monkeypatch):
    for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "omega"):
        made(name)
    set_default("omega")
    pressing(monkeypatch, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    shown = re.findall(r"│\s+(\w+)(?: \(default\))?\s+updated", out)
    assert shown == ["omega", "alpha", "beta", "delta", "epsilon"] and "and 2 more in Models" in out


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
    # The list says what it is, with the default first.
    assert re.search(r"│\s+Your models\s+│\n│\s+main \(default\)\s+updated \d+ days ago", out)
    assert re.search(r"\n│\s+llm\s+updated", out) and "more in Models" not in out
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


def scored(name: str, scores: dict[str, float]) -> Predictor:
    """A model that forecasts these AI models, each scored at this share of the simple rule's score on the same 40
    tasks, for time, tokens and cost alike. Under 1 came closer than the simple rule."""
    predictor = made(name, served=tuple(scores))
    meta = json.loads((predictor.model_dir / "model.json").read_text())
    for model, share in scores.items():
        for target in TARGETS:
            meta["report"]["models"][model][target] = {"score": share, "simple_rule_score": 1.0, "tasks_tested": 40}
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
    assert re.search(r"1\. AI models with enough data\s+claude-opus-5-5\n", out)
    assert re.search(r"2\. Choose from a list\s+every one you've used, 2 in all", out)
    assert re.search(r"○ claude-opus-5-5\s+max\n", out)  # nothing starts ticked, and enough data needs no word
    assert re.search(r"○ gpt-6\.1-sol\s+max\s+⚠ not enough data", out)
    assert "test" not in out.replace("yardstick", "")
    assert "enter forecast this one" in out
    first, second = out.split("Choose other AI models  for the same task", 1)
    table = first.split("AI model")[-1]
    assert re.search(r"claude-opus-5-5\s+max\s+60 s", table) and "gpt-6.1-sol" not in table  # only the one chosen
    assert re.search(r"gpt-6\.1-sol\s+⚠\s+max\s+60 s", second) and "Not enough data: gpt-6.1-sol." in second
    assert "1. Estimate another task" in out and "3. Back to home" in out
    assert out.count("What would you like to do?") == 2  # home only when asked for


def test_enter_picks_the_line_you_are_on_and_space_ticks_several_across_pages(capsys, monkeypatch):
    from yardstick_predict.screens import checklist

    rows = [keys.Choice(f"model {number}") for number in range(8)]
    pressing(monkeypatch, keys.DOWN, keys.ENTER)
    assert checklist.tick(rows) == [1]  # nothing ticked: Enter picks the line it's on
    pressing(monkeypatch, "3")
    assert checklist.tick(rows) == [2]  # a number picks at once, as in every menu
    pressing(monkeypatch, keys.DOWN, keys.SPACE, keys.RIGHT, "7", keys.ENTER)
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
    asked = typing(monkeypatch, "x", "2 7")  # the numbers go on from page to page
    assert checklist.tick(rows) == [1, 6]
    assert flat(asked[0]).endswith("n or p turns the page. Enter on its own goes back.")
    for back in ("", "b"):  # Enter on its own goes back, as Esc does where single keys pass
        typing(monkeypatch, back)
        assert checklist.tick(rows) is None


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
    ticks = [keys.SPACE, "2", "3", "4", "5", "6", keys.RIGHT, "7", "8", keys.ENTER]
    pressing(monkeypatch, "1", "1", *ticks, "1", "4", keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    pages = out.split("fix the parser  page ")[1:]
    assert [page[:6] for page in pages] == ["1 of 2", "2 of 2"]
    table = pages[0].split("Each forecast's first line")[0].split("AI model")[1]
    assert len(re.findall(r"claude-model-\d\s+max", table)) == 6
    assert re.search(r"1\. Next page\s+forecasts 7 on", out) and "1. Previous page" in pages[1]
    assert re.search(r"1\. AI models with enough data\s+8 AI models\n", out)  # too many to name on one line


def test_with_several_models_estimate_asks_which_and_another_can_forecast_the_same_task(
    capsys, monkeypatch, quick, ready
):
    trained("main", {"claude-opus-5-5": True})
    trained("work", {"gpt-6.1-sol": True})
    set_default("main")
    asked = typing(monkeypatch, "fix the parser")
    # Estimate, then work, its AI models with enough data and Enter. Then another of your models: main, which the
    # list starts on work for, its AI models and Enter. Then home and quit.
    pressing(monkeypatch, "1", "2", "1", keys.ENTER, "3", keys.UP, keys.ENTER, "1", keys.ENTER, "4", keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert asked == ["›"]  # the task is asked once, and kept for the other model
    assert out.count("Which of your models should forecast it?") == 2
    assert re.search(r"1\. main \(default\)\s+not trained yet\n\s+2\. work", out)
    first, second = out.split("Use another of your models  for the same task", 1)
    assert re.search(r"gpt-6\.1-sol\s+max\s+60 s", first) and "model work, updated" in first
    assert re.search(r"claude-opus-5-5\s+max\s+60 s", second) and "model main, updated" in second
    assert "4. Back to home" in out


@pytest.mark.parametrize("width", [80, 45])
def test_each_model_says_how_accurate_it_is_next_to_the_default(capsys, monkeypatch, quick, ready, width):
    from yardstick_predict.screens import estimate

    monkeypatch.setenv("COLUMNS", str(width))
    scored("main", {"claude-opus-5-5": 0.5})
    scored("llm", {"claude-opus-5-5": 0.44, "gpt-6.1-sol": 0.3})  # 12% closer on the AI model both forecast
    scored("old", {"claude-opus-5-5": 0.575})  # 15% further off
    made("thin", served=())
    set_default("main")
    typing(monkeypatch, "fix the parser", "")
    # Estimate, then back from the list of your models to the task, then home and quit.
    pressing(monkeypatch, "1", keys.ESC, keys.ESC)
    assert main([]) == 0
    out = capsys.readouterr().out
    assert widest(out) < width
    menu = plain(out).split("Which of your models should forecast it?")[1]
    assert re.search(r"1\. llm\s+12% more accurate", menu)  # cut short in a narrow terminal, as menu hints are
    assert re.search(r"2\. main \(default\)\s+updated", menu)  # the one the others are compared with
    assert re.search(r"3\. old\s+15% less accurate", menu) and re.search(r"4\. thin\s+not enough data yet", menu)
    assert estimate.ACCURACY in " ".join(menu.split())
    if width == 80:
        assert re.search(r"1\. llm\s+12% more accurate than main, updated", menu)


def test_numbered_menus_say_what_accuracy_means_too(capsys, monkeypatch, quick, ready):
    from yardstick_predict.screens import estimate

    scored("main", {"claude-opus-5-5": 0.5})
    scored("llm", {"claude-opus-5-5": 0.44})
    set_default("main")
    monkeypatch.setattr(keys, "reader", lambda: None)  # a terminal that can't pass single key presses
    typing(monkeypatch, "3")  # Back
    assert estimate.which(None) is None
    text = " ".join(plain(capsys.readouterr().out).split())
    assert "1. llm 12% more accurate than main, updated" in text and estimate.ACCURACY in text


def test_a_model_is_compared_with_the_default_only_on_the_ai_models_both_forecast(quick):
    from yardstick_predict.screens import listing

    scored("main", {"claude-opus-5-5": 0.5})
    scored("work", {"gpt-6.1-sol": 0.3})
    made("thin", served=())
    said, compared = listing.accuracy(["main", "thin", "work"], "main")
    assert said == {"thin": "not enough data yet", "work": "no AI model in common with main"} and not compared
    # A default that forecasts nothing leaves every model uncompared.
    made("empty", served=())
    assert listing.accuracy(["empty", "work"], "empty") == ({"empty": "not enough data yet"}, False)


def test_accuracy_next_to_the_default_counts_each_model_against_its_own_simple_rule():
    from yardstick_predict.screens import listing

    def report(**scores):
        """For each AI model: its score, the simple rule's and on how many tasks, the same for every target."""
        return {
            "models": {
                model: {target: {"score": a, "simple_rule_score": b, "tasks_tested": n} for target in TARGETS}
                for model, (a, b, n) in scores.items()
            }
        }

    base = report(a=(1.0, 2.0, 300), b=(3.0, 4.0, 100))
    # On the same tasks, it's the ratio of the scores. Each AI model counts as much as the tasks it was scored on.
    closer = report(a=(0.8, 2.0, 300), b=(3.0, 4.0, 100))
    assert listing.error_ratio(closer, base, ["a", "b"]) == pytest.approx(0.8**0.75)
    # Scored on harder tasks, where the simple rule came further off too, the same skill is the same accuracy.
    assert listing.error_ratio(report(a=(2.0, 4.0, 50)), base, ["a"]) == pytest.approx(1.0)
    # An AI model only one of them scored, or a score that makes no sense, counts for nothing.
    assert listing.error_ratio(closer, base, ["c"]) is None
    assert listing.error_ratio(report(a=(0.0, 2.0, 300)), base, ["a"]) is None
    said = [listing.compared(ratio, "main") for ratio in (0.88, 0.99, 1.019, 1.15, 2.5)]
    assert said == [
        "12% more accurate than main",
        "about as accurate as main",
        "about as accurate as main",
        "15% less accurate than main",
        "far less accurate than main",
    ]


def test_estimate_says_when_a_model_cant_forecast_and_another_can_be_chosen(capsys, monkeypatch, quick, ready):
    trained("main", {"claude-opus-5-5": True})
    Predictor.create("new", [], {"model": "voyage-4", "dimensions": 256}, features=["prompt"], model={"type": "quick"})
    set_default("main")
    typing(monkeypatch, "fix the parser")
    pressing(monkeypatch, "1", "2", "1", "1", keys.ENTER, keys.ESC, keys.ESC)  # new, which can't, then main
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "new needs training before it can forecast. Choose Update on the home screen to train it." in captured.err
    assert re.search(r"claude-opus-5-5\s+max\s+60 s", plain(captured.out))


def test_estimate_with_no_ai_model_that_has_enough_data_says_so(capsys, monkeypatch, quick, ready):
    trained("main", {"claude-opus-5-5": False, "gpt-6.1-sol": False})
    typing(monkeypatch, "fix the parser", "")
    # AI models with enough data, which says there are none, then back to the task, home and quit.
    pressing(monkeypatch, "1", "1", keys.ESC, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert re.search(r"1\. AI models with enough data\s+none yet", out)
    assert "No AI model has enough data yet." in out


def test_estimate_with_ai_models_that_ran_too_few_tasks_says_so_and_goes_home(capsys, monkeypatch, quick, ready):
    predictor = trained("main", {"claude-opus-5-5": True})
    meta = json.loads((predictor.model_dir / "model.json").read_text())
    meta["effort_counts"] = {"claude-opus-5-5": [["max", 5]]}  # under the 10 runs a person's list needs
    (predictor.model_dir / "model.json").write_text(json.dumps(meta))
    typing(monkeypatch, "fix the parser")
    pressing(monkeypatch, "1", keys.ESC)
    assert main([]) == 0
    squashed = " ".join(plain(capsys.readouterr().out).split())
    assert "Not enough data to forecast with main yet. Each AI model needs 10 runs." in squashed
    assert "These ran fewer: claude-opus-5-5 (5 runs)." in squashed
    monkeypatch.setattr(cli, "terminal_output", lambda: True)
    assert main(["predict", "fix the parser"]) == 0  # the command says the same, in its table's place
    assert "Not enough data to forecast claude-opus-5-5 (5 runs)." in " ".join(plain(capsys.readouterr().out).split())
    assert main(["predict", "  "]) == 1
    assert "Give the task some text" in capsys.readouterr().err


def test_a_key_that_looks_like_a_digit_but_isnt_one_is_passed_over(capsys, monkeypatch):
    from yardstick_predict.screens import checklist

    pressing(monkeypatch, "²", "2")
    assert keys.choose(CHOICES) == 1
    pressing(monkeypatch, "²", keys.ENTER)
    assert checklist.tick([keys.Choice("a"), keys.Choice("b")]) == [0]
    assert checklist.tick([]) is None  # nothing to pick from
    monkeypatch.setattr(keys, "reader", lambda: None)
    typing(monkeypatch, "²", "3")
    assert keys.choose(CHOICES) == 2


def test_with_one_model_estimate_asks_nothing_about_models(capsys, monkeypatch, quick, ready):
    trained("main", {"claude-opus-5-5": True})
    typing(monkeypatch, "fix the parser")
    pressing(monkeypatch, "1", "1", keys.ENTER, keys.ESC, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "Which of your models" not in out and "Use another of your models" not in out


# Fitting the terminal -----------------------------------------------------------------------------------------

LONG_NAME = "a-very-long-model-name-for-testing-12345"
"""A model name as long as one can be: 40 characters."""


def widest(out: str) -> int:
    """The widest line a person sees, with each redraw of a menu counted as its own line."""
    return max(ui.width(line) for line in plain(out).replace("\r", "\n").split("\n"))


@pytest.mark.parametrize("width", [80, 45])
def test_long_lines_and_questions_break_to_fit_and_a_command_stays_whole(capsys, monkeypatch, width):
    monkeypatch.setenv("COLUMNS", str(width))
    monkeypatch.setenv("FORCE_COLOR", "1")
    command = "yardstick-predict refresh --predictor main"
    ui.note(f"This model was made by the previous version. It still forecasts. Train it again with: {command}")
    ui.ok("Forecasts are on for claude-opus-5-5, claude-sonnet-5-5, gpt-6-astra and gpt-6-luna at every level.")
    ui.table([("1", "Claude subscription", ui.dim("through Claude Code. It counts toward your plan's usage limits."))])
    asked = ui.prompt("Use your claude-code records in their default location (~/.claude/projects)?", "[Y/n]")
    out = plain(capsys.readouterr().out)
    assert widest(out) < width and widest(asked) < width
    assert any(line.strip() == command or line.endswith(f" {command}") for line in out.splitlines())  # whole
    done = out.splitlines()[out.splitlines().index(next(line for line in out.splitlines() if "✓" in line)) + 1]
    assert done.startswith("    ") and not done.startswith("     ")  # lined up after the mark
    assert plain(asked).endswith("[Y/n] ") and "\n    " in asked  # the answer goes after the last line
    assert ui.prompt("Short?", "[Y/n]") == f"  {ui.accent('?')} Short? {ui.dim('[Y/n]')} "


@pytest.mark.parametrize("width", [80, 45])
def test_the_features_menu_and_the_ways_to_pay_fit_the_terminal(capsys, monkeypatch, width):
    from yardstick_predict import features, menu

    monkeypatch.setenv("COLUMNS", str(width))
    groups = list(features.MENU)
    menu._show(groups, {g.key: {o.key for o in g.options} for g in groups if not g.always}, {}, {})
    menu.ui.table(
        [(str(n), way.title, way.says) for n, way in enumerate(llm.PROVIDERS.values(), start=1)]
    )  # as the question on how to pay shows them
    out = capsys.readouterr().out
    assert widest(out) < width
    # Each option says on or off at the end of its line, lined up. Where there's no room for that, it says so first.
    lined_up = re.search(r"^ {8}a  what its earlier tasks took and how big it is +on$", out, re.M)
    first = re.search(r"^ {8}a  on   what its earlier tasks took", out, re.M)
    assert (lined_up, first) != (None, None) and (lined_up is not None) == (width == 80)


def test_a_task_as_a_title_takes_three_lines_at_most(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "45")
    cli.show_title("add a dark mode toggle to the settings page", "page 1 of 2")
    assert plain(capsys.readouterr().out).splitlines() == [
        "  add a dark mode toggle to the settings",
        "  page  page 1 of 2",
    ]
    cli.show_title("fix the parser " * 30)
    shown = plain(capsys.readouterr().out).splitlines()
    assert len(shown) == 3 and shown[-1].endswith("…") and widest("\n".join(shown)) < 45


def test_a_long_menu_shows_6_at_a_time_and_turns_the_page(capsys, monkeypatch):
    rows = [keys.Choice(f"model {number}", "updated 1 hour ago") for number in range(12)] + [keys.Choice("Back")]
    pressing(monkeypatch, keys.RIGHT, "7")
    assert keys.choose(rows) == 6  # the first on the second page, numbered on from the first page
    out = plain(capsys.readouterr().out)
    assert "Page 1 of 3, 13 in all" in out and "Page 2 of 3, 13 in all" in out and "turn the page" in out
    assert "model 6" not in out.split("Page 1 of 3")[0]  # the first page shows 6
    assert re.search(r" 7\. model 6 ", out) and re.search(r"12\. model 11 ", out)
    pressing(monkeypatch, "1", "2")
    assert keys.choose(rows) == 11  # two digits typed quickly make 12
    pressing(monkeypatch, "1")
    assert keys.choose(rows) == 0  # one digit and then nothing is 1
    pressing(monkeypatch, keys.UP, keys.ENTER)  # up from the first goes round to Back, on the last page
    assert keys.choose(rows) == 12


def test_the_keys_line_under_a_list_breaks_only_where_it_must(capsys, monkeypatch):
    from yardstick_predict.screens import checklist

    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("FORCE_COLOR", "1")  # the dots between the keys are colored, and that takes no room
    pressing(monkeypatch, keys.ENTER)
    checklist.tick([keys.Choice("claude-opus-5-5  high"), keys.Choice("claude-opus-5-5  max")])
    out = plain(capsys.readouterr().out)
    assert "↑↓ move · enter forecast this one · space tick several · esc go back" in out


@pytest.mark.parametrize("width", [80, 45])
def test_menus_cut_long_names_to_fit(capsys, monkeypatch, width):
    from yardstick_predict.screens import checklist

    monkeypatch.setenv("COLUMNS", str(width))
    pressing(monkeypatch, keys.DOWN, keys.ENTER)
    keys.choose([keys.Choice(f"{LONG_NAME} (default)", "updated 12 hours ago"), keys.Choice("Back")])
    pressing(monkeypatch, keys.ENTER)
    checklist.tick([keys.Choice(f"{LONG_NAME}  default", ui.yellow("⚠ failed the test"))] * 2)
    assert widest(capsys.readouterr().out) < width


@pytest.mark.parametrize("width", [80, 45])
def test_the_home_screen_and_a_models_screen_fit_with_long_names(capsys, monkeypatch, width):
    monkeypatch.setenv("COLUMNS", str(width))
    made(LONG_NAME, served=tuple(f"claude-some-really-long-ai-model-name-{n}" for n in range(5)))
    made("short")
    pressing(monkeypatch, "3", "1", "2", keys.ESC, keys.ESC, keys.ESC)  # Models, the long one, Make default
    assert main([]) == 0
    out = capsys.readouterr().out
    assert widest(out) < width and "now the default" in plain(out)


def test_the_update_summary_fits_with_a_long_name_and_folder(capsys, monkeypatch, quick, ready, records, tmp_path):
    folder = tmp_path / "a" / "folder" / "of" / "records" / "with" / "a" / "path" / "long" / "enough" / "to" / "wrap"
    history(folder, tasks=40, start=START)
    monkeypatch.setenv("COLUMNS", "60")
    assert main(["create", LONG_NAME, "--express", "--archive", f"claude-code={folder}", *quick]) == 0
    new_work(folder)
    capsys.readouterr()
    monkeypatch.setattr(update, "apply", lambda job, key: pytest.fail("It went ahead"))
    pressing(monkeypatch, "2", keys.DOWN, keys.ENTER, keys.ESC)  # Update, then Cancel
    assert main([]) == 0
    out = capsys.readouterr().out
    assert widest(out) < 60
    assert re.search(r"\n  [^\n]*" + LONG_NAME + r"\n    Embed \d texts with Voyage", plain(out))  # its own line
    assert re.search(r"…[^\n]*/to/wrap", plain(out))  # the folder is cut short at its start


def test_the_forecast_table_cuts_a_long_ai_model_name_to_fit(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    made("main")
    name = "claude-some-really-long-ai-model-name-for-testing"
    forecasts = [
        {
            "model": name,
            "effort": "default",
            "seconds": 960.0,
            "tokens": 23_000_000,
            "usd": 7.42,
            "distribution": curve(math.log(900), 1.2),
            "reason": "not_accurate_enough",
        }
    ]
    cli.show_forecasts("x", {"predictor": "main", "forecasts": forecasts})
    out = capsys.readouterr().out
    assert widest(out) < 80 and "claude-some-really" in out and "Not enough data: " + name in " ".join(out.split())


def test_a_narrow_terminal_gets_each_forecast_as_a_block(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "45")
    made("main")
    forecasts = [
        {
            "model": "claude-opus-5",
            "effort": "xhigh",
            "seconds": 240.0,
            "tokens": 20_600_000,
            "usd": 11.17,
            "distribution": curve(math.log(220), 0.5),
            "reason": "not_accurate_enough",
        },
        {"model": "claude-opus-5-5", "effort": "max", "seconds": 960.0, "tokens": 23_000_000, "usd": 7.42},
    ]
    cli.show_forecasts("x", {"predictor": "main", "forecasts": forecasts}, screen=True)
    out = capsys.readouterr().out
    assert widest(out) < 45 and "AI model" not in out  # no table: it wouldn't fit
    text = plain(out)
    # A small table of its own: what to expect on average, then the 25%, median and 75% lines, each named. The
    # curve has 220 at its middle on every target, and a quarter of it under about 157.
    lines = [
        r"\s+time\s+tokens\s+cost",
        r"\s+expected\s+4 min\s+20\.6M\s+\$11\.17",
        r"\s+25%\s+3 min\s+157\s+\$157",
        r"\s+median\s+4 min\s+220\s+\$220",
        r"\s+75%\s+5 min\s+308\s+\$308",
    ]
    assert re.search(r"claude-opus-5  ⚠  xhigh\n" + r"\n".join(lines) + "\n", text)
    # Without a curve, only what to expect. Names are never cut.
    assert re.search(r"claude-opus-5-5  max\n\s+time\s+tokens\s+cost\n\s+expected\s+16 min\s+23\.0M\s+\$7\.42\n", text)
    assert cli.QUARTILES_NOTE in " ".join(text.split())


@pytest.mark.parametrize("width", [80, 45])
def test_status_says_how_close_forecasts_came_in_plain_words_and_fits(capsys, monkeypatch, quick, width):
    monkeypatch.setenv("COLUMNS", str(width))
    predictor = made("main")
    scores = {"better_than_simple_rule": 0.12, "in_90_range": 0.88}
    report = {
        "tested_on": 300,
        "min_tasks_to_test": 60,
        "models": {
            "claude-opus-5-5": {"served": True, "reason": None, "tasks_tested": 200, "tasks": 600,
                                "failed_checks": [], "seconds": scores, "tokens": scores, "usd": scores},
            "gpt-6.1-sol": {"served": False, "reason": "not_accurate_enough", "tasks_tested": 100, "tasks": 330,
                            "failed_checks": ["seconds.simple_rule"],
                            "seconds": {"better_than_simple_rule": -0.42, "in_90_range": 0.45}},
            "claude-haiku-5-5": {"served": False, "reason": "not_enough_data", "tasks_tested": 0, "tasks": 6,
                                 "failed_checks": []},
        },
    }  # fmt: skip
    meta = {"tasks": 600, "report": report, "categories": {"model": list(report["models"])}}
    (predictor.model_dir / "model.json").write_text(json.dumps(meta))
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    text = " ".join(plain(out).split())
    assert "How close its forecasts came on each AI model's newest tasks, 300 in all" in text
    assert re.search(r"claude-opus-5-5 (✓ )?forecasts on(,| ·) checked on 200 tasks", text)
    assert re.search(r"gpt-6\.1-sol (⚠ )?not enough data(,| ·) checked on 100 tasks", text)
    assert "Time forecasts were 42% worse than the simple rule." in text
    assert re.search(r"claude-haiku-5-5 (⚠ )?not enough data(,| ·) 6 tasks, needs 60", text)
    assert not re.search(r"\btest", text.lower()) and widest(out) < width  # no word about a test


def test_the_setup_features_screen_fits_80_columns(capsys, monkeypatch):
    from yardstick_predict import menu

    monkeypatch.setenv("COLUMNS", "80")
    typing(monkeypatch, "")
    menu.ask({})
    assert widest(capsys.readouterr().out) < 80


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
    upgrade = f"Update with: {version.upgrade_command()}"
    assert f"yardstick-predict 99.0.0 is out. You have {__version__}. {upgrade}" in " ".join(out.split())
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


def test_the_notice_names_the_update_command_for_the_way_it_was_installed(monkeypatch):
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    uv = version.upgrade_command("/Users/me/.local/share/uv/tools/yardstick-predict")
    windows_uv = version.upgrade_command("C:/Users/me/AppData/Roaming/uv/data/tools/yardstick-predict")
    pipx = version.upgrade_command("/Users/me/.local/pipx/venvs/yardstick-predict")
    assert uv == windows_uv == "uv tool upgrade yardstick-predict"
    assert pipx == "pipx upgrade yardstick-predict"
    assert version.upgrade_command("/Users/me/project/.venv") == "pip install --upgrade yardstick-predict"
    monkeypatch.setenv("UV_TOOL_DIR", "/opt/tools")
    assert version.upgrade_command("/opt/tools/yardstick-predict") == "uv tool upgrade yardstick-predict"


def test_the_check_is_off_with_do_not_track_or_the_setting(monkeypatch):
    monkeypatch.delenv(version.OFF, raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    assert version.wanted()
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert not version.wanted()
    monkeypatch.delenv("DO_NOT_TRACK")
    store.save_setting("update_check", False)
    assert not version.wanted()


def test_versions_compare_by_their_numbers():
    assert version.newer("0.10.0", "0.9.1") and version.newer("1.0", "0.9.9")
    assert not version.newer("0.4.0", "0.4.0") and not version.newer("0.4.0", "0.5.0.dev0")


def curve(mu: float, sigma: float) -> dict:
    """A forecast's curve with no chance of a direct answer: one bell on the log scale for each of its targets."""
    part = {"mu": mu, "sigma": sigma}
    every = {target: {"direct": part, "work": part} for target in TARGETS}
    return {"shape": "two_part_lognormal", "direct_probability": 0.0, **every}


TARGETS = ("seconds", "tokens", "usd")


def test_the_default_effort_level_is_never_offered_to_a_person(capsys):
    trained("main", {"claude-opus-5-5": True, "claude-sonnet-5": True, "claude-sonnet-4-6": True})
    meta = json.loads((Predictor("main").model_dir / "model.json").read_text())
    meta["effort_counts"] = {
        "claude-opus-5-5": [["max", 200], ["default", 70], ["high", 40]],
        "claude-sonnet-5": [["default", 58], ["high", 5]],  # its named level is rare, but it's the one shown
        "claude-sonnet-4-6": [["default", 40]],  # it ran only at the default level, so no level is named
    }
    pairs, runs, _ = cli.every_pair(Predictor("main"), None, meta)
    assert pairs == [
        ("claude-opus-5-5", "max"),
        ("claude-opus-5-5", "high"),
        ("claude-sonnet-5", "high"),
        ("claude-sonnet-4-6", None),
    ]
    assert cli.for_people([("m", "max"), ("m", "default"), ("n", "default")]) == [("m", "max"), ("n", None)]
    forecasts = [{"model": "claude-sonnet-4-6", "effort": "default", "seconds": 60.0, "tokens": 1000, "usd": 0.1}]
    made("other")
    cli.show_forecasts("x", {"predictor": "other", "forecasts": forecasts}, screen=True)
    assert "default" not in plain(capsys.readouterr().out).split("AI model")[1].split("Learned")[0]


def test_the_25_median_and_75_lines_are_where_the_curve_holds_that_share_of_results():
    from yardstick_predict.forecast import quantile

    bell = curve(math.log(600), 0.5)
    assert quantile(bell, "seconds", 0.5) == pytest.approx(600, rel=1e-3)
    assert quantile(bell, "seconds", 0.25) == pytest.approx(600 * math.exp(-0.6745 * 0.5), rel=1e-3)
    assert quantile(bell, "seconds", 0.75) == pytest.approx(600 * math.exp(0.6745 * 0.5), rel=1e-3)
    assert quantile(None, "seconds", 0.5) is None and quantile({"shape": "x"}, "seconds", 0.5) is None
    # A quarter of this curve is a quick direct answer. So its median sits a third of the way into the rest.
    parts = {"direct": {"mu": math.log(60), "sigma": 0.5}, "work": {"mu": math.log(600), "sigma": 0.5}}
    mixed = {"shape": "two_part_lognormal", "direct_probability": 0.25, "seconds": parts}
    assert quantile(mixed, "seconds", 0.5) == pytest.approx(600 * math.exp(-0.4307 * 0.5), rel=1e-3)


def test_the_forecast_table_shows_the_25_median_and_75_lines_and_warns_on_rough_forecasts(capsys):
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
    # Under it, the lines a quarter, half and three quarters of results come in under: the curve has 900 at its
    # middle on every target, a quarter of it under about 687 and three quarters under about 1,179.
    assert re.search(r"\n\s+25%\s+11 min\s+687\s+\$687\s*\n\s+median\s+15 min\s+900\s+\$900\s*\n", out)
    assert re.search(r"\n\s+75%\s+20 min\s+1k\s+\$1\.2k\s*\n", out)
    assert re.search(r"⚠\s+high\s+5 min\s+7\.7M\s+\$3\.18", out)  # marked, with why under the table
    squashed = " ".join(out.split())
    assert "⚠ Not enough data: claude-opus-5-5 at high, gpt-6.1-sol and gpt-6-astra." in squashed
    assert "Not enough data to forecast claude-haiku-5-5 (6 runs) and claude-opus-4-7 (3 runs)." in squashed
    assert cli.QUARTILES_NOTE in squashed
    assert "Those are a rough guide. Plan with a wider margin there." in out
    assert "Create a fun game" not in out and "model main, updated" in out and "--json" not in out
    cli.show_forecasts("Create a fun game", result)  # the predict command says the task, and names the predictor
    out = plain(capsys.readouterr().out)
    squashed = " ".join(out.split())
    assert "Create a fun game" in out and "predictor main, updated" in out and "add --json for details" in squashed


def test_a_program_never_gets_a_forecast_with_not_enough_data_but_a_person_sees_it_marked(
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
    assert re.search(r"gpt-6\.1-sol\s+⚠\s+60 s", out) and "Not enough data: gpt-6.1-sol." in out


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
    assert "llm is now the default." in out and "Estimate and Claude Code's status line use it." in out
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


def test_delete_says_what_goes_asks_with_keep_it_first_and_deletes(capsys, monkeypatch):
    made("main")
    made("old")
    set_default("main")
    # Models, old, Delete, and Enter keeps it. Then Delete again, down to Delete it for good and Enter. Then home.
    pressing(monkeypatch, "3", "2", "4", keys.ENTER, "4", keys.DOWN, keys.ENTER, keys.ESC, keys.ESC)
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    squashed = " ".join(out.split())
    assert re.search(r"4\. Delete\s+Remove it for good", out) and "Delete old" in out
    assert "This deletes its folder" in squashed and "its settings" in squashed
    assert "Your records and the feature store stay, and so do your other models." in squashed
    assert "1. Keep it" in out and "2. Delete it for good" in out and "esc keep it" in out
    assert "Nothing was deleted." in out and "Deleted old." in out
    assert predictor_names() == ["main"] and default_name() == "main"
    assert "1. main (default)" in out.split("Deleted old.")[1]  # back on the list, which shows what's left


def test_deleting_the_default_model_says_which_is_the_default_now(capsys, monkeypatch):
    for name in ("a", "b", "c"):
        made(name)
    set_default("a")
    pressing(monkeypatch, "3", "1", "3", keys.DOWN, keys.ENTER, keys.ESC, keys.ESC)  # Models, a, Delete it for good
    assert main([]) == 0
    squashed = " ".join(plain(capsys.readouterr().out).split())
    assert "It's your default model." in squashed
    assert "It was your default. There's no default now." in squashed and default_name() is None
    set_default("b")
    pressing(monkeypatch, "3", "1", "3", keys.DOWN, keys.ENTER, keys.ESC, keys.ESC)  # b, the default, of two
    assert main([]) == 0
    assert "c is your only model now. It's the default." in plain(capsys.readouterr().out)
    assert predictor_names() == ["c"] and default_name() == "c"
    made("default")
    made("x")
    set_default("c")  # of three, so the one called default becomes the default once c goes
    pressing(monkeypatch, "3", "1", "3", keys.DOWN, keys.ENTER, keys.ESC, keys.ESC)  # c, the default
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "Deleted c." in out and "default is the default now." in out and "only model" not in out


def test_a_model_being_updated_isnt_deleted_from_its_screen(capsys, monkeypatch):
    from yardstick_predict import locks

    made("other")
    made("work")
    with locks.held(store.predictor_folder("work") / "refresh.lock"):
        pressing(monkeypatch, "3", "2", "4", keys.ESC, keys.ESC, keys.ESC)  # Models, work, Delete
        assert main([]) == 0
    out = plain(capsys.readouterr().out)
    busy = "work is being set up or updated by another command. Delete it once that's done."
    assert busy in " ".join(out.split())
    assert "Keep it" not in out and predictor_names() == ["other", "work"]


@pytest.mark.parametrize("width", [80, 45])
def test_the_delete_question_fits_with_a_long_name(capsys, monkeypatch, width):
    monkeypatch.setenv("COLUMNS", str(width))
    made(LONG_NAME)
    made("short")
    pressing(monkeypatch, "3", "1", "4", keys.ESC, keys.ESC, keys.ESC, keys.ESC)  # Models, the long one, Delete
    assert main([]) == 0
    out = capsys.readouterr().out
    assert widest(out) < width and "Nothing was deleted." in plain(out) and LONG_NAME in predictor_names()


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
    known = Predictor("main").known_files()
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
    assert "1. Update main" in out and "2. Cancel" in out
    assert "Nothing was spent or saved." in out
    # Cancel spent nothing and saved nothing.
    assert len(ready.requests) == sent and not calls
    assert Predictor("main").count_tasks() == 40 and Predictor("main").known_files() == known


def test_update_all_tries_voyage_once_when_it_cant_be_reached(capsys, monkeypatch, quick, ready, records):
    from yardstick_predict import embeddings

    assert main(["create", "--express", *quick]) == 0
    assert main(["create", "other", "--express", *quick]) == 0
    new_work(records)
    capsys.readouterr()
    monkeypatch.setenv("VOYAGE_API_BASE", "http://127.0.0.1:1/v1")  # nothing answers there
    sent, real = [], embeddings._post
    monkeypatch.setattr(embeddings, "_post", lambda *args, **kwargs: sent.append(1) or real(*args, **kwargs))
    pressing(monkeypatch, "2", keys.ENTER, keys.ENTER, keys.ESC)  # Update, Update all, go ahead
    assert main([]) == 0
    out = " ".join(plain(capsys.readouterr().out).split())
    assert "main stopped: Couldn't reach Voyage (" in out and out.count("Couldn't reach Voyage") == 1
    assert "Voyage said 0" not in out and "Check your internet connection." in out
    assert "other wasn't embedded: Voyage couldn't be reached a moment ago." in out
    assert len(sent) == embeddings.PATIENT.attempts  # every try was for the first model


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
    assert "main updated, forecasts on for claude-opus-5-5" in out
    assert Predictor("main").count_tasks() == 43 and len(Predictor("main").embedded("prompt")) == 43
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
    known = Predictor("main").known_files()
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
    assert Predictor("main").known_files() == known and Predictor("main").count_tasks() == 40
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


def test_refresh_with_a_spending_limit_keeps_to_it_on_the_update_screen(capsys, monkeypatch, quick, ready, records):
    assert main(["create", "--express", *quick]) == 0
    new_work(records)
    capsys.readouterr()
    pressing(monkeypatch)  # no key is pressed: it stops before the question
    assert main(["refresh", "--max-cost", "0"]) == 1
    assert "more than your limit of $0. Nothing was spent or saved." in " ".join(capsys.readouterr().err.split())
    assert Predictor("main").count_tasks() == 40
    pressing(monkeypatch, keys.ENTER)  # Update default
    assert main(["refresh", "--max-cost", "1"]) == 0
    assert "Found 3 new tasks" in plain(capsys.readouterr().out) and Predictor("main").count_tasks() == 43


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
    assert "Found 3 new tasks" in capsys.readouterr().out and Predictor("main").count_tasks() == 43


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


def test_add_new_on_the_home_screen_calls_it_a_model(capsys, monkeypatch):
    made("main")
    pressing(monkeypatch, "4", keys.ESC)  # Add New, then Quit once it stops
    asked = typing(monkeypatch, "work", "")  # its name, then not the default
    monkeypatch.setattr(cli, "choose_archives", lambda can_ask, given: [])  # no records, so it stops there
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "New model" in out and "predictor" not in out.split("New model")[1].split("No records")[0]
    assert [flat(question) for question in asked] == [
        "Name this model:",
        "Make it your default model? Commands use it when they name none. [y/N]",
    ]


def test_setup_suggests_main_for_the_first_model(monkeypatch):
    asked = typing(monkeypatch, "")
    assert cli.ask_name([]) == "main" and asked == ["Name this predictor: [main]"]
    asked = typing(monkeypatch, "", "work")  # a later one gets no name it could clash with
    assert cli.ask_name(["main"]) == "work" and asked == ["Name this predictor:", "Name this predictor:"]


@pytest.mark.skipif(os.name == "nt", reason="a pseudo terminal is for macOS and Linux")
def test_keys_pressed_quickly_each_count(monkeypatch):
    import pty
    import tty

    sent, received = pty.openpty()
    tty.setcbreak(received)  # nothing typed is shown back, as no one reads what the terminal shows
    monkeypatch.setattr("sys.stdin", os.fdopen(received, "r"))
    monkeypatch.setattr(keys, "_ahead", bytearray())
    # An arrow and Enter at once, two arrows at once, an arrow of the other mode, j, a key that types ², Esc and k
    # at once, then Esc on its own.
    os.write(sent, b"\x1b[B\r\x1b[A\x1b[A\x1bOCj\xc2\xb2\x1bk\x1b")
    try:
        got = [keys._posix_key(timeout=1) for _ in range(10)]
    finally:
        os.close(sent)
    assert got == [keys.DOWN, keys.ENTER, keys.UP, keys.UP, keys.RIGHT, keys.DOWN, "²", keys.ESC, keys.UP, keys.ESC]


# Windows ---------------------------------------------------------------------------------------------------------


class FakeConsole:
    """A stand-in for Windows' keyboard reader, msvcrt, that gives these characters in turn."""

    def __init__(self, *typed: str) -> None:
        self.typed = list(typed)

    def kbhit(self) -> bool:
        return bool(self.typed)

    def getwch(self) -> str:
        return self.typed.pop(0)


@pytest.mark.parametrize(
    ("typed", "key"),
    [
        (("\xe0", "H"), keys.UP),
        (("\x00", "P"), keys.DOWN),  # the arrows on the number pad
        (("\xe0", "M"), keys.RIGHT),
        (("\xe0", "K"), keys.LEFT),
        (("\r",), keys.ENTER),
        (("\x1b",), keys.ESC),
        ((" ",), keys.SPACE),
        (("7",), "7"),
        (("\xe0", "G"), ""),  # Home, which no menu uses
    ],
)
def test_windows_keys_are_read_as_menus_name_them(monkeypatch, typed, key):
    import sys

    monkeypatch.setitem(sys.modules, "msvcrt", FakeConsole(*typed))
    assert keys._windows_key() == key


def test_windows_ctrl_c_stops_and_a_wait_for_a_second_digit_ends(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "msvcrt", FakeConsole("\x03"))
    with pytest.raises(KeyboardInterrupt):
        keys._windows_key()
    monkeypatch.setitem(sys.modules, "msvcrt", FakeConsole())
    assert keys._windows_key(timeout=0.05) == ""  # no second digit came


def test_esc_stops_a_long_step_on_windows_but_an_arrow_doesnt(monkeypatch):
    import sys

    monkeypatch.setattr(keys, "_terminal", lambda: "windows")
    console = FakeConsole("\xe0", "H")
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    with keys.watching() as watch:
        watch.stop_if_asked()  # an arrow key: it goes on
        console.typed.append("\x1b")
        with pytest.raises(keys.Stopped):
            watch.stop_if_asked()


def test_a_terminal_that_cant_move_the_cursor_gets_numbered_menus(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert not ui.codes_work() and keys._terminal() is None


# Numbered menus on request --------------------------------------------------------------------------------------


class Untouched:
    """Standard input that fails a test if anything asks whether it's a terminal."""

    def isatty(self) -> bool:
        pytest.fail("It looked at the terminal")


def test_numbered_menus_can_be_asked_for_and_nothing_redraws_itself(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv(ui.NUMBERED, "1")
    ui.numbered.cache_clear()
    monkeypatch.setattr("sys.stdin", Untouched())
    assert keys._terminal() is None and keys.reader() is None  # every menu is typed into
    with keys.watching() as watch:
        watch.stop_if_asked()  # Esc can't stop a long step: Ctrl+C still does
    ui.working("Reading your records")
    ui.progress(500, 1000, "bytes")
    ui.ok("Read 1 KB.")
    assert plain(capsys.readouterr().out) == "  Reading your records...\n  ✓ Read 1 KB.\n"  # no bar, no line drawn over


def test_numbered_menus_are_a_setting_too_and_the_home_screen_has_no_box(capsys, monkeypatch):
    made("main")
    store.save_setting("numbered_menus", True)
    monkeypatch.setattr(cli, "interactive", lambda: True)
    typing(monkeypatch, "5")  # Quit, typed
    assert main([]) == 0
    out = plain(capsys.readouterr().out)
    assert "╭" not in out and "│" not in out and "5. Quit" in out and "Your models" in out
    store.save_setting("numbered_menus", False)
    ui.numbered.cache_clear()
    assert not ui.numbered()
    monkeypatch.setenv(ui.NUMBERED, "0")  # the variable says no, whatever the setting
    store.save_setting("numbered_menus", True)
    ui.numbered.cache_clear()
    assert not ui.numbered()


# Text from outside ---------------------------------------------------------------------------------------------


def test_text_from_outside_is_shown_without_codes_a_terminal_acts_on(capsys):
    link = "\x1b]8;;https://x\x1b\\link\x1b]8;;\x1b\\"  # a link the terminal opens on a click
    said = f"said: \x1b[31mred\x1b[0m \x1b]0;a title\x07done\r over \u202eback {link} end"
    assert ui.clean(said) == "said: red done over back link end"
    assert ui.clean("two\nlines\tand a tab") == "two\nlines\tand a tab"  # new lines and tabs stay
    ui.error("It broke: \x1b]0;owned\x07for good")
    assert capsys.readouterr().err == "Error: It broke: for good\n"


def test_a_version_from_pypi_shows_only_when_it_looks_like_one(monkeypatch):
    monkeypatch.delenv(version.OFF, raising=False)
    monkeypatch.setattr(version, "_looking", True)  # no new look at PyPI
    for latest, shown in (("99.0.0\x1b]0;a title\x07", False), ("99.0.0", True), ("99.0.0rc1", True)):
        store.private_home()
        (store.home() / "update-check.json").write_text(json.dumps({"at": time.time(), "latest": latest}))
        assert (version.notice() is not None) == shown, latest


def test_the_lines_of_cheap_tasks_still_say_something():
    assert [cli.short_money(usd) for usd in (0.0042, 0.0087, 0.0094)] == ["$0.0042", "$0.0087", "$0.0094"]
    assert cli.cost_lines(embeddings_cost(5_000), "voyage-4", 3)[1] == "It costs <$0.01 at Voyage's list price."


def embeddings_cost(tokens: int):
    from yardstick_predict import embeddings

    return embeddings.Cost(tokens, True, 0.06, None)
