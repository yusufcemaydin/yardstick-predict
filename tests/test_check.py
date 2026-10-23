"""The feature check: which of a predictor's features help, as setup offers it and the check command runs it."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace

import numpy as np
import pytest
from builders import history
from conftest import KEY
from test_cli import fake_analysis, quick_train
from test_model import ONE_WINDOW, kinds, rows_for, synthetic

from yardstick_predict import benchmarks, check, cli, estimate, features, llm, locks, menu, model_types, store
from yardstick_predict import model as M
from yardstick_predict.cli import main
from yardstick_predict.history import Past
from yardstick_predict.store import Predictor

START = 1_780_000_000.0
ANSI = re.compile(r"\033\[[0-9;]*[mK]")


# Leaving one feature out ------------------------------------------------------------------------------------


def every_feature_row() -> M.Features:
    """A row with something from every feature: what's always used, each optional feature and two of your own."""
    numbers = {
        "prompt_lines": 0.7,
        "effort_level": 3.0,
        "project_seconds": 4.2,
        "project_tasks_before": 1.1,
        **{name: 1.0 for name in M.CONVERSATION},
        M.CLOSENESS: 0.4,
        "benchmark_price_input": 0.7,
        "benchmark_released": 5.5,
        "days_since_2020": 2_300.5,
        "hour_of_day": 13.75,
        "day_of_week": 2.0,
        "llm_minutes": 1.6,
        "llm_kind_feature": 1.0,
        "llm_clarity": 0.0,
        "words": 12.0,  # your own number feature
    }
    names = ("prompt", "context", "skills", "llm", M.JOINT, "project")  # the last is your own text feature
    vectors = {name: np.full(4, 0.5, dtype=np.float32) for name in names}
    return M.Features("claude-opus-5-5", "high", "claude-code", "person", 40, vectors, numbers)


GONE = {
    "conversation": ({*M.CONVERSATION, M.CLOSENESS}, {M.JOINT}),
    "context": (set(), {"context"}),
    "skills": (set(), {"skills"}),
    "benchmarks": ({"benchmark_price_input", "benchmark_released"}, set()),
    "time": ({"days_since_2020", "hour_of_day", "day_of_week"}, set()),
    "llm": ({"llm_minutes", "llm_kind_feature", "llm_clarity"}, {"llm"}),
    "words": ({"words"}, set()),
    "project": (set(), {"project"}),
}
"""What leaving out each feature takes from every_feature_row: its numbers and its vectors."""


def test_every_built_in_feature_can_be_left_out():
    assert set(features.BUILT_IN) <= set(GONE)


@pytest.mark.parametrize("name", list(GONE))
def test_leaving_a_feature_out_removes_exactly_its_numbers_and_vectors(name):
    row = every_feature_row()
    numbers, vectors = dict(row.numbers), set(row.vectors)
    after = M.without([row], name)[0]
    assert set(row.numbers) - set(after.numbers) == GONE[name][0]
    assert set(row.vectors) - set(after.vectors) == GONE[name][1]
    assert all(after.numbers[key] == value for key, value in row.numbers.items() if key in after.numbers)
    assert row.numbers == numbers and set(row.vectors) == vectors  # the row it was given stays as it was
    assert M.has_feature(row, name) and not M.has_feature(after, name)


def test_each_feature_leaves_out_the_numbers_the_pipeline_really_gives_it():
    listing = {
        "data": [
            {
                "id": "anthropic/claude-opus-5.5",
                "pricing": {"prompt": "0.000005", "completion": "0.000025", "input_cache_read": "0.0000005"},
                "context_length": 1_000_000,
                "top_provider": {"max_completion_tokens": 128_000},
                "created": 1_770_000_000,
                "supported_parameters": ["reasoning"],
            }
        ]
    }
    table = benchmarks.parse(listing)
    said = {"kind": "feature", "files": 2, "steps": 3, "runs_code": True, "minutes": 5.0, "clarity": "clear"}
    info = features.TaskInfo(prompt="Fix the login page", model="claude-opus-5-5", started_at=START, llm=said)
    plain = features.numbers([], info, table)
    for name in ("benchmarks", "time", "llm"):
        found = features.numbers([name], info, table)
        assert len(found) > len(plain), name
        row = M.Features("claude-opus-5-5", "default", "", "person", 18, numbers=found)
        assert M.without([row], name)[0].numbers == plain, name
    assert set(features.time_numbers(START, ["trend", "clock"])) == set(features.TIME_NUMBERS)
    # The conversation's numbers come from what came before the task.
    tasks = synthetic(40)
    past = Past(tasks)
    at = tasks[12]
    with_it = past.numbers("demo", at.session, at.started_at, "person", 30_000, conversation=True)
    without_it = past.numbers("demo", at.session, at.started_at, "person", 30_000, conversation=False)
    row = M.Features(at.model, "high", "", "person", 18, numbers=with_it)
    assert len(with_it) > len(without_it) and M.without([row], "conversation")[0].numbers == without_it


def test_the_model_without_llm_analysis_leaves_it_out_the_same_way():
    row = every_feature_row()
    plain, left_out = M._without_llm([row])[0], M.without([row], "llm")[0]
    assert (set(plain.numbers), set(plain.vectors)) == (set(left_out.numbers), set(left_out.vectors))
    assert M._read_by_llm(row) and not M._read_by_llm(plain)


# The verdicts ---------------------------------------------------------------------------------------------


def results(**scores: float) -> dict:
    """An AI model's results in a test, with a score for each head named."""
    return {"tasks_tested": 50, "heads": list(scores), **{head: {"score": score} for head, score in scores.items()}}


def test_a_feature_helped_when_the_test_scored_more_than_1_percent_worse_without_it():
    with_all = results(seconds=1.0)
    assert check.verdict(with_all, results(seconds=1.02), 0.9) == {
        "verdict": "helped",
        "worse_without": 0.02,
        "share": 0.9,
    }
    assert check.verdict(with_all, results(seconds=1.01), 0.9)["verdict"] == "no_help"  # 1% isn't more than 1%
    assert check.verdict(with_all, results(seconds=0.95), 0.9) == {
        "verdict": "no_help",
        "worse_without": -0.05,
        "share": 0.9,
    }
    # Each head both tests scored counts alike: 3% worse time and 3% better cost is no change at all.
    assert check.verdict(results(seconds=1.0, usd=2.0), results(seconds=1.03, usd=1.94), 1.0)["worse_without"] == 0
    # A head only one of the tests scored doesn't count.
    assert check.verdict(results(seconds=1.0, usd=2.0), results(seconds=1.05), 1.0)["worse_without"] == 0.05


def test_without_enough_data_there_is_no_verdict():
    tested, untested = results(seconds=1.0), {"tasks_tested": 0, "heads": []}
    assert check.verdict(untested, untested, 1.0) == {"verdict": "not_enough_data", "worse_without": None, "share": 1.0}
    # Under 20% of the AI model's test tasks have the feature: whatever the scores say, the test can't tell.
    assert check.verdict(tested, results(seconds=1.5), 0.19)["verdict"] == "not_enough_data"
    assert check.verdict(tested, results(seconds=1.5), 0.2)["verdict"] == "helped"
    assert check.verdict(tested, results(seconds=1.5), None)["verdict"] == "not_enough_data"
    assert check.verdict(tested, {}, 1.0)["verdict"] == "not_enough_data"


def test_each_feature_gets_a_verdict_for_each_ai_model_the_test_knows():
    untested = {"tasks_tested": 0, "heads": []}
    full = {"tested_on": 130, "models": {"a": results(seconds=1.0), "b": {**results(seconds=2.0), "tasks_tested": 80}}}
    full["models"]["c"] = untested
    reports = {
        "": full,
        "context": {"models": {"a": results(seconds=1.1), "b": results(seconds=2.0), "c": untested}},
        "time": full,
    }
    shares = {"context": {"a": 1.0, "b": 0.5}, "time": {"a": 1.0, "b": 1.0}}
    result = check.judge(reports, shares, ["context", "time"])
    assert result.models == ["b", "a", "c"]  # the most tested first
    verdicts = {name: {model: v["verdict"] for model, v in found.items()} for name, found in result.verdicts.items()}
    assert verdicts == {
        "context": {"a": "helped", "b": "no_help", "c": "not_enough_data"},
        "time": {"a": "no_help", "b": "no_help", "c": "not_enough_data"},
    }
    assert result.helped() == ["context"] and result.tasks_tested("b") == 80 and result.tasks_tested("z") == 0


def test_the_check_tests_with_every_feature_then_without_each_and_fits_no_final_model(monkeypatch):
    tasks, prompt, context = kinds(600)
    noise = np.random.default_rng(5).normal(size=len(tasks))
    rows = [
        replace(r, numbers={"noise": float(n)})
        for r, n in zip(rows_for(tasks, {"prompt": prompt, "context": context}), noise)
    ]
    fitted, real = [], M.fit
    monkeypatch.setattr(M, "fit", lambda *args, **kwargs: fitted.append(len(args[0])) or real(*args, **kwargs))
    told: list[tuple[int, int]] = []
    names = ["context", "noise", "skills"]
    reports, shares = M.compare(tasks, rows, ONE_WINDOW, names, lambda done, total: told.append((done, total)))
    # One trial for the one test window: with every feature, without the context and without the noise. No task
    # has skills, so the test without them would be the same as with every feature, and doesn't run again.
    assert len(fitted) == 3 and told[0] == (0, 3) and told[-1] == (3, 3)
    assert reports["skills"] is reports[""] and set(reports) == {"", *names}
    assert shares == {name: {"claude-opus-5-5": share} for name, share in zip(names, (1.0, 1.0, 0.0))}
    result = check.judge(reports, shares, names)
    verdicts = {name: found["claude-opus-5-5"]["verdict"] for name, found in result.verdicts.items()}
    assert verdicts == {"context": "helped", "noise": "no_help", "skills": "not_enough_data"}
    assert result.verdicts["context"]["claude-opus-5-5"]["worse_without"] > 0.1


def test_llm_analysis_reads_a_sample_first_the_newest_30_percent_and_at_most_500():
    assert check.sample(list(range(180))) == list(range(54))
    assert check.sample(list(range(1_000))) == list(range(300))
    assert check.sample(list(range(5_000))) == list(range(500))
    assert check.sample(list(range(3))) == [0] and check.sample([]) == []


def test_each_ai_model_gets_its_own_newest_tasks_in_the_sample():
    # Newest first: an AI model used lately, then one used mostly long ago.
    model_of = {**{n: "lately" for n in range(0, 800)}, **{n: "long-ago" for n in range(800, 1_000)}}
    picked = check.sample(list(range(1_000)), model_of)
    assert len(picked) == 300 and picked == sorted(picked)
    assert sum(model_of[n] == "long-ago" for n in picked) == 60  # 30% of its 200, though they're the oldest
    assert picked[-60:] == list(range(800, 860))  # its newest
    many = {n: ("lately" if n % 2 else "long-ago") for n in range(10_000)}
    capped = check.sample(list(range(10_000)), many)
    assert len(capped) == 500 and {sum(many[n] == m for n in capped) for m in ("lately", "long-ago")} == {250}


def test_the_estimate_counts_the_check_as_a_step_of_its_own():
    tasks = synthetic(400)
    found = estimate.of(tasks, {"features": ["prompt", "context", "time"]}, 0, 0, 0, checked=2)
    sizes = M.fit_sizes(tasks, {})
    trials = sum(estimate.TRAIN_SECONDS_PER_MODEL + estimate.TRAIN_SECONDS_PER_TASK * size for size in sizes[:-1])
    assert list(found.steps) == ["feature check", "training"]
    assert found.steps["feature check"] == pytest.approx(3 * trials)  # every feature, then without each of two
    assert "feature check" not in estimate.of(tasks, {"features": ["prompt"]}, 0, 0, 0).steps
    said = estimate.line(estimate.Estimate({"embedding": 120, "feature check": 120, "training": 60}, 1e6, 1e9))
    assert said.startswith("It takes about 4 to 8 minutes: embedding 2 min, feature check 2 min, training 1 min.")


# Remembering the verdicts ----------------------------------------------------------------------------------


def checked(verdicts: dict[str, dict[str, str]], at: float = 100.0) -> check.Result:
    """What a check found, with these verdicts by feature and AI model."""
    found = {name: {model: {"verdict": v} for model, v in by_model.items()} for name, by_model in verdicts.items()}
    return check.Result(list(verdicts), {}, found, checked_at=at)


def test_the_latest_verdicts_are_remembered_with_their_date_and_predictor():
    assert check.latest() == {}  # nothing was checked yet
    assert check.remember(checked({"context": {"a": "helped", "b": "no_help"}}, at=100.0), "work")
    later = {"context": {"a": "not_enough_data", "b": "helped"}, "time": {"a": "not_enough_data"}}
    assert check.remember(checked(later, at=200.0), "lab")
    saved = check.latest()
    # Not enough data tells nothing new, so it never replaces a verdict that says whether a feature helped.
    assert saved["context"]["a"] == {"verdict": "helped", "checked_at": 100.0, "predictor": "work"}
    assert saved["context"]["b"] == {"verdict": "helped", "checked_at": 200.0, "predictor": "lab"}
    assert saved["time"] == {"a": {"verdict": "not_enough_data", "checked_at": 200.0, "predictor": "lab"}}
    assert check.path() == store.home() / "features" / "checks.json"
    if os.name == "posix":
        assert check.path().parent.stat().st_mode & 0o777 == 0o700  # its folder is yours alone
        assert check.path().stat().st_mode & 0o777 == 0o600


def test_a_missing_or_broken_file_of_verdicts_means_none_and_never_stops_anything():
    check.path().parent.mkdir(parents=True)
    for text in ("", "{not json", "[]", '{"features": []}', '{"features": {"context": {"a": {"verdict": "great"}}}}'):
        check.path().write_text(text)
        assert check.latest() == {}, text
    assert check.remember(checked({"time": {"a": "helped"}}), "work")  # a broken file is replaced
    assert check.latest()["time"]["a"]["verdict"] == "helped"
    # A file a newer version saved is read as far as it can be, and never saved over in this version's format.
    entry = {"verdict": "helped", "checked_at": 1.0, "predictor": "work", "more": "a newer field"}
    newer = {"format": check.FORMAT + 1, "features": {"context": {"a": entry}}}
    check.path().write_text(json.dumps(newer))
    assert check.latest() == {"context": {"a": entry}}
    assert not check.remember(checked({"time": {"a": "no_help"}}), "work")
    assert json.loads(check.path().read_text()) == newer


def test_the_features_screen_shows_the_latest_verdicts_and_starts_from_a_choice(monkeypatch, capsys):
    monkeypatch.setenv("NO_COLOR", "1")
    verdicts = {
        "context": {"claude-opus-5-5": "helped", "gpt-6.1-sol": "no_help"},
        "llm": {"claude-opus-5-5": "not_enough_data"},
    }
    check.remember(checked(verdicts), "work")
    monkeypatch.setattr("builtins.input", lambda question="": "")
    how = {"provider": "codex", "model": llm.DEFAULT_CODEX}
    picked = menu.ask(start=menu.Picked(["prompt", "context", "llm"], {"llm": ["kind"]}, how))
    out = capsys.readouterr().out
    assert "    Last check: helped claude-opus-5-5; didn't help gpt-6.1-sol." in out
    assert "    Last check: not enough data for claude-opus-5-5." in out
    assert out.index("the end of the conversation before each prompt") < out.index("Last check: helped")
    assert re.search(r" 4  Conversation +off", out) and re.search(r" 5  Context +on", out)
    assert picked.features == ["prompt", "context", "llm"] and picked.options == {"llm": ["kind"]}
    assert picked.llm == how and "How should it be paid for?" not in out  # LLM analysis isn't asked about again


# Setup offers the check -----------------------------------------------------------------------------------


@pytest.fixture
def records(tmp_path):
    """180 Claude Code tasks on two AI models: 90 each, enough to test each one."""
    root = tmp_path / "claude-projects"
    history(root, tasks=180, start=START, models=("claude-opus-5-5", "claude-sonnet-5-5"))
    return root


@pytest.fixture
def ready(voyage, monkeypatch):
    """A Voyage key and the stand-in Voyage server."""
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    return voyage


QUESTION = "Check which of these features actually help? Recommended."


def person(monkeypatch, **replies: str | list[str]) -> list[str]:
    """A person at a terminal. A question with words from a reply's name gets that reply, or the next of its
    replies. Every other question gets Enter. Returns every question asked."""
    asked: list[str] = []
    waiting = {words.replace("_", " "): [r] if isinstance(r, str) else list(r) for words, r in replies.items()}

    def answer(question: str = "") -> str:
        asked.append(question.strip())
        for words, left in waiting.items():
            if words in question and left:
                return left.pop(0) if len(left) > 1 else left[0]
        return ""

    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", answer)
    return asked


def stand_in_check(helped: set[str]):
    """A stand-in for the model's comparison: the features in `helped` scored 5% worse without them, on every AI
    model, and every task had every feature."""

    def compare(tasks, rows, settings=None, names=(), on_progress=None):
        models = sorted({task.model for task in tasks})

        def report(worse: float = 0.0) -> dict:
            return {"tested_on": 60, "models": {model: results(seconds=1.0 + worse) for model in models}}

        reports = {"": report(), **{name: report(0.05 if name in helped else 0.0) for name in names}}
        if on_progress is not None:
            on_progress(1, 1)
        return reports, {name: {model: 1.0 for model in models} for name in names}

    return compare


def test_setup_offers_the_check_and_trains_with_what_the_person_keeps(capsys, monkeypatch, records, ready):
    asked = person(monkeypatch)
    assert main(["create", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0
    out = capsys.readouterr().out
    assert sum(QUESTION in question for question in asked) == 1
    assert "Some features carry no signal for your kind of work, and with few tasks a feature can't show" in out
    # The question to go ahead says the check comes next, and the estimate counts it.
    assert any("Embed them, then check the features?" in question for question in asked)
    assert "feature check" in out
    assert re.search(r"Conversation +(helped|didn't help|not enough data) +(helped|didn't help|not enough data)", out)
    assert "Tested on each AI model's newest tasks, with every feature and then without each one." in out
    # The Features screen comes again, with the features that helped on, and the person pressed Enter to keep them.
    assert sum("Turn on or off" in question for question in asked) == 2
    saved = check.latest()
    helped = [name for name in saved if any(found["verdict"] == "helped" for found in saved[name].values())]
    assert Predictor("default").config["features"] == ["prompt", *helped]
    assert {entry["predictor"] for by_model in saved.values() for entry in by_model.values()} == {"default"}
    assert "Trained on" in out and "Ready" in out


def test_llm_analysis_reads_a_sample_for_the_check_and_the_rest_once_it_helps(capsys, monkeypatch, records, ready):
    read: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(read))
    monkeypatch.setattr(M, "compare", stand_in_check({"context", "llm"}))
    asked = person(monkeypatch, Turn_on_or_off=["9", "", ""], How_should_it_be_paid_for="4")
    assert main(["create", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0
    out = ANSI.sub("", capsys.readouterr().out)
    # Before the check, LLM analysis reads only the newest 30% of the 180 tasks, with its cost shown first.
    assert read == [54, 126]
    assert "LLM analysis: 54 tasks with Codex's own model through your chatgpt subscription" in out
    assert "For the check, it reads only your newest tasks." in out
    # The features that helped start on, the one that didn't starts off.
    after = out[out.index("The features that helped are on below.") :]
    assert re.search(r" 4  Conversation +off", after) and re.search(r" 5  Context +on", after)
    assert re.search(r" 9  LLM analysis +on", after)
    assert "Last check: helped claude-opus-5-5, claude-sonnet-5-5." in after
    # Then the rest, with what it costs, once the person agrees.
    assert "For the check, LLM analysis read only a sample of your tasks. To read the rest:" in after
    assert "LLM analysis: 126 tasks with Codex's own model" in after
    assert sum("Analyze them now?" in question for question in asked) == 1
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt", "context", "llm"]
    assert len(predictor.llm_labels()) == 180 and len(predictor.embedded("llm")) == 180
    assert sum("How should it be paid for?" in question for question in asked) == 1  # not asked again


def test_llm_analysis_is_left_out_when_the_person_wont_have_it_read_the_rest(capsys, monkeypatch, records, ready):
    read: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(read))
    monkeypatch.setattr(M, "compare", stand_in_check({"llm"}))
    person(monkeypatch, Turn_on_or_off=["9", "", ""], How_should_it_be_paid_for="4", Analyze_them_now="n")
    assert main(["create", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0
    out = capsys.readouterr().out
    assert read == [54] and "LLM analysis is left out of this predictor." in out
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt"] and "llm" not in predictor.config
    assert "llm" not in (predictor.config.get("options") or {}) and "Trained on" in out


def test_when_the_check_cant_test_any_ai_model_the_features_stay_as_chosen(capsys, monkeypatch, records, ready):
    read: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(read))

    def untested(tasks, rows, settings=None, names=(), on_progress=None):
        models = {"claude-opus-5-5": {"tasks_tested": 0, "heads": []}}
        report = {"tested_on": 0, "min_tasks_to_test": 60, "models": models}
        return {"": report, **{name: report for name in names}}, {name: {} for name in names}

    monkeypatch.setattr(M, "compare", untested)
    asked = person(monkeypatch, Turn_on_or_off=["9", ""], How_should_it_be_paid_for="4")
    assert main(["create", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0
    out = capsys.readouterr().out
    assert "There aren't enough tasks to check the features yet. Each AI model needs 60 of its own." in out
    # Two answers on the first Features screen, and no second screen: there's nothing to suggest.
    assert sum("Turn on or off" in question for question in asked) == 2 and "The features that helped" not in out
    # LLM analysis read a sample for the check. It still offers to read the rest, with its cost.
    assert read == [54, 126] and "To read the rest:" in out
    assert Predictor("default").config["features"] == ["prompt", "conversation", "context", "llm"]


def test_a_feature_turned_on_after_the_check_shows_its_cost_first(capsys, monkeypatch, records, ready):
    monkeypatch.setattr(M, "compare", stand_in_check(set()))
    # Only the conversation is checked. Then the context is turned on, and its texts were never embedded.
    asked = person(monkeypatch, Turn_on_or_off=["5", "", "5", ""], Embed_them_and_train="n")
    assert main(["create", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0
    out = capsys.readouterr().out
    assert "None of them helped, so they're off below." in out
    # Embedding the context is priced and asked first. Here the answer is no.
    assert sum("Embed them and train the predictor now?" in question for question in asked) == 1
    # Once for the check, and once for the context of the tasks that have one.
    assert re.findall(r"Embedding ([\d,]+) tasks with voyage-4 uses", out) == ["180", "144"]
    assert "Stopped. Your choice of features is kept." in out and "refresh --predictor default" in out
    predictor = Predictor("default")
    assert predictor.config["features"] == ["prompt", "context"] and not (predictor.model_dir / "model.json").exists()


def test_the_check_is_offered_only_in_a_guided_setup_with_something_to_check(capsys, monkeypatch, records, tmp_path):
    def offered(*flags: str, **replies) -> bool:
        asked = person(monkeypatch, Embed_them="n", **replies)
        main(["create", *flags])
        capsys.readouterr()
        return any(QUESTION in question for question in asked)

    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    monkeypatch.setattr(cli, "get_key", lambda model, can_ask: KEY)  # no request to check it
    archive = ["--archive", f"claude-code={records}", "--embedding", "voyage-4"]
    assert offered(*archive)
    assert not offered(*archive, Turn_on_or_off=["4 5", ""])  # only the features that are always on
    assert not offered(*archive, "--features", "context")  # the person named the features
    monkeypatch.setitem(model_types._TYPES, "quick", (quick_train, lambda folder: None))
    assert not offered(*archive, "--model-type", "quick")  # a model type from a plugin has no test to run
    few = tmp_path / "few"
    history(few, tasks=50, start=START)
    assert not offered("--archive", f"claude-code={few}", "--embedding", "voyage-4")  # too few tasks to test


def test_yes_and_express_never_check(capsys, monkeypatch, records, ready):
    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda question="": pytest.fail(f"It asked: {question}"))
    monkeypatch.setattr(M, "compare", lambda *args, **kwargs: pytest.fail("It checked"))
    assert main(["create", "--yes", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0
    assert "Feature check" not in capsys.readouterr().out
    assert main(["create", "other", "--express", "--test-windows", "1", "--archive", f"claude-code={records}"]) == 0


# The check command ------------------------------------------------------------------------------------------


@pytest.fixture
def made(capsys, monkeypatch, records, ready):
    """A trained predictor with the context and the time features."""
    flags = ["--features", "context,time", "--test-windows", "1", "--archive", f"claude-code={records}"]
    assert main(["create", "--yes", *flags]) == 0
    capsys.readouterr()
    return Predictor("default")


def kept(predictor: Predictor) -> dict[str, bytes]:
    """Every file of the predictor, to see that nothing changed."""
    return {str(path): path.read_bytes() for path in sorted(predictor.dir.rglob("*")) if path.is_file()}


def test_check_shows_a_table_of_verdicts_and_changes_nothing(capsys, monkeypatch, made):
    before = kept(made)
    monkeypatch.setenv("NO_COLOR", "1")
    assert main(["check", "default"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"Checked 2 features on [\d,]+ test tasks\.", out)
    header = next(line for line in out.splitlines() if "claude-opus-5-5" in line and "claude-sonnet-5-5" in line)
    assert header.strip().startswith("claude-")  # a column for each AI model tested
    verdict = r"(helped|didn't help|not enough data)"
    assert re.search(rf"  Context +{verdict} +{verdict}\n", out) and re.search(rf"  Time +{verdict} +{verdict}\n", out)
    assert "A feature helped when an AI model's test scored more than 1% worse without it." in out
    assert "This is advice only: nothing changed." in out
    assert kept(made) == before
    saved = check.latest()
    assert set(saved) == {"context", "time"} and set(saved["time"]) == {"claude-opus-5-5", "claude-sonnet-5-5"}


def test_check_answers_a_program_in_json(capsys, monkeypatch, made):
    monkeypatch.setattr(M, "compare", stand_in_check({"time"}))
    assert main(["check", "--json"]) == 0  # the default predictor
    found = json.loads(capsys.readouterr().out)
    assert found["version"] == 1 and found["predictor"] == "default" and found["features"] == ["context", "time"]
    assert found["tested_on"] == 60 and set(found["models"]) == {"claude-opus-5-5", "claude-sonnet-5-5"}
    assert found["models"]["claude-opus-5-5"]["tasks_tested"] == 50
    assert found["verdicts"]["time"]["claude-opus-5-5"] == {"verdict": "helped", "worse_without": 0.05, "share": 1.0}
    assert found["verdicts"]["context"]["claude-sonnet-5-5"]["verdict"] == "no_help"
    assert isinstance(found["checked_at"], float)


def test_check_errors_reach_a_program_as_json(capsys):
    def error(*args: str) -> str:
        assert main(["check", *args, "--json"]) == 1
        return json.loads(capsys.readouterr().out)["error"]["code"]

    assert error() == "no_predictor"
    embedding = {"model": "voyage-4", "dimensions": 256}
    Predictor.create("plain", [], embedding, features=["prompt"])
    assert error("nope") == "unknown_predictor"
    Predictor.create("other", [], embedding, features=["prompt", "context"], model={"type": "quick"})
    assert error() == "no_default_predictor"
    assert error("other") == "unsupported_model_type"
    with locks.held(store.predictor_folder("plain") / "refresh.lock"):
        assert error("plain") == "busy"
    assert main(["check", "plain", "--json"]) == 0  # nothing to check, which isn't an error
    found = json.loads(capsys.readouterr().out)
    assert found["features"] == [] and found["verdicts"] == {} and found["models"] == {}
    assert main(["check", "plain"]) == 0
    assert "There's nothing to check: it uses only the features that are always on." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "help its forecasts. Advice only: it changes nothing." in " ".join(capsys.readouterr().out.split())


def test_check_shows_a_bar_in_a_terminal(capsys, monkeypatch, made):
    from yardstick_predict import ui

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr(ui, "_bar", None)
    assert main(["check"]) == 0
    out = ANSI.sub("", capsys.readouterr().out)
    assert re.search(r" (\d+) / \1 steps", out)
