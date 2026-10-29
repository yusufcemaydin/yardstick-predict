"""The yardstick-predict command.

    yardstick-predict              In a terminal, the home screen: Estimate, Update, Models and Add New.
    yardstick-predict create       Create a predictor from your past runs.
    yardstick-predict refresh      Add new runs, embed them and retrain.
    yardstick-predict predict      Forecast a task: a table in a terminal, JSON for programs.
    yardstick-predict status       Show your predictors, their features and how well they forecast.
    yardstick-predict list         List your predictors, one line each.
    yardstick-predict delete       Delete a predictor. Your records and the feature store stay.
    yardstick-predict check        Check which of a predictor's features help its forecasts. Changes nothing.
    yardstick-predict features     List each predictor's features: what's stored, and how many tasks each covers.
                                   features delete ID deletes a stored feature.
    yardstick-predict default      Show or choose the predictor used when a command names none.
    yardstick-predict rename       Give a predictor another name.

It only asks questions when a person runs it in a terminal. When a program runs it, it never asks.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import getpass
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__, embeddings, locks, ui
from .archives import DEFAULT_ARCHIVES, FORMATS, archive_path
from .store import (
    BrokenPredictor,
    Predictor,
    default_name,
    llm_answers,
    name_lock,
    name_problem,
    predictor_folder,
    predictor_names,
    private_home,
    set_default,
)

SIMPLE_RULE = (
    "The simple rule gives every task on an AI model the same forecast: the spread of that AI model's past tasks."
)
REASONS = {
    "unknown_model": "no data",
    "not_enough_data": "not enough data",
    "not_accurate_enough": "not enough data",
}
"""Why an AI model in a person's table has no forecast, in plain words. A program gets the reason's code."""


class CommandError(SystemExit):
    """A command can't go on: a message for a person, and a code for a program that asked for JSON. It stops the
    command as a SystemExit with the message does, so a person sees the message."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def for_program(args, run) -> int:
    """Run a command for a program that asked for JSON. What the command says on the way goes to the error stream,
    so standard output holds the JSON answer alone. `run` fills the answer it's given and returns the exit code.
    When the command can't go on, the answer is the error, with its code."""
    answer: dict = {"version": 1}
    stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        code = run(answer)
    except CommandError as error:
        answer, code = {"version": 1, "error": {"code": error.reason, "message": error.message}}, 1
    except BrokenPredictor as error:
        answer, code = {"version": 1, "error": {"code": "broken_predictor", "message": str(error)}}, 1
    except SystemExit as stop:
        if not isinstance(stop.code, str):
            raise
        answer, code = {"version": 1, "error": {"code": "failed", "message": stop.code}}, 1
    finally:
        sys.stdout = stdout
    print(json.dumps(answer, indent=2))
    return code


_limit = None
"""The spending limit of the setup or refresh running now, from --max-cost, as a spending.Limit. None without one."""


@contextlib.contextmanager
def limited(args):
    """Keep to the --max-cost limit while the block runs: each step that costs money checks it first."""
    global _limit
    usd = getattr(args, "max_cost", None)
    if usd is not None and (not math.isfinite(usd) or usd < 0):
        raise CommandError("invalid_request", "--max-cost is a number of US dollars, 0 or more, such as 1.50.")
    from . import spending

    _limit = spending.Limit(usd) if usd is not None else None
    try:
        yield _limit
    finally:
        _limit = None


def keep_to_limit(cost, what: str, more_usd: float = 0.0) -> None:
    """Stop before anything is spent when what's planned may cost more than the --max-cost limit allows. `cost` is a
    spending.Cost, or a list of them for an update of several models. `what` is the command, such as refresh.
    `more_usd` is what else it spends, such as the key check."""
    from .spending import dollars

    if _limit is None:
        return
    costs = cost if isinstance(cost, list) else [cost]
    unknown = next((found for found in costs if found.most_usd is None), None)
    if unknown is not None:
        raise CommandError(
            "cost_unknown",
            f"The price of {unknown.analysis_model} isn't in yardstick's list. A spending limit can't be kept without "
            "it. Nothing was spent or saved. To go ahead, leave out --max-cost.",
        )
    most = sum(found.most_usd for found in costs) + more_usd
    if not _limit.fits(most):
        raise CommandError(
            "over_max_cost",
            f"This {what} would cost up to {dollars(most)} at list price, more than your limit of "
            f"{dollars(_limit.usd)}. Nothing was spent or saved. Raise --max-cost to go ahead.",
        )


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def terminal_output() -> bool:
    return sys.stdout.isatty()


def millions(tokens: int) -> str:
    return f"{tokens / 1e6:.1f} million" if tokens >= 1e6 else f"{tokens:,}"


def ago(moment: float | None) -> str:
    if not moment:
        return "never"
    seconds = time.time() - moment
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            count = int(seconds // size)
            return f"{count} {unit}{'s' if count != 1 else ''} ago"
    return "just now"


def duration(seconds: float) -> str:
    """A time read at a glance: 45 s, 4 min, 1.5 h."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def short_tokens(tokens: int) -> str:
    if tokens >= 1e6:
        return f"{tokens / 1e6:.1f}M"
    if tokens >= 1e3:
        return f"{tokens / 1e3:.0f}k"
    return str(tokens)


def money(usd: float) -> str:
    return "<$0.01" if 0 < usd < 0.005 else f"${usd:.2f}"


# Talking about cost ------------------------------------------------------------------------------------


def cost_lines(cost: embeddings.Cost, model: str, tasks: int) -> list[str]:
    """What embedding costs, in plain words. Free tokens count only on an account said to be new."""
    about = "" if cost.exact else "about "
    lines = [f"Embedding {tasks:,} tasks with {model} uses {about}{cost.tokens:,} tokens."]
    if not cost.free_left:
        lines.append(f"It costs {ui.bold(money(cost.list_usd))} at Voyage's list price.")
    elif cost.free_left >= cost.tokens:
        lines.append(
            f"Your free tokens cover it, so it costs {ui.bold('nothing')}. "
            f"{millions(cost.free_left - cost.tokens)} free tokens will be left."
        )
    else:
        lines.append(
            f"Your free tokens cover {millions(cost.free_left)} of them. "
            f"The rest costs {ui.bold(money(cost.your_usd))}."
        )
    if not cost.exact:
        lines.append("The token count is an estimate, because Voyage's tokenizer didn't download.")
    return lines


def explain(error: embeddings.VoyageError) -> str:
    if error.status == embeddings.SETTINGS:
        return error.detail
    detail = error.detail.lower()
    if any(word in detail for word in ("payment", "billing", "credit", "balance", "quota", "free token")):
        return (
            f"Voyage said: {error.detail}\nYour account may be out of free tokens with no payment method. "
            "Add one at https://dashboard.voyageai.com and run this again."
        )
    if error.status == 401:
        return "Voyage refused the key. Check that you copied all of it."
    if error.status == 403:
        return "Voyage refused requests from this network. It may block your internet address."
    if error.status == 429:
        advice = "Try again later, or add a payment method to raise your limits."
        # Voyage's own reason, or the embedding code's, such as one text being more than a minute allows.
        reason = error.detail.strip()
        if not reason:
            return f"Voyage kept limiting requests. {advice}"
        return f"Voyage kept limiting requests: {reason}{'' if reason.endswith('.') else '.'}\n{advice}"
    if error.status == 0:
        return f"{error}. Check your internet connection."
    return str(error)


def damaged_usage_note(key: str | None) -> None:
    """Say once that the record of Voyage usage was damaged and moved aside, so costs count no free tokens.

    yardstick keeps its own count of each key's free tokens. When that file was damaged, the count is lost,
    so every cost shows at the list price from then on. It's said once for each time it happens.
    """
    from .store import home, save_setting, settings

    aside = embeddings.account(key).get(embeddings.SET_ASIDE) if key else None
    if not isinstance(aside, dict) or settings().get("usage_set_aside_told") == aside.get("at"):
        return
    kept = f" The damaged copy is {home() / aside['file']}." if aside.get("file") else ""
    ui.note(f"Your record of Voyage usage was damaged, so it was moved aside.{kept}")
    ui.note("Costs are now shown at Voyage's list price, without any free tokens.")
    save_setting("usage_set_aside_told", aside.get("at"))


# The Voyage key ----------------------------------------------------------------------------------------


def get_key(model: str, can_ask: bool, check: bool = True) -> str | None:
    """Find a working key. A person can paste one, and it is saved in the computer's password store. None when a
    person pastes none. It stops the command when there's no key and none can be asked for, or when the key fails.
    Without `check`, the key isn't tried yet, so nothing goes to Voyage: verify_key tries it once it's needed."""
    key, source = embeddings.find_key()
    pasted = False
    ui.heading("Voyage key")
    if key:
        ui.line(f"Using the key from {source}.")
    elif can_ask:
        ui.line("Paste your Voyage API key. It stays on this computer and is only ever sent to Voyage.")
        ui.note("You can make one at https://dashboard.voyageai.com")
        key = getpass.getpass(ui.prompt("Voyage API key (hidden as you type):")).strip()
        pasted = True
        if not key:
            return None
    else:
        raise CommandError(
            "no_embedding_key",
            "No Voyage key found. Set VOYAGE_API_KEY. Or run this command in a terminal without --yes or --express: "
            "it asks for one.",
        )
    if check:
        verify_key(key, model)
    if pasted:
        if embeddings.save_key(key):
            ui.ok("Saved in your computer's password store, so you won't need to paste it again.")
        else:
            ui.note("This computer has no password store to save it in. Set VOYAGE_API_KEY to skip pasting next time.")
    if can_ask and not embeddings.account(key).get("asked"):
        ui.line()
        ui.note(
            "Voyage gives every account 200 million free tokens. yardstick can't see your balance. It keeps its own "
            "count."
        )
        ui.out(ui.prompt("Has this account used any of its free tokens before?").rstrip())
        ui.line(f"    {ui.accent('n')}   no, it's a new account")
        ui.line(f"    {ui.accent('y')}   yes or not sure. Costs then count no free tokens.")
        # Pressing Enter counts no free tokens, so a cost is never shown lower than it may be.
        reply = input(f"    {ui.dim('[n/Y]')} ").strip().lower()
        embeddings.set_free_tokens(key, "new" if reply in ("n", "no") else "unknown")
    return key


def verify_key(key: str, model: str) -> None:
    """Try the key with one tiny request, and stop the command when it doesn't work. Under a spending limit, the
    tokens it sends count toward it."""
    ui.working("Checking the key")
    try:
        used = embeddings.check_key(key, model)
    except embeddings.VoyageError as error:
        raise CommandError("embedding_failed", f"The key didn't work.\n{explain(error)}") from None
    if _limit is not None:
        from . import spending

        _limit.spent += spending.voyage_usd(used, model)
    ui.ok("The key works.")


# Questions ---------------------------------------------------------------------------------------------


def ask_name(existing: list[str]) -> str:
    """A name for a new predictor. The first one is called main unless you pick another name."""
    suggestion = "" if existing else "main"
    while True:
        reply = input(ui.prompt("Name this model:", f"[{suggestion}]" if suggestion else "")).strip()
        name = reply or suggestion
        if not name:
            ui.note("Type a name, such as work.")
        elif name_problem(name):
            ui.note(name_problem(name))
        elif name in existing:
            ui.note(f"There's already a model called {name}. Pick another name.")
        else:
            return name


def parse_archive(spec: str) -> dict:
    """'claude-code' and 'codex' mean the usual folders. FORMAT=PATH adds a folder of records in that format."""
    for archive in DEFAULT_ARCHIVES:
        if spec == archive["name"]:
            return dict(archive)
    if "=" in spec:
        fmt, path = spec.split("=", 1)
        if fmt not in FORMATS:
            raise CommandError("invalid_request", f"Unknown record format {fmt!r}. Use one of: {', '.join(FORMATS)}.")
        path = os.path.abspath(os.path.expanduser(path))
        return {"name": os.path.basename(path.rstrip("/\\")) or "custom", "format": fmt, "path": path}
    raise CommandError(
        "invalid_request", f"Can't read --archive {spec!r}. Use claude-code, codex, or FORMAT=PATH like codex=~/runs."
    )


def unique_names(archives: list[dict]) -> list[dict]:
    """Give each archive its own name, such as sessions and sessions-2, so counts and tasks don't mix."""
    taken: set[str] = set()
    for archive in archives:
        name, number = archive["name"], 2
        while name in taken:
            name, number = f"{archive['name']}-{number}", number + 1
        archive["name"] = name
        taken.add(name)
    return archives


def no_folder(path: str, names: list[str]) -> str:
    """What to say of a folder of records that isn't there, as when its records moved or its drive isn't plugged
    in. `names` are the predictors that read from it."""
    one = len(names) == 1
    return (
        f"There's no folder at {path}. {in_words(names)} {'keeps' if one else 'keep'} the tasks "
        f"{'it' if one else 'they'} read from there. {'It reads' if one else 'They read'} nothing new there until "
        "the records are back."
    )


def given_archives(given: list[str]) -> list[dict]:
    """The folders of records named with --archive. One that isn't there stops the command: read, it would look like
    a folder with no tasks in it."""
    archives = [parse_archive(spec) for spec in given]
    missing = list(dict.fromkeys(a["path"] for a in archives if not archive_path(a).is_dir()))
    if missing:
        there = "There's no folder at" if len(missing) == 1 else "There are no folders at"
        raise CommandError("invalid_request", f"{there} {in_words(missing)}. Nothing was read or spent.")
    return archives


def choose_archives(can_ask: bool, given: list[str]) -> list[dict]:
    if given:
        return given_archives(given)
    found = [dict(a) for a in DEFAULT_ARCHIVES if archive_path(a).is_dir()]
    if not can_ask:
        return found
    chosen = []
    for archive in found:
        if ui.ask(f"Use your {archive['name']} records in their default location ({archive['path']})?", True):
            chosen.append(archive)
    for archive in DEFAULT_ARCHIVES:
        if not archive_path(archive).is_dir():
            ui.note(f"No {archive['name']} records in their default location ({archive['path']}).")
    while True:
        # With no records chosen yet, a folder is the only way to get some, so it isn't optional.
        optional = f"{ui.dim('(Optional)')} " if chosen else ""
        question = f"{optional}Add a folder of records in Claude Code or Codex format? Paste its path, or press Enter:"
        extra = input(ui.prompt(question)).strip()
        if not extra:
            break
        fmt = input(ui.prompt("Is it claude-code or codex format?", "[claude-code]")).strip() or "claude-code"
        if fmt not in FORMATS:
            ui.note(f"Use one of: {', '.join(FORMATS)}.")
            continue
        archive = parse_archive(f"{fmt}={extra}")
        if not archive_path(archive).is_dir():
            ui.note(f"There's no folder at {archive['path']}.")
            continue
        chosen.append(archive)
    return chosen


def choose_features(can_ask: bool, args):
    """The features a person picks on the Features screen, or a program names with flags."""
    from . import menu

    if args.features is not None or not can_ask:
        try:
            return menu.from_flags(args.features, args.llm_provider, args.llm_model)
        except ValueError as error:  # it's checked before a record is read
            raise CommandError("invalid_request", f"{error} Nothing was read or spent.") from None
    try:
        return menu.ask(custom_features())
    except ValueError as error:
        raise CommandError("invalid_request", str(error)) from None


def custom_features() -> dict[str, str]:
    """The features plugins add, with what each is, for the Features screen."""
    from . import features

    return {name: says for name, says in features.available().items() if name not in features.BUILT_IN}


def feature_summary(config: dict) -> str:
    """A predictor's features in plain words, with the options it uses and how LLM analysis runs."""
    from . import features, llm

    found = [group.title for group in features.MENU if group.always]
    for name in config.get("features") or ["prompt"]:
        group = next((g for g in features.MENU if g.key == name), None)
        if name == "prompt":
            continue
        if group is None:
            found.append(name)
            continue
        label, chosen = group.title, features.options(config, name)
        if len(group.options) > 1 and len(chosen) < len(group.options):
            label += f" ({', '.join(o.says for o in group.options if o.key in chosen)})"
        how = config.get("llm") or {}
        if name == "llm" and how.get("provider") in llm.PROVIDERS:
            label += f", by {llm.title(how['model'])} through {llm.PROVIDERS[how['provider']].title.lower()}"
        found.append(label)
    return in_words(found)


def unknown_features_note(predictor: Predictor) -> None:
    """Say in one line which features the predictor lists that this version doesn't know. They're skipped.

    The predictor's plugins must be loaded first, or their features would count as unknown.
    """
    from .pipeline import unknown_features

    names = unknown_features(predictor)
    if names:
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        it = "it" if len(names) == 1 else "them"
        ui.note(f"Skipping {listed}: this version doesn't know {it}. A newer version or a plugin added {it}.")


def newer_settings_note(predictor: Predictor) -> None:
    """Say that a newer version saved the predictor's settings, when it did. This version only reads them."""
    if predictor.newer:
        ui.note("A newer version of yardstick-predict saved this model. This version reads what it can of it.")


def analysis_lines(asks: int, how: dict, tasks: int | None = None) -> list[str]:
    """What LLM analysis of this many tasks costs, or how it counts toward a subscription, in plain words.

    `tasks`, when given, is how many different tasks the `asks` readings are of. It's fewer when some tasks are
    read more than once, as when predictors that share them ask for different parts.
    """
    from . import llm

    provider = llm.PROVIDERS[how["provider"]]
    model = llm.title(how["model"])
    read = f"{asks:,} tasks" if tasks is None or tasks == asks else f"{tasks:,} tasks, read {asks:,} times in all,"
    if not provider.billed:
        requests = -(-asks // provider.batch)
        return [
            f"LLM analysis: {read} with {model} through your {provider.title}, "
            f"in about {requests:,} requests. It counts toward your plan's usage limits."
        ]
    usd = llm.cost(asks, how["model"])
    price = f"about ${usd:,.2f} at list price" if usd is not None else "its price isn't in yardstick's list"
    return [f"LLM analysis: {read} with {model}, {price}."]


def say_estimate(
    tasks: list, config: dict, plan, asks: list, new_tasks: list | None = None, dataset: float = 0, checked: int = 0
) -> None:
    """One line on how long the rest takes, what it's made of and the room the dataset takes, before the
    question to go ahead. Then a note on memory, only when training may use a lot of this computer's.

    `tasks` are all the predictor's tasks once the new ones are in, `config` its settings, `plan` what goes to
    Voyage and `asks` the tasks LLM analysis reads. `new_tasks` are the ones the dataset doesn't have yet, all of
    `tasks` by default, and `dataset` is how many bytes its data takes now. `checked` is how many features the feature
    check tests the model without, when it runs.
    """
    from . import estimate

    found = estimate.of(tasks, config, plan.cost.tokens, len(plan.items), len(asks), new_tasks, dataset, checked)
    ui.line(estimate.line(found))
    memory = estimate.memory_note(found, estimate.total_memory())
    if memory:
        ui.note(memory)


def analyze_with_progress(predictor: Predictor, asks: list, reserve: float = 0.0) -> tuple[int, str | None]:
    """Run LLM analysis with a progress bar. When it can't run, training goes on without it. Returns how many tasks
    it analyzed, and what went wrong, when something did.

    Under a spending limit, LLM analysis paid with an API key gets what's left of it, less `reserve`, what the
    embedding still to come may cost. No request starts that may take it past that, and what it really cost counts
    toward the limit.
    """
    from . import llm
    from .pipeline import run_analysis

    how = predictor.config.get("llm") or {}
    budget = None
    if _limit is not None and llm.PROVIDERS[how["provider"]].billed:
        budget = llm.Budget(_limit.left() - reserve)
    ui.heading("LLM analysis")
    ui.note("Each batch is saved as it arrives, so you can stop and resume later.")
    try:
        answered, problem = run_analysis(predictor, asks, lambda done, total: ui.progress(done, total, "tasks"), budget)
    finally:
        if budget is not None:
            _limit.spent += budget.spent
    if not answered and problem:
        ui.fail(problem)
        ui.note("Training goes on without LLM analysis for now. Once that's fixed, a refresh tries again.")
        return answered, problem
    missed = len(asks) - answered
    ui.ok(f"Analyzed {answered:,} tasks." + (f" {missed:,} weren't answered." if missed else ""))
    if problem:
        ui.note(f"{problem} A refresh tries the rest again, up to twice for a task it refused.")
    return answered, problem


NUMBER_FLAGS = {"dropout": "0.1", "test_share": "0.3", "validation_share": "0.15", "half_life": "30"}
"""The advanced settings given as numbers, each with an example of a good value."""


def check_numbers(args) -> None:
    """Refuse a number flag that isn't an ordinary number, such as nan or inf, with a plain reason.

    A half-life is a number of days, 0 or more. Each model type checks the ranges of its own settings.
    """
    for name, example in NUMBER_FLAGS.items():
        value = getattr(args, name, None)
        flag = "--" + name.replace("_", "-")
        if value is not None and not math.isfinite(value):
            raise CommandError("invalid_request", f"{flag} must be a number, such as {example}, not {value}.")
    if getattr(args, "half_life", None) is not None and args.half_life < 0:
        raise CommandError("invalid_request", "--half-life is a number of days: 0 or more. 0 counts every task alike.")


def advanced_settings(args) -> dict:
    """The model settings a program or an expert gave with flags. The rest keep their defaults."""
    check_numbers(args)
    from . import model

    given = {
        "type": args.model_type,
        "heads": args.heads,
        "head": args.head,
        "family": args.shape,
        "loss": args.loss,
        "dropout": args.dropout,
        "test_share": args.test_share,
        "test_windows": args.test_windows,
        "validation_share": args.validation_share,
    }
    settings = {k: v for k, v in given.items() if v is not None}
    if args.half_life is not None:
        settings["half_life_days"] = args.half_life if args.half_life > 0 else None
    if settings.get("type", "xgboost") != "xgboost":
        return settings
    try:
        return model.settings_for(settings)
    except ValueError as error:
        raise CommandError("invalid_request", str(error)) from None


# Reports -----------------------------------------------------------------------------------------------


def tested_models(report: dict) -> list[str]:
    """Every AI model the test knows about, the most tested first."""
    models = report.get("models") or {}
    return sorted(models, key=lambda name: -(models[name].get("tasks_tested") or 0))


def served_models(report: dict) -> list[str]:
    """The AI models whose forecasts passed the test, the most tested first."""
    return [name for name in tested_models(report) if report["models"][name].get("served")]


THRESHOLDS = {"min_test_tasks": 30, "tolerance": 0.02, "min_range_held": 0.75, "totals_range": [0.5, 2.0]}
"""The test's thresholds in this version, for a report that doesn't carry its own when XGBoost can't load."""


def thresholds(report: dict) -> dict:
    """The thresholds the test used, to say why a check failed: the report's own, else the model module's.

    The model module needs XGBoost to load. Status works without it, so then this version's values are used.
    """
    found = report.get("thresholds")
    found = found if isinstance(found, dict) else {}
    if all(name in found for name in THRESHOLDS):
        return found
    try:
        from . import model

        current = {
            "min_test_tasks": model.MIN_TEST_TASKS,
            "tolerance": model.TOLERANCE,
            "min_range_held": model.MIN_RANGE_HELD,
            "totals_range": list(model.TOTALS_RANGE),
        }
    except Exception:  # XGBoost can't load on this computer
        current = dict(THRESHOLDS)
    return {**current, **found}


def print_report(meta: dict) -> None:
    """How close the forecasts came on each AI model's newest tasks, from the meta a trained model saves. It's said
    in plain words: an AI model whose forecasts weren't close enough, or had too few tasks to show it, has not
    enough data."""
    report = meta.get("report") or {}
    needed = report.get("min_tasks_to_test") or 60
    if not report.get("tested_on"):
        wrapped_note(
            f"There's not enough data to check its forecasts yet. Each AI model needs {needed:,} tasks of its own. "
            "Until then, it gives no forecasts."
        )
        return
    checks = thresholds(report)
    least = checks["min_test_tasks"]
    for part in ui.wrapped(
        f"How close its forecasts came on each AI model's newest tasks, {report['tested_on']:,} in all, "
        "as if they hadn't happened yet:"
    ):
        ui.line(part)
    ui.line()
    names = {"seconds": "time", "tokens": "tokens", "usd": "cost"}
    results = [(name, report["models"][name]) for name in tested_models(report)]
    if not results:
        return
    pad = max(len(name) for name, _ in results)
    flags = {"ok": ui.green(ui.mark("ok")), "thin": ui.yellow(ui.mark("warn"))}
    lead = pad + 3 + (2 if ui.styled() else 0)

    def flag(kind: str) -> str:
        return f"{flags[kind]} " if ui.styled() else ""

    def verdict(name: str, label: str, mark: str, words: list[str]) -> None:
        """An AI model's line: its name, then what came of its forecasts. Too wide, the words go under the name."""
        shown = f"{label}   {mark}" + ui.join(words)
        if ui.width(shown) < ui.columns() - 2:
            ui.line(shown)
            return
        ui.line(ui.bold(ui.cut(name, ui.columns() - 3)))
        for part in ui.wrapped(", ".join(words), 4 + ui.width(mark)):
            ui.line(f"  {mark}{part}")
            mark = " " * ui.width(mark)

    def why(text: str) -> None:
        """A reason under an AI model's line, broken to fit the terminal. It lines up under the AI model's words
        where there's room for that, else it starts closer to the edge."""
        indent = lead if 2 + lead + 20 < ui.columns() else 4
        for part in ui.wrapped(text, 2 + indent):
            ui.line(" " * indent + ui.dim(part))

    for name, result in results:
        tested = result["tasks_tested"]
        label = ui.bold(name) + " " * (pad - len(name))
        if result["served"]:
            verdict(name, label, flag("ok"), ["forecasts on", f"checked on {tested:,} tasks"])
            scores = [(names[t], result[t]) for t in names if t in result]
            if scores:
                better = [f"{n} {s['better_than_simple_rule']:.0%}" for n, s in scores]
                rows = [(ui.dim("better than the simple rule"), *better)]
                if all(s.get("in_90_range") is not None for _, s in scores):
                    rows.append((ui.dim("inside the 90% range"), *(f"{n} {s['in_90_range']:.0%}" for n, s in scores)))
                # Lined up under the AI model's words when that fits the terminal, else closer to the edge. A
                # terminal too narrow for the table gets each line as words broken to fit.
                wide = sum(max(ui.width(row[i]) for row in rows) + 3 for i in range(len(rows[0]))) - 3
                if 4 + wide < ui.columns():
                    ui.table(rows, indent=2 + lead if 2 + lead + wide < ui.columns() else 4)
                else:
                    for label, *values in rows:
                        for part in ui.wrapped(f"{ui.plain(label)}: {', '.join(values)}", 4):
                            ui.line(f"  {ui.dim(part)}")
        elif result["reason"] == "not_enough_data":
            count = result.get("tasks")
            if tested == 0 and isinstance(count, int) and count < needed:
                short = f"{count:,} tasks, needs {needed}"
            elif tested == 0:
                short = f"needs {least} recent tasks"
            elif tested < least:
                short = f"{tested} recent tasks, needs {least}"
            else:
                short = "too few known times, tokens or costs"
            verdict(name, label, flag("thin"), ["not enough data", short])
        else:
            verdict(name, label, flag("thin"), ["not enough data", f"checked on {tested:,} tasks"])
            for failed in result.get("failed_checks", []):
                why(failed_check(failed, result, names, checks))
    ui.line()
    wrapped_note(SIMPLE_RULE)


def failed_check(failed: str, result: dict, names: dict, checks: dict | None = None) -> str:
    """Why an AI model's forecasts weren't close enough, in plain words. `checks` holds the bars they had to clear,
    as thresholds() gives them."""
    checks = checks or thresholds({})
    target, check = failed.split(".")
    scores = result[target]
    label = names[target].capitalize()
    if check == "simple_rule":
        worse = -scores["better_than_simple_rule"]
        return (
            f"{label} forecasts were {worse:.0%} worse than the simple rule. "
            f"They may be at most {checks['tolerance']:.0%} worse."
        )
    if check == "range_held":
        return (
            f"{label}: only {scores['in_90_range']:.0%} of real results fell inside the 90% range. "
            f"It needs {checks['min_range_held']:.0%}."
        )
    ratio = scores["expected_over_real_total"]
    low, high = checks["totals_range"]
    return (
        f"{label}: the expected values added up to {ratio:.2f} times the real total. "
        f"It must be between {low} and {high} times."
    )


RELAUNCHED = "YARDSTICK_PREDICT_OPENMP"
"""Set when the command restarted itself to find OpenMP, so it never restarts twice."""


def bundled_openmp() -> str | None:
    """The folder of the OpenMP library that scikit-learn ships on a Mac, when it's installed."""
    spec = importlib.util.find_spec("sklearn")
    if spec is None or not spec.origin:
        return None
    folder = Path(spec.origin).parent / ".dylibs"
    return str(folder) if (folder / "libomp.dylib").is_file() else None


def relaunch_with_openmp(argv: list[str]) -> None:
    """On a Mac without Homebrew's OpenMP, start this command again so XGBoost finds scikit-learn's copy.

    XGBoost asks the Mac for OpenMP by name. The Mac also searches a fallback list of folders, which
    a program can only set when it starts. So the command restarts once with scikit-learn's folder on
    that list. It returns only when it can't restart.

    Python's -P keeps the current folder off the list of places Python imports from, as it is for the
    installed command. Otherwise a file there, such as a json.py, would run in place of Python's own.
    """
    if sys.platform != "darwin" or os.environ.get(RELAUNCHED) or not sys.executable:
        return
    folder = bundled_openmp()
    if folder is None:
        return
    env = dict(os.environ)
    fallback = env.get("DYLD_FALLBACK_LIBRARY_PATH") or f"{Path.home()}/lib:/usr/local/lib:/usr/lib"
    env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{folder}:{fallback}"
    env[RELAUNCHED] = "1"
    os.execve(sys.executable, [sys.executable, "-P", "-m", "yardstick_predict", *argv], env)


def xgboost_problem(argv: list[str]) -> str | None:
    """Why XGBoost can't load, in plain words, or None when it loads."""
    from .model_types import without_scikit_learn

    try:
        with without_scikit_learn():
            import xgboost  # noqa: F401
    except Exception as error:
        if "libomp" in str(error):
            relaunch_with_openmp(argv)
            return (
                "XGBoost can't find the OpenMP library on this Mac. Reinstall yardstick-predict to bring "
                "in the copy it comes with, or install one with: brew install libomp"
            )
        first_line = str(error).strip().splitlines()[0] if str(error).strip() else repr(error)
        return f"XGBoost couldn't load: {first_line}"
    return None


# Commands ----------------------------------------------------------------------------------------------


def pick_predictor(name: str | None) -> Predictor:
    names = predictor_names()
    if name:
        if name not in names:
            message = f"There's no model called {name!r}. Your models: {', '.join(names) or 'none'}."
            raise CommandError("unknown_predictor", message)
        return Predictor(name)
    if not names:
        raise CommandError("no_predictor", "There's no model yet. Run yardstick-predict to create one.")
    chosen = default_name()
    if chosen is None:
        raise CommandError(
            "no_default_predictor",
            f"You have several models and no default: {', '.join(names)}. Name one with --predictor, "
            "or choose a default with: yardstick-predict default NAME",
        )
    return Predictor(chosen)


def cmd_home(args) -> int:
    if interactive():
        from .screens import home

        return home.run()
    names = predictor_names()
    if not names:
        ui.out(ui.bold("yardstick-predict"))
        ui.out("Forecast how long a coding-agent task will take and what it will cost.")
        ui.out(ui.dim("It learns from your own past Claude Code and Codex runs. It sends your prompts to"))
        ui.out(ui.dim("Voyage, which turns them into numbers. Keys and passwords in them are hidden first."))
        ui.out()
        ui.out("No model yet. Set VOYAGE_API_KEY, then create one with: yardstick-predict create --express")
        return 1
    for name in names:
        try:
            predictor = Predictor(name)
        except BrokenPredictor as error:
            ui.out(f"{name}: {error}")
            continue
        facts = [
            f"{predictor.embedding_model} embeddings",
            f"{predictor.count_tasks():,} tasks",
            f"updated {ago(predictor.config.get('refreshed_at'))}",
        ]
        ui.out(f"{name}: {', '.join(facts)}")
    return 0


def already_exists(name: str) -> str:
    return f"A model called {name!r} already exists. Refresh it with: yardstick-predict refresh --predictor {name}"


def unknown_embedding(model: str) -> str:
    known = list(embeddings.MODELS)
    return f"Unknown embedding model {model!r}. Use {', '.join(known[:-1])} or {known[-1]}."


def cmd_create(args) -> int:
    from contextlib import ExitStack

    def create(answer: dict) -> int:
        # Setup takes locks as it goes, and holds them until it's done.
        with ExitStack() as held:
            return _create(args, held, answer)

    def limited_create(answer: dict) -> int:
        with limited(args):
            return create(answer)

    if getattr(args, "json", False):
        return for_program(args, limited_create)
    return limited_create({})


def _create(args, held, answer: dict) -> int:
    """Set up a new predictor: read the records, choose the features, then embed, analyze and train. `held` keeps
    the locks it takes until it's done. `answer` is filled with what a program reads: the predictor, what it costs
    and what was done."""
    from . import model_types, spending

    if args.express:
        args.yes = True
    # With --yes or --express it asks nothing, even in a terminal. Every choice takes its default. Nor does a
    # program that asked for JSON.
    can_ask = interactive() and not args.yes and not getattr(args, "json", False)
    given_archives(args.archive)  # a folder named that isn't there is said before anything is asked
    existing = predictor_names()
    if can_ask:
        ui.out()
        ui.out(ui.bold("New model"))
    name = args.name or (ask_name(existing) if can_ask else None)
    if name is None:  # a program that names none gets main, as a person's first one is called
        if "main" in existing:
            raise CommandError("invalid_request", "Give the new model a name, like: yardstick-predict create work")
        name = "main"
    problem = name_problem(name)
    if problem:
        raise CommandError("invalid_request", problem)
    if name in existing:
        raise CommandError("predictor_exists", already_exists(name))
    first = not existing
    make_default = first or (
        can_ask and ui.ask("Make it your default model? Commands use it when they name none.", False)
    )
    answer.update(
        predictor=name,
        done=False,
        default=bool(make_default),
        found_tasks=0,
        cost=None,
        analyzed=0,
        embedded=0,
        trained=False,
        learned_from=None,
        forecasts_for=[],
        max_cost=None if _limit is None else _limit.usd,
        spent=None,
        notes=[],
    )
    try:
        # Each plugin is saved by where it was found, so it loads the same from any folder later.
        plugins = [model_types.find_plugin(plugin) for plugin in args.plugin]
        model_types.load_plugins(plugins)
    except model_types.PluginError as error:
        raise CommandError("plugin_failed", f"{error} Nothing was read or spent.") from None
    model_settings = advanced_settings(args)
    try:
        model_types.get(model_settings.get("type", "xgboost"))
    except (ValueError, ImportError, AttributeError) as error:
        raise CommandError("invalid_request", f"{error} Nothing was read or spent.") from None
    if args.embedding is not None and args.embedding not in embeddings.MODELS:
        raise CommandError("invalid_request", f"{unknown_embedding(args.embedding)} Nothing was read or spent.")
    from .model import MIN_TASKS

    # The built-in model needs 30 tasks it can learn from. A model type from a plugin sets its own needs.
    built_in = model_settings.get("type", "xgboost") == "xgboost"
    least = MIN_TASKS if built_in else 1
    if args.max_tasks is not None and args.max_tasks < least:
        raise CommandError(
            "invalid_request", f"Use --max-tasks {least} or more. Training needs at least {least} tasks."
        )
    # Features named with flags are checked before a record is read, so a mistyped one is said at once.
    picked = choose_features(can_ask, args) if args.features is not None or not can_ask else None

    ui.heading("Your past runs")
    archives = unique_names(choose_archives(can_ask, args.archive))
    if not archives:
        raise CommandError("no_records", "No records to learn from. Add a folder with --archive FORMAT=PATH.")
    ui.working("Reading your records. A big history takes a minute")
    from . import features, llm
    from .pipeline import (
        analysis_plan,
        embedding_items,
        newest_that_fit,
        plan_embeddings,
        price,
        read_all,
        refresh_benchmarks,
        save_scan,
        train,
    )
    from .store import Embedded

    found = read_all(archives, on_progress=lambda done, total: ui.progress(done, total, "bytes"))
    counts = [(archive["name"], found.per_archive.get(archive["name"], 0)) for archive in archives]
    total = sum(count for _, count in counts)
    if len(counts) == 1:
        ui.ok(f"Read {total:,} tasks from {counts[0][0]}.")
    else:
        ui.ok(f"Read {total:,} tasks: " + ui.join([f"{source} {count:,}" for source, count in counts]))
    answer["found_tasks"] = len(found.new_tasks)
    if not found.new_tasks:
        raise CommandError("no_records", "Found no finished tasks in those records.")
    from .pipeline import learnable

    ready = len(learnable(found.new_tasks, args.max_tasks))
    if built_in and ready < least:
        raise CommandError(
            "not_enough_data",
            f"Only {ready} of those tasks can be learned from. Training needs at least {least}. "
            "Nothing was spent. Add more records, or try again after more runs.",
        )
    picked = picked or choose_features(can_ask, args)
    chosen, settings = picked.features, picked.config()
    # From here, setup reads the feature store to plan what to make, and then saves to it. It holds the lock of the
    # store's users until it's done, so no features delete takes away what it planned with or saved.
    from . import feature_store

    held.enter_context(feature_store.writing())
    # What the same AI model said about these tasks for another predictor is in the feature store already,
    # descriptions included, so they're embedded with the rest.
    answers = llm_answers(settings)
    # The check is advice for a person setting up step by step. A program, or a person who named the features
    # with flags, gets the features they named.
    checking = can_ask and args.features is None and built_in and offer_check(chosen, found.new_tasks, args.max_tasks)
    ids = {task.id for task in found.new_tasks}

    def items(model: str) -> list:
        """The texts to embed with this Voyage model. None the feature store holds a vector of, made from the same
        text for another predictor, as a refresh counts them. So a second predictor pays for nothing it has."""
        stored = {
            name: Embedded(
                feature_store.vector_hashes(feature_store.embedding(name, model, embeddings.DIMENSIONS), ids)
            )
            for name in chosen
            if features.is_text(name)
        }
        return embedding_items(found.new_tasks, chosen, stored, args.max_tasks, settings, answers)

    model_name = args.embedding or (None if can_ask else "voyage-4")
    plans: dict = {}
    if model_name is None:
        ui.heading("Embedding model")
        ui.working("Working out what each Voyage model would cost")
        plans = {m: price(items(m), m, None) for m in embeddings.MODELS}
        options = list(embeddings.MODELS)
        rows = [
            (
                ui.accent(str(number)),
                ui.bold(option),
                f"${plans[option].cost.list_usd:.2f}",
                f"{plans[option].cost.tokens:,} tokens",
                ui.dim(embeddings.MODELS[option].note),
            )
            for number, option in enumerate(options, start=1)
        ]
        ui.table(rows, align="llrrl")
        ui.note("The model keeps this Voyage model for good. To change it later, make a new model.")
        while model_name is None:
            reply = input(ui.prompt("Which Voyage model should embed your tasks?", "[1]")).strip() or "1"
            if ui.number_typed(reply) and 1 <= int(reply) <= len(options):
                model_name = options[int(reply) - 1]
            else:
                ui.note(f"Type a number from 1 to {len(options)}.")
    if model_name not in embeddings.MODELS:
        raise CommandError("invalid_request", unknown_embedding(model_name))

    # A person's key is tried at once, so a pasted one is saved only when it works. A program's is tried once it goes
    # ahead, so its preview sends nothing.
    key = get_key(model_name, can_ask, check=can_ask)
    if not key:
        return 1
    # Each Voyage model's tokens are counted once. The key only changes how many of them are free.
    plan = (
        with_key(plans[model_name], model_name, key)
        if model_name in plans
        else price(items(model_name), model_name, key)
    )
    ui.heading("Cost")
    damaged_usage_note(key)
    for text in cost_lines(plan.cost, model_name, plan.tasks):
        ui.line(text)
    limit = args.max_tasks
    free = plan.cost.free_left
    if can_ask and free is not None and plan.cost.tokens > free and free > 0:
        # The descriptions LLM analysis writes are embedded too. Leave room for them.
        described = "llm" in chosen and "description" in features.options(settings, "llm")
        fit = newest_that_fit(plan, free, llm.DESCRIPTION_TOKENS if described else 0)
        if fit >= least and ui.ask(f"Only the newest {fit:,} tasks fit in your free tokens. Use just those?", True):
            limit = fit
        elif fit < least:
            ui.note(f"Your free tokens cover only {fit:,} tasks, fewer than the {least} training needs.")
    if limit and limit != args.max_tasks:
        plan = newest_only(plan, found.new_tasks, limit, model_name, key)
        for text in cost_lines(plan.cost, model_name, plan.tasks):
            ui.line(text)
    asks = analysis_plan(found.new_tasks, settings, answers, limit)
    if checking:
        from . import check

        # The rest only once the check shows LLM analysis helps.
        asks = check.sample(asks, {task.id: task.model for task in found.new_tasks})
    if asks:
        for text in analysis_lines(len(asks), settings["llm"]):
            ui.line(text)
        if checking:
            ui.note("For the check, it reads only your newest tasks. If it helps, you can have it read the rest after.")
        if "description" in features.options(settings, "llm"):
            ui.note("Its descriptions are embedded too, which adds a little to the Voyage cost.")
    described = len(asks) if asks and "description" in features.options(settings, "llm") else 0
    cost = spending.cost_of(settings, model_name, plan, asks, described)
    answer["cost"] = cost.as_json()
    check_later = not can_ask and bool(plan.texts or described)  # the key is tried before anything goes to Voyage
    keep_to_limit(cost, "setup", spending.voyage_usd(embeddings.KEY_CHECK_TOKENS, model_name) if check_later else 0.0)
    if _limit is not None:
        answer["spent"] = 0.0
    checked = len(chosen) - 1 if checking else 0  # every feature but the prompt
    say_estimate(
        found.new_tasks, {**settings, "model": model_settings, "max_tasks": limit}, plan, asks, checked=checked
    )
    if checking:
        question = (
            "Analyze and embed them, then check the features?" if asks else "Embed them, then check the features?"
        )
    else:
        question = (
            "Analyze them, embed them and train the model now?" if asks else "Embed them and train the model now?"
        )
    if can_ask and not ui.ask(question, True):
        ui.note("Stopped. Nothing was created.")
        return 0
    if not can_ask and not args.yes:
        ui.line("Add --yes to go ahead.")
        return 2
    if check_later:
        verify_key(key, model_name)  # before LLM analysis, which would be paid for in vain with a key that fails

    config: dict = {**settings, "model": model_settings, "plugins": plugins}
    if limit:
        config["max_tasks"] = int(limit)
    if make_default and not first:
        config["default_when_ready"] = True  # it becomes your default once it's trained
    embedding = {"model": model_name, "dimensions": embeddings.DIMENSIONS}
    folder = predictor_folder(name)
    private_home()
    # A delete of an older predictor with this name holds the name's lock until its folder is gone. Setup waits for
    # it, so the delete never takes away what setup makes. The refresh lock is taken before the predictor exists,
    # so no refresh can start on it until setup is done.
    with name_lock(name):
        folder.mkdir(parents=True, exist_ok=True)
        if not held.enter_context(locks.held(folder / "refresh.lock", wait=False)):
            raise CommandError("busy", f"A model called {name!r} is being set up by another command.")
        try:
            predictor = Predictor.create(name, archives, embedding, **config)
        except ValueError:
            raise CommandError("predictor_exists", already_exists(name)) from None
        # The dataset closes as setup ends, before the predictor's lock goes. Otherwise it stays open until Python
        # tidies up, and Windows can't delete an open file, as a delete in the same program, such as a test, would.
        held.callback(predictor.close)
    if first:
        set_default(name)  # the first predictor is the default, even when setup stops before it's trained
    save_scan(predictor, found)
    warning = refresh_benchmarks(predictor)
    if warning:
        ui.note(warning)
        answer["notes"].append(warning)
    try:
        if asks:
            reserve = spending.embedding_most_usd(plan, model_name, described)  # kept for the embedding after it
            answer["analyzed"], problem = analyze_with_progress(predictor, asks, reserve)
            if problem:
                answer["notes"].append(problem)
            plan = plan_embeddings(predictor, key)  # now with the descriptions LLM analysis wrote
        if plan.items:
            answer["embedded"] = embed_with_progress(predictor, plan, key)
        if checking:
            stop = check_and_choose(predictor, key)
            if stop is not None:
                return stop
    finally:
        if _limit is not None:
            answer["spent"] = round(_limit.spent, 6)
    model = train_and_report(predictor, train)
    default_when_ready(predictor)
    say_how_to_use(model, name, screen=getattr(args, "screen", False))
    answer.update(
        done=True, trained=True, learned_from=model.meta.get("tasks"), forecasts_for=served_models(model.meta["report"])
    )
    return 0


def with_key(plan, model: str, key: str | None):
    """The same plan, priced for a key, so its free tokens count. Its tokens aren't counted again."""
    return dataclasses.replace(plan, cost=embeddings.cost(plan.cost.tokens, plan.cost.exact, model, key))


def newest_only(plan, tasks: list, limit: int, model: str, key: str | None):
    """The plan for only the newest `limit` tasks the model learns from, with the token counts it has already.

    The plan's items are the tasks' texts. Its texts are the ones that go to Voyage, each once, with their
    counts. A text that only older tasks need is left out.
    """
    from .pipeline import learnable

    kept = {task.id for task in learnable(tasks, limit)}
    items = [item for item in plan.items if item[0] in kept]
    needed = {text for _, _, text in items}
    count_of = dict(zip(plan.texts, plan.counts))
    texts = [text for text in plan.texts if text in needed]
    counts = [count_of[text] for text in texts]
    cost = embeddings.cost(sum(counts), plan.cost.exact, model, key)
    return dataclasses.replace(plan, items=items, texts=texts, counts=counts, cost=cost)


def default_when_ready(predictor: Predictor) -> None:
    """Make a predictor your default once it's trained, when you asked for that at setup."""
    if predictor.config.pop("default_when_ready", False):
        set_default(predictor.name)
        predictor.save_config()


def embed_with_progress(predictor: Predictor, plan, key: str) -> int:
    """Embed what the plan holds with a progress bar, and return how many texts went to Voyage. When Voyage stops
    it, it stops the command: what was embedded so far is saved."""
    from . import spending
    from .pipeline import run_embeddings

    most = spending.embedding_most_usd(plan, predictor.embedding_model)
    if _limit is not None and not _limit.fits(most):
        raise CommandError(
            "over_max_cost",
            f"Embedding would take the spending past your limit of {spending.dollars(_limit.usd)}. "
            f"{spending.dollars(_limit.spent)} is spent already. Embedding may cost up to {spending.dollars(most)} at "
            "list price. What was done so far is saved. With a higher --max-cost, the next refresh goes on from there.",
        )
    ui.heading("Embedding")
    ui.note("Each batch is saved as it arrives, so you can stop and resume later.")
    seen = {"total": 0}

    def progress(done: int, total: int) -> None:
        seen["total"] = total
        ui.progress(done, total, "texts")

    try:
        run_embeddings(predictor, plan, key, progress, on_wait=ui.note)
    except embeddings.VoyageError as error:
        if "free token" in error.detail.lower() or "payment" in error.detail.lower():
            embeddings.set_free_tokens(key, "used")
        raise CommandError(
            "embedding_failed",
            f"{explain(error)}\nWhat was embedded so far is saved. "
            f"Run yardstick-predict refresh --predictor {predictor.name} to finish.",
        ) from None
    if _limit is not None:
        _limit.spent += most
    ui.ok(f"Embedded {seen['total']:,} texts." if seen["total"] else "Embedded.")
    return seen["total"]


def train_and_report(predictor: Predictor, train):
    """Train, test and show how each AI model did. With too little to train on, it stops the command."""
    from .model import NotEnoughData

    ui.heading("Training")
    # The built-in model's bar takes this line's place. A model type from a plugin keeps it.
    ui.working("Training, then checking its forecasts. This takes a minute")
    try:
        model = train(predictor, lambda done, total: ui.progress(done, total, "steps"))
    except NotEnoughData as error:
        raise CommandError("not_enough_data", str(error)) from None
    ui.ok(f"Trained on {model.meta['tasks']:,} tasks.")
    for name in model.meta.get("features_without_values") or []:
        ui.note(
            f"Your feature {name} had no value for any past task, so the model can't learn from it. A feature "
            "that reads only a request's extra has none for past tasks: work it out from the task itself too."
        )
    print_report(model.meta)
    return model


def say_how_to_use(model, name: str, full: bool = True, screen: bool = False) -> None:
    """Which AI models get forecasts, and how to ask for one, with their real names. `screen` is for the home screen,
    where Update adds new tasks."""
    served = served_models(model.meta["report"])
    listed = (served[0] if len(served) == 1 else ", ".join(served[:-1]) + " and " + served[-1]) if served else ""
    if not full:
        if served:
            ui.ok(f"Forecasts are on for {listed}.")
        return
    is_default = default_name() == name
    pick = "" if is_default else f" --predictor {name}"
    again = "choose Update on the home screen" if screen else f"run yardstick-predict refresh{pick}"
    ui.heading("Ready")
    if not served:
        ui.line("No AI model has enough data yet. Until one has, forecasts come back empty, with the reason.")
        ui.note(f"Once you have more tasks, {again}. It adds them and retrains.")
        return
    ui.line(f"Forecasts are on for {listed}.")
    ui.line("Before you start a task, ask what it will take:")
    ui.line(ui.accent(f'  yardstick-predict predict{pick} "Add a login page with tests"'))
    if is_default:
        ui.note("It's your default model. Commands use it when they name none.")
    ui.note("Programs send JSON to yardstick-predict predict and read JSON back.")
    ui.note(f"Every week or so, {again} to add new tasks and retrain.")


# The feature check -------------------------------------------------------------------------------------


def offer_check(chosen: list[str], tasks: list, limit: int | None) -> bool:
    """Ask whether to check which of the chosen features help, before anything is spent. Enter says yes.

    It's asked only when an optional feature is chosen and at least one AI model has enough tasks for the test
    to judge it. Otherwise the check couldn't tell anything.
    """
    from collections import Counter

    from .model import min_tasks_to_test
    from .pipeline import learnable

    if len(chosen) < 2:  # only the prompt, which is always used
        return False
    counts = Counter(task.model for task in learnable(tasks, limit))
    if max(counts.values(), default=0) < min_tasks_to_test():
        return False
    ui.heading("Feature check")
    ui.note("Some features carry no signal for your kind of work. With few tasks, a feature can't show anything.")
    return ui.ask("Check which of these features actually help? Recommended.", True)


def run_check(predictor: Predictor):
    """Run the feature check with a progress bar, show what it found and remember it.

    Returns what it found, or None when it couldn't test any AI model, so the features stay as they are.
    """
    from . import check
    from .model import NotEnoughData

    ui.heading("Feature check")
    # The bar takes this line's place once the first trial model is being fitted.
    ui.working("Comparing forecasts with every feature, then without each one")
    try:
        result = check.run(check.inputs(predictor), lambda done, total: ui.progress(done, total, "steps"))
    except NotEnoughData as error:
        ui.note(f"It couldn't run. {error}")
        return None
    check.remember(result, predictor.name)
    say_checked(result)
    show_check(result)
    return result if any(result.tasks_tested(model) for model in result.models) else None


def say_checked(result) -> None:
    """Say how many features the check tested, on how many of the AI models' newest tasks."""
    count = len(result.features)
    tested = result.report.get("tested_on", 0)
    tasks = f"{tested:,} newest task{'' if tested == 1 else 's'}"
    ui.ok(f"Checked {count} feature{'' if count == 1 else 's'} on your {tasks}.")


def show_check(result) -> None:
    """The feature check's verdicts as a small table: a row for each feature, a column for each AI model tested."""
    from . import check

    tested = [model for model in result.models if result.tasks_tested(model)]
    if not tested:
        needed = result.report.get("min_tasks_to_test") or 60
        wrapped_note(
            f"There's not enough data to check the features yet. Each AI model needs {needed:,} tasks of its own."
        )
        return
    for part in ui.wrapped("Forecasts of each AI model's newest tasks, with every feature and then without each one:"):
        ui.line(part)
    ui.line()
    looks = {check.HELPED: ui.green, check.NO_HELP: str, check.NOT_ENOUGH: ui.dim}
    rows = [("", *(ui.dim(model) for model in tested))]
    for name in result.features:
        found = [result.verdicts[name][model]["verdict"] for model in tested]
        rows.append((ui.bold(check.title(name)), *(looks[kind](check.WORDS[kind]) for kind in found)))
    wide = 2 + sum(max(ui.width(row[i]) for row in rows) + 3 for i in range(len(rows[0]))) - 3
    if wide < ui.columns():
        ui.table(rows)
    else:
        # A column for each AI model would run past the terminal's edge, so each feature says it in words.
        for name in result.features:
            ui.line(ui.bold(check.title(name)))
            by_model = {model: result.verdicts[name][model] for model in tested}
            for part in ui.wrapped(check.sentences(by_model), 4):
                ui.line(f"  {part}")
    ui.line()
    wrapped_note(f"A feature helped when an AI model's forecasts were over {check.WORSE_BY:.0%} worse without it.")
    wrapped_note(
        f"Not enough data: the AI model had too few tasks. Or under {check.LEAST_SHARE:.0%} of its newest tasks had "
        "the feature, or of the tasks before them."
    )
    short = check.sampled_too_little(result)
    if "llm" in result.alone and len(short) < len(tested):
        wrapped_note(
            "LLM analysis read only some of your tasks. So it was judged on those alone: the forecasts with it and "
            "without it learned from and forecast only the tasks it read."
        )
    untested = [model for model in result.models if not result.tasks_tested(model)]
    if untested:
        wrapped_note(f"Not enough data: {in_words(untested)}.")
    if short:
        wrapped_note(
            f"LLM analysis read only a sample of your tasks, too few to judge it for {in_words(short)}. To judge "
            "it fully, keep it, let it read the rest, then run: yardstick-predict check"
        )


def check_and_choose(predictor: Predictor, key: str) -> int | None:
    """In setup: run the feature check, then ask once whether to keep the features that helped. No opens the Features
    screen to change them.

    The features that helped at least one AI model stay on. When the check can't test any AI model, the features
    stay as chosen. Whatever the final choice still needs is shown with its cost and asked for first: LLM analysis
    of the tasks its sample left out, and embedding any text the check didn't have. Returns None to go on and
    train, or the command's exit code when it stops.
    """
    from . import features, menu
    from .pipeline import plan_embeddings, refresh_benchmarks

    config = predictor.config
    sampled = "llm" in config["features"]  # LLM analysis read only a sample for the check
    result = run_check(predictor)
    if result is not None:
        helped = result.helped()
        options = {name: chosen for name, chosen in (config.get("options") or {}).items() if name in helped}
        start = menu.Picked([features.REQUIRED, *helped], options, dict(config.get("llm") or {}))
        ui.line()
        if keep_what_helped(config["features"], helped):
            picked = menu.Picked(start.features, start.options, start.llm if "llm" in helped else {})
        else:
            ui.note("Change them below. Press Enter when they're as you want them.")
            picked = menu.ask(custom_features(), start)
        config.pop("llm", None)
        config.update(picked.config())
        predictor.save_config()
    if "benchmarks" in config["features"] and not (predictor.dir / "benchmarks.json").is_file():
        warning = refresh_benchmarks(predictor)
        if warning:
            ui.note(warning)
    if "llm" in config["features"] and not analyze_the_rest(predictor, sampled):
        config["features"] = [name for name in config["features"] if name != "llm"]
        (config.get("options") or {}).pop("llm", None)
        config.pop("llm", None)
        predictor.save_config()
        ui.note("LLM analysis is left out of this model.")
    plan = plan_embeddings(predictor, key)
    # The descriptions LLM analysis just wrote were in the cost of analyzing the rest. Anything else is new.
    if any(name != "llm" for _, name, _ in plan.items):
        ui.heading("Cost")
        for text in cost_lines(plan.cost, predictor.embedding_model, plan.tasks):
            ui.line(text)
        if not ui.ask("Embed them and train the model now?", True):
            ui.note(
                "Stopped. Your choice of features is kept. Embed and train later with: "
                f"yardstick-predict refresh --predictor {predictor.name}"
            )
            return 0
    if plan.items:
        embed_with_progress(predictor, plan, key)
    return None


def keep_what_helped(chosen: list[str], helped: list[str]) -> bool:
    """After the feature check, in setup: say which of the chosen features stay on and which go off, then ask once
    whether to go on with that. Enter says yes. No opens the Features screen to change them first."""
    from .screens.listing import LEARNS_FROM

    optional = [name for name in chosen if name != "prompt"]
    kept = [LEARNS_FROM.get(name, name) for name in optional if name in helped]
    dropped = [LEARNS_FROM.get(name, name) for name in optional if name not in helped]
    if kept:
        ui.line("The features that helped an AI model stay on. The rest go off.")
        ui.line(f"On: {in_words(kept)}.")
    else:
        ui.line("None of the features you chose helped an AI model. The model learns from the three always on.")
    if dropped:
        ui.line(f"Off: {in_words(dropped)}.")
    return ui.ask("Keep this and go on to make the model?", True)


def analyze_the_rest(predictor: Predictor, sampled: bool) -> bool:
    """Show what LLM analysis of the tasks it hasn't read costs, ask, and run it. False when the person says no.

    With `sampled`, it read a sample for the feature check. Turned on only after the check, it read nothing yet.
    """
    from . import features
    from .pipeline import analysis_plan, plan_embeddings

    labels = predictor.llm_labels()
    rest = analysis_plan(predictor.tasks(), predictor.config, labels, predictor.config.get("max_tasks"))
    if not rest:
        return True
    ui.heading("Cost")
    if sampled:
        ui.line("For the check, LLM analysis read only a sample of your tasks. To read the rest:")
    for text in analysis_lines(len(rest), predictor.config["llm"]):
        ui.line(text)
    if "description" in features.options(predictor.config, "llm"):
        ui.note("Its descriptions are embedded too, which adds a little to the Voyage cost.")
    reserve = 0.0
    if _limit is not None:
        from . import spending

        described = len(rest) if "description" in features.options(predictor.config, "llm") else 0
        cost = spending.cost_of(
            predictor.config, predictor.embedding_model, plan_embeddings(predictor, None), rest, described
        )
        if cost.most_usd is None or not _limit.fits(cost.most_usd):
            raise CommandError(
                "over_max_cost",
                "Reading the rest with LLM analysis would take the spending past your limit. Setup stopped here, "
                "with your choice of features kept. With a higher --max-cost, finish with: "
                f"yardstick-predict refresh --predictor {predictor.name}",
            )
        reserve = cost.embedding_most_usd  # kept for embedding what's still to come
    if not ui.ask("Analyze them now?", True):
        return False
    analyze_with_progress(predictor, rest, reserve)
    return True


def cmd_refresh(args) -> int:
    def run(answer: dict) -> int:
        with limited(args):
            return refresh(args, answer)

    if args.json:
        return for_program(args, run)
    return run({})


def refresh(args, answer: dict) -> int:
    """Add a predictor's new tasks, embed them and train it again. `answer` is filled with what a program reads:
    what was found, what it costs, and what was done."""
    predictor = pick_predictor(args.predictor)
    answer["predictor"] = predictor.name
    if predictor.newer:
        # A refresh would save the settings, the dataset and the model the way this older version does.
        raise CommandError(
            "newer_version",
            f"{predictor.name} was saved by a newer version of yardstick-predict, so this version leaves it as it "
            "is. Upgrade with: uv tool upgrade yardstick-predict",
        )
    # The Update screen asks once, with every cost. It keeps to a --max-cost limit too.
    if interactive() and not args.yes and not args.json:
        from .screens import update

        predictor.close()
        return update.run([predictor.name])
    # A predictor deleted just as this command found it isn't saved again, with none of its tasks.
    deleted = f"There's no model called {predictor.name!r} anymore: it was deleted."
    if gone(predictor):
        raise CommandError("unknown_predictor", deleted)
    from . import feature_store

    # Two refreshes at once could each train, and the slower one could save an older model over a newer one.
    with locks.held(predictor.dir / "refresh.lock", wait=False) as mine:
        if not mine:
            raise CommandError(
                "busy", f"{predictor.name} is being set up or refreshed by another command. Try again when it's done."
            )
        if gone(predictor):
            raise CommandError("unknown_predictor", deleted)
        # It reads the feature store to plan what to make, and saves to it, so no features delete runs meanwhile.
        with feature_store.writing():
            return _refresh(args, predictor, answer)


BEHIND_NOTES = {
    "unfinished": ["Training didn't finish last time. The model is out of date."],
    "outdated": ["This model was made by an older version, which trained it another way. It needs training again."],
    "previous": [
        "This model was made by the previous version. It still forecasts, and training it again brings",
        "this version's improvements.",
    ],
    "damaged": ["A file of this model is missing or can't be read. Training it again repairs it."],
}
"""What a refresh says when the model is trained again even with nothing new, by the reason plan_refresh gives."""


def retrain_reason(planned) -> str | None:
    """Why a refresh trains the model, for a program: new_work, no_model, unfinished, outdated, previous or damaged.
    None when it's up to date."""
    if planned.up_to_date:
        return None
    if planned.model is None:
        return "no_model"
    return planned.behind or "new_work"


def gone(predictor: Predictor) -> bool:
    """Whether the predictor was deleted after this command found it. Its settings go last."""
    return not (predictor.dir / "predictor.json").is_file()


def _refresh(args, predictor: Predictor, answer: dict) -> int:
    from . import model_types, spending
    from .pipeline import (
        load_plugins,
        model_meta,
        plan_embeddings,
        plan_refresh,
        refresh_benchmarks,
        save_scan,
        scan,
        train,
    )

    answer.update(
        done=False,
        up_to_date=False,
        new_tasks=0,
        changed_tasks=0,
        removed_tasks=0,
        cost=None,
        retrain_reason=None,
        analyzed=0,
        embedded=0,
        trained=False,
        forecasts_for=trained_facts(predictor, model_meta(predictor))["forecasts_for"],
        max_cost=None if _limit is None else _limit.usd,
        spent=None,
        notes=[],
    )
    try:
        load_plugins(predictor)
    except model_types.PluginError as error:
        raise CommandError("plugin_failed", str(error)) from None
    # With --yes it asks nothing, even in a terminal, and nor does a program that asked for JSON.
    can_ask = interactive() and not getattr(args, "yes", False) and not getattr(args, "json", False)
    ui.heading(predictor.name)
    ui.note(f"Last updated {ago(predictor.config.get('refreshed_at'))}.")
    unknown_features_note(predictor)
    ui.working("Looking for new tasks")
    found = scan(predictor, on_progress=lambda done, total: ui.progress(done, total, "bytes"))
    counts = [f"{source} {count:,}" for source, count in found.per_archive.items() if count]
    ui.ok(f"Found {len(found.new_tasks):,} new tasks" + (f": {ui.join(counts)}" if counts else "."))
    from .screens.update import short_path

    for archive in predictor.config["archives"]:
        if not archive_path(archive).is_dir():  # read, it would look like a folder with nothing new
            text = no_folder(short_path(archive), [predictor.name])
            ui.warn(text)
            answer["notes"].append(text)
    if getattr(found, "reader", None) is not None and predictor.config.get("reader") != found.reader:
        # A newer version reads records more accurately. It read them all again, once.
        ui.note(
            "This version reads your records more accurately, so it read them all again. "
            f"{len(found.updated_tasks):,} saved tasks changed."
        )
    elif found.updated_tasks:
        grew = len(found.updated_tasks)
        ui.note(
            f"{grew:,} saved tasks changed, because they or their helper agents finished or stopped after the last "
            "refresh."
        )
    removed = getattr(found, "removed", [])
    if removed:
        ui.note(f"{len(removed):,} saved tasks aren't in their records anymore, so they're taken out.")
    answer.update(new_tasks=len(found.new_tasks), changed_tasks=len(found.updated_tasks), removed_tasks=len(removed))
    model_name = predictor.embedding_model
    key, _ = embeddings.find_key()
    planned = plan_refresh(predictor, found, key)
    for text in BEHIND_NOTES.get(planned.behind if planned.model is not None else None, []):
        ui.note(text)
    everything, plan, asks = planned.tasks, planned.embed, planned.asks
    cost = spending.cost_of(predictor.config, model_name, plan, asks, planned.described)
    answer["cost"] = cost.as_json()
    answer["retrain_reason"] = retrain_reason(planned)
    if _limit is not None:
        answer["spent"] = 0.0
    if planned.up_to_date:
        # A record file that changed with nothing new in it, such as one that was only touched, isn't read
        # again next time. Nor is every file, after a new version of the reader read them all and found
        # nothing new.
        predictor.remember_files(found.files)
        if getattr(found, "reader", None) is not None:
            predictor.config["reader"] = found.reader
            predictor.save_config()
        ui.ok("Nothing new. The model is up to date.")
        answer.update(done=True, up_to_date=True)
        return 0
    if plan.items:
        damaged_usage_note(key)
        for text in cost_lines(plan.cost, model_name, plan.tasks):
            ui.line(text)
    if asks:
        for text in analysis_lines(len(asks), predictor.config["llm"]):
            ui.line(text)
    if planned.described:
        from .llm import DESCRIPTION_TOKENS

        tokens = planned.described * DESCRIPTION_TOKENS
        ui.note(f"Its descriptions are embedded too: about {tokens:,} more tokens, priced as above.")
    dataset = predictor.dir / "dataset.sqlite"
    size = (dataset.stat().st_size if dataset.is_file() else 0) + planned.stored
    say_estimate(everything, predictor.config, plan, asks, found.new_tasks, size)
    keep_to_limit(cost, "refresh")
    if can_ask:
        if not ui.ask("Refresh now? This adds the new tasks, embeds them and retrains.", True):
            ui.note("Keeping the current model.")
            return 0
    elif not args.yes:
        ui.line("Add --yes to go ahead.")
        return 2
    if plan.texts or planned.described:  # a key only for what goes to Voyage: LLM analysis needs none of it
        key = key or get_key(model_name, can_ask)
        if not key:
            return 1
    save_scan(predictor, found)
    warning = refresh_benchmarks(predictor)
    if warning:
        ui.note(warning)
        answer["notes"].append(warning)
    try:
        if asks:
            # What embedding may cost after it is kept out of what LLM analysis may spend.
            reserve = spending.embedding_most_usd(plan, model_name, planned.described)
            answer["analyzed"], problem = analyze_with_progress(predictor, asks, reserve)
            if problem:
                answer["notes"].append(problem)
        # The plan priced above is what's left to embed now that the tasks are saved, so its counts are reused.
        if plan.items:
            answer["embedded"] += embed_with_progress(predictor, plan, key)
        # Each text the plan above needed has its vector now, so this one pays only for the descriptions LLM
        # analysis just wrote.
        rest = plan_embeddings(predictor, key)
        if rest.items:
            if rest.texts:
                key = key or get_key(model_name, can_ask)
                if not key:
                    return 1
            answer["embedded"] += embed_with_progress(predictor, rest, key)
    finally:
        if _limit is not None:
            answer["spent"] = round(_limit.spent, 6)
    model = train_and_report(predictor, train)
    default_when_ready(predictor)
    say_how_to_use(model, predictor.name, full=False)
    answer.update(done=True, trained=True, forecasts_for=served_models(model.meta["report"]))
    return 0


def cmd_predict(args) -> int:
    from .forecast import RequestError

    prompt = args.task if args.task is not None else args.prompt
    table = prompt is not None and terminal_output() and not args.json and not args.distribution
    predictor = None
    try:
        # A person's table shows every AI model, with a warning on the rough ones. A program gets only tested ones.
        predictor, result = forecast_request(args, prompt, rough=table)
        code = 0
    except RequestError as error:
        result, code = error.as_answer(), 1
    except Exception as error:  # a program still gets an answer in the format, never a traceback
        first_line = (str(error).strip().splitlines() or [""])[0]
        message = f"Something went wrong that yardstick-predict didn't expect: {type(error).__name__}: {first_line}"
        result, code = RequestError("internal_error", message.rstrip(": ")).as_answer(), 1
    person = prompt is None and args.request is None and sys.stdin.isatty() and terminal_output() and not args.json
    if table or person:
        if code:
            ui.error(result["error"]["message"])
        else:
            show_forecasts(prompt, result, predictor)
        return code
    print(json.dumps(result, indent=2 if terminal_output() else None))
    return code


def forecast_request(args, prompt: str | None, rough: bool = False) -> tuple[Predictor, dict]:
    """The predictor and the answer for a prompt on the command line, or for a JSON request from a file or standard
    input.

    With `rough`, for a person's table: every AI model the predictor knows, at each effort level it often ran at,
    with the curves the table's ranges come from. One that failed the test or wasn't tested keeps its reason.
    """
    from .forecast import RequestError, answer, load, parse

    if prompt is not None and getattr(args, "request", None) is not None:
        raise RequestError("invalid_request", "Give the task in quotes or a request with --request, not both.")
    if prompt is not None and not prompt.strip():
        raise RequestError(
            "invalid_request", 'Give the task some text, like: yardstick-predict predict "Fix the login bug"'
        )
    if prompt is None:
        request = read_request(getattr(args, "request", None))
        tasks, distribution, chosen = parse(request)
        predictor = request_predictor(args.predictor or chosen, from_request=True)
        return predictor, answer(predictor, tasks, distribution or args.distribution)
    predictor = request_predictor(args.predictor)
    # The model is loaded once. Its meta picks the AI models and effort levels, and it forecasts them.
    model = load(predictor)
    runs, too_few = {}, []
    if rough and not args.model:
        pairs, runs, too_few = every_pair(predictor, args.effort, model.meta)
    else:
        models = list(dict.fromkeys(args.model)) or forecast_models(predictor, model.meta)
        pairs = effort_pairs(predictor, models, args.effort, model.meta)
        pairs = for_people(pairs) if rough else pairs
    if rough and not pairs:  # no AI model ran enough tasks to show: the table says so, with how many each ran
        empty = {"version": 1, "predictor": predictor.name, "embedding_model": predictor.embedding_model}
        return predictor, {**empty, "forecasts": [], "too_few": too_few}
    result = forecast_pairs(predictor, model, prompt, pairs, args, rough, runs)
    if rough:
        result["too_few"] = too_few
    return predictor, result


def read_request(path: str | None):
    """The JSON request: from the file `path` names, or from standard input when there's no path or it's -."""
    from .forecast import RequestError

    try:
        if path is not None and path != "-":
            with open(os.path.expanduser(path), encoding="utf-8") as found:
                return json.load(found)
        if sys.stdin.isatty():
            raise RequestError("invalid_request", 'Give the task in quotes, like: yardstick-predict predict "Fix it"')
        return json.load(sys.stdin)
    except OSError as error:
        raise RequestError(
            "invalid_request", f"Can't read the request file {path}: {error.strerror or error}."
        ) from None
    except ValueError as error:
        raise RequestError("invalid_request", f"The request isn't valid JSON: {error}") from None
    except RecursionError:
        raise RequestError("invalid_request", "The request is nested too deeply to read.") from None


def forecast_pairs(predictor: Predictor, model, prompt: str, pairs: list, args, rough: bool = False, runs=None) -> dict:
    """The answer for a prompt a person typed, on each AI model at each effort level in `pairs`. `args` holds the
    conversation, the skills and whether to give the curves. With `rough`, as forecast_request says, each forecast
    also says how many tasks its AI model ran at its effort level, from `runs`."""
    from .forecast import answer, parse

    asked = []
    for number, (name, effort) in enumerate(pairs, start=1):
        task = {"id": "task" if len(pairs) == 1 else f"task-{number}", "prompt": prompt, "models": [name]}
        # A prompt given on the command line was typed by a person. A JSON request says who for itself.
        # The folder it's asked from is the project, so the project's recent tasks count, as in training.
        task.update(context=args.context or "", skills=args.skill, origin="person", project=Path.cwd().name)
        if effort:
            task["effort"] = effort
        asked.append(task)
    tasks, distribution, _ = parse({"version": 1, "distribution": args.distribution or rough, "tasks": asked})
    result = answer(predictor, tasks, distribution, model=model, rough=rough)
    # One prompt is one task, whatever models and effort levels it's forecast on.
    for forecast in result["forecasts"]:
        forecast["task"] = "task"
        if rough:
            # A pair asked with no effort level comes back with the level the AI model ran at most.
            found, name = runs or {}, forecast["model"]
            forecast["runs"] = found.get((name, forecast.get("effort")), found.get((name, None)))
    return result


SHOWN_FROM = 10
"""A person's table shows an AI model at an effort level once it ran this many tasks at it."""
FEW_RUNS = 30
"""Below this many tasks at an effort level, the table warns that there's not enough data."""
DEFAULT_EFFORT = "default"
"""What the records call the effort level of a task run without one set. No one picks it, so a person's lists
leave it out when the AI model ran at a named level too, and never show the word."""


def for_people(pairs: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """The pairs a person's table shows: none at the default effort level when the AI model has a named one too. An
    AI model that ran only at the default level is forecast with no level named."""
    named = {model for model, effort in pairs if effort and effort != DEFAULT_EFFORT}
    return [
        (model, None if effort == DEFAULT_EFFORT else effort)
        for model, effort in pairs
        if not (effort == DEFAULT_EFFORT and model in named)
    ]


def every_pair(
    predictor: Predictor, effort: str | None, meta: dict
) -> tuple[list[tuple[str, str | None]], dict, list[tuple[str, int]]]:
    """For a person's table: every AI model the predictor knows, at each named effort level it ran 10 tasks at, or
    at its most common named level when its tasks spread over several. The default level is left out. An AI model
    that ran only at the default level is forecast with no level named. The ones with enough data come first. Returns
    the pairs, how many tasks each pair had, and the AI models with too few tasks to forecast at all, with how many
    they had.
    """
    report = meta.get("report") or {}
    verdicts = report.get("models") or {}
    counts = meta.get("effort_counts")
    if not isinstance(counts, dict):
        counts = predictor.effort_counts()
    known = list((meta.get("categories") or {}).get("model") or verdicts)

    def total(model: str) -> int:
        return sum(count for _, count in counts.get(model, []))

    def rank(model: str) -> tuple:
        verdict = verdicts.get(model) or {}
        group = 0 if verdict.get("served") else 1 if verdict.get("reason") == "not_accurate_enough" else 2
        return group, -(verdict.get("tasks_tested") or 0), -total(model)

    pairs, runs, too_few = [], {}, []
    for model in sorted(known, key=rank):
        levels = [(level, count) for level, count in counts.get(model, []) if count]
        if effort:
            pairs.append((model, effort))
            runs[(model, effort)] = dict(levels).get(effort, 0)
            continue
        if total(model) < SHOWN_FROM:
            too_few.append((model, total(model)))
            continue
        named = [(level, count) for level, count in levels if level != DEFAULT_EFFORT]
        often = [(level, count) for level, count in named if count >= SHOWN_FROM] or named[:1]
        for level, count in often or [(None, total(model))]:
            pairs.append((model, level))
            runs[(model, level)] = count
    return pairs, runs, too_few


def request_predictor(chosen: str | None, from_request: bool = False) -> Predictor:
    """The predictor a forecast request names, or your default one, with errors in the answer format.

    `from_request` says the request came as JSON, so a message about choosing a predictor names its field.
    """
    from .forecast import RequestError

    names = predictor_names()
    if not names:
        raise RequestError("no_predictor", "There's no model yet. Run yardstick-predict to create one.")
    if chosen and chosen not in names:
        raise RequestError("unknown_predictor", f"There's no model called {chosen!r}. Your models: {', '.join(names)}.")
    name = chosen or default_name()
    if name is None:
        where = 'Name one in the request\'s "predictor"' if from_request else "Name one with --predictor"
        raise RequestError(
            "no_default_predictor",
            f"There are several models and none is the default: {', '.join(names)}. {where}, "
            "or choose a default with: yardstick-predict default NAME",
        )
    try:
        return Predictor(name)
    except BrokenPredictor as error:
        raise RequestError("broken_predictor", str(error)) from None


def effort_pairs(
    predictor: Predictor, models: list[str], effort: str | None, meta: dict | None = None
) -> list[tuple[str, str | None]]:
    """Each AI model with each effort level to forecast: the one asked for, or every level it ran at often.

    A level counts as often when the model ran 30 tasks at it. A model with no such level is forecast at
    the level it ran at most often. The counts are of the tasks the model was trained on, from its `meta`,
    or from the dataset for a model saved before it counted them.
    """
    from .model import MIN_TEST_TASKS
    from .pipeline import model_meta

    if effort:
        return [(model, effort) for model in models]
    meta = model_meta(predictor) if meta is None else meta
    counts = (meta or {}).get("effort_counts")
    if not isinstance(counts, dict):
        counts = predictor.effort_counts()
    pairs: list[tuple[str, str | None]] = []
    for model in models:
        common = [level for level, count in counts.get(model, []) if count >= MIN_TEST_TASKS]
        pairs += [(model, level) for level in common] or [(model, None)]
    return pairs


def forecast_models(predictor: Predictor, meta: dict | None = None) -> list[str]:
    """The AI models to forecast for a request that names none: the ones that passed the test, the most tested first.

    When none has passed yet, every AI model the predictor knows, so each comes back with the reason it has
    no forecast. `meta` is what the saved model says about itself, when it's read already.
    """
    from .forecast import RequestError, not_trained
    from .pipeline import model_meta

    meta = model_meta(predictor) if meta is None else meta
    if meta is None:
        raise RequestError("not_trained", not_trained(predictor.name))
    report = meta.get("report") or {}
    known = served_models(report) or tested_models(report) or list((meta.get("categories") or {}).get("model") or [])
    if not known:
        raise RequestError("invalid_request", "This model knows no AI model yet. Name one with --model.")
    return known


def show_title(task: str, after: str = "") -> None:
    """A task as a title, in bold, broken to fit the terminal on 3 lines at most. A longer one is cut short on the
    third. `after` follows it, dimmed, on its last line, or on a line of its own where there's no room."""
    parts = ui.wrapped(ui.clean(task))
    if len(parts) > 3:
        parts = [*parts[:2], ui.cut(" ".join(parts[2:]), ui.columns() - 3)]
    for part in parts[:-1]:
        ui.line(ui.bold(part))
    if after and ui.width(parts[-1]) + 2 + ui.width(after) <= ui.columns() - 3:
        ui.line(f"{ui.bold(parts[-1])}  {ui.dim(after)}")
        return
    ui.line(ui.bold(parts[-1]))
    if after:
        ui.line(ui.dim(after))


FORECAST_COLUMNS = ("", "effort", "", "time", "tokens", "cost")
"""The forecast table's columns after the AI model's name. The first holds the warning mark, when there is one. The
third names each line of a forecast: expected, then 25%, median and 75%."""
QUARTILES = (("25%", 0.25), ("median", 0.5), ("75%", 0.75))
"""The lines under each forecast in a person's table, each with the share of real results that come in under it."""
QUARTILES_NOTE = (
    "The expected line is what to expect on average. A quarter of real results come in under the 25% line, half "
    "under the median and three quarters under the 75% line."
)
"""What the lines of each forecast mean, in the note under the table."""
WARNINGS = {
    "not_accurate_enough": "not enough data",
    "not_enough_data": "not enough data",
    "unknown_model": "no data",
}
"""The words of a rough forecast's warning, by the reason code a program gets. Not enough data covers an AI model
whose forecasts were often wrong on your past tasks, and one with too few of them to check."""


def show_forecasts(prompt: str, result: dict, predictor: Predictor | None = None, screen: bool = False) -> None:
    """The forecasts as a small table, for a person in a terminal. Each forecast's first line is what to expect on
    average. Under it, in grey, are the values a quarter, half and three quarters of real results come in under,
    when the forecast has a curve. A column of its own names each line, after the effort level, which shows only on
    the first. A forecast with not enough data behind it is marked, and the AI models that have too little are named
    under the table. It fits the terminal at any width.

    `screen` is for the Estimate screen: the task was typed just above, so it isn't said again.
    """
    from .forecast import quantile

    pad = max([len("AI model"), *(len(forecast["model"]) for forecast in result["forecasts"])])
    rows, missing, previous, spread, blocks = [], {}, None, False, []
    thin: dict[str, bool] = {}
    """The AI models with not enough data, and the effort levels with too few runs, to name under the table."""
    for forecast in result["forecasts"]:
        model = forecast["model"]
        seconds, tokens, usd = forecast["seconds"], forecast["tokens"], forecast["usd"]
        if seconds is None and tokens is None and usd is None:
            reason = REASONS.get(forecast.get("reason", ""), "no forecast")
            missing.setdefault(model, reason)
            continue
        # A model forecast at several effort levels is named once, on its first row. The name is fitted below.
        name = model if model != previous else ""
        previous = model
        warning = rough_warning(forecast)
        effort = shown_effort(forecast.get("effort"))
        if warning:
            # Too little for the whole AI model names it alone. Too few runs at one level names that level.
            thin[model if forecast.get("reason") or not effort else f"{model} at {effort}"] = True
        mark = ui.yellow(ui.mark("warn")) if warning else ""
        values = (
            duration(seconds) if seconds is not None else ui.dim("-"),
            short_tokens(tokens) if tokens is not None else ui.dim("-"),
            money(usd) if usd is not None else ui.dim("-"),
        )
        rows.append((name, mark, effort, ui.dim("expected"), *values))
        # The 25%, median and 75% values each go on a row of their own, named in the column after the effort level.
        # The effort column stays empty there. A line's name never looks like an effort level.
        curve = forecast.get("distribution")
        found = [[quantile(curve, target, share) for target in ("seconds", "tokens", "usd")] for _, share in QUARTILES]
        words = []
        if any(value is not None for line in found for value in line):
            spread = True
            shows = (duration, whole_tokens, short_money)
            for (label, _), line in zip(QUARTILES, found):
                words.append(tuple("" if value is None else show(value) for value, show in zip(line, shows)))
                rows.append(("", "", "", ui.dim(label), *(ui.dim(word) for word in words[-1])))
        blocks.append((model, mark, effort, values, words))
    titles = FORECAST_COLUMNS
    if not spread:  # with one line to each forecast, there's no line to name
        titles = (*titles[:2], *titles[3:])
        rows = [(*r[:3], *r[4:]) for r in rows]
    # A table needs room for every column, its titles too. A terminal too narrow for it gets each forecast as a
    # block of its own.
    others = sum(max([len(titles[i - 1]), *(ui.width(r[i]) for r in rows)]) + 2 for i in range(1, len(titles) + 1))
    stacked = 2 + pad + others >= ui.columns()
    rows = [(ui.bold(r[0].ljust(pad)) if r[0] else " " * pad, *r[1:]) for r in rows]
    if not screen:
        ui.out()
        show_title(prompt)
    ui.out()
    if rows and not stacked:
        header = tuple(ui.dim(title) for title in ("AI model".ljust(pad), *titles))
        ui.table([header, *rows], align="l" * (len(titles) - 2) + "rrr", gap=2)
    elif rows:
        show_blocks(blocks)
    for model, reason in missing.items():
        # cut colors and all, only when it can't fit with them
        ui.line(ui.cut(f"{ui.bold(ui.cut(model, pad).ljust(pad))}   {ui.dim(reason)}", ui.columns() - 3))
    too_few = result.get("too_few") or []
    if too_few:
        listed = in_words([f"{m} ({n:,} run{'' if n == 1 else 's'})" for m, n in too_few])
        wrapped_note(f"Not enough data to forecast {listed}.")
    predictor = predictor or Predictor(result["predictor"])
    ui.out()
    # An AI model named on its own covers its levels, so a level of it isn't named again.
    names = [who for who in thin if " at " not in who or who.split(" at ")[0] not in thin]
    if names:
        mark = ui.yellow(ui.mark("warn"))
        for number, part in enumerate(ui.wrapped(f"Not enough data: {in_words(names)}.", 5)):  # names stay whole
            ui.line(f"{mark if number == 0 else ' '} {part}")
        wrapped_note("Those are a rough guide. Plan with a wider margin there.")
    if spread:
        wrapped_note(f"{QUARTILES_NOTE} Costs are at API prices.")
    footer = [
        "Learned from your own runs",
        f"model {predictor.name}, updated {ago(predictor.config.get('refreshed_at'))}",
        *([] if screen else ["add --json for details"]),
    ]
    wrapped_note((f" {ui.mark('dot')} " if ui.styled() else ", ").join(footer))
    newer_settings_note(predictor)
    unknown_features_note(predictor)


def columns() -> int:
    """How wide the terminal is, 80 when it doesn't say."""
    return shutil.get_terminal_size((80, 24)).columns


def wrapped_note(text: str) -> None:
    """A note broken to fit the terminal, so a long one never runs past its edge. It never breaks a name at a hyphen,
    such as claude-sonnet-4-6, nor inside a command."""
    for part in ui.wrapped(text, 3):
        ui.note(part)


def show_blocks(blocks: list) -> None:
    """Each forecast as a block of its own, for a terminal too narrow for the table: the AI model with its mark and
    effort level, then a small table of its time, tokens and cost. Its first line is what to expect on average. The
    25%, median and 75% lines follow, each named."""
    for number, (model, mark, effort, values, words) in enumerate(blocks):
        if number:
            ui.out()
        name = ui.bold(ui.cut(model, ui.columns() - 16))
        ui.line(f"{name}{f'  {mark}' if mark else ''}{f'  {effort}' if effort else ''}")
        rows = [("", *(ui.dim(title) for title in ("time", "tokens", "cost"))), (ui.dim("expected"), *values)]
        rows += [(ui.dim(label), *(ui.dim(word) for word in line)) for (label, _), line in zip(QUARTILES, words)]
        ui.table(rows, indent=4, align="lrrr", gap=2)


def rough_warning(forecast: dict) -> str | None:
    """Why a forecast in a person's table is only a rough guide, in a few words, or None when there's enough data
    behind it: for its AI model, and at its effort level."""
    if forecast.get("reason"):
        return WARNINGS.get(forecast["reason"], "not enough data")
    runs = forecast.get("runs")
    if isinstance(runs, int) and runs < FEW_RUNS:
        return "not enough data"
    return None


def shown_effort(effort: str | None) -> str:
    """An effort level as a person's table shows it. The default level isn't one a person picks, so it's left out."""
    return "" if not effort or effort == DEFAULT_EFFORT else effort


def short_money(usd: float) -> str:
    """Dollars at a glance, such as the lines under a forecast: $0.0042, $0.12, $29.17, $424, $59.0k. Under a cent,
    the two digits that matter show, so the lines of a cheap task still say something."""
    from .spending import dollars

    if usd >= 999.5:
        return f"${usd / 1000:.1f}k"
    if usd >= 99.5:
        return f"${usd:.0f}"
    return dollars(usd) if 0 < usd < 0.01 else money(usd)


def whole_tokens(value: float) -> str:
    """Tokens at a glance, such as the lines under a forecast, which a curve gives as fractions."""
    return short_tokens(int(round(value)))


def trained_facts(predictor: Predictor, meta: dict | None) -> dict:
    """What list and status say about a predictor's model: how many tasks it learned from, when it was last trained
    and the AI models it gives forecasts for, from what the saved model says about itself. A model that has to be
    trained again first gives none. Without a model, it learned from none and was never trained."""
    from .pipeline import outdated

    if not meta:
        return {"learned_from": None, "trained_at": None, "forecasts_for": []}
    learned = meta.get("tasks")
    report = meta.get("report") if isinstance(meta.get("report"), dict) else {}
    try:
        forecasts = [] if outdated(predictor, meta) else served_models(report)
    except (KeyError, TypeError, ValueError, AttributeError):  # a report this version can't read
        forecasts = []
    return {
        "learned_from": learned if isinstance(learned, int) and not isinstance(learned, bool) else None,
        "trained_at": predictor.config.get("refreshed_at") or meta.get("trained_at"),
        "forecasts_for": forecasts,
    }


def prepared_bytes(predictor: Predictor) -> int | None:
    """How big the prepared dataset is, what went into the last training. None before the first training."""
    try:
        return predictor.prepared_path.stat().st_size
    except OSError:
        return None


def coverage_of(predictor: Predictor) -> tuple[dict | None, str | None]:
    """The share of the tasks a predictor can learn from that have a value for each of its features, counted as
    the features listing counts it, for status. None, with the reason, when it can't be counted now.

    It loads the predictor's plugins, and needs XGBoost, as training does. Like the listing, it moves the vectors
    and LLM answers of a predictor an older version made into the feature store, once.
    """
    from . import model_types
    from .pipeline import load_plugins

    try:
        load_plugins(predictor)
        total, uses = feature_uses_of(predictor, count=True)
    except model_types.PluginError as error:
        return None, f"Its coverage can't be counted until its plugin loads. {error}"
    except StoreBusy:
        return None, "Its coverage can't be counted while another command is using the feature store."
    except BlockingIOError:
        return None, "Its coverage can't be counted while another command sets it up or refreshes it."
    except Exception as error:  # such as a plugin's feature that fails on a task: the rest of status still shows
        first = (str(error).strip().splitlines() or [""])[0]
        return None, f"Its coverage can't be counted: {type(error).__name__}: {first}".rstrip(": ")
    listed = predictor_json({"name": predictor.name, "default": False, "tasks": total, "uses": uses})
    return {"tasks": total, "features": [use for use in listed["features"] if use["kept"] != "unknown"]}, None


def show_coverage(coverage: dict | None, problem: str | None) -> None:
    """One line: the share of the tasks a predictor can learn from that have a value for each of its features."""
    if coverage is None:
        if problem:
            ui.note(problem)
        return
    if not coverage["tasks"]:
        for part in ui.wrapped("Coverage: it has no tasks to learn from yet."):
            ui.line(part)
        return
    shares = [f"{use['title']} {use['coverage']:.0%}" for use in coverage["features"] if use["coverage"] is not None]
    for part in ui.wrapped(f"Coverage of the {coverage['tasks']:,} tasks it can learn from: {', '.join(shares)}"):
        ui.line(part)


def cmd_status(args) -> int:
    """Each predictor, its features with their coverage, and how its test went. The test is what the saved model
    says about itself, so it shows without XGBoost and without the predictor's plugins. Coverage is counted from
    what training reads, so it needs both: without them, the rest still shows."""
    from .pipeline import model_meta

    names = predictor_names()
    if args.predictor and args.predictor not in names:
        message = f"There's no model called {args.predictor!r}."
        if args.json:
            print(json.dumps({"version": 1, "error": {"code": "unknown_predictor", "message": message}}, indent=2))
            return 1
        raise SystemExit(message)
    without = getattr(args, "xgboost", None)  # coverage needs XGBoost. Without it, the rest still shows.
    default = default_name()
    found, metas, predictors, problems = [], {}, {}, {}
    notes: list[str] = []
    quiet = args.json and not without
    if quiet:
        from . import feature_store

        feature_store.tell = notes.append  # a program reads them in the answer, not between its lines
    try:
        for name in [args.predictor] if args.predictor else names:
            try:
                predictor = Predictor(name)
            except BrokenPredictor as error:
                found.append({"name": name, "default": name == default, "error": str(error)})
                continue
            metas[name] = meta = model_meta(predictor)
            predictors[name] = predictor
            entry = {
                "name": name,
                "default": name == default,
                "embedding_model": predictor.embedding_model,
                "features": predictor.config.get("features") or ["prompt"],
                "options": predictor.config.get("options") or {},
                "llm": predictor.config.get("llm"),
                "model": predictor.config.get("model") or {},
                "archives": predictor.config["archives"],
                "tasks": predictor.count_tasks(),
                "refreshed_at": predictor.config.get("refreshed_at"),
                "report": (meta or {}).get("report"),
                **trained_facts(predictor, meta),
                "prepared_bytes": prepared_bytes(predictor),
                "coverage": None,
            }
            if not without:
                entry["coverage"], problems[name] = coverage_of(predictor)
            found.append(entry)
    finally:
        if quiet:
            feature_store.tell = None
    code = 1 if any("error" in entry for entry in found) else 0  # a predictor it shows can't be used
    if args.json:
        print(json.dumps({"version": 1, "predictors": found, "notes": notes}, indent=2))
        return code
    if not found:
        ui.out("There's no model yet. Run yardstick-predict to create one.")
        return 0
    from .pipeline import from_previous_version, outdated

    for entry in found:
        ui.out()
        title = [ui.bold(entry["name"])] + ([ui.dim("default model")] if entry["default"] else [])
        ui.out(ui.join(title))
        if "error" in entry:
            ui.line(entry["error"])
            continue
        facts = [
            f"{entry['embedding_model']} embeddings",
            f"{entry['tasks']:,} tasks",
            f"updated {ago(entry['refreshed_at'])}",
        ]
        if entry["prepared_bytes"] is not None:
            facts.append(f"prepared dataset {ui.size(entry['prepared_bytes'])}")
        predictor = predictors[entry["name"]]
        joined = ui.join(facts)
        for part in [joined] if ui.width(joined) < ui.columns() - 2 else ui.wrapped(", ".join(facts)):
            ui.line(part)
        for part in ui.wrapped(f"Features: {feature_summary(predictor.config)}"):
            ui.line(part)
        show_coverage(entry["coverage"], problems.get(entry["name"]))
        newer_settings_note(predictor)
        if not predictor.config.get("plugins") or entry["coverage"] is not None:
            # Without plugins, every feature this version knows is built in. With them, they're loaded by now.
            unknown_features_note(predictor)
        meta = metas.get(entry["name"])
        if meta and outdated(predictor, meta):
            ui.note("An older version trained it. It gives no forecasts until it's trained again.")
            ui.note(f"Train it again with: yardstick-predict refresh --predictor {entry['name']}")
        elif meta:
            if from_previous_version(predictor, meta):
                ui.note(
                    "The previous version trained it. It still forecasts. Train it again to get this "
                    f"version's improvements: yardstick-predict refresh --predictor {entry['name']}"
                )
            try:
                ui.line(f"Trained on {meta.get('tasks', 0):,} tasks.")
                print_report(meta)
            except (KeyError, TypeError, ValueError, AttributeError):  # a report this version can't read
                ui.note("Its forecast report can't be shown. A newer version of yardstick-predict may have saved it.")
    if without and predictors:
        ui.out()
        ui.note("Coverage isn't shown: it's counted from what training reads, and training needs XGBoost.")
        ui.note(without)
    return code


def short_features(config: dict) -> str:
    """A predictor's optional features, as the Features screen names them, or that it uses only the ones always on.
    A feature this version doesn't know keeps its own name."""
    from . import features

    titles = {group.key: group.title for group in features.MENU}
    listed = config.get("features") or []
    chosen = [str(name) for name in listed if name != features.REQUIRED] if isinstance(listed, list) else []
    return ", ".join(titles.get(name, name) for name in chosen) or "only the ones always on"


def cmd_list(args) -> int:
    """One line for each predictor: its features, how many tasks it learned from, when it was last trained and the
    AI models it gives forecasts for. Like status, it reads only what each saved model says about itself, so it
    needs neither XGBoost nor the predictors' plugins. Status shows the same, in detail."""
    from .pipeline import model_meta, outdated

    default = default_name()
    found, retrain = [], set()
    for name in predictor_names():
        entry: dict = {"name": name, "default": name == default}
        try:
            predictor = Predictor(name)
        except BrokenPredictor as error:
            entry["error"] = str(error)
        else:
            meta = model_meta(predictor)
            entry["features"] = predictor.config.get("features") or ["prompt"]
            entry.update(trained_facts(predictor, meta))
            if meta and outdated(predictor, meta):
                retrain.add(name)
        found.append(entry)
    code = 1 if any("error" in entry for entry in found) else 0  # a predictor it shows can't be used
    if args.json:
        print(json.dumps({"version": 1, "predictors": found}, indent=2))
        return code
    if not found:
        ui.out("There's no model yet. Run yardstick-predict to create one.")
        return 0
    rows = [("", *(ui.dim(words) for words in ("features", "learned from", "trained", "forecasts for")))]
    for entry in found:
        name = ui.bold(entry["name"]) + (f" {ui.dim('(default)')}" if entry["default"] else "")
        if "error" in entry:
            rows.append((name, ui.dim("can't be used, see below")))
            continue
        learned, trained = entry["learned_from"], entry["trained_at"]
        if entry["name"] in retrain:
            forecasts = ui.dim("none until it's trained again")
        else:
            forecasts = ", ".join(entry["forecasts_for"]) or ui.dim("none yet")
        rows.append(
            (
                name,
                short_features({"features": entry["features"]}),
                f"{learned:,} tasks" if learned is not None else ui.dim("-"),
                ago(trained) if trained else ui.dim("not yet"),
                forecasts,
            )
        )
    wide = 2 + sum(max(ui.width(row[i]) for row in rows if i < len(row)) + 3 for i in range(5)) - 3
    if wide < ui.columns():
        ui.table(rows)
    else:
        # A line for each would run past the terminal's edge, so each gets a few lines of its own.
        for row, entry in zip(rows[1:], found):
            ui.line(row[0])
            if "error" in entry:
                ui.line(f"  {row[1]}")
                continue
            trained = f"trained {ui.plain(row[3])}" if entry["trained_at"] else "not trained yet"
            learned = [f"learned from {row[2]}"] if entry["learned_from"] is not None else []
            ui.line(f"  {ui.join([row[1], *learned, trained])}")
            ui.line(f"  {ui.dim('forecasts for')} {row[4]}")
    for entry in found:
        if "error" in entry:
            ui.out()
            ui.line(entry["error"])
    return code


def in_words(names: list[str]) -> str:
    """Names as a person lists them: a, b and c. Nothing for no names."""
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def answer_error(args, code: str, message: str, exit_code: int = 1) -> int:
    """Stop a command that answers programs in JSON with --json: an error in that format, or the message."""
    if args.json:
        print(json.dumps({"version": 1, "error": {"code": code, "message": message}}, indent=2))
        return exit_code
    if exit_code == 2:  # it needs --yes: not an error, but a step only you can take
        ui.line(message)
        return exit_code
    raise SystemExit(message)


def cmd_delete(args) -> int:
    """Delete a predictor: its folder, with its settings, its dataset of tasks, its prepared dataset and its model.
    Your records, the feature store and your other predictors stay. In a terminal it says what goes and asks
    first. A program adds --yes, and --json for an answer in JSON."""
    from .store import Busy, being_refreshed, contents, delete_predictor

    names = predictor_names()
    name = args.name
    if name not in names:
        message = f"There's no model called {name!r}. Your models: {', '.join(names) or 'none'}."
        return answer_error(args, "unknown_predictor", message)
    # A setup or refresh of it would make it again, so it's refused before anything is asked.
    if being_refreshed(name):
        busy = f"{name} is being set up or refreshed by another command. Try again when it's done."
        return answer_error(args, "busy", busy)
    found = contents(name)
    default = default_name()
    if not args.json:
        say_what_goes(name, found, default == name)
    if interactive() and not args.yes and not args.json:
        if not ui.ask(f"Delete {name} for good?", False):
            ui.note("Nothing was deleted.")
            return 0
    elif not args.yes:
        return answer_error(args, "needs_yes", "Add --yes to delete it. Nothing was deleted.", exit_code=2)
    try:
        delete_predictor(name)
    except Busy as error:
        return answer_error(args, "busy", str(error))
    except OSError as error:
        return answer_error(
            args,
            "delete_failed",
            f"Couldn't delete all of {name}: {error.strerror or error}. It's still listed, so you can try again "
            f"when no other program is using its files: yardstick-predict delete {name}",
        )
    now = default_name()
    if args.json:
        deleted = {
            "name": name,
            "folder": str(found.folder),
            "link_to": None if found.link_to is None else str(found.link_to),  # only the link went
            "tasks": found.tasks,
            "bytes": found.size,
        }
        print(json.dumps({"version": 1, "deleted": deleted, "default": now}, indent=2))
        return 0
    if found.link_to is not None:
        ui.ok(f"Deleted the link {name}. {found.link_to} stays as it was.")
    else:
        ui.ok(f"Deleted {name}.")
    say_default_after(name, default, now)
    return 0


def say_what_goes(name: str, found, is_default: bool) -> None:
    """Exactly what deleting a predictor removes, and what stays."""
    ui.heading(f"Delete {name}")
    for quiet, text in what_goes(name, found, is_default):
        (ui.note if quiet else ui.line)(text)


def what_goes(name: str, found, is_default: bool) -> list[tuple[bool, str]]:
    """Exactly what deleting a predictor removes, and what stays, as lines, each with whether it's a quiet note."""
    default = [(True, "It's your default model.")] if is_default else []
    if found.link_to is not None:
        return [
            (False, f"{name} is a link to {found.link_to}, not a folder of its own."),
            (False, "This deletes only the link. What it links to stays, with everything in it."),
            *default,
        ]
    goes = ["its settings"] if found.settings else []
    if found.dataset:
        tasks = "" if found.tasks is None else f" of {found.tasks:,} task{'' if found.tasks == 1 else 's'}"
        goes.append(f"its dataset{tasks}, with their prompts ({ui.size(found.dataset)})")
    if found.prepared:
        goes.append(f"its prepared dataset, what went into its last training ({ui.size(found.prepared)})")
    if found.model:
        goes.append(f"what it learned ({ui.size(found.model)})")
    return [
        (False, f"This deletes its folder, {found.folder}, and everything in it:"),
        *((False, f"  {part}") for part in goes),
        (False, "Your records and the feature store stay, and so do your other models."),
        (True, "The embeddings and LLM answers it made stay in the feature store, for any model that uses them."),
        *default,
    ]


def say_default_after(name: str, before: str | None, after: str | None) -> None:
    """Say which predictor commands use now that one is deleted, when that changed. When one predictor is left,
    commands use it without a default."""
    left = predictor_names()
    if not left:
        ui.note("There's no model now. Run yardstick-predict to create one.")
    elif after is None and before == name:
        ui.note("It was your default model, so there's no default now.")
        ui.note("Choose one with: yardstick-predict default NAME")
    elif after is not None and after != before:
        if len(left) == 1:
            ui.note(f"{after} is your only model now, so commands use it when they name none.")
        else:
            ui.note(
                f"Commands now use {after} when they name none. Choose another with: yardstick-predict default NAME"
            )


def cmd_check(args) -> int:
    """Check which of a predictor's features help its forecasts, and show the verdicts. It's advice only: it reads
    the predictor's tasks and features, embeds nothing, asks no AI model and keeps none of the models it tests.
    It remembers the verdicts, so the Features screen can show them. With --json, the feature store's notes, such
    as about a file it can't read, come in the answer, so nothing is printed before it."""
    from . import check, feature_store, model_types
    from .model import NotEnoughData
    from .pipeline import load_plugins, settings

    notes: list[str] = []

    def stop(code: str, message: str) -> int:
        if args.json:
            answer = {"version": 1, "error": {"code": code, "message": message}}
            print(json.dumps({**answer, "notes": notes} if notes else answer, indent=2))
            return 1
        raise SystemExit(message)

    names = predictor_names()
    if args.name and args.name not in names:
        return stop(
            "unknown_predictor",
            f"There's no model called {args.name!r}. Your models: {', '.join(names) or 'none'}.",
        )
    if not names:
        return stop("no_predictor", "There's no model yet. Run yardstick-predict to create one.")
    name = args.name or default_name()
    if name is None:
        return stop(
            "no_default_predictor",
            f"You have several models and no default: {', '.join(names)}. Name one, like: "
            f"yardstick-predict check {names[0]}",
        )
    try:
        predictor = Predictor(name)
        load_plugins(predictor)
    except BrokenPredictor as error:
        return stop("broken_predictor", str(error))
    except model_types.PluginError as error:
        return stop("plugin_failed", str(error))
    kind = settings(predictor).get("type", "xgboost")
    if kind != "xgboost":
        return stop(
            "unsupported_model_type",
            f"{name} uses the {kind} model type from a plugin. The check works with the built-in model only.",
        )
    person = not args.json
    if person:
        ui.heading(name)
        ui.working("Reading its tasks")
    # No setup or refresh changes the tasks while they're read. The tests then run on what was read, so one can
    # start meanwhile.
    if gone(predictor):
        return stop("unknown_predictor", f"There's no model called {name!r} anymore: it was deleted.")
    with locks.held(predictor.dir / "refresh.lock", wait=False) as mine:
        if not mine:
            return stop("busy", f"{name} is being set up or refreshed by another command. Try again when it's done.")
        if gone(predictor):
            return stop("unknown_predictor", f"There's no model called {name!r} anymore: it was deleted.")
        told = feature_store.tell
        if args.json:
            feature_store.tell = notes.append  # a program reads them in the answer, not before it
        try:
            with moving_in(predictor) as free:
                if not free:
                    return stop("busy", STORE_BUSY)
                given = check.inputs(predictor)
        finally:
            feature_store.tell = told
    if not given.features:
        if args.json:
            print(json.dumps({**check.as_json(check.judge({"": {}}, {}, []), name), "notes": notes}, indent=2))
            return 0
        ui.line("There's nothing to check: it uses only the features that are always on.")
        return 0
    if person:
        ui.working("Comparing forecasts with every feature, then without each one")
    try:
        result = check.run(given, (lambda done, total: ui.progress(done, total, "steps")) if person else None)
    except NotEnoughData as error:
        return stop("not_enough_data", f"The check can't run. {error}")
    check.remember(result, name)
    if args.json:
        print(json.dumps({**check.as_json(result, name), "notes": notes}, indent=2))
        return 0
    say_checked(result)
    show_check(result)
    ui.line()
    ui.note("This is advice only: nothing changed. To use other features, create a new model with them.")
    return 0


def stored_title(feature) -> str:
    """What a stored feature is and what made it, in plain words: prompt embeddings by voyage-4, 256 numbers."""
    from . import llm

    made = feature.made_with
    if feature.kind == "embedding":
        field = {"llm": "LLM description"}.get(made.get("field"), made.get("field"))
        return f"{field} embeddings by {made.get('model')}, {made.get('dimensions')} numbers"
    provider = llm.PROVIDERS.get(made.get("provider"))
    how = f", through your {provider.title}" if provider else f", through {made.get('provider')}"
    return f"LLM answers by {llm.title(str(made.get('model')))}{how}"


def stored_facts(found) -> list[str]:
    """How many tasks a stored feature covers, its size on disk and when it last changed, or what's wrong with it."""
    if found.missing:
        return ["missing: its folder is gone, and the next refresh that uses it makes it again"]
    tasks = f"{found.tasks:,} task{'s' if found.tasks != 1 else ''}"
    facts = [tasks, ui.size(found.size), f"changed {ago(found.changed_at)}"]
    if found.unreadable:
        files = len(found.unreadable)
        facts.append(f"skips {files} file{'s' if files != 1 else ''} it can't read: {', '.join(found.unreadable)}")
    return facts


def feature_uses_of(predictor: Predictor, count: bool):
    """The predictor's features and how many of its tasks each covers, as pipeline.feature_uses gives them.

    A predictor an older version made has its vectors and LLM answers moved to the feature store first, once. A
    refresh or setup does that too, so while one runs, this waits for none: it raises BlockingIOError. While a
    features delete runs, it raises StoreBusy.
    """
    from .pipeline import feature_uses

    if predictor.config.get("feature_store"):
        return feature_uses(predictor, count)
    with locks.held(predictor.dir / "refresh.lock", wait=False) as mine:
        if not mine:
            raise BlockingIOError
        with moving_in(predictor) as free:
            if not free:
                raise StoreBusy
            predictor.move_in()
            return feature_uses(predictor, count)


class StoreBusy(BlockingIOError):
    """Another command is using the feature store, as a features delete does, so a predictor an older version made
    can't move its vectors and LLM answers in now."""


STORE_BUSY = "Another command is using the feature store. Try again when it's done."


def moving_in(predictor: Predictor):
    """The lock of the feature store's users, shared and without waiting, when the predictor still has its
    vectors and LLM answers to move into the store, as one an older version made does. It yields False when a
    features delete is running. For a predictor that moved in already, it holds nothing and yields True."""
    from contextlib import nullcontext

    if predictor.config.get("feature_store"):
        return nullcontext(True)
    from . import feature_store

    return feature_store.writing(wait=False)


def cmd_features(args) -> int:
    """Every feature of every predictor, in plain words: what it is, whether the feature store keeps it or
    training works it out, and how many of the tasks the predictor learns from have a value for it.

    It reads and changes nothing, except that a predictor an older version made has its vectors and LLM
    answers moved into the feature store, once, as a refresh would. `features delete ID` deletes a stored feature.
    """
    from . import feature_store, model_types
    from .pipeline import load_plugins

    if getattr(args, "action", None) == "delete":
        said: list[str] = []
        if args.json:
            feature_store.tell = said.append  # a program reads them in the answer, not between its lines
        try:
            return delete_feature(args, said)
        finally:
            feature_store.tell = None

    names = predictor_names()
    if args.predictor and args.predictor not in names:
        message = f"There's no model called {args.predictor!r}."
        if args.json:
            print(json.dumps({"version": 1, "error": {"code": "unknown_predictor", "message": message}}, indent=2))
            return 1
        raise SystemExit(message)
    without = getattr(args, "xgboost", None)  # coverage needs XGBoost. Without it, the rest is still listed.
    notes: list[str] = []
    if args.json:
        feature_store.tell = notes.append  # a program reads them in the answer, not between its lines
    default = default_name()
    predictors = []
    try:
        for name in [args.predictor] if args.predictor else names:
            entry: dict = {"name": name, "default": name == default}
            predictors.append(entry)
            try:
                predictor = Predictor(name)
                load_plugins(predictor)
                entry["tasks"], entry["uses"] = feature_uses_of(predictor, count=not without)
            except BrokenPredictor as error:
                entry["error"] = str(error)
            except model_types.PluginError as error:
                entry["error"] = f"{error} Its features can't be listed until the plugin loads."
            except StoreBusy:
                entry["error"] = STORE_BUSY
            except BlockingIOError:
                entry["error"] = f"{name} is being set up or refreshed by another command. Try again when it's done."
        found = feature_store.stored()
    finally:
        feature_store.tell = None
    used_by: dict[str, list[str]] = {}
    for entry in predictors:
        for use in entry.get("uses", []):
            for feature in use.stored:
                used_by.setdefault(feature.id, []).append(entry["name"])
    code = 1 if any("error" in entry for entry in predictors) else 0  # a predictor it shows can't be used
    if args.json:
        listing = {
            "version": 1,
            "folder": str(feature_store.folder()),
            "stored": [stored_json(stored, used_by) for stored in found],
            "predictors": [predictor_json(entry) for entry in predictors],
            "notes": notes,
        }
        print(json.dumps(listing, indent=2))
        return code
    show_stored(found, used_by)
    for entry in predictors:
        show_uses(entry, {stored.feature.id: stored for stored in found})
    if not predictors:
        ui.out()
        ui.out("There's no model yet. Run yardstick-predict to create one.")
    if without:
        ui.out()
        ui.note("Coverage isn't shown: it's counted from what training reads, and training needs XGBoost.")
        ui.note(without)
    return code


def delete_feature(args, notes: list[str]) -> int:
    """Delete a stored feature: its folder in the feature store and its line in the index. It says which
    predictors use it and what making it again costs, and in a terminal it asks first. A program adds --yes. It's
    refused while any command that saves to the store runs, such as a setup or a refresh: that may be saving to
    it, or have planned with it. `notes` collects what the store says, for an answer in JSON."""
    from . import feature_store

    if not args.id:
        return answer_error(
            args,
            "invalid_request",
            "Name the stored feature to delete, such as: yardstick-predict features delete "
            "embedding.context.voyage-4.256. yardstick-predict features lists each one with its id.",
        )
    found = next((stored for stored in feature_store.stored() if stored.feature.id == args.id), None)
    if found is None:
        message = f"There's no stored feature {args.id!r}. yardstick-predict features lists each one with its id."
        return answer_error(args, "unknown_feature", message)

    def refused() -> int:
        message = (
            "Another command that saves to the feature store is running, such as a setup or a refresh, and it may be "
            "using this feature. Try again when it's done."
        )
        return answer_error(args, "busy", message)

    # Each command that saves to the store holds its users' lock, shared, while it runs. A delete holds it alone.
    with feature_store.deleting() as free:
        pass
    if not free:
        return refused()
    users, unsure = users_of(found.feature.id)
    remade = None
    if not args.xgboost:  # which tasks get it again is worked out as training picks them, with XGBoost
        from .pipeline import remake

        remade = remake(found.feature, users)
    if not args.json:
        say_what_feature_goes(found, users, unsure, remade)
    if interactive() and not args.yes and not args.json:
        if not ui.ask("Delete it for good?", False):
            ui.note("Nothing was deleted.")
            return 0
    elif not args.yes:
        return answer_error(args, "needs_yes", "Add --yes to delete it. Nothing was deleted.", exit_code=2)
    with feature_store.deleting() as free:
        if free:
            try:
                feature_store.delete(found.feature.id)
            except OSError as error:
                message = f"Couldn't delete all of it: {error.strerror or error}. Try again when no other program "
                return answer_error(args, "delete_failed", message + "is using its files.")
    if not free:
        return refused()
    if args.json:
        deleted = stored_json(found, {found.feature.id: [predictor.name for predictor in users]})
        remaking = dataclasses.asdict(remade) if remade is not None else None
        print(json.dumps({"version": 1, "deleted": deleted, "remake": remaking, "notes": notes}, indent=2))
        return 0
    ui.ok(f"Deleted {stored_title(found.feature)}.")
    return 0


def users_of(feature_id: str) -> tuple[list[Predictor], list[str]]:
    """The predictors that use a stored feature, with their plugins loaded, and a note on each that can't tell.
    A predictor whose settings can't be read uses nothing."""
    from . import model_types
    from .pipeline import feature_uses, load_plugins

    users, unsure = [], []
    for name in predictor_names():
        try:
            predictor = Predictor(name)
            load_plugins(predictor)
        except BrokenPredictor:
            continue
        except model_types.PluginError as error:
            unsure.append(f"Whether {name} uses it can't be told until its plugin loads. {error}")
            continue
        _, uses = feature_uses(predictor, count=False)
        if any(feature.id == feature_id for use in uses for feature in use.stored):
            users.append(predictor)
    return users, unsure


def say_what_feature_goes(found, users: list[Predictor], unsure: list[str], remade) -> None:
    """Exactly what deleting a stored feature removes, who makes it again and what that costs, and what stays."""
    ui.heading("Delete a stored feature")
    ui.line(stored_title(found.feature))
    ui.line(ui.dim(ui.join(stored_facts(found))))
    for text in remake_lines(found.feature, users, remade):
        ui.line(text)
    for text in unsure:
        ui.note(text)
    if users:
        now = "those models forecast as they do" if len(users) > 1 else "that model forecasts as it does"
        ui.note(f"Until then, {now} now: a forecast never reads the feature store.")
    ui.line("Your records and your models stay, and so does the rest of the feature store.")


def remake_lines(feature, users: list[Predictor], remade) -> list[str]:
    """Who makes a deleted stored feature again, and what that costs, in plain words. `remade` is what pipeline.remake
    found, or None when it couldn't be worked out."""
    from . import features, llm

    if not users:
        return ["No model uses it, so nothing makes it again."]
    names = in_words([predictor.name for predictor in users])
    who = f"Used by {names}. " + (
        "The next refresh of each makes it again" if len(users) > 1 else "Its next refresh makes it again"
    )
    if remade is None:
        return [f"{who}.", ui.dim("What that costs can't be worked out without XGBoost, which training needs too.")]
    asks = "A refresh shows what it costs and asks before it spends anything, unless you give it --yes."
    if feature.kind == "llm":
        how = {"provider": feature.made_with.get("provider"), "model": str(feature.made_with.get("model"))}
        found = [f"{who}, for {remade.tasks:,} task{'' if remade.tasks == 1 else 's'}."]
        if remade.analyses_least < remade.analyses:
            found.append(
                "They ask for different parts of the analysis, so a task can be read more than once: from "
                f"{remade.analyses_least:,} to {remade.analyses:,} readings in all when they refresh one at a time, "
                "depending on which goes first. Below is the most."
            )
        elif remade.analyses > remade.tasks:
            found.append("They ask for different parts of the analysis, so their tasks are read once for each.")
        if how["provider"] in llm.PROVIDERS and remade.analyses:
            found += analysis_lines(remade.analyses, how, remade.tasks)
        if any("description" in features.options(predictor.config, "llm") for predictor in users):
            found.append("It describes the tasks anew, and embedding the new descriptions costs Voyage tokens.")
        return [*found, asks]
    texts, cached = remade.texts, remade.cached
    found = [f"{who}, for {remade.tasks:,} task{'' if remade.tasks == 1 else 's'}."]
    paid = f"{'' if remade.exact else 'about '}{remade.tokens:,} tokens"
    if remade.usd is not None:
        paid += f", {money(remade.usd)} at Voyage's list price"
    if not texts:
        found.append("None of their tasks has its text now, so nothing is embedded.")
    elif cached == texts:
        held = "that text" if texts == 1 else f"all {texts:,} of their texts"
        found.append(f"The cache of vectors holds {held}, so it costs nothing.")
    elif cached:
        found.append(f"The cache of vectors holds {cached:,} of their {texts:,} texts, which cost nothing.")
        found.append(f"The other {texts - cached:,} go to Voyage: {paid}. {asks}")
    else:
        found.append(f"The cache of vectors holds none of their {texts:,} texts, so they go to Voyage: {paid}.")
        found.append(asks)
    return found


def stored_json(found, used_by: dict[str, list[str]]) -> dict:
    """A stored feature, as features --json shows it."""
    return {
        "id": found.feature.id,
        "kind": found.feature.kind,
        "made_with": found.feature.made_with,
        "missing": found.missing,
        "tasks": found.tasks,
        "rows": found.rows,
        "bytes": found.size,
        "changed_at": found.changed_at,
        "unreadable": found.unreadable,
        "used_by": used_by.get(found.feature.id, []),
    }


def predictor_json(entry: dict) -> dict:
    """A predictor's features, as features --json shows them. `tasks` is how many it learns from."""
    if "error" in entry:
        return {"name": entry["name"], "default": entry["default"], "error": entry["error"]}
    total = entry["tasks"]
    found = []
    for use in entry["uses"]:
        kept = "stored" if use.stored else "training" if use.known else "unknown"
        coverage = round(use.covered / total, 4) if use.covered is not None and total else None
        found.append(
            {
                "name": use.name,
                "title": use.title,
                "says": use.says,
                "kept": kept,
                "stored": [feature.id for feature in use.stored],
                "tasks_with_value": use.covered,
                "coverage": coverage,
            }
        )
    return {"name": entry["name"], "default": entry["default"], "tasks": total, "features": found}


def show_stored(found: list, used_by: dict[str, list[str]]) -> None:
    """The feature store's features, each with what made it, what it covers and which predictors use it."""
    from . import feature_store

    ui.out(ui.bold("Stored features"))
    ui.line(f"Embeddings and LLM answers are made once and kept in {feature_store.folder()}.")
    ui.line("Every model that uses one shares it. Features that cost nothing are worked out at training.")
    if not found:
        ui.out()
        ui.line("Nothing is stored yet.")
        return
    ui.out()
    rows = []
    for stored in found:
        users = used_by.get(stored.feature.id)
        who = f"used by {', '.join(users)}" if users else "used by no model"
        rows.append((stored_title(stored.feature), ui.dim(ui.join([*stored_facts(stored), who]))))
        rows.append((ui.dim(f"  id {stored.feature.id}"),))
    ui.table(rows)
    ui.out()
    ui.note("Delete one with: yardstick-predict features delete ID")


def show_uses(entry: dict, stored: dict) -> None:
    """One predictor's features, each with its coverage and whether it's stored or worked out at training."""
    ui.out()
    ui.out(ui.join([ui.bold(entry["name"])] + ([ui.dim("default model")] if entry["default"] else [])))
    if "error" in entry:
        ui.line(entry["error"])
        return
    total = entry["tasks"]
    if total:
        ui.line(f"It learns from {total:,} tasks. Coverage is the share of them with a value for each feature.")
    elif total == 0:
        ui.line("It has no tasks to learn from yet.")
    ui.out()
    rows = []
    for use in entry["uses"]:
        if not use.known:
            rows.append((use.title, "", ui.dim("this version doesn't know it, so it's skipped")))
            continue
        share = f"{use.covered / total:.0%}" if use.covered is not None and total else ui.dim("-")
        if use.stored:
            made = " and ".join(stored_title(feature) for feature in use.stored)
            missing = [feature for feature in use.stored if feature.id not in stored]
            how = f"stored: {made}" + (", not made yet" if missing and len(missing) == len(use.stored) else "")
        else:
            how = "worked out at training, free"
        rows.append((use.title, share, ui.dim(how)))
    ui.table(rows, align="lrl")


def cmd_claude_code(args) -> int:
    """Add the status line and its hooks to Claude Code's settings, or take them out again."""
    from . import claude_code

    path = claude_code.settings_path(args.here)
    try:
        settings = claude_code.read_settings(path)
    except ValueError:
        raise SystemExit(f"Claude Code's settings in {path} aren't valid JSON, so they were left alone.") from None
    except OSError as error:
        raise SystemExit(f"Claude Code's settings in {path} can't be read: {error.strerror or error}.") from None
    settings = {} if settings is None else settings
    if not isinstance(settings, dict):
        raise SystemExit(f"Claude Code's settings in {path} aren't a JSON object, so they were left alone.")
    where = "Claude Code sessions in this folder" if args.here else "all your Claude Code sessions"
    if args.action == "remove":
        changed = claude_code.removed(settings)
        if changed == settings:
            ui.out(f"yardstick isn't in the settings for {where}. Nothing to remove.")
            return 0
        home = claude_code.installed_home(settings)  # the folder the hook kept its forecasts in
        changed, restored = claude_code.put_back_status_line(settings, changed, path, home)
        backup = claude_code.save_settings(path, changed)
        ui.ok(f"Removed the forecasts from {where}. New sessions won't show them.")
        if restored is not None:
            shown = claude_code.other_status_line({"statusLine": restored}) or json.dumps(restored)
            ui.note(f"Your own status line is back: {shown}")
        if backup:
            ui.note(f"A copy of the file as it was is kept next to it, in {backup.name}.")
        if not claude_code.in_use(claude_code.settings_path(not args.here)):
            claude_code.clean(home)  # no settings it knows of use the forecasts the hook kept
        return 0
    if not predictor_names():
        raise SystemExit("There's no model yet. Run yardstick-predict to create one first.")
    hooks = settings.get("hooks", {})
    unreadable = not isinstance(hooks, dict) or any(
        not isinstance(hooks.get(event, []), list) for event in ("UserPromptSubmit", *claude_code.STOPS)
    )
    if unreadable:
        # Adding only the status line would show it without ever starting a forecast.
        raise SystemExit(
            f'The "hooks" in {path} aren\'t in the form Claude Code reads, so nothing was added. '
            "Fix or remove them, then run this again."
        )
    # Claude Code may run the hook without your YARDSTICK_HOME, so the commands name the folder themselves.
    home = os.environ.get("YARDSTICK_HOME")
    home = os.path.abspath(os.path.expanduser(home)) if home else None
    for used in (sys.executable, home or ""):
        problem = claude_code.unsafe_path(used)
        if problem:
            raise SystemExit(f"{problem} Nothing was added.")
    ui.out(ui.bold("Forecasts in Claude Code"))
    ui.line(f"This adds a status line and two hooks to {where}.")
    ui.note(f"They go in {path}." + (" A copy of the file as it is now is kept next to it." if path.is_file() else ""))
    ui.note("Each prompt you send there goes to Voyage to be embedded: a few tokens each.")
    if home:
        ui.note(f"They keep their forecasts in {home}, the folder YARDSTICK_HOME names.")
    other = claude_code.other_status_line(settings)
    if other:
        ui.note(f"It replaces your current status line: {other}")
        ui.note("Removing yardstick puts yours back.")
    if interactive() and not args.yes:
        if not ui.ask("Add them?", True):
            ui.note("Nothing changed.")
            return 0
    elif not args.yes:
        ui.line("Add --yes to go ahead.")
        return 2
    claude_code.remember_status_line(settings, path, home)
    claude_code.save_settings(path, claude_code.added(settings, sys.executable, home))
    ui.ok("Added. Start a new Claude Code session there to see forecasts in its status line.")
    ui.note("Remove them with: yardstick-predict claude-code remove" + (" --here" if args.here else ""))
    # The hook finds the Voyage key as every command does: in VOYAGE_API_KEY, then in the password store.
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    stored, answer = claude_code.stored_key(claude_code.KEY_SECONDS)
    if answer == "no_answer":
        ui.note("Your password store didn't answer in time. If it's locked, unlock it: forecasts look there.")
    elif not key and not stored:
        ui.note(
            "No Voyage key was found. Forecasts need one. Set VOYAGE_API_KEY, then run this again in a terminal: it "
            "offers to keep the key in your computer's password store."
        )
    elif not stored:
        ui.line()
        ui.line("Your Voyage key is only in the VOYAGE_API_KEY environment variable.")
        ui.note("Claude Code may start without that variable, and then no forecast can run.")
        if answer == "no_store":
            ui.note("This computer has no password store to keep it in. Set VOYAGE_API_KEY where Claude Code starts.")
        elif answer != "ok":
            ui.note("Your password store couldn't be opened. Set VOYAGE_API_KEY where Claude Code starts.")
        elif interactive() and not args.yes:
            if not ui.ask("Save the key in your computer's password store, where the hook can find it?", True):
                ui.note("Not saved. Set VOYAGE_API_KEY where Claude Code starts, or forecasts can't run.")
            elif embeddings.save_key(key):
                ui.ok("Saved in your computer's password store.")
            else:
                ui.fail("It couldn't be saved. Set VOYAGE_API_KEY where Claude Code starts.")
        else:
            ui.note(
                "With --yes it isn't saved anywhere. To keep it in your computer's password store, run this again "
                "in a terminal without --yes. Or set VOYAGE_API_KEY where Claude Code starts."
            )
    return 0


def cmd_rename(args) -> int:
    """Give a predictor another name. The default stays the default."""
    from .store import Busy, rename_predictor

    names = predictor_names()
    if args.old not in names:
        raise SystemExit(f"There's no model called {args.old!r}. Your models: {', '.join(names) or 'none'}.")
    was_default = default_name() == args.old
    try:
        rename_predictor(args.old, args.new)
    except (ValueError, Busy) as error:
        raise SystemExit(str(error)) from None
    except OSError as error:
        raise SystemExit(f"Couldn't rename {args.old}: {error.strerror or error}. Nothing changed.") from None
    say_renamed(args.old, args.new, was_default)
    return 0


def say_renamed(old: str, new: str, was_default: bool) -> None:
    ui.ok(f"Renamed {old} to {new}.")
    if was_default:
        ui.note(f"{new} is still the default.")


def cmd_default(args) -> int:
    names = predictor_names()
    if not args.name:
        current = default_name()
        if current:
            ui.out(f"Your default model is {ui.bold(current)}. Commands use it when they name none.")
        elif names:
            ui.out(f"You have no default model. Choose one of {', '.join(names)} with: yardstick-predict default NAME")
        else:
            ui.out("There's no model yet. Run yardstick-predict to create one.")
        return 0
    if args.name not in names:
        raise SystemExit(f"There's no model called {args.name!r}. Your models: {', '.join(names) or 'none'}.")
    set_default(args.name)
    ui.out(f"{ui.bold(args.name)} is now your default model. Commands use it when they name none.")
    return 0


class BadArguments(Exception):
    """A command line argparse can't read. `parser` is the command's parser, for its usage line."""

    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        super().__init__(message)
        self.parser = parser
        self.message = message


class Parser(argparse.ArgumentParser):
    """An argument parser that raises BadArguments instead of exiting, so predict can say what's wrong in JSON."""

    def error(self, message: str):
        raise BadArguments(self, message)


def bad_arguments(problem: BadArguments, argv: list[str]) -> int:
    """Say what's wrong with the command line: in JSON to a program that runs predict, or that asked for JSON
    from any command."""
    command = next((word for word in argv if not word.startswith("-")), None)
    forecasting = command == "predict" and not terminal_output()
    if forecasting or "--json" in argv:
        message = f"The command line isn't right: {problem.message}."
        answer = {"version": 1, "error": {"code": "invalid_request", "message": message}}
        print(json.dumps(answer, indent=None if forecasting else 2))
        return 1
    problem.parser.print_usage(sys.stderr)
    print(f"{problem.parser.prog}: error: {problem.message}", file=sys.stderr)
    return 2


MAX_COST_HELP = (
    "Spend at most this many US dollars at list price, such as 1.50. It goes ahead only when the cost fits, and stops "
    "before any step that could take it past the limit."
)


def make_parser() -> argparse.ArgumentParser:
    from . import features

    added = " and ".join(name for name in features.DEFAULT if name != features.REQUIRED) or "none"
    parser = Parser(
        prog="yardstick-predict",
        description="Forecast how long a coding-agent task will take and what it will cost, before it runs.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"yardstick-predict {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", title="commands")

    create = commands.add_parser("create", help="Create a model from your past runs.")
    create.add_argument("name", nargs="?", help="The model's name. In a terminal it asks. Otherwise: main.")
    create.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="SPEC",
        help="claude-code, codex, or FORMAT=PATH for a folder of records. Repeat to add more.",
    )
    create.add_argument("--embedding", help="voyage-4 (the default), voyage-4-lite or voyage-4-large.")
    create.add_argument("--max-tasks", type=int, help="Embed and learn from only the newest this many tasks.")
    create.add_argument(
        "--yes",
        action="store_true",
        help="Go ahead without asking anything, even when it costs money. Every choice takes its default.",
    )
    create.add_argument(
        "--express",
        action="store_true",
        help=f"Take every default and go ahead without questions: all your records, the prompt with {added}, voyage-4.",
    )
    create.add_argument(
        "--json",
        action="store_true",
        help="Answer in JSON, for programs. It asks nothing: without --yes it only says what it would do and cost.",
    )
    create.add_argument("--max-cost", type=float, metavar="USD", help=MAX_COST_HELP)
    create.add_argument(
        "--features",
        help="The optional features, such as conversation,context,skills:named,benchmarks,time,llm. A feature alone "
        f"gets all its options. Default: {added}.",
    )
    create.add_argument(
        "--llm-provider",
        choices=["anthropic", "openai", "claude", "codex"],
        help="How LLM analysis is paid for: an Anthropic or OpenAI API key, or a Claude or ChatGPT subscription "
        "through Claude Code or Codex. Default: anthropic.",
    )
    create.add_argument("--llm-model", help="The AI model LLM analysis uses. Default: claude-opus-5-5 with a key.")
    create.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="MODULE",
        help="A module that adds features or model types: its name, or the path of its .py file. "
        "A module in this folder is found by its name.",
    )
    advanced = create.add_argument_group("advanced settings", "For experts. Every one has a sensible default.")
    advanced.add_argument("--model-type", help="xgboost (the default), or one a plugin adds.")
    advanced.add_argument("--heads", help="What to forecast: any of seconds,tokens,usd. Default: all three.")
    advanced.add_argument(
        "--head", choices=["distribution", "expected"], help="A full curve, or only the expected value."
    )
    advanced.add_argument("--shape", choices=["lognormal", "weibull"], help="The curve's shape. Default: lognormal.")
    advanced.add_argument("--loss", choices=["log_score", "crps", "gamma", "tweedie"], help="What training minimizes.")
    advanced.add_argument("--dropout", type=float, help="Drop this share of trees each round (XGBoost's DART).")
    advanced.add_argument(
        "--test-share", type=float, help="The share of each AI model's newest tasks held back as the test."
    )
    advanced.add_argument("--test-windows", type=int, help="How many windows each AI model's test is split into.")
    advanced.add_argument("--validation-share", type=float, help="The share of training tasks that picks rounds.")
    advanced.add_argument("--half-life", type=float, help="Days until a task counts half. 0 counts every task alike.")

    refresh = commands.add_parser("refresh", help="Add new runs, embed them and retrain.")
    refresh.add_argument("--predictor", metavar="NAME", help="Which model. Default: your default model.")
    refresh.add_argument(
        "--yes", action="store_true", help="Go ahead without asking anything, even when it costs money."
    )
    refresh.add_argument(
        "--json",
        action="store_true",
        help="Answer in JSON, for programs. It asks nothing: without --yes it only says what it would do and cost.",
    )
    refresh.add_argument("--max-cost", type=float, metavar="USD", help=MAX_COST_HELP)

    predict = commands.add_parser("predict", help="Forecast a task. Give it in quotes, or send a JSON request.")
    predict.add_argument("task", nargs="?", help="The task's prompt, in quotes.")
    predict.add_argument("--prompt", help="The task's prompt, the same as giving it in quotes.")
    predict.add_argument("--predictor", metavar="NAME", help="Which model. Default: your default model.")
    predict.add_argument(
        "--model",
        action="append",
        default=[],
        help="An AI model to forecast. Repeat for more. Default: every AI model your model forecasts.",
    )
    predict.add_argument("--context", help="The conversation so far, if there is one.")
    predict.add_argument("--effort", help="The thinking level, such as high or max.")
    predict.add_argument(
        "--skill",
        action="append",
        default=[],
        help="A skill the prompt invokes, or that the conversation used before it. Repeat for more.",
    )
    predict.add_argument(
        "--request",
        metavar="FILE",
        help="Read the JSON request from this file, not from standard input. - means standard input.",
    )
    predict.add_argument("--distribution", action="store_true", help="Also give each forecast's full curve, in JSON.")
    predict.add_argument("--json", action="store_true", help="Print the JSON answer that programs get.")

    status = commands.add_parser("status", help="Show your models, their features and how well they forecast.")
    status.add_argument("--predictor", metavar="NAME", help="Show only this model.")
    status.add_argument("--json", action="store_true", help="Print JSON.")

    every = commands.add_parser("list", help="List your models, one line each.")
    every.add_argument("--json", action="store_true", help="Print JSON.")

    delete = commands.add_parser(
        "delete",
        help="Delete a model: its settings, dataset and what it learned. Your records and the feature store stay.",
    )
    delete.add_argument("name", help="The model to delete.")
    delete.add_argument("--yes", action="store_true", help="Delete it without asking.")
    delete.add_argument("--json", action="store_true", help="Print JSON. Add --yes too: it never asks.")

    check = commands.add_parser(
        "check", help="Check which of a model's features help its forecasts. Advice only: it changes nothing."
    )
    check.add_argument("name", nargs="?", help="The model to check. Default: your default model.")
    check.add_argument("--json", action="store_true", help="Print JSON.")
    listing = commands.add_parser(
        "features",
        help="List each model's features: what's stored, and how many of its tasks each covers. "
        "features delete ID deletes a stored feature.",
    )
    listing.add_argument("action", nargs="?", choices=["delete"], help="delete: delete the stored feature ID.")
    listing.add_argument("id", nargs="?", help="With delete: the stored feature's id, as the listing shows it.")
    listing.add_argument("--predictor", metavar="NAME", help="Show only this model.")
    listing.add_argument("--yes", action="store_true", help="With delete: delete it without asking.")
    listing.add_argument("--json", action="store_true", help="Print JSON.")

    default = commands.add_parser("default", help="Show or choose the model used when a command names none.")
    default.add_argument("name", nargs="?", help="The model to make the default.")

    rename = commands.add_parser("rename", help="Give a model another name. The default stays the default.")
    rename.add_argument("old", help="The model's name now.")
    rename.add_argument("new", help="Its new name.")

    claude = commands.add_parser("claude-code", help="Show forecasts in Claude Code's status line, or stop.")
    claude.add_argument("action", choices=["install", "remove"], help="Add the status line, or take it out.")
    claude.add_argument("--here", action="store_true", help="Only for Claude Code sessions in this folder.")
    claude.add_argument("--yes", action="store_true", help="Go ahead without asking.")
    return parser


TRAINS_OR_FORECASTS = ("create", "refresh", "predict", "check")
"""The commands that need XGBoost. The others work on a computer where it can't load, so you can still
look at your predictors, choose a default and take the status line out of Claude Code."""


def needs_xgboost(args) -> bool:
    # In a terminal the bare command opens the home screen, where you can estimate and update. Otherwise it lists.
    return args.command in TRAINS_OR_FORECASTS or (args.command is None and interactive())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    ui.numbered.cache_clear()  # whether numbered menus were asked for is looked up once a command
    try:
        args = make_parser().parse_args(argv)
    except BadArguments as problem:
        return bad_arguments(problem, argv)
    handlers = {
        "create": cmd_create,
        "refresh": cmd_refresh,
        "predict": cmd_predict,
        "status": cmd_status,
        "list": cmd_list,
        "delete": cmd_delete,
        "check": cmd_check,
        "features": cmd_features,
        "default": cmd_default,
        "rename": cmd_rename,
        "claude-code": cmd_claude_code,
    }
    if args.command in ("features", "status"):
        # Coverage, and what making a deleted feature again costs, need XGBoost. The rest shows without it.
        args.xgboost = xgboost_problem(argv)
    problem = xgboost_problem(argv) if needs_xgboost(args) else None
    if problem:
        forecasting = args.command == "predict"
        person = forecasting and terminal_output() and bool(args.task or args.prompt)
        answer = {"version": 1, "error": {"code": "missing_library", "message": problem}}
        if forecasting and not (person and not args.json and not args.distribution):
            print(json.dumps(answer))
        elif getattr(args, "json", False):  # any command that a program asked for JSON
            print(json.dumps(answer, indent=2))
        else:
            ui.error(problem)
        return 1
    try:
        return handlers.get(args.command, cmd_home)(args)
    except (KeyboardInterrupt, EOFError):  # Ctrl+C, or Ctrl+D at a question
        ui.stopped()
        return 130
    except BrokenPredictor as error:
        ui.error(str(error))
        return 1
    except SystemExit as stop:
        if isinstance(stop.code, str):
            ui.error(stop.code)
            return 1
        raise
    except BrokenPipeError:
        # The program reading the output stopped early, as head does. Nothing more can reach it, so nothing is said.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        return 141
    except Exception as error:
        return crashed(error, args)


DEBUG = "YARDSTICK_DEBUG"
"""Set to also print the details of a crash, the traceback, where it happens."""
ISSUES = "https://github.com/yusufcemaydin/yardstick-predict/issues"
CRASH_FILE = "last-crash.txt"
"""The file in the yardstick folder that keeps the details of the last crash."""


def crashed(error: Exception, args) -> int:
    """Something went wrong that the command didn't expect. Say so in a line, with where the details are kept and
    where to report it, rather than a page of Python. A program that asked for JSON gets it as an error in JSON."""
    import traceback

    details = "".join(traceback.format_exception(error))
    first = (str(error).strip().splitlines() or [""])[0]
    what = f"Something went wrong that yardstick-predict didn't expect: {type(error).__name__}: {first}".rstrip(": ")
    kept = keep_crash(details, args.command)
    if getattr(args, "json", False):
        print(json.dumps({"version": 1, "error": {"code": "internal_error", "message": ui.clean(what)}}, indent=2))
        return 1
    if os.environ.get(DEBUG) or kept is None:
        print(details, end="", file=sys.stderr)
    lines = [what]
    if kept is not None:
        lines.append(f"The details are in {kept}. They may hold text from your work. Read them before you share them.")
    lines.append(f"Please report it at {ISSUES}")
    ui.error("\n".join(lines))
    return 1


def keep_crash(details: str, command: str | None) -> Path | None:
    """Save the details of a crash in the yardstick folder, over the last one's, and return where. None when they
    can't be saved. They name the command, never what it was given, such as a prompt."""
    from . import files

    lines = [
        f"yardstick-predict {__version__}, Python {sys.version.split()[0]}, {sys.platform}",
        f"Command: {command or 'the home screen'}",
        time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "",
        details,
    ]
    try:
        path = private_home() / CRASH_FILE
        files.write_text(path, "\n".join(lines))
    except OSError:
        return None
    return path


if __name__ == "__main__":
    sys.exit(main())
