"""The yardstick-predict command.

    yardstick-predict              Set up, or check for new runs and refresh.
    yardstick-predict create       Create a predictor from your past runs.
    yardstick-predict refresh      Add new runs, embed them and retrain.
    yardstick-predict predict      Forecast a task: a table in a terminal, JSON for programs.
    yardstick-predict status       Show your predictors and how well they forecast.
    yardstick-predict default      Show or choose the predictor used when a command names none.

It only asks questions when a person runs it in a terminal. When a program runs it, it never asks.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from . import __version__, embeddings, ui
from .archives import DEFAULT_ARCHIVES, FORMATS, archive_path
from .store import NAME_PATTERN, Predictor, default_name, predictor_names, set_default

SIMPLE_RULE = "The simple rule gives every task on a model the same forecast: the spread of that model's past tasks."
REASONS = {
    "unknown_model": "no forecast: no past tasks on this model",
    "not_enough_data": "no forecast yet: too few tasks on this model",
    "not_accurate_enough": "no forecast: it didn't pass the test",
}


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
        lines.append(f"It costs {ui.bold(f'${cost.list_usd:.2f}')} at Voyage's list price.")
    elif cost.free_left >= cost.tokens:
        lines.append(
            f"Your free tokens cover it, so it costs {ui.bold('nothing')}. "
            f"{millions(cost.free_left - cost.tokens)} free tokens will be left."
        )
    else:
        lines.append(
            f"Your free tokens cover {millions(cost.free_left)} of them. "
            f"The rest costs {ui.bold(f'${cost.your_usd:.2f}')}."
        )
    if not cost.exact:
        lines.append("The token count is an estimate, because Voyage's tokenizer didn't download.")
    return lines


def explain(error: embeddings.VoyageError) -> str:
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
        return "Voyage kept limiting requests. Try again later, or add a payment method to raise your limits."
    if error.status == 0:
        return f"Couldn't reach Voyage. Check your internet connection. ({error.detail})"
    return str(error)


# The Voyage key ----------------------------------------------------------------------------------------


def get_key(model: str, can_ask: bool) -> str | None:
    """Find a working key. A person can paste one, and it is saved in the computer's password store."""
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
        ui.fail("No Voyage key found. Set VOYAGE_API_KEY, or run yardstick-predict in a terminal to paste one.")
        return None
    ui.working("Checking the key")
    try:
        embeddings.check_key(key, model)
    except embeddings.VoyageError as error:
        ui.fail(f"The key didn't work.\n{explain(error)}")
        return None
    ui.ok("The key works.")
    if pasted:
        if embeddings.save_key(key):
            ui.ok("Saved in your computer's password store, so you won't need to paste it again.")
        else:
            ui.note("This computer has no password store to save it in. Set VOYAGE_API_KEY to skip pasting next time.")
    if can_ask and not embeddings.account(key).get("asked"):
        ui.line()
        ui.note("Voyage gives every account 200 million free tokens. yardstick can't see your balance, so it keeps")
        ui.note("its own count.")
        ui.out(ui.prompt("Has this account used any of its free tokens before?").rstrip())
        ui.line(f"    {ui.accent('n')}   no, it's a new account")
        ui.line(f"    {ui.accent('y')}   yes, or I'm not sure - report costs assuming no credits")
        # Pressing Enter assumes no credits, so a cost is never shown lower than it may be.
        reply = input(f"    {ui.dim('[n/Y]')} ").strip().lower()
        embeddings.set_free_tokens(key, "new" if reply in ("n", "no") else "unknown")
    return key


# Questions ---------------------------------------------------------------------------------------------


def ask_name(existing: list[str]) -> str:
    """A name for a new predictor. The first one is called 'default' unless you pick another name."""
    suggestion = "" if "default" in existing else "default"
    while True:
        reply = input(ui.prompt("Name this predictor:", f"[{suggestion}]" if suggestion else "")).strip()
        name = reply or suggestion
        if not name:
            ui.note("Type a name, such as work.")
        elif not NAME_PATTERN.match(name):
            ui.note("Use 1 to 40 lowercase letters, digits, dashes or underscores.")
        elif name in existing:
            ui.note(f"There's already a predictor called {name}. Pick another name.")
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
            raise SystemExit(f"Unknown record format {fmt!r}. Use one of: {', '.join(FORMATS)}.")
        path = os.path.abspath(os.path.expanduser(path))
        return {"name": os.path.basename(path.rstrip("/\\")) or "custom", "format": fmt, "path": path}
    raise SystemExit(f"Can't read --archive {spec!r}. Use claude-code, codex, or FORMAT=PATH like codex=~/runs.")


def choose_archives(can_ask: bool, given: list[str]) -> list[dict]:
    if given:
        return [parse_archive(spec) for spec in given]
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


def choose_features(can_ask: bool, given: str | None) -> list[str]:
    """The prompt, plus the features a person picks or a program names."""
    from . import features

    try:
        if given is not None:
            return features.check([name.strip() for name in given.split(",") if name.strip()])
        if not can_ask:
            return features.check(list(features.DEFAULT))
        ui.heading("Features")
        ui.note("The prompt, the AI model and its effort level are always used. You can add:")
        chosen = ["prompt"]
        for name, description in features.available().items():
            if ui.ask(f"{ui.bold(name)}: {description}. Add it?", name in features.DEFAULT):
                chosen.append(name)
        return features.check(chosen)
    except ValueError as error:
        raise SystemExit(str(error)) from None


def advanced_settings(args) -> dict:
    """The model settings a program or an expert gave with flags. The rest keep their defaults."""
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
        raise SystemExit(str(error)) from None


# Reports -----------------------------------------------------------------------------------------------


def served_models(report: dict) -> list[str]:
    """The AI models whose forecasts passed the test, the most tested first."""
    tested = sorted(report.get("models", {}).items(), key=lambda kv: -kv[1]["tasks_tested"])
    return [name for name, result in tested if result["served"]]


def print_report(model) -> None:
    from .model import MIN_TEST_TASKS

    report = model.meta["report"]
    if not report.get("tested_on"):
        ui.note("There aren't enough tasks to test forecasts yet. It needs 100. Until then, it gives no forecasts.")
        return
    ui.line(f"Tested on the newest {report['tested_on']:,} tasks, as if they hadn't happened yet.")
    ui.line()
    names = {"seconds": "time", "tokens": "tokens", "usd": "cost"}
    results = sorted(report["models"].items(), key=lambda kv: -kv[1]["tasks_tested"])
    pad = max(len(name) for name, _ in results)
    flags = {"ok": ui.green(ui.mark("ok")), "fail": ui.red(ui.mark("fail")), "none": ui.dim("-")}
    lead = pad + 3 + (2 if ui.styled() else 0)

    def flag(kind: str) -> str:
        return f"{flags[kind]} " if ui.styled() else ""

    for name, result in results:
        tested = result["tasks_tested"]
        label = ui.bold(name) + " " * (pad - len(name))
        if result["served"]:
            ui.line(f"{label}   {flag('ok')}" + ui.join(["forecasts on", f"tested on {tested:,} tasks"]))
            scores = [(names[t], result[t]) for t in names if t in result]
            if scores:
                better = [f"{n} {s['better_than_simple_rule']:.0%}" for n, s in scores]
                rows = [(ui.dim("better than the simple rule"), *better)]
                if all(s.get("in_90_range") is not None for _, s in scores):
                    rows.append((ui.dim("inside the 90% range"), *(f"{n} {s['in_90_range']:.0%}" for n, s in scores)))
                ui.table(rows, indent=2 + lead)
        elif result["reason"] == "not_enough_data":
            ui.line(f"{label}   {flag('none')}" + ui.join(["not yet", f"{tested} test tasks, needs {MIN_TEST_TASKS}"]))
        else:
            ui.line(f"{label}   {flag('fail')}" + ui.join(["no forecasts", f"tested on {tested:,} tasks"]))
            for failed in result.get("failed_checks", []):
                ui.line(" " * lead + ui.dim(failed_check(failed, result, names)))
    ui.line()
    ui.note(SIMPLE_RULE)


def failed_check(failed: str, result: dict, names: dict) -> str:
    """One failed check, in plain words."""
    from .model import MIN_RANGE_HELD, TOLERANCE, TOTALS_RANGE

    target, check = failed.split(".")
    scores = result[target]
    label = names[target].capitalize()
    if check == "simple_rule":
        worse = -scores["better_than_simple_rule"]
        return (
            f"{label} forecasts were {worse:.0%} worse than the simple rule. They may be at most {TOLERANCE:.0%} worse."
        )
    if check == "range_held":
        return (
            f"{label}: only {scores['in_90_range']:.0%} of real results fell inside the 90% range. "
            f"It needs {MIN_RANGE_HELD:.0%}."
        )
    ratio = scores["expected_over_real_total"]
    return (
        f"{label}: the expected values added up to {ratio:.2f} times the real total. "
        f"It must be between {TOTALS_RANGE[0]} and {TOTALS_RANGE[1]} times."
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
    os.execve(sys.executable, [sys.executable, "-m", "yardstick_predict", *argv], env)


def xgboost_problem(argv: list[str]) -> str | None:
    """Why XGBoost can't load, in plain words, or None when it loads."""
    try:
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
            raise SystemExit(f"There's no predictor called {name!r}. Your predictors: {', '.join(names) or 'none'}.")
        return Predictor(name)
    if not names:
        raise SystemExit("There's no predictor yet. Run yardstick-predict to create one.")
    chosen = default_name()
    if chosen is None:
        raise SystemExit(
            f"You have several predictors and no default: {', '.join(names)}. Name one with --predictor, "
            "or choose a default with: yardstick-predict default NAME"
        )
    return Predictor(chosen)


def cmd_home(args) -> int:
    names = predictor_names()
    if not names:
        ui.out(ui.bold("yardstick-predict"))
        ui.out("Forecast how long a coding-agent task will take and what it will cost.")
        ui.out(ui.dim("It learns from your own past Claude Code and Codex runs. Nothing leaves your computer except"))
        ui.out(ui.dim("the text you choose to embed, such as your prompts, and that goes only to Voyage."))
        ui.out()
        if not interactive():
            ui.out("No predictor yet. Set VOYAGE_API_KEY, then create one with: yardstick-predict create --express")
            return 1
        if ui.ask("There's no predictor yet. Create one now?", True):
            return cmd_create(make_parser().parse_args(["create"]))
        return 0
    if not interactive():
        for name in names:
            predictor = Predictor(name)
            facts = [
                f"{predictor.embedding_model} embeddings",
                f"{predictor.count_tasks():,} tasks",
                f"updated {ago(predictor.config.get('refreshed_at'))}",
            ]
            ui.out(f"{name}: {', '.join(facts)}")
        return 0
    for name in names:
        cmd_refresh(argparse.Namespace(predictor=name, yes=False, ask_first=True))
    ui.out()
    ui.out(
        ui.join([ui.dim("New predictor: yardstick-predict create"), ui.dim("Default: yardstick-predict default NAME")])
    )
    return 0


def cmd_create(args) -> int:
    from . import model_types

    if args.express:
        args.yes = True
    can_ask = interactive() and not args.express
    existing = predictor_names()
    if can_ask:
        ui.out()
        ui.out(ui.bold("New predictor"))
    name = args.name or (ask_name(existing) if can_ask else None)
    if name is None:
        if "default" in existing:
            raise SystemExit("Give the new predictor a name, like: yardstick-predict create work")
        name = "default"
    if not NAME_PATTERN.match(name):
        raise SystemExit("Use 1 to 40 lowercase letters, digits, dashes or underscores for the name.")
    if name in existing:
        raise SystemExit(f"A predictor called {name!r} already exists. Refresh it with: yardstick-predict refresh")
    make_default = not existing or (
        can_ask and ui.ask("Make it your default predictor? Commands use it when they name none.", False)
    )
    try:
        model_types.load_plugins(args.plugin)
    except ImportError as error:
        raise SystemExit(f"Couldn't load the plugin: {error}") from None
    model_settings = advanced_settings(args)

    ui.heading("Your past runs")
    archives = choose_archives(can_ask, args.archive)
    if not archives:
        raise SystemExit("No records to learn from. Add a folder with --archive FORMAT=PATH.")
    ui.working("Reading your records. A big history takes a minute")
    from .pipeline import embedding_items, newest_that_fit, price, read_all, refresh_benchmarks, save_scan, train

    found = read_all(archives)
    counts = [(archive["name"], found.per_archive.get(archive["name"], 0)) for archive in archives]
    total = sum(count for _, count in counts)
    if len(counts) == 1:
        ui.ok(f"Read {total:,} tasks from {counts[0][0]}.")
    else:
        ui.ok(f"Read {total:,} tasks: " + ui.join([f"{source} {count:,}" for source, count in counts]))
    if not found.new_tasks:
        raise SystemExit("Found no finished tasks in those records.")
    chosen = choose_features(can_ask, args.features)
    items = embedding_items(found.new_tasks, chosen, limit=args.max_tasks)

    model_name = args.embedding or (None if can_ask else "voyage-4")
    if model_name is None:
        ui.heading("Embedding model")
        ui.working("Working out what each Voyage model would cost")
        plans = {m: price(items, m, None) for m in embeddings.MODELS}
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
        ui.note("The predictor keeps this model for good. To change it later, create a new predictor.")
        while model_name is None:
            reply = input(ui.prompt("Which Voyage model should embed your tasks?", "[1]")).strip() or "1"
            if reply.isdigit() and 1 <= int(reply) <= len(options):
                model_name = options[int(reply) - 1]
            else:
                ui.note(f"Type a number from 1 to {len(options)}.")
    if model_name not in embeddings.MODELS:
        raise SystemExit(f"Unknown embedding model {model_name!r}. Use voyage-4, voyage-4-lite or voyage-4-large.")

    key = get_key(model_name, can_ask)
    if not key:
        return 1
    plan = price(items, model_name, key)
    ui.heading("Cost")
    for text in cost_lines(plan.cost, model_name, plan.tasks):
        ui.line(text)
    limit = args.max_tasks
    free = plan.cost.free_left
    if can_ask and free is not None and plan.cost.tokens > free and free > 0:
        fit = newest_that_fit(plan, free)
        if fit and ui.ask(f"Only the newest {fit:,} tasks fit in your free tokens. Use just those?", True):
            limit = fit
    if limit and limit != args.max_tasks:
        items = embedding_items(found.new_tasks, chosen, limit=limit)
        plan = price(items, model_name, key)
        for text in cost_lines(plan.cost, model_name, plan.tasks):
            ui.line(text)
    if can_ask and not ui.ask("Embed them and train the predictor now?", True):
        ui.note("Stopped. Nothing was created.")
        return 0
    if not can_ask and not args.yes:
        ui.line("Add --yes to go ahead.")
        return 2
    args.max_tasks = limit

    predictor = Predictor.create(name, archives, {"model": model_name, "dimensions": embeddings.DIMENSIONS})
    predictor.config.update(features=chosen, model=model_settings, plugins=list(args.plugin))
    if args.max_tasks:
        predictor.config["max_tasks"] = int(args.max_tasks)
    predictor.save_config()
    save_scan(predictor, found)
    warning = refresh_benchmarks(predictor)
    if warning:
        ui.note(warning)
    if plan.items and not embed_with_progress(predictor, plan, key):
        return 1
    model = train_and_report(predictor, train)
    if model is None:
        return 1
    if make_default:
        set_default(name)
    say_how_to_use(model, name)
    return 0


def embed_with_progress(predictor: Predictor, plan, key: str) -> bool:
    from .pipeline import run_embeddings

    ui.heading("Embedding")
    ui.note("Each batch is saved as it arrives, so you can stop and resume later.")
    seen = {"total": 0}

    def progress(done: int, total: int) -> None:
        seen["total"] = total
        ui.progress(done, total, "texts")

    try:
        run_embeddings(predictor, plan, key, progress, on_wait=ui.note)
    except embeddings.VoyageError as error:
        ui.fail(explain(error))
        if "free token" in error.detail.lower() or "payment" in error.detail.lower():
            embeddings.set_free_tokens(key, "used")
        ui.note("What was embedded so far is saved. Run yardstick-predict refresh to finish.")
        return False
    ui.ok(f"Embedded {seen['total']:,} texts." if seen["total"] else "Embedded.")
    return True


def train_and_report(predictor: Predictor, train):
    """Train, test and show how each AI model did. None when there's too little to train on."""
    from .model import NotEnoughData

    ui.heading("Training")
    ui.working("Training and testing. This takes a minute")
    try:
        model = train(predictor)
    except NotEnoughData as error:
        ui.fail(str(error))
        return None
    ui.ok(f"Trained on {model.meta['tasks']:,} tasks.")
    print_report(model)
    return model


def say_how_to_use(model, name: str, full: bool = True) -> None:
    """Which AI models get forecasts, and how to ask for one, with their real names."""
    served = served_models(model.meta["report"])
    listed = (served[0] if len(served) == 1 else ", ".join(served[:-1]) + " and " + served[-1]) if served else ""
    if not full:
        if served:
            ui.ok(f"Forecasts are on for {listed}.")
        return
    ui.heading("Ready")
    if not served:
        ui.line("No AI model passed the test yet, so forecasts come back empty, with the reason.")
        ui.note("Run yardstick-predict again once you have more tasks. It adds them and retrains.")
        return
    is_default = default_name() == name
    pick = "" if is_default else f" --predictor {name}"
    ui.line(f"Forecasts are on for {listed}.")
    ui.line("Before you start a task, ask what it will take:")
    ui.line(ui.accent(f'  yardstick-predict predict{pick} "Add a login page with tests"'))
    if is_default:
        ui.note("It's your default predictor, so commands use it when they name none.")
    ui.note("Programs send JSON to yardstick-predict predict and read JSON back.")
    ui.note("Run yardstick-predict every week or so to add new tasks and retrain.")


def cmd_refresh(args) -> int:
    from .pipeline import chosen_features, load_plugins, plan_embeddings, refresh_benchmarks, save_scan, scan, train

    predictor = pick_predictor(args.predictor)
    load_plugins(predictor)
    can_ask = interactive()
    ask_first = getattr(args, "ask_first", False)
    ui.heading(predictor.name)
    ui.note(f"Last updated {ago(predictor.config.get('refreshed_at'))}.")
    ui.working("Looking for new tasks")
    found = scan(predictor)
    counts = [f"{source} {count:,}" for source, count in found.per_archive.items() if count]
    ui.ok(f"Found {len(found.new_tasks):,} new tasks" + (f": {ui.join(counts)}" if counts else "."))
    if found.updated_tasks:
        grew = len(found.updated_tasks)
        ui.note(f"{grew:,} saved tasks grew, because their helper agents finished after the last refresh.")
    model_name = predictor.embedding_model
    key, _ = embeddings.find_key()
    from . import features
    from .pipeline import embedding_items, price

    chosen = chosen_features(predictor)
    have = {name: predictor.embedded(name) for name in chosen if features.is_text(name)}
    everything = predictor.tasks() + found.new_tasks
    pending = embedding_items(everything, chosen, have, predictor.config.get("max_tasks"))
    plan = price(pending, model_name, key)
    nothing_new = not found.new_tasks and not found.updated_tasks and not plan.items
    if nothing_new and (predictor.model_dir / "model.json").is_file():
        ui.ok("Nothing new. The model is up to date.")
        return 0
    if plan.items:
        for text in cost_lines(plan.cost, model_name, plan.tasks):
            ui.line(text)
    if can_ask:
        if not ui.ask("Refresh now? This adds the new tasks, embeds them and retrains.", not ask_first):
            ui.note("Keeping the current model.")
            return 0
    elif not args.yes:
        ui.line("Add --yes to go ahead.")
        return 2
    if plan.items:
        key = key or get_key(model_name, can_ask)
        if not key:
            return 1
    save_scan(predictor, found)
    warning = refresh_benchmarks(predictor)
    if warning:
        ui.note(warning)
    if plan.items:
        plan = plan_embeddings(predictor, key)
        if not embed_with_progress(predictor, plan, key):
            return 1
    model = train_and_report(predictor, train)
    if model is None:
        return 1
    say_how_to_use(model, predictor.name, full=False)
    return 0


def cmd_predict(args) -> int:
    from .forecast import RequestError, answer, parse

    prompt = args.task if args.task is not None else args.prompt
    table = prompt is not None and terminal_output() and not args.json and not args.distribution
    try:
        if prompt is not None:
            predictor = request_predictor(args.predictor)
            pairs = effort_pairs(predictor, args.model or forecast_models(predictor), args.effort)
            asked = []
            for number, (model, effort) in enumerate(pairs, start=1):
                task = {"id": "task" if len(pairs) == 1 else f"task-{number}", "prompt": prompt, "models": [model]}
                task.update(context=args.context or "", skills=args.skill)
                if effort:
                    task["effort"] = effort
                asked.append(task)
            tasks, distribution, _ = parse({"version": 1, "distribution": args.distribution, "tasks": asked})
        else:
            if sys.stdin.isatty():
                raise RequestError(
                    "invalid_request", 'Give the task in quotes, like: yardstick-predict predict "Fix it"'
                )
            try:
                request = json.load(sys.stdin)
            except ValueError as error:
                raise RequestError("invalid_request", f"The request isn't valid JSON: {error}") from None
            tasks, distribution, chosen = parse(request)
            predictor = request_predictor(args.predictor or chosen)
        result = answer(predictor, tasks, distribution or args.distribution)
        code = 0
    except RequestError as error:
        result, code = error.as_answer(), 1
    if table:
        if code:
            ui.error(result["error"]["message"])
        else:
            show_forecasts(prompt, result)
        return code
    print(json.dumps(result, indent=2 if terminal_output() else None))
    return code


def request_predictor(chosen: str | None) -> Predictor:
    """The predictor a forecast request names, or your default one, with errors in the answer format."""
    from .forecast import RequestError

    names = predictor_names()
    if chosen and chosen not in names:
        raise RequestError("unknown_predictor", f"There's no predictor called {chosen!r}.")
    if not names:
        raise RequestError("no_predictor", "There's no predictor yet. Run yardstick-predict to create one.")
    try:
        return pick_predictor(chosen)
    except SystemExit as error:
        raise RequestError("no_predictor", str(error)) from None


def effort_pairs(predictor: Predictor, models: list[str], effort: str | None) -> list[tuple[str, str | None]]:
    """Each AI model with each effort level to forecast: the one asked for, or every level it ran at often.

    A level counts as often when the model ran 30 tasks at it. A model with no such level is forecast at
    the level it ran at most often.
    """
    from .model import MIN_TEST_TASKS

    if effort:
        return [(model, effort) for model in models]
    counts = predictor.effort_counts()
    pairs: list[tuple[str, str | None]] = []
    for model in models:
        common = [level for level, count in counts.get(model, []) if count >= MIN_TEST_TASKS]
        pairs += [(model, level) for level in common] or [(model, None)]
    return pairs


def forecast_models(predictor: Predictor) -> list[str]:
    """The AI models a predictor forecasts, for a request that names none."""
    from .forecast import RequestError

    model_file = predictor.model_dir / "model.json"
    report = json.loads(model_file.read_text()).get("report", {}) if model_file.is_file() else {}
    served = served_models(report)
    if not served:
        raise RequestError("invalid_request", "No AI model has forecasts yet. Name one with --model.")
    return served


def show_forecasts(prompt: str, result: dict) -> None:
    """The forecasts as a small table, for a person in a terminal."""
    pad = max(len(forecast["model"]) for forecast in result["forecasts"])
    rows, missing, previous = [], {}, None
    for forecast in result["forecasts"]:
        model = forecast["model"]
        seconds, tokens, usd = forecast["seconds"], forecast["tokens"], forecast["usd"]
        if seconds is None and tokens is None and usd is None:
            reason = REASONS.get(forecast.get("reason", ""), "no forecast")
            missing.setdefault(model, f"{ui.bold(model.ljust(pad))}   {ui.dim(reason)}")
            continue
        # A model forecast at several effort levels is named once, on its first row.
        name = ui.bold(model.ljust(pad)) if model != previous else " " * pad
        previous = model
        rows.append(
            (
                name,
                forecast.get("effort", ""),
                duration(seconds) if seconds is not None else ui.dim("-"),
                f"{short_tokens(tokens)} tokens" if tokens is not None else ui.dim("-"),
                money(usd) if usd is not None else ui.dim("-"),
            )
        )
    shown = prompt if len(prompt) <= 72 else prompt[:71].rstrip() + ui.mark("wait")
    ui.out()
    ui.line(ui.bold(shown))
    ui.out()
    ui.table(rows, align="llrrr")
    for text in missing.values():
        ui.line(text)
    predictor = Predictor(result["predictor"])
    footer = [
        "Expected values from your own runs",
        f"predictor {predictor.name}, updated {ago(predictor.config.get('refreshed_at'))}",
        "add --json for details",
    ]
    ui.out()
    ui.line(ui.join([ui.dim(part) for part in footer]))


def cmd_status(args) -> int:
    names = [args.predictor] if args.predictor else predictor_names()
    default = default_name()
    found = []
    for name in names:
        if name not in predictor_names():
            raise SystemExit(f"There's no predictor called {name!r}.")
        predictor = Predictor(name)
        entry = {
            "name": name,
            "default": name == default,
            "embedding_model": predictor.embedding_model,
            "features": predictor.config.get("features") or ["prompt"],
            "model": predictor.config.get("model") or {},
            "archives": predictor.config["archives"],
            "tasks": predictor.count_tasks(),
            "refreshed_at": predictor.config.get("refreshed_at"),
            "report": None,
        }
        model_file = predictor.model_dir / "model.json"
        if model_file.is_file():
            entry["report"] = json.loads(model_file.read_text()).get("report")
        found.append(entry)
    if args.json:
        print(json.dumps({"version": 1, "predictors": found}, indent=2))
        return 0
    if not found:
        ui.out("There's no predictor yet. Run yardstick-predict to create one.")
        return 0
    from .pipeline import load_model

    for entry in found:
        ui.out()
        title = [ui.bold(entry["name"])] + ([ui.dim("default predictor")] if entry["default"] else [])
        ui.out(ui.join(title))
        facts = [
            f"{entry['embedding_model']} embeddings",
            f"{entry['tasks']:,} tasks",
            f"updated {ago(entry['refreshed_at'])}",
        ]
        ui.line(ui.join(facts))
        ui.line(f"Features: {', '.join(entry['features'])}")
        predictor = Predictor(entry["name"])
        if (predictor.model_dir / "model.json").is_file():
            model = load_model(predictor)
            ui.line(f"Trained on {model.meta['tasks']:,} tasks.")
            print_report(model)
    return 0


def cmd_default(args) -> int:
    names = predictor_names()
    if not args.name:
        current = default_name()
        if current:
            ui.out(f"Your default predictor is {ui.bold(current)}. Commands use it when they name none.")
        elif names:
            ui.out(
                f"You have no default predictor. Choose one of {', '.join(names)} with: yardstick-predict default NAME"
            )
        else:
            ui.out("There's no predictor yet. Run yardstick-predict to create one.")
        return 0
    if args.name not in names:
        raise SystemExit(f"There's no predictor called {args.name!r}. Your predictors: {', '.join(names) or 'none'}.")
    set_default(args.name)
    ui.out(f"{ui.bold(args.name)} is now your default predictor. Commands use it when they name none.")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yardstick-predict",
        description="Forecast how long a coding-agent task will take and what it will cost, before it runs.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"yardstick-predict {__version__}")
    commands = parser.add_subparsers(dest="command")

    create = commands.add_parser("create", help="Create a predictor from your past runs.")
    create.add_argument("name", nargs="?", help="The predictor's name. In a terminal it asks. Otherwise: default.")
    create.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="SPEC",
        help="claude-code, codex, or FORMAT=PATH for a folder of records. Repeat to add more.",
    )
    create.add_argument("--embedding", help="voyage-4 (the default), voyage-4-lite or voyage-4-large.")
    create.add_argument("--max-tasks", type=int, help="Embed and learn from only the newest this many tasks.")
    create.add_argument("--yes", action="store_true", help="Go ahead without asking, even when it costs money.")
    create.add_argument(
        "--express",
        action="store_true",
        help="Take every default and go ahead without questions: all your records, the prompt, voyage-4.",
    )
    create.add_argument(
        "--features", help="The features to add to the prompt, such as context,skills,benchmarks. Default: none."
    )
    create.add_argument(
        "--plugin", action="append", default=[], metavar="MODULE", help="A module that adds features or model types."
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
    advanced.add_argument("--test-share", type=float, help="The share of newest tasks held back as the test.")
    advanced.add_argument("--test-windows", type=int, help="How many windows the test is split into.")
    advanced.add_argument("--validation-share", type=float, help="The share of training tasks that picks rounds.")
    advanced.add_argument("--half-life", type=float, help="Days until a task counts half. 0 counts every task alike.")

    refresh = commands.add_parser("refresh", help="Add new runs, embed them and retrain.")
    refresh.add_argument("--predictor", help="Which predictor. Default: your default predictor.")
    refresh.add_argument("--yes", action="store_true", help="Go ahead without asking, even when it costs money.")

    predict = commands.add_parser("predict", help="Forecast a task. Give it in quotes, or send a JSON request.")
    predict.add_argument("task", nargs="?", help="The task's prompt, in quotes.")
    predict.add_argument("--prompt", help="The task's prompt, the same as giving it in quotes.")
    predict.add_argument("--predictor", help="Which predictor. Default: your default predictor.")
    predict.add_argument(
        "--model",
        action="append",
        default=[],
        help="An AI model to forecast. Repeat for more. Default: every model the predictor forecasts.",
    )
    predict.add_argument("--context", help="The conversation so far, if there is one.")
    predict.add_argument("--effort", help="The thinking level, such as high or max.")
    predict.add_argument("--skill", action="append", default=[], help="A skill the agent will use. Repeat for more.")
    predict.add_argument("--distribution", action="store_true", help="Also give each forecast's full curve, in JSON.")
    predict.add_argument("--json", action="store_true", help="Print the JSON answer that programs get.")

    status = commands.add_parser("status", help="Show your predictors and how well they forecast.")
    status.add_argument("--predictor", help="Show only this predictor.")
    status.add_argument("--json", action="store_true", help="Print JSON.")

    default = commands.add_parser("default", help="Show or choose the predictor used when a command names none.")
    default.add_argument("name", nargs="?", help="The predictor to make the default.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    handlers = {
        "create": cmd_create,
        "refresh": cmd_refresh,
        "predict": cmd_predict,
        "status": cmd_status,
        "default": cmd_default,
    }
    problem = xgboost_problem(sys.argv[1:] if argv is None else argv)
    if problem:
        if args.command == "predict" and not (terminal_output() and (args.task or args.prompt) and not args.json):
            print(json.dumps({"version": 1, "error": {"code": "missing_library", "message": problem}}))
        else:
            ui.error(problem)
        return 1
    try:
        return handlers.get(args.command, cmd_home)(args)
    except KeyboardInterrupt:
        ui.out()
        ui.note("Stopped.")
        return 130
    except SystemExit as stop:
        if isinstance(stop.code, str):
            ui.error(stop.code)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
