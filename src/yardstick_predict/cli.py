"""The yardstick-predict command.

    yardstick-predict              Set up, or check for new runs and refresh.
    yardstick-predict create       Create a predictor from your past runs.
    yardstick-predict refresh      Add new runs, embed them and retrain.
    yardstick-predict predict      Forecast tasks. Reads and writes JSON, and never asks questions.
    yardstick-predict status       Show your predictors and how well they forecast.

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

from . import __version__, embeddings
from .archives import DEFAULT_ARCHIVES, FORMATS, archive_path
from .store import NAME_PATTERN, Predictor, predictor_names

SIMPLE_RULE = "The simple rule gives every task on a model the same forecast: the spread of that model's past tasks."


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def say(text: str = "") -> None:
    print(text, flush=True)


def ask(question: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        reply = input(f"{question} {hint} ").strip().lower()
        if not reply:
            return default
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        say("Please answer y or n.")


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


# Talking about cost ------------------------------------------------------------------------------------


def cost_lines(cost: embeddings.Cost, model: str) -> list[str]:
    counted = "counted with Voyage's own tokenizer" if cost.exact else "estimated, because the tokenizer didn't load"
    lines = [
        f"That is {cost.tokens:,} tokens, {counted}.",
        f"At Voyage's list price for {model}, it costs ${cost.list_usd:.2f}.",
    ]
    if cost.free_left is None:
        lines.append(
            "Voyage gives every account 200 million free tokens. If yours has some left, you pay less or nothing. "
            "Your Voyage dashboard shows what's left."
        )
    elif cost.free_left >= cost.tokens:
        lines.append(
            f"By yardstick's count, your account has {millions(cost.free_left)} free tokens left. You pay $0.00."
        )
    else:
        lines.append(
            f"By yardstick's count, your account has {millions(cost.free_left)} free tokens left. "
            f"You pay ${cost.your_usd:.2f}."
        )
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
    if key:
        say(f"Using the Voyage key from {source}.")
    elif can_ask:
        say("Paste your Voyage API key. It stays on this computer and is only ever sent to Voyage.")
        say("You can make one at https://dashboard.voyageai.com.")
        key = getpass.getpass("Voyage API key (hidden as you type): ").strip()
        pasted = True
        if not key:
            return None
    else:
        say("No Voyage key found. Set the VOYAGE_API_KEY environment variable, or run yardstick-predict in a terminal.")
        return None
    print("Checking the key... ", end="", flush=True)
    try:
        embeddings.check_key(key, model)
    except embeddings.VoyageError as error:
        say("it didn't work.")
        say(explain(error))
        return None
    say("it works.")
    if pasted:
        if embeddings.save_key(key):
            say("Saved it in your computer's password store, so you won't need to paste it again.")
        else:
            say("This computer has no password store to save it in. Set VOYAGE_API_KEY to skip pasting next time.")
    if can_ask and not embeddings.account(key).get("asked"):
        say()
        say("Voyage gives every account 200 million free tokens. yardstick can't see your balance, so it keeps its")
        say("own count. Has this Voyage account used any of its free tokens before?")
        say("  n   no, it's a new account")
        say("  y   yes, or I'm not sure")
        reply = input("[n/y] ").strip().lower()
        embeddings.set_free_tokens(key, "new" if reply in ("", "n", "no") else "unknown")
    return key


# Archives ----------------------------------------------------------------------------------------------


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
    say("Where your past runs are:")
    for archive in found:
        if ask(f"  Use your {archive['name']} records in {archive['path']}?", True):
            chosen.append(archive)
    for archive in DEFAULT_ARCHIVES:
        if not archive_path(archive).is_dir():
            say(f"  No {archive['name']} records in {archive['path']}.")
    while True:
        extra = input(
            "  Add a folder of records in Claude Code or Codex format? Paste its path, or press Enter: "
        ).strip()
        if not extra:
            break
        fmt = input("  Is it claude-code or codex format? [claude-code] ").strip() or "claude-code"
        if fmt not in FORMATS:
            say(f"  Use one of: {', '.join(FORMATS)}.")
            continue
        archive = parse_archive(f"{fmt}={extra}")
        if not archive_path(archive).is_dir():
            say(f"  There's no folder at {archive['path']}.")
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
        say("Features. The prompt is always used. You can add:")
        chosen = ["prompt"]
        for name, description in features.available().items():
            if ask(f"  {name}: {description}. Add it?", name in features.DEFAULT):
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


def print_report(model) -> None:
    from .model import MIN_TEST_TASKS

    report = model.meta["report"]
    say(f"Trained on {model.meta['tasks']:,} tasks.")
    if not report.get("tested_on"):
        say("There aren't enough tasks to test forecasts yet. It needs 100. Until then, it gives no forecasts.")
        return
    say(f"Tested on the newest {report['tested_on']:,} tasks, as if they hadn't happened yet. {SIMPLE_RULE}")
    names = {"seconds": "time", "tokens": "tokens", "usd": "cost"}
    for name, result in sorted(report["models"].items(), key=lambda kv: -kv[1]["tasks_tested"]):
        tested = result["tasks_tested"]
        if result["served"]:
            say(f"  {name}: forecasts on. Tested on {tested:,} tasks.")
        elif result["reason"] == "not_enough_data":
            say(f"  {name}: no forecasts yet. It had {tested} test tasks and needs {MIN_TEST_TASKS}.")
        else:
            say(f"  {name}: no forecasts. Tested on {tested:,} tasks, they didn't pass:")
            for failed in result.get("failed_checks", []):
                say(f"    {failed_check(failed, result, names)}")
        scores = [(names[t], result[t]) for t in names if t in result]
        if result["served"] and scores:
            better = ", ".join(f"{label} {s['better_than_simple_rule']:.0%}" for label, s in scores)
            say(f"    Better than the simple rule: {better}.")
            ranges = [(label, s["in_90_range"]) for label, s in scores if s.get("in_90_range") is not None]
            if ranges:
                say("    Real results inside the 90% range: " + ", ".join(f"{la} {v:.0%}" for la, v in ranges) + ".")


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
    if "default" in names:
        return Predictor("default")
    if len(names) == 1:
        return Predictor(names[0])
    raise SystemExit(f"You have several predictors. Pick one with --predictor: {', '.join(names)}.")


def cmd_home(args) -> int:
    names = predictor_names()
    if not names:
        say("yardstick-predict forecasts how long a coding-agent task will take and what it will cost.")
        say("It learns from your own past Claude Code and Codex runs. Nothing leaves your computer except")
        say("your prompts, which are sent to Voyage to embed.")
        say()
        if not interactive():
            say("No predictor yet. Set VOYAGE_API_KEY, then create one with: yardstick-predict create --yes")
            return 1
        if ask("There's no predictor yet. Create one now?", True):
            return cmd_create(make_parser().parse_args(["create"]))
        return 0
    if not interactive():
        for name in names:
            predictor = Predictor(name)
            say(
                f"Predictor {name!r}: {predictor.embedding_model} embeddings, {predictor.count_tasks():,} tasks, "
                f"last refreshed {ago(predictor.config.get('refreshed_at'))}."
            )
        return 0
    for name in names:
        cmd_refresh(argparse.Namespace(predictor=name, yes=False, ask_first=True))
        say()
    say('Forecast with: yardstick-predict predict --model claude-opus-5-5 --prompt "Add a login page"')
    return 0


def cmd_create(args) -> int:
    from . import model_types

    if args.express:
        args.yes = True
    can_ask = interactive() and not args.express
    name = args.name or "default"
    if not NAME_PATTERN.match(name):
        raise SystemExit("Use 1 to 40 lowercase letters, digits, dashes or underscores for the name.")
    if name in predictor_names():
        raise SystemExit(f"A predictor called {name!r} already exists. Refresh it with: yardstick-predict refresh")
    try:
        model_types.load_plugins(args.plugin)
    except ImportError as error:
        raise SystemExit(f"Couldn't load the plugin: {error}") from None
    model_settings = advanced_settings(args)
    if can_ask:
        say(f"Creating predictor {name!r}.")
        say()
    archives = choose_archives(can_ask, args.archive)
    if not archives:
        raise SystemExit("No records to learn from. Add a folder with --archive FORMAT=PATH.")
    say("Reading your records. A big history takes a minute...")
    from .pipeline import embedding_items, newest_that_fit, price, read_all, refresh_benchmarks, save_scan, train

    found = read_all(archives)
    for archive in archives:
        say(f"  {archive['name']}: {found.per_archive.get(archive['name'], 0):,} tasks")
    if not found.new_tasks:
        raise SystemExit("Found no finished tasks in those records.")
    if can_ask:
        say()
    chosen = choose_features(can_ask, args.features)
    items = embedding_items(found.new_tasks, chosen, limit=args.max_tasks)

    model_name = args.embedding or (None if can_ask else "voyage-4")
    if model_name is None:
        say()
        say("Each embedded feature, like the prompt, is turned into numbers that capture what it means. The")
        say("forecasts then use what the most similar past tasks really took.")
        say("Counting tokens with each Voyage model's tokenizer...")
        plans = {m: price(items, m, None) for m in embeddings.MODELS}
        options = list(embeddings.MODELS)
        for number, option in enumerate(options, start=1):
            plan = plans[option]
            note = embeddings.MODELS[option].note
            say(f"  {number}  {option:<15} ${plan.cost.list_usd:<7.2f} {plan.cost.tokens:>12,} tokens  {note}")
        say("The predictor keeps this embedding model for good. To change it, create a new predictor.")
        while model_name is None:
            reply = input("Choose [1]: ").strip() or "1"
            if reply.isdigit() and 1 <= int(reply) <= len(options):
                model_name = options[int(reply) - 1]
            else:
                say(f"Type a number from 1 to {len(options)}.")
    if model_name not in embeddings.MODELS:
        raise SystemExit(f"Unknown embedding model {model_name!r}. Use voyage-4, voyage-4-lite or voyage-4-large.")

    say()
    key = get_key(model_name, can_ask)
    if not key:
        return 1
    plan = price(items, model_name, key)
    say()
    say(f"Embedding {plan.tasks:,} tasks with {model_name}.")
    for line in cost_lines(plan.cost, model_name):
        say(line)
    limit = args.max_tasks
    free = plan.cost.free_left
    if can_ask and free is not None and plan.cost.tokens > free and free > 0:
        fit = newest_that_fit(plan, free)
        if fit and ask(f"Only the newest {fit:,} tasks fit in your free tokens. Use just those?", True):
            limit = fit
    if limit and limit != args.max_tasks:
        items = embedding_items(found.new_tasks, chosen, limit=limit)
        plan = price(items, model_name, key)
        for line in cost_lines(plan.cost, model_name):
            say(line)
    if can_ask and not ask("Go ahead?", True):
        say("Stopped. Nothing was created.")
        return 0
    if not can_ask and not args.yes:
        say("Add --yes to go ahead.")
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
        say(warning)
    if plan.items and not embed_with_progress(predictor, plan, key):
        return 1
    return train_and_report(predictor, train)


def embed_with_progress(predictor: Predictor, plan, key: str) -> bool:
    from .pipeline import run_embeddings

    show = sys.stdout.isatty()

    def progress(done: int, total: int) -> None:
        if show:
            print(f"\rEmbedding: {done:,} of {total:,} texts", end="", flush=True)

    say("Embedding. Each batch is saved as it arrives, so you can stop and resume later.")
    try:
        run_embeddings(predictor, plan, key, progress, on_wait=lambda message: say(f"\n{message}"))
    except embeddings.VoyageError as error:
        say()
        say(explain(error))
        if "free token" in error.detail.lower() or "payment" in error.detail.lower():
            embeddings.set_free_tokens(key, "used")
        say("What was embedded so far is saved. Run yardstick-predict refresh to finish.")
        return False
    if show:
        say()
    return True


def train_and_report(predictor: Predictor, train) -> int:
    from .model import NotEnoughData

    say("Training...")
    try:
        model = train(predictor)
    except NotEnoughData as error:
        say(str(error))
        return 1
    print_report(model)
    say()
    say('Forecast with: yardstick-predict predict --model claude-opus-5-5 --prompt "Add a login page"')
    return 0


def cmd_refresh(args) -> int:
    from .pipeline import chosen_features, load_plugins, plan_embeddings, refresh_benchmarks, save_scan, scan, train

    predictor = pick_predictor(args.predictor)
    load_plugins(predictor)
    can_ask = interactive()
    ask_first = getattr(args, "ask_first", False)
    say(f"Predictor {predictor.name!r} was last refreshed {ago(predictor.config.get('refreshed_at'))}.")
    say("Looking for new tasks...")
    found = scan(predictor)
    per_archive = ", ".join(f"{name} {count:,}" for name, count in found.per_archive.items())
    say(f"Found {len(found.new_tasks):,} new tasks ({per_archive}).")
    if found.updated_tasks:
        grew = len(found.updated_tasks)
        say(f"{grew:,} saved tasks grew, because their helper agents finished after the last refresh.")
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
        say("Nothing new. The model is up to date.")
        return 0
    if plan.items:
        say(f"Embedding the new tasks with {model_name}:")
        for line in cost_lines(plan.cost, model_name):
            say("  " + line)
    if can_ask:
        question = "Refresh now? This adds the new tasks, embeds them and retrains." if ask_first else "Go ahead?"
        if not ask(question, not ask_first):
            say("Keeping the current model.")
            return 0
    elif not args.yes:
        say("Add --yes to go ahead.")
        return 2
    if plan.items:
        key = key or get_key(model_name, can_ask)
        if not key:
            return 1
    save_scan(predictor, found)
    warning = refresh_benchmarks(predictor)
    if warning:
        say(warning)
    if plan.items:
        plan = plan_embeddings(predictor, key)
        if not embed_with_progress(predictor, plan, key):
            return 1
    return train_and_report(predictor, train)


def cmd_predict(args) -> int:
    from .forecast import RequestError, answer, parse

    try:
        if args.prompt is not None:
            if not args.model:
                raise RequestError("invalid_request", "Give at least one --model with --prompt.")
            request = {
                "version": 1,
                "distribution": args.distribution,
                "tasks": [
                    {
                        "id": "task",
                        "prompt": args.prompt,
                        "models": args.model,
                        "context": args.context or "",
                        "skills": args.skill,
                    }
                ],
            }
            if args.effort:
                request["tasks"][0]["effort"] = args.effort
        else:
            if sys.stdin.isatty():
                raise RequestError("invalid_request", "Pipe a JSON request in, or use --model and --prompt.")
            try:
                request = json.load(sys.stdin)
            except ValueError as error:
                raise RequestError("invalid_request", f"The request isn't valid JSON: {error}") from None
        tasks, distribution, chosen = parse(request)
        names = predictor_names()
        chosen = args.predictor or chosen
        if chosen and chosen not in names:
            raise RequestError("unknown_predictor", f"There's no predictor called {chosen!r}.")
        if not names:
            raise RequestError("no_predictor", "There's no predictor yet. Run yardstick-predict to create one.")
        try:
            predictor = pick_predictor(chosen)
        except SystemExit as error:
            raise RequestError("no_predictor", str(error)) from None
        result = answer(predictor, tasks, distribution or args.distribution)
        code = 0
    except RequestError as error:
        result, code = error.as_answer(), 1
    indent = 2 if (args.pretty or sys.stdout.isatty()) else None
    print(json.dumps(result, indent=indent))
    return code


def cmd_status(args) -> int:
    names = [args.predictor] if args.predictor else predictor_names()
    found = []
    for name in names:
        if name not in predictor_names():
            raise SystemExit(f"There's no predictor called {name!r}.")
        predictor = Predictor(name)
        entry = {
            "name": name,
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
        say("There's no predictor yet. Run yardstick-predict to create one.")
        return 0
    from .pipeline import load_model

    for entry in found:
        say(
            f"{entry['name']}: {entry['embedding_model']} embeddings, {entry['tasks']:,} tasks, "
            f"last refreshed {ago(entry['refreshed_at'])}."
        )
        say(f"Features: {', '.join(entry['features'])}.")
        predictor = Predictor(entry["name"])
        if (predictor.model_dir / "model.json").is_file():
            print_report(load_model(predictor))
        say()
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yardstick-predict",
        description="Forecast how long a coding-agent task will take and what it will cost, before it runs.",
    )
    parser.add_argument("--version", action="version", version=f"yardstick-predict {__version__}")
    commands = parser.add_subparsers(dest="command")

    create = commands.add_parser("create", help="Create a predictor from your past runs.")
    create.add_argument("name", nargs="?", default="default", help="The predictor's name. Default: default.")
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
    refresh.add_argument("--predictor", help="Which predictor. Default: the only one, or 'default'.")
    refresh.add_argument("--yes", action="store_true", help="Go ahead without asking, even when it costs money.")

    predict = commands.add_parser("predict", help="Forecast tasks. Reads a JSON request on stdin, or use --prompt.")
    predict.add_argument("--predictor", help="Which predictor. Default: the only one, or 'default'.")
    predict.add_argument("--model", action="append", default=[], help="An AI model to forecast. Repeat for more.")
    predict.add_argument("--prompt", help="The task's prompt.")
    predict.add_argument("--context", help="The conversation so far, if there is one.")
    predict.add_argument("--effort", help="The thinking level, such as high or max.")
    predict.add_argument("--skill", action="append", default=[], help="A skill the agent will use. Repeat for more.")
    predict.add_argument("--distribution", action="store_true", help="Also give each forecast's full curve.")
    predict.add_argument("--pretty", action="store_true", help="Indent the JSON.")

    status = commands.add_parser("status", help="Show your predictors and how well they forecast.")
    status.add_argument("--predictor", help="Show only this predictor.")
    status.add_argument("--json", action="store_true", help="Print JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    handlers = {"create": cmd_create, "refresh": cmd_refresh, "predict": cmd_predict, "status": cmd_status}
    problem = xgboost_problem(sys.argv[1:] if argv is None else argv)
    if problem:
        if args.command == "predict":
            print(json.dumps({"version": 1, "error": {"code": "missing_library", "message": problem}}))
        else:
            say(problem)
        return 1
    try:
        return handlers.get(args.command, cmd_home)(args)
    except KeyboardInterrupt:
        say()
        say("Stopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
