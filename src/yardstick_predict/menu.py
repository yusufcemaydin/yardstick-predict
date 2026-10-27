"""The Features step of setup: each feature with its options under it, turned on and off by number.

The first three features are always on: they cost nothing and send nothing anywhere. The others are
choices. Most cost embedding tokens or send text to another company. The conversation costs nothing, but
it suits a person's sessions better than a program's one-off runs. LLM analysis also asks how it's paid
for and which AI model reads the tasks. Next to each feature the feature check tested, the screen shows
what the check said last.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import check, features, llm, ui


@dataclass
class Picked:
    """What a person or a program chose: the features, each feature's options and how LLM analysis runs."""

    features: list[str]
    options: dict[str, list[str]] = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    def config(self) -> dict:
        """The predictor settings these choices make."""
        found: dict = {"features": self.features, "options": self.options}
        if self.llm:
            found["llm"] = self.llm
        return found


def from_flags(given: str | None, provider: str | None = None, model: str | None = None) -> Picked:
    """The features a program names, like "context,skills:named,llm". A feature alone gets all its options.

    Raises ValueError with a plain reason when a name or an option doesn't exist.
    """
    names = [part.strip() for part in (given or "").split(",") if part.strip()] if given is not None else []
    chosen, options = [], {}
    for part in names if given is not None else list(features.DEFAULT):
        name, _, wanted = part.partition(":")
        chosen.append(name)
        every = [o.key for o in _group(name).options] if _group(name) else []
        if wanted:
            picked = [o for o in wanted.split("+") if o]
            unknown = [o for o in picked if o not in every]
            if unknown or not every:
                raise ValueError(
                    f"{name} has no option {unknown[0] if unknown else wanted!r}. Its options: "
                    f"{', '.join(every) or 'none'}."
                )
            options[name] = picked
        elif len(every) > 1:
            options[name] = every
    found = Picked(features.check(chosen), options)
    if "llm" in found.features:
        found.llm = llm.access(provider, model)
    elif provider or model:
        raise ValueError("--llm-provider and --llm-model go with the llm feature. Add llm to --features.")
    return found


def _group(name: str) -> features.Group | None:
    return next((g for g in features.MENU if g.key == name), None)


def ask(
    custom: dict[str, str] | None = None, start: Picked | None = None, verdicts: dict[str, dict] | None = None
) -> Picked:
    """Show every feature with its options, and let a person turn them on and off until they press Enter.

    The features `start` picks start on, with their options. Without it, the default features do, with all their
    options. LLM analysis runs as `start` says, and is asked about only when `start` doesn't say. Next to each
    feature go the feature check's `verdicts` on it, by default the latest ones remembered.
    """
    groups = list(features.MENU)
    verdicts = check.latest() if verdicts is None else verdicts
    chosen_first = features.DEFAULT if start is None else start.features
    on: dict[str, set[str]] = {}
    for g in groups:
        if g.always:
            continue
        every = {o.key for o in g.options}
        kept = (start.options.get(g.key) if start is not None else None) or every
        on[g.key] = every & set(kept) if g.key in chosen_first else set()
    for name in custom or {}:
        on[name] = {"on"} if start is not None and name in start.features else set()
    while True:
        _show(groups, on, custom or {}, verdicts)
        reply = input(ui.prompt("Turn on or off", "[Enter when done]")).strip().lower()
        if not reply:
            break
        for token in re.split(r"[\s,]+", reply):
            _toggle(token, groups, on, custom or {})
    chosen = ["prompt"] + [g.key for g in groups if not g.always and on[g.key]]
    chosen += [name for name in custom or {} if on[name]]
    options = {
        g.key: sorted(on[g.key], key=[o.key for o in g.options].index)
        for g in groups
        if len(g.options) > 1 and on.get(g.key)
    }
    picked = Picked(features.check(chosen), options)
    if "llm" in picked.features:
        picked.llm = dict(start.llm) if start is not None and start.llm else ask_access()
    return picked


def _show(
    groups: list[features.Group], on: dict[str, set[str]], custom: dict[str, str], verdicts: dict[str, dict]
) -> None:
    ui.heading("Features")
    _note("What the predictor learns from. The first three are always on, cost nothing and send nothing anywhere.")
    ui.line()
    entries = [(g.title, g.says, g.always, g.key, g.options) for g in groups]
    entries += [(name, says, False, name, ()) for name, says in custom.items()]
    title_width = max(len(title) for title, *_ in entries) + 3
    option_width = max((len(o.says) for *_, options in entries for o in options), default=0) + 3
    # Each option says on or off at the end of its line, lined up with the others. Where the terminal is too
    # narrow for that, it says so first, and its words wrap after it.
    lined_up = 2 + 9 + option_width + 3 <= ui.columns() - 1

    def state(is_on: bool) -> str:
        return ui.green("on") if is_on else ui.dim("off")

    for number, (title, says, always, key, options) in enumerate(entries, start=1):
        mark = ui.dim("always on") if always else state(bool(on[key]))
        ui.line(f"{ui.accent(str(number).rjust(2))}  {ui.bold(title.ljust(title_width))}{mark}")
        for text in (says, check.summary(verdicts.get(key) or {})):
            for part in ui.wrapped(text, 6) if text else []:  # broken to fit the terminal
                ui.line(f"    {ui.dim(part)}")
        for letter, option in zip("abcdefgh", () if always else options):
            is_on = option.key in on[key]
            if lined_up:
                ui.line(f"      {ui.accent(letter)}  {option.says.ljust(option_width)}{state(is_on)}")
                continue
            for row, part in enumerate(ui.wrapped(option.says, 16)):
                first = f"      {ui.accent(letter)}  {state(is_on)}{' ' * (5 - len('on' if is_on else 'off'))}"
                ui.line((first if row == 0 else " " * 14) + part)
    ui.line()
    _note("Type a number to turn a feature on or off, or a number and a letter for one option, like 9 or 6b.")


def _note(text: str) -> None:
    """A note broken to fit the terminal."""
    for part in ui.wrapped(text):
        ui.note(part)


def _toggle(token: str, groups: list[features.Group], on: dict[str, set[str]], custom: dict[str, str]) -> None:
    match = re.fullmatch(r"(\d+)([a-h]?)", token)
    names = [g.key for g in groups] + list(custom)
    if not match or not 1 <= int(match.group(1)) <= len(names):
        ui.note(f"{token!r} isn't one of the numbers. Type a number from 1 to {len(names)}, like 9 or 6b.")
        return
    name = names[int(match.group(1)) - 1]
    group = _group(name)
    if group is not None and group.always:
        ui.note(f"{group.title} is always on. It costs nothing.")
        return
    every = [o.key for o in group.options] if group else ["on"]
    letter = match.group(2)
    if not letter:
        on[name] = set() if on[name] else set(every)
        return
    position = "abcdefgh".index(letter)
    if group is None or position >= len(every):
        ui.note(f"{name} has no option {letter}.")
        return
    on[name] ^= {every[position]}


def ask_access() -> dict:
    """How LLM analysis is paid for, and which AI model reads the tasks."""
    ui.heading("LLM analysis")
    _note("It sends each task's prompt and the end of the conversation before it to the AI model you pick.")
    ui.line()
    ways = list(llm.PROVIDERS.values())
    ui.table([(ui.accent(str(n)), ui.bold(way.title), ui.dim(way.says)) for n, way in enumerate(ways, start=1)])
    provider = ways[_pick("How should it be paid for?", len(ways)) - 1]
    models = provider.models
    if len(models) == 1:
        return {"provider": provider.key, "model": models[0]}
    ui.line()
    rows = []
    for n, name in enumerate(models, start=1):
        cost = llm.cost_per_thousand(name) if provider.billed else None
        price = f"about ${cost:,.2f} per 1,000 tasks" if cost is not None else ""
        rows.append((ui.accent(str(n)), ui.bold(llm.title(name)), ui.dim(price)))
    ui.table(rows)
    return {"provider": provider.key, "model": models[_pick("Which model?", len(models)) - 1]}


def _pick(question: str, count: int) -> int:
    while True:
        reply = input(ui.prompt(question, "[1]")).strip() or "1"
        if ui.number_typed(reply) and 1 <= int(reply) <= count:
            return int(reply)
        ui.note(f"Type a number from 1 to {count}.")
