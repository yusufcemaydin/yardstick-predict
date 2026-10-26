"""The summary an update shows before it spends anything: what reading found, then what each model will analyze,
embed and train, with every cost and time. One question follows it."""

from __future__ import annotations

from .. import cli, embeddings, estimate, llm, ui
from . import keys, listing


def ask(work: list, key: str | None) -> bool:
    """Show what each model finds and does, with every cost and time, then ask once. True to go ahead."""
    ui.out()
    ui.line(found_line(work))
    ui.out()
    usd = seconds = disk = 0.0
    memory: list[str] = []
    names = [job.predictor.name for job in work]
    pad = max(len(name) for name in names)
    for job in work:
        lines, cost, found = block(job, key)
        usd += cost
        if found is not None:
            seconds += found.seconds
            disk += found.disk
            note = estimate.memory_note(found, estimate.total_memory())
            if note:
                memory.append(note)
        for number, text in enumerate(lines):
            name = ui.bold(job.predictor.name.ljust(pad)) if number == 0 else " " * pad
            ui.line(f"{name}  {text}")
    todo = [job for job in work if not job.planned.up_to_date]
    took = "under a minute" if seconds < 60 else f"about {estimate.rough(seconds)}"
    ui.out()
    their = "Their" if len(todo) > 1 else "Its"
    ui.line(f"{took[0].upper()}{took[1:]} in all. {their} data grows to {ui.size(disk)}.")
    for note in dict.fromkeys(memory):
        ui.note(note)
    ui.out()
    label = "Update all" if len(todo) > 1 else f"Update {todo[0].predictor.name}"
    choices = [keys.Choice(label, ui.join([cli.money(usd), took])), keys.Choice("Cancel")]
    return keys.choose(choices, footer=[keys.hint("esc cancel")]) == 0


def found_line(work: list) -> str:
    """What reading found, in all the models together: how many new tasks, how many changed and where."""
    new, changed, removed, where_from = {}, {}, set(), {}
    titles = {a["name"]: listing.title(a) for job in work for a in job.predictor.config.get("archives") or []}
    for job in work:
        found = job.planned.found
        new.update({task.id: task for task in found.new_tasks})
        changed.update({task.id: task for task in found.updated_tasks})
        removed |= set(found.removed)
    for task in [*new.values(), *changed.values()]:
        where_from[titles.get(task.archive, task.archive)] = True
    parts = []
    if new:
        parts.append(f"{len(new):,} new task{'' if len(new) == 1 else 's'}")
    if changed:
        kind = "one" if new else "task"
        parts.append(f"{len(changed):,} changed {kind}{'' if len(changed) == 1 else 's'}")
    if not parts:
        text = "No new tasks."
    else:
        places = list(where_from)
        text = f"Found {' and '.join(parts)}"
        text += f", all in {places[0]}." if len(places) == 1 else f" in {cli.in_words(places)}."
    if removed:
        count = len(removed)
        text += f" {count:,} saved task{' is' if count == 1 else 's are'} gone from their records and taken out."
    return text


def block(job, key: str | None) -> tuple[list[str], float, estimate.Estimate | None]:
    """One model's lines in the summary: what it analyzes, embeds and trains, with each cost and time. Returns
    them with what it costs in dollars and how long and how much room it takes."""
    predictor, planned = job.predictor, job.planned
    if planned.up_to_date:
        return ["Nothing new. It's up to date."], 0.0, None
    config = predictor.config
    lines, usd = [], 0.0
    asks = planned.asks
    if asks:
        how = config["llm"]
        provider = llm.PROVIDERS[how["provider"]]
        lines.append(f"Analyze {len(asks):,} task{'' if len(asks) == 1 else 's'} with {analyst(how)}")
        if provider.billed:
            price = llm.cost(len(asks), how["model"])
            usd += price or 0.0
            lines.append(f"{cli.money(price)} at list price" if price is not None else "its price isn't known")
        else:
            requests = -(-len(asks) // provider.batch)
            plan = provider.title.replace("subscription", "plan")
            lines.append(ui.join([f"your {plan}", f"about {requests:,} request{'' if requests == 1 else 's'}"]))
    texts = len(planned.embed.texts) + planned.described
    if texts:
        # The descriptions LLM analysis writes are embedded once they exist, so their tokens are an estimate.
        tokens = planned.embed.cost.tokens + planned.described * llm.DESCRIPTION_TOKENS
        exact = planned.embed.cost.exact and not planned.described
        cost = embeddings.cost(tokens, exact, predictor.embedding_model, key)
        paid = cost.your_usd if cost.your_usd is not None else cost.list_usd
        usd += paid
        lines.append(f"Embed {texts:,} text{'' if texts == 1 else 's'} with Voyage")
        lines.append(ui.join([cli.money(paid), f"{'' if exact else 'about '}{tokens:,} tokens"]))
    dataset = predictor.dir / "dataset.sqlite"
    size = dataset.stat().st_size if dataset.is_file() else 0
    found = estimate.of(
        planned.tasks,
        config,
        planned.embed.cost.tokens,
        len(planned.embed.items),
        len(asks),
        planned.found.new_tasks,
        size,
    )
    lines.append(ui.join(["Train again", estimate.rough(found.steps.get("training", 0.0))]))
    why = BEHIND.get(planned.behind) if planned.model is not None else None
    if why:
        lines.append(ui.dim(why))
    return lines, usd, found


BEHIND = {
    "unfinished": "Training didn't finish last time.",
    "outdated": "An older version made its model. It gives no forecasts until it's trained again.",
    "previous": "The previous version made its model. Training again brings this one's improvements.",
}


def analyst(how: dict) -> str:
    """Who reads the tasks for LLM analysis: Codex or Claude Code for a subscription, else the AI model."""
    if how["provider"] == "codex":
        return "Codex"
    if how["provider"] == "claude":
        return "Claude Code"
    return llm.title(how["model"])
