"""What a setup or a refresh costs before it spends anything, and the limit --max-cost puts on it.

Every paid step counts at its list price: LLM analysis paid with an API key, and Voyage's embeddings. Voyage's free
tokens aren't taken off, as yardstick can't see how many are left. LLM analysis paid by a subscription adds no
bill, so it counts as nothing.

A limit is kept three ways. Before anything is spent, what's planned must fit in it, with every text Voyage gets at
the most tokens it can take. While LLM analysis runs, no request starts unless the most it can cost fits in what's
left, as llm.Budget says. And before each step that embeds, the most it may cost must fit in what's left. A step
that doesn't fit stops the command, and what was done is kept.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import embeddings, llm


@dataclass(frozen=True)
class Cost:
    """What a setup or a refresh costs, step by step, at list price, in US dollars."""

    analysis_tasks: int
    """How many tasks LLM analysis reads."""
    analysis_model: str | None
    paid_with: str | None
    """How LLM analysis is paid for: anthropic or openai for an API key, claude or codex for a subscription."""
    analysis_usd: float | None
    """None when the AI model's price isn't in yardstick's list. 0 for a subscription."""
    embedding_model: str
    embedding_texts: int
    """How many texts go to Voyage, each once. The descriptions LLM analysis writes count too."""
    embedding_tokens: int
    embedding_exact: bool
    """False when the tokens are an estimate: Voyage's tokenizer didn't download, or descriptions are still to come."""
    embedding_usd: float
    embedding_most_usd: float
    """The most embedding may cost: at the token count when it's exact, else at the most tokens each text can take."""

    @property
    def usd(self) -> float | None:
        """Every paid step together. None when the price of LLM analysis isn't known."""
        return None if self.analysis_usd is None else self.analysis_usd + self.embedding_usd

    @property
    def most_usd(self) -> float | None:
        """What a spending limit counts: LLM analysis at the price shown, and embedding at the most it may cost."""
        return None if self.analysis_usd is None else self.analysis_usd + self.embedding_most_usd

    def as_json(self) -> dict:
        analysis = None
        if self.analysis_tasks:
            analysis = {
                "tasks": self.analysis_tasks,
                "model": self.analysis_model,
                "paid_with": self.paid_with,
                "usd": _rounded(self.analysis_usd),
            }
        embedding = {
            "texts": self.embedding_texts,
            "tokens": self.embedding_tokens,
            "exact": self.embedding_exact,
            "model": self.embedding_model,
            "usd": _rounded(self.embedding_usd),
        }
        return {"usd": _rounded(self.usd), "analysis": analysis, "embedding": embedding}


def cost_of(config: dict, embedding_model: str, plan, asks: list, described: int) -> Cost:
    """What a setup or refresh costs: LLM analysis of the `asks`, and embedding the texts the `plan` sends to Voyage
    with the `described` descriptions LLM analysis writes."""
    how = config.get("llm") or {}
    provider = llm.PROVIDERS.get(how.get("provider")) if asks else None
    if provider is None:
        analysis_usd: float | None = 0.0
    elif provider.billed:
        analysis_usd = llm.cost(len(asks), how["model"])
    else:
        analysis_usd = 0.0  # a subscription pays: no bill
    tokens = plan.cost.tokens + described * llm.DESCRIPTION_TOKENS
    price = _price(embedding_model)
    return Cost(
        analysis_tasks=len(asks) if provider is not None else 0,
        analysis_model=how.get("model") if provider is not None else None,
        paid_with=provider.key if provider is not None else None,
        analysis_usd=analysis_usd,
        embedding_model=embedding_model,
        embedding_texts=len(plan.texts) + described,
        embedding_tokens=tokens,
        embedding_exact=plan.cost.exact and not described,
        embedding_usd=tokens / 1e6 * price,
        embedding_most_usd=embedding_most_usd(plan, embedding_model, described),
    )


def embedding_most_usd(plan, model: str, described: int = 0) -> float:
    """The most embedding a plan's texts may cost at list price, with `described` descriptions still to come. Each
    text counts at its tokens when the count is exact, else at its length in bytes, which no count can pass. A
    description counts at the most one takes."""
    if plan.cost.exact:
        tokens = plan.cost.tokens
    else:
        tokens = sum(embeddings.most_tokens(text) for text in plan.texts)
    return (tokens + described * llm.DESCRIPTION_TOKENS) / 1e6 * _price(model)


class Limit:
    """A limit on what one setup or refresh spends, in US dollars at list price, with what it has spent so far."""

    def __init__(self, usd: float) -> None:
        self.usd = usd
        self.spent = 0.0

    def fits(self, usd: float) -> bool:
        """Whether spending this much more stays within the limit."""
        return self.spent + usd <= self.usd + 1e-9

    def left(self) -> float:
        return max(self.usd - self.spent, 0.0)


def dollars(usd: float) -> str:
    """A cost or a limit in words: with cents, or for an amount under a cent with its first two digits that aren't
    0: $1.42, $0.0030, $0.0000059. Nothing at all is $0."""
    if usd <= 0:
        return "$0"
    if usd >= 0.01:
        return f"${usd:.2f}"
    return f"${usd:.{1 - math.floor(math.log10(usd))}f}"


def _price(model: str) -> float:
    return embeddings.MODELS[model].usd_per_million if model in embeddings.MODELS else 0.0


def _rounded(usd: float | None) -> float | None:
    return None if usd is None else round(usd, 6)
