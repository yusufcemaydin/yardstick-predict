"""What a setup or a refresh costs before it spends anything.

Every paid step counts at its list price: LLM analysis paid with an API key, and Voyage's embeddings. Voyage's free
tokens aren't taken off, as yardstick can't see how many are left. LLM analysis paid by a subscription adds no
bill, so it counts as nothing.
"""

from __future__ import annotations

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

    @property
    def usd(self) -> float | None:
        """Every paid step together. None when the price of LLM analysis isn't known."""
        return None if self.analysis_usd is None else self.analysis_usd + self.embedding_usd

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
    price = embeddings.MODELS[embedding_model].usd_per_million if embedding_model in embeddings.MODELS else 0.0
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
    )


def _rounded(usd: float | None) -> float | None:
    return None if usd is None else round(usd, 6)
