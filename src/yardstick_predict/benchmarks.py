"""Public facts about AI models, used as a feature: prices, context size, longest answer and release date.

They come from OpenRouter's public list of models, which describes old and new models in the same
standard way and needs no account. The list is downloaded when a predictor is created or refreshed,
and saved with the predictor, so training and forecasting use the same numbers.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from datetime import datetime, timezone

SOURCE = "https://openrouter.ai/api/v1/models"
FACTS = (
    "benchmark_price_input",
    "benchmark_price_output",
    "benchmark_price_cache_read",
    "benchmark_context_window",
    "benchmark_longest_answer",
    "benchmark_released",
    "benchmark_reasons",
)
_EPOCH = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()


def normalize(name: str) -> str:
    """One spelling for every source: anthropic/claude-opus-5.5 and claude-opus-5-5 both become claude-opus-5-5."""
    name = name.strip().lower().split("/")[-1].split(":")[0]
    name = re.sub(r"\[.*\]$", "", name)
    name = re.sub(r"-\d{8}$", "", name)
    return name.replace(".", "-")


def _per_million(value: object) -> float | None:
    try:
        price = float(value)  # OpenRouter gives dollars per token
    except (TypeError, ValueError):
        return None
    return price * 1e6 if price >= 0 else None


def _facts(entry: dict) -> dict[str, float]:
    pricing = entry.get("pricing") or {}
    top = entry.get("top_provider") or {}
    found: dict[str, float] = {}
    for fact, value in (
        ("benchmark_price_input", _per_million(pricing.get("prompt"))),
        ("benchmark_price_output", _per_million(pricing.get("completion"))),
        ("benchmark_price_cache_read", _per_million(pricing.get("input_cache_read"))),
    ):
        if value is not None:
            found[fact] = math.log10(1 + value)
    for fact, value in (
        ("benchmark_context_window", entry.get("context_length") or top.get("context_length")),
        ("benchmark_longest_answer", top.get("max_completion_tokens")),
    ):
        if isinstance(value, (int, float)) and value > 0:
            found[fact] = math.log10(value)
    if isinstance(entry.get("created"), (int, float)):
        found["benchmark_released"] = (entry["created"] - _EPOCH) / (365.25 * 86400)
    parameters = entry.get("supported_parameters") or []
    found["benchmark_reasons"] = 1.0 if ("reasoning" in parameters or entry.get("reasoning")) else 0.0
    return found


def parse(listing: dict) -> dict[str, dict[str, float]]:
    """The facts of every model in an OpenRouter listing, by normalized name."""
    table: dict[str, dict[str, float]] = {}
    for entry in listing.get("data") or []:
        model_id = str(entry.get("id") or "")
        if not model_id or ":" in model_id:  # skip variants such as :batch and :free
            continue
        table.setdefault(normalize(model_id), _facts(entry))
    return table


def download(timeout: float = 30) -> dict[str, dict[str, float]]:
    """Download the current facts. Raises OSError when OpenRouter can't be reached."""
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "yardstick-predict"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse(json.loads(response.read()))


def facts(table: dict[str, dict[str, float]], model: str) -> dict[str, float]:
    """The facts for one model. Empty when the list doesn't have it, which the model reads as missing."""
    return dict(table.get(normalize(model), {}))
