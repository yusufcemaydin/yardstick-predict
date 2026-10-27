"""Public facts about AI models, used as a feature: prices, context size, longest answer and release date.

They come from OpenRouter's public list of models, which describes old and new models in the same
standard way and needs no account. The list is downloaded when a predictor is created or refreshed,
and saved with the predictor, so training and forecasting use the same numbers.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import urllib.request
from datetime import UTC, datetime

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
_EPOCH = datetime(2020, 1, 1, tzinfo=UTC).timestamp()


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
    return price * 1e6 if math.isfinite(price) and price >= 0 else None


def _facts(entry: dict) -> dict[str, float]:
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    top = entry.get("top_provider") if isinstance(entry.get("top_provider"), dict) else {}
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
        if _number(value) and value > 0:
            found[fact] = math.log10(value)
    if _number(entry.get("created")):
        found["benchmark_released"] = (entry["created"] - _EPOCH) / (365.25 * 86400)
    parameters = entry.get("supported_parameters")
    parameters = parameters if isinstance(parameters, list) else []
    found["benchmark_reasons"] = 1.0 if ("reasoning" in parameters or entry.get("reasoning")) else 0.0
    return found


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse(listing: dict) -> dict[str, dict[str, float]]:
    """The facts of every model in an OpenRouter listing, by normalized name.

    Raises ValueError when the listing isn't in the shape OpenRouter sends, so a refresh keeps the facts
    it saved before. An entry in another shape is left out.
    """
    if not isinstance(listing, dict) or not isinstance(listing.get("data"), list):
        raise ValueError("the model list isn't in the shape OpenRouter sends")
    table: dict[str, dict[str, float]] = {}
    for entry in listing["data"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        model_id = entry["id"]
        if not model_id or ":" in model_id:  # skip variants such as :batch and :free
            continue
        table.setdefault(normalize(model_id), _facts(entry))
    return table


def download(timeout: float = 30) -> dict[str, dict[str, float]]:
    """Download the current facts. Raises OSError when OpenRouter can't be reached, and ValueError for an odd answer."""
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "yardstick-predict"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except http.client.HTTPException as error:  # the connection broke while OpenRouter answered
        raise OSError(f"the connection to OpenRouter broke ({type(error).__name__})") from None
    return parse(json.loads(raw))


def facts(table: dict[str, dict[str, float]], model: str) -> dict[str, float]:
    """The facts for one model. Empty when the list doesn't have it, which the model reads as missing."""
    return dict(table.get(normalize(model), {}))
