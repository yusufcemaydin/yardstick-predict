"""Work out what a model call cost from its tokens and the model's list price."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(resources.files("yardstick_predict").joinpath("prices.json").read_text())


def normalize(model: str) -> str:
    """Drop a release date or a bracketed variant, so claude-opus-4-5-20251101 matches claude-opus-4-5."""
    model = re.sub(r"\[.*\]$", "", model.strip())
    return re.sub(r"-\d{8}$", "", model)


def price(model: str) -> dict | None:
    """The model's prices in dollars per million tokens, or None when the model isn't in the list."""
    return _table()["models"].get(normalize(model))


def call_cost(
    model: str,
    new_input: int,
    cache_write_5m: int,
    cache_write_1h: int,
    cache_read: int,
    output: int,
    fast: bool = False,
) -> float | None:
    """The dollar cost of one model call. None when the model's price isn't known."""
    base = price(model)
    if base is None:
        return None
    rates = base
    long_context = base.get("long_context")
    if long_context and new_input + cache_write_5m + cache_write_1h + cache_read > long_context["above"]:
        rates = {**base, **long_context}
    write_5m = rates.get("cache_write_5m", rates["input"])
    write_1h = rates.get("cache_write_1h", write_5m)
    read = rates.get("cache_read", rates["input"])
    usd = (
        new_input * rates["input"]
        + cache_write_5m * write_5m
        + cache_write_1h * write_1h
        + cache_read * read
        + output * rates["output"]
    ) / 1e6
    if fast:
        multiplier = base.get("fast_multiplier")
        if multiplier is None:
            return None
        usd *= multiplier
    return usd
