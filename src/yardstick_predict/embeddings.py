"""Turn task text into embeddings with Voyage, and keep track of what that costs.

An embedding is a list of numbers that captures what a text means. Similar texts get similar numbers.

Your Voyage key stays on your computer. It is read from the VOYAGE_API_KEY environment variable or
from your computer's password store, and it is sent only to Voyage.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

from . import store

DIMENSIONS = 256
"""Voyage can return shorter vectors. 256 numbers keep the dataset small and lose little."""

FREE_TOKENS = 200_000_000
"""Voyage gives every account 200 million free tokens. It may count them per model; we count them
as one shared allowance, so we never promise more free tokens than you have."""

MAX_TEXTS_PER_REQUEST = 1000
KEY_SERVICE = "yardstick-predict"
KEY_USER = "voyage-api-key"


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    usd_per_million: float
    tokens_per_request: int
    note: str


MODELS = {
    "voyage-4": EmbeddingModel("voyage-4", 0.06, 320_000, "recommended"),
    "voyage-4-lite": EmbeddingModel("voyage-4-lite", 0.02, 1_000_000, "cheaper, slightly weaker in Voyage's tests"),
    "voyage-4-large": EmbeddingModel("voyage-4-large", 0.12, 120_000, "strongest, costs the most"),
}


class VoyageError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Voyage said {status}: {detail}")
        self.status = status
        self.detail = detail


def api_base() -> str:
    """Where requests go. Only an encrypted address may receive the key, or a test server on this computer."""
    base = os.environ.get("VOYAGE_API_BASE", "https://api.voyageai.com/v1").rstrip("/")
    address = urllib.parse.urlparse(base)
    if address.scheme != "https" and address.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise VoyageError(0, "VOYAGE_API_BASE must start with https://, so your key is never sent unencrypted")
    return base


# Counting tokens before anything is sent ---------------------------------------------------------------


TOKENIZER_REVISIONS = {
    "voyage-4": "44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9",
    "voyage-4-lite": "0335ddf7698395712e3220733b4079006951cfef",
    "voyage-4-large": "bb931c2635a93efe400c24741363d8ff61d7bb32",
}
"""Each tokenizer is downloaded at one fixed version, so a later change upstream can't change counts unseen."""


@lru_cache(maxsize=4)
def _tokenizer(model: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(f"voyageai/{model}", revision=TOKENIZER_REVISIONS[model])


def count_tokens(texts: list[str], model: str) -> tuple[list[int], bool]:
    """Count each text's tokens with Voyage's own tokenizer. The second value is False for an estimate.

    The tokenizer is downloaded from Hugging Face once. Without it, the count is an estimate.
    """
    if not texts:
        return [], True
    try:
        tokenizer = _tokenizer(model)
        return [len(e.ids) for e in tokenizer.encode_batch(texts)], True
    except Exception:
        return [math.ceil(len(t) / 3) + 1 for t in texts], False


# The key -------------------------------------------------------------------------------------------------


def find_key() -> tuple[str | None, str]:
    """The Voyage key and where it came from."""
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if key:
        return key, "the VOYAGE_API_KEY environment variable"
    try:
        import keyring

        key = (keyring.get_password(KEY_SERVICE, KEY_USER) or "").strip()
    except Exception:
        key = ""
    return (key, "your computer's password store") if key else (None, "")


def save_key(key: str) -> bool:
    """Keep the key in the computer's password store. False when there is no password store to use."""
    try:
        import keyring

        keyring.set_password(KEY_SERVICE, KEY_USER, key)
        return True
    except Exception:
        return False


def fingerprint(key: str) -> str:
    """A short, one-way label for a key, so usage can be counted without storing the key."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# Talking to Voyage ---------------------------------------------------------------------------------------


def _post(body: dict, key: str, timeout: float = 120) -> dict:
    request = urllib.request.Request(
        api_base() + "/embeddings",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read()).get("detail") or error.reason
        except Exception:
            detail = error.reason
        raise VoyageError(error.code, str(detail)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise VoyageError(0, f"couldn't reach Voyage ({error})") from None


def _post_with_retries(body: dict, key: str, on_wait: Callable[[str], None] | None = None, attempts: int = 8) -> dict:
    """Send one request. Wait and try again when Voyage is busy or limiting requests."""
    for attempt in range(attempts):
        try:
            return _post(body, key)
        except VoyageError as error:
            retry = error.status == 429 or error.status >= 500 or error.status == 0
            if not retry or attempt == attempts - 1:
                raise
            delay = min(60.0, 2.0**attempt)
            if on_wait:
                if error.status == 429:
                    on_wait(
                        f"Voyage is limiting requests, so this waits {delay:.0f} seconds. Accounts without a "
                        "payment method get low limits. Adding one raises them, and your free tokens still apply."
                    )
                else:
                    on_wait(f"Voyage didn't answer ({error.detail}). Trying again in {delay:.0f} seconds.")
            time.sleep(delay)
    raise AssertionError("unreachable")


def check_key(key: str, model: str) -> None:
    """Make one tiny request to confirm the key works. Raises VoyageError when it doesn't."""
    body = {"input": ["yardstick-predict key check"], "model": model, "output_dimension": DIMENSIONS}
    reply = _post_with_retries(body, key)
    record_usage(key, model, int((reply.get("usage") or {}).get("total_tokens") or 0))


def embed(
    texts: list[str],
    model: str,
    key: str,
    counts: list[int] | None = None,
    on_wait: Callable[[str], None] | None = None,
) -> Iterator[tuple[list[int], list[list[float]]]]:
    """Embed texts in as few requests as Voyage allows. Yields each request's positions and vectors."""
    limit = MODELS[model].tokens_per_request if model in MODELS else 120_000
    counts = counts or count_tokens(texts, model)[0]
    batch: list[int] = []
    batch_tokens = 0
    for position, tokens in enumerate(counts):
        if batch and (len(batch) == MAX_TEXTS_PER_REQUEST or batch_tokens + tokens > limit * 0.95):
            yield from _embed_batch(batch, texts, model, key, on_wait)
            batch, batch_tokens = [], 0
        batch.append(position)
        batch_tokens += tokens
    if batch:
        yield from _embed_batch(batch, texts, model, key, on_wait)


def _embed_batch(
    batch: list[int], texts: list[str], model: str, key: str, on_wait: Callable[[str], None] | None
) -> Iterator[tuple[list[int], list[list[float]]]]:
    """Embed one batch. When Voyage keeps limiting it, split it in half and send smaller requests.

    An account without a payment method may allow fewer tokens a minute than one full batch holds.
    Smaller requests are slower, but they finish. Each half is yielded as it arrives, so when a later
    half fails, the vectors already paid for are not lost.
    """
    body = {"input": [texts[i] for i in batch], "model": model, "output_dimension": DIMENSIONS, "truncation": True}
    try:
        reply = _post_with_retries(body, key, on_wait, attempts=4 if len(batch) > 1 else 8)
    except VoyageError as error:
        if error.status != 429 or len(batch) == 1:
            raise
        if on_wait:
            on_wait("Voyage keeps limiting requests this size, so this sends smaller ones.")
        half = len(batch) // 2
        yield from _embed_batch(batch[:half], texts, model, key, on_wait)
        yield from _embed_batch(batch[half:], texts, model, key, on_wait)
        return
    record_usage(key, model, int((reply.get("usage") or {}).get("total_tokens") or 0))
    data = sorted(reply.get("data") or [], key=lambda item: item.get("index", 0))
    if len(data) != len(batch):
        raise VoyageError(200, f"expected {len(batch)} embeddings and got {len(data)}")
    yield batch, [item["embedding"] for item in data]


# What it costs, counting the free tokens ------------------------------------------------------------------


def _ledger_path():
    return store.home() / "voyage-usage.json"


@contextmanager
def _ledger_held() -> Iterator[None]:
    """Keep other processes out of the ledger until this one is done with it.

    Two forecasts can record usage at the same moment. Without this, one could read the ledger while
    the other is still writing it, and the count it saves would undo the other's.
    """
    lock = os.open(store.private_home() / "voyage-usage.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(lock, msvcrt.LK_LOCK, 1)
                    break
                except OSError as error:  # Windows stops trying after ten seconds. Keep waiting.
                    if error.errno != errno.EDEADLOCK:
                        raise
            try:
                yield
            finally:
                msvcrt.locking(lock, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)  # closing the file lets go
            yield
    finally:
        os.close(lock)


def _ledger() -> dict:
    try:
        return json.loads(_ledger_path().read_text())
    except (OSError, ValueError):
        return {}


def _save_ledger(ledger: dict) -> None:
    store.private_home()
    _ledger_path().write_text(json.dumps(ledger, indent=2) + "\n")


def account(key: str) -> dict:
    """What yardstick knows about this key's Voyage account. The key itself is never stored."""
    with _ledger_held():
        ledger = _ledger()
    return ledger.get(fingerprint(key), {"free_tokens": "unknown", "asked": False, "used": {}})


def set_free_tokens(key: str, status: str) -> None:
    """Record whether the account had its free tokens when yardstick first used it: 'new', 'used' or 'unknown'."""
    with _ledger_held():
        ledger = _ledger()
        entry = ledger.setdefault(fingerprint(key), {"used": {}, "first_seen": time.time()})
        entry["free_tokens"] = status
        entry["asked"] = True
        _save_ledger(ledger)


def record_usage(key: str, model: str, tokens: int) -> None:
    with _ledger_held():
        ledger = _ledger()
        entry = ledger.setdefault(fingerprint(key), {"free_tokens": "unknown", "used": {}, "first_seen": time.time()})
        entry["used"][model] = entry["used"].get(model, 0) + tokens
        _save_ledger(ledger)


def free_tokens_left(key: str | None) -> int | None:
    """Free tokens left on the account, by yardstick's own count. None when yardstick can't know."""
    if not key:
        return None
    entry = account(key)
    if entry.get("free_tokens") == "used":
        return 0
    if entry.get("free_tokens") == "new":
        return max(FREE_TOKENS - sum(entry.get("used", {}).values()), 0)
    return None


@dataclass
class Cost:
    """What embedding some tokens costs: at Voyage's list price, and after the free tokens."""

    tokens: int
    exact: bool
    usd_per_million: float
    free_left: int | None

    @property
    def list_usd(self) -> float:
        return self.tokens / 1e6 * self.usd_per_million

    @property
    def your_usd(self) -> float | None:
        """What you pay after free tokens. None when yardstick can't tell whether any are left."""
        if self.free_left is None:
            return None
        return max(self.tokens - self.free_left, 0) / 1e6 * self.usd_per_million


def cost(tokens: int, exact: bool, model: str, key: str | None) -> Cost:
    price = MODELS[model].usd_per_million if model in MODELS else 0.0
    return Cost(tokens, exact, price, free_tokens_left(key))
