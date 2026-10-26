"""Turn task text into embeddings with Voyage, and keep track of what that costs.

An embedding is a list of numbers that captures what a text means. Similar texts get similar numbers.

Your Voyage key stays on your computer. It is read from the VOYAGE_API_KEY environment variable or
from your computer's password store, and it is sent only to Voyage.

Every vector Voyage sends is kept in a private cache, by a one-way hash of its text and never the text
itself. A text embedded once, for training or for a forecast, is never paid for again.
"""

from __future__ import annotations

import email.utils
import hashlib
import http.client
import json
import logging
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from array import array
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

from . import files, locks, store

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


SETTINGS = -1
"""The status of an error in a setting on this computer. No retry can fix it, so it's never retried."""


class VoyageError(Exception):
    def __init__(self, status: int, detail: str, retry_after: float | None = None) -> None:
        from . import ui

        detail = ui.clean(detail)  # Voyage's own words, shown as they are, but never codes a terminal acts on
        super().__init__(detail if status == SETTINGS else f"Voyage said {status}: {detail}")
        self.status = status
        self.detail = detail
        self.retry_after = retry_after
        """How many seconds Voyage asked to wait before trying again, when it said."""


def api_base() -> str:
    """Where requests go. Only an encrypted address may receive the key, or a test server on this computer."""
    base = os.environ.get("VOYAGE_API_BASE", "https://api.voyageai.com/v1").rstrip("/")
    address = urllib.parse.urlparse(base)
    local = address.scheme == "http" and address.hostname in ("127.0.0.1", "localhost", "::1")
    if address.scheme != "https" and not local:
        raise VoyageError(SETTINGS, "VOYAGE_API_BASE must start with https://, so your key is never sent unencrypted")
    return base


def _now() -> float:
    """Seconds on a clock that only goes forward. Tests replace it, and time.sleep, to pass minutes at once."""
    return time.monotonic()


def cache_folder():
    """The private folder for what yardstick keeps to save time and money, inside the yardstick folder."""
    folder = store.private_home() / "cache"
    folder.mkdir(exist_ok=True)
    if os.name == "posix":
        os.chmod(folder, 0o700)
    return folder


# Counting tokens before anything is sent ---------------------------------------------------------------


TOKENIZER_REVISIONS = {
    "voyage-4": "44f3b2ae4ddf33403ed4dd66bec3fa48ff7dbbf9",
    "voyage-4-lite": "0335ddf7698395712e3220733b4079006951cfef",
    "voyage-4-large": "bb931c2635a93efe400c24741363d8ff61d7bb32",
}
"""Each tokenizer is downloaded at one fixed version, so a later change upstream can't change counts unseen."""

TOKENIZER_PAUSE = 3600.0
"""After the tokenizer fails to download, counts are estimates for an hour before the download is tried again.

Without a connection to Hugging Face, each try waits about 20 seconds before it fails. Setup counts the
tokens for each Voyage model, so without this pause it would wait that long again and again.
"""

_tokenizers: dict[str, object] = {}
_tokenizer_failed_at: float | None = None
"""When the tokenizer last failed to download in this process. The note on disk tells other processes."""


def _tokenizer_note():
    return store.home() / "cache" / "tokenizer-failed.json"


def _load_tokenizer(model: str, revision: str):
    """Download a tokenizer from Hugging Face, or load it from the computer's copy of an earlier download."""
    from tokenizers import Tokenizer

    try:  # the download goes through huggingface_hub, which sets up its logger as this loads
        from huggingface_hub.utils import logging as _hub_logging  # noqa: F401
    except ImportError:
        pass
    hub = logging.getLogger("huggingface_hub")
    level = hub.level
    hub.setLevel(logging.ERROR)  # its note about each retry would fill the screen
    try:
        return Tokenizer.from_pretrained(f"voyageai/{model}", revision=revision)
    finally:
        hub.setLevel(level)


def _tokenizer_failed_recently() -> bool:
    now = time.time()
    if _tokenizer_failed_at is not None and 0 <= now - _tokenizer_failed_at < TOKENIZER_PAUSE:
        return True
    try:
        failed_at = float(json.loads(_tokenizer_note().read_text())["failed_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return 0 <= now - failed_at < TOKENIZER_PAUSE


def _tokenizer(model: str):
    """Voyage's own tokenizer for a model. A failed download is remembered, in this process and on disk."""
    global _tokenizer_failed_at
    if model in _tokenizers:
        return _tokenizers[model]
    revision = TOKENIZER_REVISIONS[model]  # a KeyError means no tokenizer is known for this model
    if _tokenizer_failed_recently():
        raise RuntimeError("Voyage's tokenizer couldn't be downloaded a short while ago")
    try:
        loaded = _load_tokenizer(model, revision)
    except Exception:
        _tokenizer_failed_at = time.time()
        try:
            cache_folder()
            files.write_text(_tokenizer_note(), json.dumps({"failed_at": _tokenizer_failed_at}) + "\n")
        except OSError:
            pass  # this process still remembers
        raise
    _tokenizer_failed_at = None
    _tokenizer_note().unlink(missing_ok=True)  # the connection works again, so other models may try too
    _tokenizers[model] = loaded
    return loaded


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


def most_tokens(text: str) -> int:
    """A cheap count that no text's tokens can go over: its length in bytes, and room for start and end marks.

    Every token stands for at least one byte of text. So a request sized by this is never too big for Voyage,
    and nothing has to wait for the tokenizer to load.
    """
    return len(text.encode("utf-8", "surrogatepass")) + 2


# The key -------------------------------------------------------------------------------------------------


NO_ANSWER = "no_answer"
"""Where find_key says the key came from when the password store didn't answer in time."""


def find_key(wait: float | None = None) -> tuple[str | None, str]:
    """The Voyage key and where it came from.

    With `wait`, the password store gets at most that many seconds to answer. A locked one, such as a macOS
    Keychain, waits for you to unlock it. When it doesn't answer in time, the key is None and it came from
    NO_ANSWER.
    """
    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if key:
        return key, "the VOYAGE_API_KEY environment variable"
    found: list[str] = []

    def look() -> None:
        try:
            import keyring

            found.append((keyring.get_password(KEY_SERVICE, KEY_USER) or "").strip())
        except Exception:
            found.append("")

    if wait is None:
        look()
    else:
        import threading

        asking = threading.Thread(target=look, daemon=True)  # a store that never answers can't keep it from ending
        asking.start()
        asking.join(wait)
        if not found:
            return None, NO_ANSWER
    return (found[0], "your computer's password store") if found[0] else (None, "")


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
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def _check_key_text(key: str) -> None:
    """Refuse a key with a character no Voyage key has, such as a space or a hidden mark copied along with it."""
    if not key or not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
        raise VoyageError(
            SETTINGS,
            "The Voyage key has a character no key has, such as a space or a hidden mark copied along with it. "
            "Copy the key again from https://dashboard.voyageai.com",
        )


# Talking to Voyage ---------------------------------------------------------------------------------------


WINDOW = 60.0
"""Voyage counts an account's tokens and requests over a minute."""
LONGEST_WAIT = 120.0
"""The longest one wait may be, whatever Voyage asks for."""
REFUSALS = 12
"""How many times in a row Voyage may refuse a request for the account's requests a minute before this gives up."""


@dataclass(frozen=True)
class Patience:
    """How long to keep trying when Voyage is busy, limits the account or can't be reached."""

    timeout: float
    """Seconds one request may take."""
    attempts: int
    """Tries for one request when Voyage fails or can't be reached."""
    deadline: float | None
    """Seconds all the waiting in one run may add up to. None waits as long as the account's limits need."""


PATIENT = Patience(timeout=120.0, attempts=8, deadline=None)
"""For setup and refresh. They wait out a busy Voyage and an account's low limits."""
QUICK = Patience(timeout=8.0, attempts=2, deadline=8.0)
"""For forecasts. When Voyage can't be reached, a forecast says so within seconds, not minutes."""


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect. The key goes to the one address it was meant for, or nowhere."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _retry_after(headers) -> float | None:
    """The wait Voyage asks for in its Retry-After header, in seconds: a number, or a date to wait until."""
    value = headers.get("Retry-After") if headers is not None else None
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return min(max(seconds, 0.0), LONGEST_WAIT) if math.isfinite(seconds) else None


def _post(body: dict, key: str, timeout: float = 120) -> dict:
    _check_key_text(key)
    request = urllib.request.Request(
        api_base() + "/embeddings",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.build_opener(_NoRedirects).open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            to = urllib.parse.urlparse(error.headers.get("Location") or "").hostname
            raise VoyageError(
                SETTINGS,
                f"Voyage's address answered with a redirect{f' to {to}' if to else ''}. yardstick never follows "
                "one, so your key isn't sent anywhere else. Check VOYAGE_API_BASE.",
            ) from None
        try:
            detail = json.loads(error.read()).get("detail") or error.reason
        except Exception:
            detail = error.reason
        raise VoyageError(error.code, str(detail), _retry_after(error.headers)) from None
    except http.client.HTTPException as error:  # the connection broke while Voyage answered
        raise VoyageError(0, f"the connection to Voyage broke ({type(error).__name__})") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise VoyageError(0, f"couldn't reach Voyage ({error})") from None
    try:
        reply = json.loads(raw)
    except ValueError:
        reply = None
    if not isinstance(reply, dict):
        raise VoyageError(
            200,
            "its answer wasn't the JSON Voyage sends. Something between this computer and Voyage, such as a "
            "network filter, may have changed it.",
        )
    return reply


def _limit(error: VoyageError) -> str | None:
    """Which of the account's limits Voyage says a 429 is for: 'tokens' or 'requests' a minute. None when unclear.

    Voyage says which limit it hit, as in "You have exceeded your TPM rate limit". A message that names
    both limits, or neither, doesn't say which one it was.
    """
    tokens = re.search(r"\bTPM\b|tokens per minute", error.detail, re.IGNORECASE) is not None
    requests = re.search(r"\bRPM\b|requests per minute", error.detail, re.IGNORECASE) is not None
    if tokens != requests:
        return "tokens" if tokens else "requests"
    return None


def _tokens_limit(error: VoyageError) -> bool:
    """Whether Voyage refused a request for the account's limit on tokens a minute, not on requests."""
    return error.status == 429 and _limit(error) == "tokens"


SLOWER = "Accounts without a payment method get low limits. Adding one raises them, and your free tokens still apply."


class _TooBig(Exception):
    """Voyage refused a request for tokens a minute even after the account's whole minute came back."""

    def __init__(self, error: VoyageError) -> None:
        super().__init__(str(error))
        self.error = error


@dataclass
class _Room:
    """What one run has learned about how many tokens a request may hold, and how long it has waited.

    A request starts at Voyage's own limit for one request. An account without a payment method may allow
    fewer tokens a minute than that. When Voyage refuses a request for tokens a minute, this first waits
    for the account's minute to come back. Only a request that is refused again then is bigger than the
    account allows in a minute. Later requests stay smaller than it, but never smaller than the biggest
    request that got through.
    """

    cap: float
    patience: Patience
    fits: float = 0.0
    """The biggest request that got through, in tokens."""
    too_big: float = math.inf
    """The smallest request found bigger than the account allows in a minute."""
    rested: float = 0.0
    """Seconds waited since the last request that got through."""
    waited: float = 0.0
    """Seconds waited in this run, all told."""
    last: float | None = None
    """When the last request that got through was sent."""
    spent: list[tuple[float, float]] = field(default_factory=list)
    """When each request that got through in the last minute was sent, and its tokens."""
    explained: bool = False

    def tokens(self) -> float:
        """How many tokens the next request may hold."""
        if self.too_big == math.inf:
            return self.cap
        if not self.fits:
            return self.too_big / 2
        if self.too_big > self.fits * 1.1:
            return (self.fits + self.too_big) / 2  # between what got through and what didn't, to find the limit
        return self.fits

    def rested_fully(self) -> bool:
        """Whether the account's whole minute has come back since this run last spent tokens."""
        if self.rested >= WINDOW:
            return True
        return self.last is not None and _now() - self.last >= WINDOW

    def got_through(self, tokens: float) -> None:
        self.fits = max(self.fits, tokens)
        self.rested = 0.0
        self.last = now = _now()
        self.spent = [(when, n) for when, n in self.spent if now - when < WINDOW] + [(now, tokens)]

    def found_too_big(self, tokens: float) -> None:
        self.too_big = min(self.too_big, tokens)
        if self.fits >= self.too_big:  # what got through before doesn't now: the account's limit went down
            self.fits = 0.0

    def wait(self, seconds: float, on_wait: Callable[[str], None] | None, message: str | None) -> bool:
        """Wait, unless that would go past the run's deadline. False when it would."""
        if self.patience.deadline is not None and self.waited + seconds > self.patience.deadline:
            return False
        if on_wait and message:
            on_wait(message)
        time.sleep(seconds)
        self.waited += seconds
        self.rested += seconds
        return True

    def pace(self, tokens: float, on_wait: Callable[[str], None] | None) -> None:
        """Before a request, wait for the account's minute to come back when the request surely won't fit now.

        It only waits once the run knows the account allows fewer tokens a minute than a full request. Then
        it sends about one request a minute, instead of one that's refused and one that gets through.
        """
        if self.too_big == math.inf:
            return
        now = _now()
        recent = [(when, n) for when, n in self.spent if now - when < WINDOW]
        if not recent or sum(n for _, n in recent) + tokens < self.too_big:
            return
        message = None
        if not self.explained:
            self.explained = True
            message = (
                "Voyage allows this account fewer tokens a minute than a full request holds, so this sends "
                f"about one request a minute. {SLOWER}"
            )
        seconds = max(recent[-1][0] + WINDOW - now, 0.0)
        if not self.wait(seconds, on_wait, message):
            raise VoyageError(429, "this account's tokens a minute are used up for now")
        self.spent = []
        self.rested = max(self.rested, WINDOW)


def _send(body: dict, tokens: float, key: str, room: _Room, on_wait: Callable[[str], None] | None) -> dict:
    """Send one request. Wait and try again while Voyage is busy, limits the account or can't be reached.

    For a limit on requests a minute, it waits, as long as Voyage asks or longer each time, up to a minute.
    For a limit on tokens a minute, it waits for the account's minute to come back. Raises _TooBig when the
    request is refused again after that, so the caller can send smaller ones. Raises VoyageError when
    trying again can't help, or when the run's patience is used up.
    """
    failures = refusals = 0
    while True:
        room.pace(tokens, on_wait)
        try:
            reply = _post(body, key, room.patience.timeout)
        except VoyageError as error:
            if error.status == 429:
                limit = _limit(error)
                if limit != "requests" and room.rested_fully():
                    raise _TooBig(error) from None
                refusals += 1
                if limit == "requests" and refusals >= REFUSALS:
                    raise
                if error.retry_after is not None:
                    seconds = error.retry_after
                elif limit == "tokens":
                    seconds = WINDOW
                else:
                    seconds = min(WINDOW, 2.0 ** (refusals - 1))
                seconds = max(seconds, 1.0)
                if limit == "tokens":
                    message = (
                        f"Voyage limits this account's tokens a minute, so this waits {seconds:.0f} seconds for "
                        f"them to come back. {SLOWER}"
                    )
                else:
                    message = f"Voyage is limiting requests, so this waits {seconds:.0f} seconds. {SLOWER}"
            elif error.status >= 500 or error.status == 0:
                failures += 1
                if failures >= room.patience.attempts:
                    raise
                seconds = min(WINDOW, 2.0 ** (failures - 1))
                message = f"Voyage didn't answer ({error.detail}). Trying again in {seconds:.0f} seconds."
            else:
                raise
            if not room.wait(seconds, on_wait, message):
                raise
            continue
        room.got_through(tokens)
        return reply


def check_key(key: str, model: str) -> None:
    """Make one tiny request to confirm the key works. Raises VoyageError when it doesn't."""
    body = {"input": ["yardstick-predict key check"], "model": model, "output_dimension": DIMENSIONS}
    try:
        reply = _send(body, 8, key, _Room(cap=math.inf, patience=PATIENT), None)
    except _TooBig as refused:
        raise refused.error from None
    record_usage(key, model, _tokens_used(reply))


def _tokens_used(reply: dict) -> int:
    usage = reply.get("usage")
    tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    return int(tokens) if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and tokens > 0 else 0


def _vectors(reply: dict, expected: int) -> list[list[float]]:
    """The vectors in Voyage's answer, in the order of the texts. Raises VoyageError for an answer that's wrong."""
    data = reply.get("data")
    items = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    if len(items) != expected:
        raise VoyageError(200, f"expected {expected} embeddings and got {len(items)}")
    items.sort(key=lambda item: item.get("index", 0) if isinstance(item.get("index"), int) else 0)
    vectors = []
    for item in items:
        vector = item.get("embedding")
        if not isinstance(vector, list) or len(vector) != DIMENSIONS:
            size = len(vector) if isinstance(vector, list) else 0
            raise VoyageError(200, f"it sent a vector of {size} numbers, not {DIMENSIONS}")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vector):
            raise VoyageError(200, "it sent a vector with something other than numbers in it")
        vectors.append(vector)
    return vectors


def embed(
    texts: list[str],
    model: str,
    key: str,
    counts: list[int] | None = None,
    on_wait: Callable[[str], None] | None = None,
    patience: Patience = PATIENT,
) -> Iterator[tuple[list[int], list[list[float]]]]:
    """Embed texts in as few requests as Voyage allows. Yields each request's positions and vectors.

    Without `counts`, each text counts as its most_tokens, so nothing waits for the tokenizer to load. Each
    request's vectors are kept in the cache as they arrive, so a vector paid for is never lost, even when
    a later request fails.
    """
    if counts is None:
        counts = [most_tokens(text) for text in texts]
    if len(counts) != len(texts):
        raise ValueError("embed needs one count for each text")
    limit = MODELS[model].tokens_per_request if model in MODELS else 120_000
    room = _Room(cap=limit * 0.95, patience=patience)
    waiting = deque(range(len(texts)))
    refused_alone: dict[int, int] = {}
    while waiting:
        batch: list[int] = []
        size = 0
        while waiting and len(batch) < MAX_TEXTS_PER_REQUEST:
            position = waiting[0]
            if batch and (size + counts[position] > room.tokens() or position in refused_alone):
                break
            batch.append(waiting.popleft())
            size += counts[position]
            if position in refused_alone:
                break  # a text refused on its own goes on its own
        try:
            vectors = _embed_batch([texts[i] for i in batch], size, model, key, room, on_wait)
        except _TooBig as refused:
            waiting.extendleft(reversed(batch))  # they go again first, in smaller requests or after a wait
            if len(batch) > 1:
                room.found_too_big(size)
                if on_wait:
                    on_wait(
                        "A request this size is more than Voyage allows this account in a minute, "
                        "so this sends smaller ones."
                    )
                continue
            # One text alone was refused after a whole minute. Another program may have used that minute,
            # so it gets two more minutes before this gives up on it.
            refused_alone[batch[0]] = refused_alone.get(batch[0], 0) + 1
            message = f"Voyage still limits this account's tokens a minute, so this waits {WINDOW:.0f} seconds."
            if refused_alone[batch[0]] >= 3 or not room.wait(WINDOW, on_wait, message):
                raise VoyageError(
                    429,
                    f"{refused.error.detail} One text alone is more than this account allows in a minute.",
                ) from None
            continue
        yield batch, vectors


def _embed_batch(
    batch: list[str],
    size: float,
    model: str,
    key: str,
    room: _Room,
    on_wait: Callable[[str], None] | None,
) -> list[list[float]]:
    """Embed one request's texts, keep their vectors in the cache, and count what they cost."""
    body = {"input": batch, "model": model, "output_dimension": DIMENSIONS, "truncation": True}
    reply = _send(body, size, key, room, on_wait)
    try:
        vectors = _vectors(reply, len(batch))
    except VoyageError:
        record_usage(key, model, _tokens_used(reply))  # an answer that's wrong may still have cost tokens
        raise
    remember(model, dict(zip(batch, vectors)))  # first, so even if counting fails, nothing is paid for twice
    record_usage(key, model, _tokens_used(reply))
    return vectors


# The cache of vectors already paid for --------------------------------------------------------------------


def text_hash(text: str) -> str:
    """A one-way fingerprint of a text. The cache and the dataset keep it in place of the text itself."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _open_cache() -> sqlite3.Connection | None:
    """The cache, shared by every yardstick process on this computer. None when it can't be opened.

    It lives in the private yardstick folder and holds, for each text, only the hash of the text and its
    vector, by embedding model and size. Several processes can use it at once, such as a refresh and the
    status line's forecasts: each change is a short transaction, and a process waits its turn.
    """
    try:
        db = sqlite3.connect(cache_folder() / "embeddings.sqlite", timeout=30)
    except (OSError, sqlite3.Error):
        return None
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS vectors "
            "(model TEXT, dimensions INTEGER, hash TEXT, vector BLOB, PRIMARY KEY (model, dimensions, hash))"
        )
    except sqlite3.Error:
        db.close()
        return None
    return db


def _chunks(values: list, size: int = 500) -> Iterator[list]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _lookup(model: str, texts: list[str], with_vectors: bool) -> dict[str, list[float] | None]:
    hashes = {text_hash(text): text for text in texts}
    found: dict[str, list[float] | None] = {}
    if not hashes:
        return found
    db = _open_cache()
    if db is None:
        return found
    try:
        column = "vector" if with_vectors else "NULL"
        for chunk in _chunks(list(hashes)):
            marks = ", ".join("?" for _ in chunk)
            rows = db.execute(
                f"SELECT hash, {column} FROM vectors WHERE model = ? AND dimensions = ? AND hash IN ({marks})",
                (model, DIMENSIONS, *chunk),
            )
            for digest, blob in rows:
                if not with_vectors:
                    found[hashes[digest]] = None
                elif isinstance(blob, bytes) and len(blob) == 4 * DIMENSIONS:
                    found[hashes[digest]] = array("f", blob).tolist()
    except sqlite3.Error:
        pass  # what was found so far still holds
    finally:
        db.close()
    return found


def cached(model: str, texts: list[str]) -> dict[str, list[float]]:
    """The vectors the cache already holds for these texts, by text."""
    return _lookup(model, texts, with_vectors=True)  # type: ignore[return-value]


def in_cache(model: str, texts: list[str]) -> set[str]:
    """Which of these texts the cache already holds a vector for."""
    return set(_lookup(model, texts, with_vectors=False))


def remember(model: str, vectors: dict[str, list[float]]) -> None:
    """Keep vectors in the cache, by the hash of their text. A cache that can't be written is skipped."""
    if not vectors:
        return
    db = _open_cache()
    if db is None:
        return
    try:
        rows = [(model, DIMENSIONS, text_hash(text), array("f", vector).tobytes()) for text, vector in vectors.items()]
        db.executemany("INSERT OR IGNORE INTO vectors VALUES (?, ?, ?, ?)", rows)
        db.commit()
    except (sqlite3.Error, OverflowError, TypeError):
        pass
    finally:
        db.close()


# What it costs, counting the free tokens ------------------------------------------------------------------


SET_ASIDE = "set_aside"
"""A key in the ledger that no key's fingerprint can be: when a damaged ledger was moved aside, and where to."""


def _ledger_path():
    return store.home() / "voyage-usage.json"


@contextmanager
def _ledger_held() -> Iterator[None]:
    """Keep other processes out of the ledger until this one is done with it.

    Two forecasts can record usage at the same moment. Without this, one could read the ledger while
    the other is still writing it, and the count it saves would undo the other's.
    """
    with locks.held(store.private_home() / "voyage-usage.lock"):
        yield


def _well_formed(ledger: object) -> bool:
    if not isinstance(ledger, dict):
        return False
    for name, entry in ledger.items():
        if not isinstance(entry, dict):
            return False
        if name == SET_ASIDE:
            continue
        used = entry.get("used", {})
        if not isinstance(used, dict) or not all(
            isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0 for n in used.values()
        ):
            return False
        if entry.get("free_tokens", "unknown") not in ("new", "used", "unknown"):
            return False
    return True


def _ledger() -> dict:
    """The ledger of how many tokens each key used. Call it while holding the ledger.

    A ledger that is damaged or can't be read is moved aside, and a fresh one notes that it was. It can't
    say how many free tokens are left, or whether the account was new. Starting over as if nothing were
    known would ask again whether the account is new, and could promise all 200 million free tokens a
    second time. So after a damaged ledger, every key counts as 'unknown', which assumes no free tokens,
    and yardstick doesn't ask again.
    """
    path = _ledger_path()
    try:
        ledger = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return _set_aside(path)
    return ledger if _well_formed(ledger) else _set_aside(path)


def _set_aside(path) -> dict:
    aside = path.with_name(f"voyage-usage.damaged-{datetime.now():%Y%m%d-%H%M%S-%f}.json")
    try:
        os.replace(path, aside)
    except OSError:
        aside = None
    fresh = {SET_ASIDE: {"at": time.time(), "file": aside.name if aside else None}}
    try:
        _save_ledger(fresh)
    except OSError:
        pass  # each later read finds the damage again, and comes to the same answer
    return fresh


def _save_ledger(ledger: dict) -> None:
    store.private_home()
    files.write_text(_ledger_path(), json.dumps(ledger, indent=2) + "\n")


def _entry(ledger: dict, key: str) -> dict:
    """The key's entry, made when it has none. After a damaged ledger, a new entry is never asked about."""
    fresh: dict = {"free_tokens": "unknown", "used": {}, "first_seen": time.time()}
    if SET_ASIDE in ledger:
        fresh["asked"] = True
    return ledger.setdefault(fingerprint(key), fresh)


def account(key: str) -> dict:
    """What yardstick knows about this key's Voyage account. The key itself is never stored.

    It says 'asked' once the person was asked whether the account is new, or when they can't be asked
    usefully: after the ledger was damaged and moved aside, which its 'set_aside' entry says.
    """
    with _ledger_held():
        ledger = _ledger()
    entry = ledger.get(fingerprint(key))
    if entry is None:
        entry = {"free_tokens": "unknown", "asked": SET_ASIDE in ledger, "used": {}}
    elif SET_ASIDE in ledger and not entry.get("asked"):
        entry = {**entry, "asked": True}
    if SET_ASIDE in ledger:
        entry = {**entry, SET_ASIDE: ledger[SET_ASIDE]}
    return entry


def set_free_tokens(key: str, status: str) -> None:
    """Record whether the account had its free tokens when yardstick first used it: 'new', 'used' or 'unknown'."""
    with _ledger_held():
        ledger = _ledger()
        entry = _entry(ledger, key)
        entry["free_tokens"] = status
        entry["asked"] = True
        _save_ledger(ledger)


def record_usage(key: str, model: str, tokens: int) -> None:
    """Add tokens Voyage counted to the key's usage. Raises VoyageError when the count can't be saved.

    A count that couldn't be saved would make the free tokens look like more than are left, so it stops
    the run instead. What was embedded so far is already in the cache.
    """
    with _ledger_held():
        ledger = _ledger()
        entry = _entry(ledger, key)
        entry.setdefault("used", {})
        entry["used"][model] = entry["used"].get(model, 0) + tokens
        try:
            _save_ledger(ledger)
        except OSError as error:
            raise VoyageError(
                SETTINGS, f"Couldn't save the count of Voyage tokens in {_ledger_path()} ({error.strerror or error})."
            ) from None


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
