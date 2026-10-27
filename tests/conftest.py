from __future__ import annotations

import hashlib
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from yardstick_predict import embeddings as _embeddings
from yardstick_predict import features as _features

KEY = "test-key-never-real"
REAL_COUNT_TOKENS = _embeddings.count_tokens
"""The real token counter. Every test gets a stand-in that never downloads the tokenizer."""


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test gets its own yardstick folder, no real Voyage key and no access to the password store."""
    folder = tmp_path / "yardstick-home"
    monkeypatch.setenv("YARDSTICK_HOME", str(folder))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_BASE", raising=False)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("YARDSTICK_NO_UPDATE_CHECK", "1")  # the home screen never asks PyPI in a test
    monkeypatch.delenv("YARDSTICK_NUMBERED_MENUS", raising=False)
    # A terminal of 80 columns, styled only where a test asks for it, whatever the computer running the tests has.
    monkeypatch.setenv("COLUMNS", "80")
    for name in ("NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)
    from yardstick_predict import embeddings, ui

    ui.numbered.cache_clear()  # each test asks for numbered menus, or not, for itself

    # Tests never download Voyage's tokenizer. One token per word is close enough here.
    monkeypatch.setattr(embeddings, "count_tokens", lambda texts, model: ([len(t.split()) + 1 for t in texts], True))
    monkeypatch.setattr(embeddings, "_tokenizers", {})
    monkeypatch.setattr(embeddings, "_tokenizer_failed_at", None)
    _features.skill_descriptions.cache_clear()  # each test may have other skills
    yield folder
    _features.skill_descriptions.cache_clear()


@pytest.fixture
def wide(monkeypatch):
    """A terminal wide enough for the paths of a test's own folders, which are far longer than a person's."""
    monkeypatch.setenv("COLUMNS", "400")


def fake_vector(text: str) -> list[float]:
    """Similar texts get similar vectors: each word adds to a few fixed positions."""
    vector = np.zeros(256)
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode()).digest()
        for i in range(4):
            vector[digest[i]] += 1.0
    norm = np.linalg.norm(vector) or 1.0
    return (vector / norm).round(6).tolist()


def fake_tokens(texts: list[str]) -> int:
    """How many tokens the stand-in Voyage counts: one per word, and one more for each text."""
    return sum(len(t.split()) + 1 for t in texts)


class VirtualClock:
    """A clock that a wait moves forward at once, so a test can pass minutes of rate limits in an instant."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(float(seconds), 0.0)


class FakeVoyage(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []
        self.answers: list[int] = []
        """The status of each answer, in the order of the requests."""
        self.fail_next: list[tuple[int, str]] = []
        self.most_tokens: int | None = None
        """When set, a request with more tokens than this gets a rate-limit error, like a low-limit account."""
        self.clock = VirtualClock()
        self.tokens_a_minute: int | None = None
        """When set, the account takes at most this many tokens in any minute, as Voyage's limits work."""
        self.requests_a_minute: int | None = None
        """When set, the account takes at most this many requests in any minute."""
        self.retry_after = False
        """When True, a refusal for the account's limits says how long to wait, in a Retry-After header."""
        self.accepted: list[tuple[float, int]] = []
        """When each request that got through came, by the clock, and its tokens."""

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def limited(self, tokens: int) -> tuple[str, float] | None:
        """Why the account's limits refuse a request now, and how long until they wouldn't. None when they allow it."""
        now = self.clock.time()
        recent = [(when, n) for when, n in self.accepted if now - when < 60]
        if self.requests_a_minute is not None and len(recent) >= self.requests_a_minute:
            return "You have exceeded your RPM rate limit.", recent[0][0] + 60 - now
        if self.tokens_a_minute is not None and sum(n for _, n in recent) + tokens > self.tokens_a_minute:
            if tokens > self.tokens_a_minute:
                return "You have exceeded your TPM rate limit.", 60.0
            used = sum(n for _, n in recent)
            for when, n in recent:  # the oldest go first: wait until enough of them have
                used -= n
                if used + tokens <= self.tokens_a_minute:
                    return "You have exceeded your TPM rate limit.", when + 60 - now
        return None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, body: dict, wait: float | None = None):
        server: FakeVoyage = self.server  # type: ignore[assignment]
        server.answers.append(status)
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if wait is not None and server.retry_after:
            self.send_header("Retry-After", str(max(math.ceil(wait), 1)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        server: FakeVoyage = self.server  # type: ignore[assignment]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(body)
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self._send(401, {"detail": "Provided API key is invalid."})
        if server.fail_next:
            status, detail = server.fail_next.pop(0)
            return self._send(status, {"detail": detail})
        texts = body["input"]
        tokens = fake_tokens(texts)
        if server.most_tokens is not None and tokens > server.most_tokens:
            return self._send(429, {"detail": "You have exceeded your TPM rate limit."})
        refused = server.limited(tokens)
        if refused:
            return self._send(429, {"detail": refused[0]}, wait=refused[1])
        server.accepted.append((server.clock.time(), tokens))
        data = [{"object": "embedding", "embedding": fake_vector(t), "index": i} for i, t in enumerate(texts)]
        return self._send(
            200, {"object": "list", "data": data, "model": body["model"], "usage": {"total_tokens": tokens}}
        )


@pytest.fixture
def voyage(monkeypatch):
    """A stand-in for Voyage's embeddings API, running on this computer only, with a clock waits move at once."""
    server = FakeVoyage()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    monkeypatch.setenv("VOYAGE_API_BASE", server.base)
    from yardstick_predict import embeddings

    monkeypatch.setattr(embeddings.time, "sleep", server.clock.sleep)
    monkeypatch.setattr(embeddings, "_now", server.clock.time)
    yield server
    server.shutdown()
