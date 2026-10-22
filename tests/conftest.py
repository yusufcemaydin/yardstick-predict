from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

KEY = "test-key-never-real"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Every test gets its own yardstick folder, no real Voyage key and no access to the password store."""
    folder = tmp_path / "yardstick-home"
    monkeypatch.setenv("YARDSTICK_HOME", str(folder))
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_BASE", raising=False)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    from yardstick_predict import embeddings

    # Tests never download Voyage's tokenizer. One token per word is close enough here.
    monkeypatch.setattr(embeddings, "count_tokens", lambda texts, model: ([len(t.split()) + 1 for t in texts], True))
    return folder


def fake_vector(text: str) -> list[float]:
    """Similar texts get similar vectors: each word adds to a few fixed positions."""
    vector = np.zeros(256)
    for word in text.lower().split():
        digest = hashlib.sha256(word.encode()).digest()
        for i in range(4):
            vector[digest[i]] += 1.0
    norm = np.linalg.norm(vector) or 1.0
    return (vector / norm).round(6).tolist()


class FakeVoyage(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []
        self.fail_next: list[tuple[int, str]] = []
        self.most_tokens: int | None = None
        """When set, a request with more tokens than this gets a rate-limit error, like a low-limit account."""

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
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
        tokens = sum(len(t.split()) + 1 for t in texts)
        if server.most_tokens is not None and tokens > server.most_tokens:
            return self._send(429, {"detail": "You have exceeded your TPM rate limit."})
        data = [{"object": "embedding", "embedding": fake_vector(t), "index": i} for i, t in enumerate(texts)]
        return self._send(
            200, {"object": "list", "data": data, "model": body["model"], "usage": {"total_tokens": tokens}}
        )


@pytest.fixture
def voyage(monkeypatch):
    """A stand-in for Voyage's embeddings API, running on this computer only."""
    server = FakeVoyage()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("VOYAGE_API_BASE", server.base)
    from yardstick_predict import embeddings

    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: None)
    yield server
    server.shutdown()
