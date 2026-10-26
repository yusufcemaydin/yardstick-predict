from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from conftest import KEY, REAL_COUNT_TOKENS, fake_tokens, fake_vector

from yardstick_predict import embeddings, forecast, pipeline
from yardstick_predict.archives import Task
from yardstick_predict.store import Predictor

START = 1_780_000_000.0


def test_texts_are_sent_in_as_few_requests_as_voyage_allows(voyage):
    texts = [f"task number {i}" for i in range(2500)]
    found = list(embeddings.embed(texts, "voyage-4", KEY, counts=[100] * 2500))
    assert [len(positions) for positions, _ in found] == [1000, 1000, 500]
    assert sorted(p for positions, _ in found for p in positions) == list(range(2500))
    assert all(len(vector) == 256 for _, vectors in found for vector in vectors)
    assert all(r["output_dimension"] == 256 and r["model"] == "voyage-4" for r in voyage.requests)


def test_a_request_never_goes_over_voyages_token_limit(voyage):
    texts = ["a"] * 10
    found = list(embeddings.embed(texts, "voyage-4-large", KEY, counts=[50_000] * 10))
    assert all(len(positions) * 50_000 <= 120_000 for positions, _ in found)


def test_it_waits_and_tries_again_when_voyage_is_busy(voyage):
    voyage.fail_next = [(429, "Rate limit"), (503, "Busy")]
    waits = []
    found = list(embeddings.embed(["hello"], "voyage-4", KEY, on_wait=waits.append))
    assert len(found) == 1 and len(waits) == 2
    assert "payment method" in waits[0]


def test_a_low_limit_account_still_finishes_with_smaller_requests(voyage):
    voyage.most_tokens = 30
    texts = [f"task number {i} with a few words" for i in range(40)]
    waits = []
    found = list(embeddings.embed(texts, "voyage-4", KEY, counts=[8] * 40, on_wait=waits.append))
    assert sorted(p for positions, _ in found for p in positions) == list(range(40))
    assert any("smaller" in w for w in waits)
    accepted = [r for r in voyage.requests if sum(len(t.split()) + 1 for t in r["input"]) <= 30]
    assert sum(len(r["input"]) for r in accepted) == 40  # every text went through once, in a small request


def test_what_a_split_batch_embedded_is_kept_when_a_later_part_fails(voyage):
    voyage.most_tokens = 30
    texts = ["short one", "short two", "word " * 40, "word " * 40]  # the long texts never fit, even alone
    found = []
    with pytest.raises(embeddings.VoyageError):
        for positions, vectors in embeddings.embed(texts, "voyage-4", KEY):
            found.append((positions, vectors))
    assert found == [([0, 1], [fake_vector("short one"), fake_vector("short two")])]  # paid for, so not lost


def test_a_tokens_limit_waits_for_the_minute_before_it_sends_smaller_requests(voyage, monkeypatch):
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: (slept.append(seconds), voyage.clock.sleep(seconds)))
    voyage.tokens_a_minute = 10  # one text of 8 tokens fits in a minute, two don't
    texts = [f"task number {i} with a few words" for i in range(3)]
    waits = []
    found = list(embeddings.embed(texts, "voyage-4", KEY, counts=[8] * 3, on_wait=waits.append))
    assert sorted(p for positions, _ in found for p in positions) == [0, 1, 2]
    # The first refusal may only mean the minute was used up, so the same batch goes again after a minute.
    # Refused again, it is more than the account allows in a minute, and only then goes in smaller requests.
    assert [len(r["input"]) for r in voyage.requests][:3] == [3, 3, 1]
    assert slept[0] == 60.0 and "minute" in waits[0]


def test_a_low_tokens_a_minute_account_embeds_in_about_the_ideal_time(voyage):
    # An account without a payment method: 10,000 tokens and 3 requests a minute. 102,000 tokens to embed.
    voyage.tokens_a_minute, voyage.requests_a_minute = 10_000, 3
    texts = [" ".join(["word"] * 49 + [str(i)]) for i in range(2000)]
    counts = [fake_tokens([text]) for text in texts]
    assert sum(counts) == 102_000
    start = voyage.clock.now
    found = list(embeddings.embed(texts, "voyage-4", KEY, counts=counts))
    minutes = (voyage.clock.now - start) / 60
    assert sorted(p for positions, _ in found for p in positions) == list(range(2000))
    assert sum(len(r["input"]) for r, status in zip(voyage.requests, voyage.answers) if status == 200) == 2000
    # The ideal is about 11 requests in about 10 minutes. Before, this took over 6,000 requests and 7 hours.
    assert len(voyage.requests) <= 25 and minutes <= 13


def test_a_requests_a_minute_limit_waits_and_never_splits(voyage):
    voyage.requests_a_minute = 3
    texts = [f"text {i}" for i in range(5000)]
    start = voyage.clock.now
    found = list(embeddings.embed(texts, "voyage-4", KEY, counts=[3] * 5000))
    assert sorted(p for positions, _ in found for p in positions) == list(range(5000))
    assert all(len(r["input"]) == 1000 for r in voyage.requests)  # smaller requests would only need more of them
    assert len(voyage.requests) <= 12 and voyage.clock.now - start <= 120


def test_later_requests_never_shrink_below_the_biggest_one_that_got_through(voyage):
    voyage.tokens_a_minute = 100
    texts = [f"word {i}" for i in range(200)]
    list(embeddings.embed(texts, "voyage-4", KEY, counts=[3] * 200))
    sizes = [fake_tokens(r["input"]) for r, status in zip(voyage.requests, voyage.answers) if status == 200]
    assert sum(sizes) == 600
    assert all(size >= sizes[0] for size in sizes[:-1])  # the last request holds whatever was left
    assert sizes[-2] > 90  # it learned close to the account's 100 tokens a minute


def test_voyage_s_retry_after_sets_the_wait(voyage, monkeypatch):
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: (slept.append(seconds), voyage.clock.sleep(seconds)))
    monkeypatch.setattr(embeddings, "MAX_TEXTS_PER_REQUEST", 1)
    voyage.requests_a_minute, voyage.retry_after = 1, True
    list(embeddings.embed(["one", "two", "three"], "voyage-4", KEY))
    assert slept == [60.0, 60.0] and voyage.answers == [200, 429, 200, 429, 200]


def test_a_requests_limit_waits_with_the_same_batch_and_a_tokens_limit_waits_a_minute(voyage, monkeypatch):
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", slept.append)
    voyage.fail_next = [(429, "You have exceeded your RPM rate limit.")]
    list(embeddings.embed(["first text", "second text"], "voyage-4", KEY))
    assert [r["input"] for r in voyage.requests] == [["first text", "second text"]] * 2  # the same batch again
    voyage.fail_next = [(429, "You have exceeded your TPM rate limit.")]
    list(embeddings.embed(["one text"], "voyage-4", KEY))
    assert [r["input"] for r in voyage.requests[2:]] == [["one text"]] * 2
    assert slept == [1.0, 60.0]


def test_server_errors_get_the_same_tries_for_one_text_or_several(voyage):
    for texts in (["one"], ["one", "two"]):
        voyage.requests.clear()
        voyage.fail_next = [(503, "Service Unavailable")] * 4
        found = list(embeddings.embed(texts, "voyage-4", KEY))
        assert [p for positions, _ in found for p in positions] == list(range(len(texts)))
        assert len(voyage.requests) == 5


def _closed_port() -> int:
    with socket.socket() as spare:
        spare.bind(("127.0.0.1", 0))
        return spare.getsockname()[1]


def test_a_forecast_gives_up_within_seconds_when_voyage_cant_be_reached(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_BASE", f"http://127.0.0.1:{_closed_port()}/v1")
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", slept.append)
    with pytest.raises(embeddings.VoyageError) as caught:
        list(embeddings.embed(["hello"], "voyage-4", KEY, patience=embeddings.QUICK))
    assert caught.value.status == 0 and sum(slept) <= embeddings.QUICK.deadline

    # A server that never answers holds each try only as long as the short timeout.
    timeouts = []

    def silent(body, key, timeout=120):
        timeouts.append(timeout)
        raise embeddings.VoyageError(0, "couldn't reach Voyage (timed out)")

    monkeypatch.setattr(embeddings, "_post", silent)
    with pytest.raises(embeddings.VoyageError):
        list(embeddings.embed(["hello"], "voyage-4", KEY, patience=embeddings.QUICK))
    assert timeouts == [embeddings.QUICK.timeout] * embeddings.QUICK.attempts
    worst = embeddings.QUICK.timeout * embeddings.QUICK.attempts + embeddings.QUICK.deadline
    assert worst <= 30  # seconds, not minutes


def test_a_forecast_never_waits_out_an_accounts_rate_limit(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    voyage.fail_next = [(429, "You have exceeded your TPM rate limit.")] * 10
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    start = voyage.clock.now
    with pytest.raises(forecast.RequestError) as caught:
        forecast._embed(predictor, [(0, "prompt", "fix the login page")])
    assert caught.value.code == "embedding_failed" and voyage.clock.now - start <= embeddings.QUICK.deadline


def test_embedding_without_counts_never_loads_the_tokenizer(voyage, monkeypatch):
    def fail(*args):
        raise AssertionError("the tokenizer was loaded")

    monkeypatch.setattr(embeddings, "count_tokens", fail)
    monkeypatch.setattr(embeddings, "_load_tokenizer", fail)
    found = list(embeddings.embed(["fix the login page", "a\u00f1ade una p\u00e1gina"], "voyage-4", KEY))
    assert [p for positions, _ in found for p in positions] == [0, 1]
    assert embeddings.most_tokens("a\u00f1ade") == len("a\u00f1ade".encode()) + 2  # never fewer than the real tokens


class _Tokenizer:
    def encode_batch(self, texts):
        return [type("Encoding", (), {"ids": text.split()})() for text in texts]


def test_a_failed_tokenizer_download_is_remembered_in_the_process_and_on_disk(monkeypatch):
    monkeypatch.setattr(embeddings, "count_tokens", REAL_COUNT_TOKENS)
    tries = []

    def unreachable(model, revision):
        tries.append(model)
        raise OSError("no connection to Hugging Face")

    monkeypatch.setattr(embeddings, "_load_tokenizer", unreachable)
    assert embeddings.count_tokens(["one two"], "voyage-4") == ([4], False)  # an estimate
    assert embeddings.count_tokens(["one two"], "voyage-4-lite")[1] is False
    assert tries == ["voyage-4"]  # the second model didn't wait out the same failure again
    monkeypatch.setattr(embeddings, "_tokenizer_failed_at", None)  # as in another process
    assert embeddings.count_tokens(["one"], "voyage-4-large")[1] is False and tries == ["voyage-4"]

    # An hour later, the download is tried again. When it works, every model may try again.
    note = embeddings._tokenizer_note()
    note.write_text(json.dumps({"failed_at": time.time() - embeddings.TOKENIZER_PAUSE - 1}))
    monkeypatch.setattr(embeddings, "_load_tokenizer", lambda model, revision: (tries.append(model), _Tokenizer())[1])
    assert embeddings.count_tokens(["one two three"], "voyage-4") == ([3], True)
    assert tries == ["voyage-4", "voyage-4"] and not note.exists()


def test_a_wrong_key_is_refused_at_once(voyage):
    with pytest.raises(embeddings.VoyageError) as caught:
        embeddings.check_key("wrong-key", "voyage-4")
    assert caught.value.status == 401
    assert len(voyage.requests) == 1


def test_a_key_with_a_hidden_character_is_refused_before_anything_is_sent(voyage):
    for pasted in (KEY + "\u200b", "pa-key with a space", "pa-\u00e9t\u00e9"):
        with pytest.raises(embeddings.VoyageError) as caught:
            embeddings.check_key(pasted, "voyage-4")
        assert caught.value.status == embeddings.SETTINGS and "Copy the key again" in caught.value.detail
    assert voyage.requests == []


def test_a_busy_voyage_doesnt_fail_the_key_check(voyage):
    voyage.fail_next = [(503, "Busy")]
    embeddings.check_key(KEY, "voyage-4")
    assert len(voyage.requests) == 2


@contextmanager
def _server(answer, monkeypatch):
    """A stand-in for Voyage whose every answer comes from `answer(handler)`."""

    class Handler(BaseHTTPRequestHandler):
        seen: list[dict] = []

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            Handler.seen.append(dict(self.headers))
            answer(self)

        do_GET = do_POST

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    monkeypatch.setenv("VOYAGE_API_BASE", f"http://127.0.0.1:{server.server_address[1]}/v1")
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: None)
    try:
        yield Handler.seen
    finally:
        server.shutdown()


def test_a_redirect_is_never_followed_with_the_key(voyage, monkeypatch):
    elsewhere = voyage.base

    def redirect(handler):
        handler.send_response(302)
        handler.send_header("Location", f"{elsewhere}/embeddings")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    with _server(redirect, monkeypatch) as seen:
        with pytest.raises(embeddings.VoyageError) as caught:
            embeddings.check_key(KEY, "voyage-4")
    assert caught.value.status == embeddings.SETTINGS and "redirect" in str(caught.value)
    assert len(seen) == 1 and voyage.requests == []  # the key went nowhere else, and nothing was sent again


def test_a_connection_that_breaks_while_voyage_answers_is_tried_again(monkeypatch):
    def broken(handler):
        handler.send_response(200)
        handler.send_header("Content-Length", "1000")
        handler.end_headers()
        handler.wfile.write(b'{"data": [')  # and then the connection closes
        handler.close_connection = True

    with _server(broken, monkeypatch) as seen:
        with pytest.raises(embeddings.VoyageError) as caught:
            list(embeddings.embed(["hello"], "voyage-4", KEY))
    assert caught.value.status == 0 and "broke" in caught.value.detail
    assert len(seen) == embeddings.PATIENT.attempts


def test_an_answer_that_isnt_json_is_a_clean_error(monkeypatch):
    def page(handler):
        body = b"<html>Please sign in to the network</html>"
        handler.send_response(200)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    with _server(page, monkeypatch) as seen:
        with pytest.raises(embeddings.VoyageError) as caught:
            list(embeddings.embed(["hello"], "voyage-4", KEY))
    assert caught.value.status == 200 and "JSON" in caught.value.detail and len(seen) == 1


def test_usage_is_counted_against_the_free_tokens(voyage):
    assert embeddings.free_tokens_left(KEY) is None
    embeddings.set_free_tokens(KEY, "new")
    list(embeddings.embed(["one two three"], "voyage-4", KEY))
    assert embeddings.free_tokens_left(KEY) == embeddings.FREE_TOKENS - 4
    embeddings.set_free_tokens(KEY, "used")
    assert embeddings.free_tokens_left(KEY) == 0


def test_usage_recorded_by_several_processes_at_once_is_never_lost(tmp_path):
    go = tmp_path / "go"
    script = (
        "import pathlib, sys, time\n"
        "from yardstick_predict import embeddings\n"
        "print('ready', flush=True)\n"
        "while not pathlib.Path(sys.argv[1]).exists():\n"
        "    time.sleep(0.001)\n"
        "for _ in range(200):\n"
        "    embeddings.record_usage('key', 'voyage-4', 1)\n"
    )
    command = [sys.executable, "-c", script, str(go)]
    workers = [subprocess.Popen(command, stdout=subprocess.PIPE, text=True) for _ in range(4)]
    try:
        ready = [worker.stdout.readline() for worker in workers]
    finally:
        go.touch()  # all four start counting at the same moment
        codes = [worker.wait(timeout=60) for worker in workers]
        for worker in workers:
            worker.stdout.close()
    assert ready == ["ready\n"] * 4 and codes == [0] * 4
    assert embeddings.account("key")["used"] == {"voyage-4": 800}


def test_the_usage_file_is_saved_in_one_step(monkeypatch):
    saved = []
    real = embeddings.files.write_text
    monkeypatch.setattr(embeddings.files, "write_text", lambda path, text: (saved.append(path), real(path, text)))
    embeddings.record_usage(KEY, "voyage-4", 10)
    assert saved == [embeddings._ledger_path()]


def test_a_damaged_usage_file_is_set_aside_and_never_promises_the_free_tokens_again(home):
    embeddings.set_free_tokens(KEY, "new")
    embeddings.record_usage(KEY, "voyage-4", 150_000_000)
    assert embeddings.free_tokens_left(KEY) == 50_000_000
    path = embeddings._ledger_path()
    damaged = path.read_text()[:40]  # a save cut off halfway, by an older version
    path.write_text(damaged)
    assert embeddings.free_tokens_left(KEY) is None  # unknown: costs assume no free tokens
    entry = embeddings.account(KEY)
    assert entry["free_tokens"] == "unknown" and entry["asked"] is True  # so setup doesn't ask again
    aside = list(home.glob("voyage-usage.damaged-*.json"))
    assert len(aside) == 1 and aside[0].read_text() == damaged and entry["set_aside"]["file"] == aside[0].name
    embeddings.record_usage(KEY, "voyage-4", 10)
    assert embeddings.account(KEY)["asked"] is True and embeddings.free_tokens_left(KEY) is None
    assert embeddings.account("another key")["asked"] is True  # a key seen first after the damage too
    for shape in ("[]", '{"abc": {"used": {"voyage-4": "lots"}}}'):
        path.write_text(shape)
        assert embeddings.free_tokens_left(KEY) is None and embeddings.account(KEY)["asked"] is True


def test_a_count_that_cant_be_saved_stops_the_run_but_keeps_what_was_paid_for(voyage, monkeypatch):
    def full(ledger):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(embeddings, "_save_ledger", full)
    with pytest.raises(embeddings.VoyageError) as caught:
        list(embeddings.embed(["fix the login page"], "voyage-4", KEY))
    assert caught.value.status == embeddings.SETTINGS and "count" in caught.value.detail
    assert list(embeddings.cached("voyage-4", ["fix the login page"])) == ["fix the login page"]


def test_cost_counts_free_tokens_and_list_price():
    cost = embeddings.Cost(tokens=300_000_000, exact=True, usd_per_million=0.06, free_left=200_000_000)
    assert cost.list_usd == pytest.approx(18.0)
    assert cost.your_usd == pytest.approx(6.0)
    assert embeddings.Cost(1000, True, 0.06, None).your_usd is None


def test_the_key_comes_from_the_environment_and_is_never_stored(monkeypatch, home):
    monkeypatch.setenv("VOYAGE_API_KEY", "  from-env  ")
    assert embeddings.find_key() == ("from-env", "the VOYAGE_API_KEY environment variable")
    assert embeddings.save_key("anything") is False  # the test password store refuses everything
    embeddings.record_usage("from-env", "voyage-4", 10)
    stored = "".join(p.read_text() for p in home.rglob("*") if p.is_file())
    assert "from-env" not in stored


def test_the_key_is_never_sent_unencrypted(monkeypatch):
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: pytest.fail("a wrong setting was retried"))
    for base in ("http://evil.example.com/v1", "file://localhost/tmp", "ftp://127.0.0.1/v1"):
        monkeypatch.setenv("VOYAGE_API_BASE", base)
        with pytest.raises(embeddings.VoyageError, match="https") as caught:
            embeddings.check_key(KEY, "voyage-4")
        assert str(caught.value).startswith("VOYAGE_API_BASE must start with https://")


def test_the_yardstick_folder_is_private(home):
    from yardstick_predict.store import private_home

    folder = private_home()
    if __import__("os").name == "posix":
        assert folder.stat().st_mode & 0o077 == 0


# The cache of vectors already paid for --------------------------------------------------------------------


def test_a_text_embedded_once_is_never_paid_for_again(voyage):
    list(embeddings.embed(["continue", "fix the login page"], "voyage-4", KEY))
    assert set(embeddings.cached("voyage-4", ["continue", "never seen"])) == {"continue"}
    assert embeddings.cached("voyage-4", ["continue"])["continue"] == pytest.approx(fake_vector("continue"), abs=1e-6)
    assert embeddings.cached("voyage-4-lite", ["continue"]) == {}  # another model makes other vectors


def test_the_cache_keeps_hashes_and_vectors_but_never_the_text(voyage, home):
    secret = "the plan for the secret project"
    list(embeddings.embed([secret], "voyage-4", KEY))
    stored = b"".join(path.read_bytes() for path in home.rglob("*") if path.is_file())
    assert secret.encode() not in stored and embeddings.text_hash(secret).encode() in stored
    assert (home / "cache" / "embeddings.sqlite").is_file()


def test_several_processes_can_use_the_cache_at_once(tmp_path):
    go = tmp_path / "go"
    script = (
        "import pathlib, sys, time\n"
        "from yardstick_predict import embeddings\n"
        "print('ready', flush=True)\n"
        "while not pathlib.Path(sys.argv[1]).exists():\n"
        "    time.sleep(0.001)\n"
        "worker = sys.argv[2]\n"
        "for n in range(100):\n"
        "    embeddings.remember('voyage-4', {f'{worker} text {n}': [0.5] * 256})\n"
        "    assert f'{worker} text {n}' in embeddings.cached('voyage-4', [f'{worker} text {n}'])\n"
    )
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(go), str(n)], stdout=subprocess.PIPE, text=True)
        for n in range(3)
    ]
    try:
        ready = [worker.stdout.readline() for worker in workers]
    finally:
        go.touch()
        codes = [worker.wait(timeout=60) for worker in workers]
        for worker in workers:
            worker.stdout.close()
    assert ready == ["ready\n"] * 3 and codes == [0] * 3
    texts = [f"{worker} text {n}" for worker in range(3) for n in range(100)]
    assert len(embeddings.cached("voyage-4", texts)) == 300


def _task(n: int, prompt: str, context: str = "", skills: str = "") -> Task:
    return Task(
        id=f"t{n}",
        archive="test",
        agent="claude-code",
        agent_version="1",
        session="s",
        project="demo",
        origin="person",
        started_at=START + n * 600,
        model="claude-opus-5-5",
        effort="high",
        prompt=prompt,
        context=context,
        context_tokens=0,
        seconds=30.0,
        input_tokens=100,
        cache_write_tokens=0,
        cache_read_tokens=1000,
        output_tokens=100,
        usd=0.01,
        calls=1,
        direct=True,
        stopped=False,
        idle=False,
        skills=skills,
        prior_skills=skills,  # the skills feature embeds the skills known before the task ran
    )


def _predictor(features: list[str]) -> Predictor:
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.config["features"] = features
    predictor.save_config()
    return predictor


def saved_by_an_older_version(predictor: Predictor) -> None:
    """Make the predictor's settings as a version before the feature store saved them: without the mark that its
    vectors and LLM answers moved there."""
    path = predictor.dir / "predictor.json"
    config = json.loads(path.read_text())
    config.pop("feature_store", None)
    path.write_text(json.dumps(config, indent=2) + "\n")


def test_the_cost_counts_each_text_once_and_none_that_the_cache_holds(voyage):
    tasks = [_task(n, prompt) for n, prompt in enumerate(["continue", "fix it", "continue", "continue"])]
    items = pipeline.embedding_items(tasks, ["prompt"])
    assert len(items) == 4
    embeddings.remember("voyage-4", {"fix it": fake_vector("fix it")})
    plan = pipeline.price(items, "voyage-4", None)
    assert plan.texts == ["continue"] and plan.counts == [2] and plan.cost.tokens == 2 and plan.tasks == 4


def test_the_newest_tasks_that_fit_pay_for_a_shared_text_once():
    newest_first = ["continue", "add a page", "continue", "write the docs"]
    tasks = [_task(n, prompt) for n, prompt in enumerate(reversed(newest_first))]
    plan = pipeline.price(pipeline.embedding_items(tasks, ["prompt"]), "voyage-4", None)
    assert [text for _, _, text in plan.items] == newest_first
    # continue costs 2, add a page 4 and write the docs 4. The second continue is already paid for.
    assert pipeline.newest_that_fit(plan, 6) == 3
    assert pipeline.newest_that_fit(plan, 10) == 4


def test_every_task_gets_its_vector_though_each_text_is_sent_once(voyage):
    predictor = _predictor(["prompt", "context"])
    prompts = ["continue", "fix it", "continue", "continue"]
    predictor.add_tasks([_task(n, prompt, context="we talked") for n, prompt in enumerate(prompts)])
    plan = pipeline.plan_embeddings(predictor, KEY)
    assert sorted(plan.texts) == ["continue", "fix it", "we talked"]
    pipeline.run_embeddings(predictor, plan, KEY)
    assert sorted(text for r in voyage.requests for text in r["input"]) == ["continue", "fix it", "we talked"]
    assert predictor.embedded("prompt") == {"t0", "t1", "t2", "t3"} and len(predictor.embedded("context")) == 4
    vectors = predictor.get_embeddings("prompt")  # by task, then by the hash of the text
    continued = embeddings.text_hash("continue")
    assert np.array_equal(vectors["t0"][continued], vectors["t3"][continued])
    assert predictor.embedded("prompt").hashes["t1"] == {embeddings.text_hash("fix it")}


def test_a_prompt_a_forecast_embedded_costs_nothing_at_the_next_refresh(voyage, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    predictor = _predictor(["prompt"])
    forecast._embed(predictor, [(0, "prompt", "fix the login page")])
    predictor.add_tasks([_task(1, "fix the login page")])
    plan = pipeline.plan_embeddings(predictor, KEY)
    assert plan.texts == [] and plan.cost.tokens == 0 and plan.tasks == 1
    pipeline.run_embeddings(predictor, plan, KEY)
    assert len(voyage.requests) == 1 and predictor.embedded("prompt") == {"t1"}


def test_a_forecast_takes_what_the_cache_holds_and_sends_only_new_texts(voyage, monkeypatch):
    predictor = _predictor(["prompt"])
    embeddings.remember("voyage-4", {"continue": fake_vector("continue")})
    vectors = forecast._embed(predictor, [(0, "prompt", "continue"), (1, "prompt", "continue")])
    assert voyage.requests == [] and np.allclose(vectors["prompt"][1], fake_vector("continue"), atol=1e-6)
    with pytest.raises(forecast.RequestError) as caught:  # a new text needs the key
        forecast._embed(predictor, [(0, "prompt", "continue"), (1, "prompt", "add tests")])
    assert caught.value.code == "no_embedding_key"
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    forecast._embed(predictor, [(0, "prompt", "continue"), (1, "prompt", "add tests")])
    assert [r["input"] for r in voyage.requests] == [["add tests"]]


def test_a_task_whose_text_changed_gets_a_new_vector_and_the_others_keep_theirs(voyage):
    predictor = _predictor(["prompt", "context"])
    tasks = [_task(n, f"task {n}", context="we talked") for n in range(3)]
    predictor.add_tasks(tasks)
    pipeline.run_embeddings(predictor, pipeline.plan_embeddings(predictor, KEY), KEY)
    predictor.replace_tasks([replace(tasks[1], context="we talked, then planned the page")])  # a fuller copy
    plan = pipeline.plan_embeddings(predictor, KEY)
    assert [(task_id, name) for task_id, name, _ in plan.items] == [("t1", "context")]
    pipeline.run_embeddings(predictor, plan, KEY)
    expected = np.array(fake_vector("we talked, then planned the page"), dtype=np.float32)
    found = predictor.get_embeddings("context")["t1"]
    assert np.allclose(found[embeddings.text_hash("we talked, then planned the page")], expected)
    assert len(found) == 2  # the vector of the text it had before stays in the feature store, unused
    assert pipeline.plan_embeddings(predictor, KEY).items == []
    row = pipeline.feature_rows(predictor, [task for task in predictor.tasks() if task.id == "t1"])[0]
    assert np.allclose(row.vectors["context"], expected)


def test_a_vector_whose_task_no_longer_has_a_text_for_it_is_no_longer_used(voyage):
    predictor = _predictor(["prompt", "context"])
    tasks = [_task(n, f"task {n}", context="we talked") for n in range(2)]
    predictor.add_tasks(tasks)
    pipeline.run_embeddings(predictor, pipeline.plan_embeddings(predictor, KEY), KEY)
    predictor.replace_tasks([replace(tasks[0], context="")])
    assert pipeline.plan_embeddings(predictor, KEY).items == []
    # The vector stays in the feature store, for any predictor whose task still has that text. Training here
    # finds it made from another text, so it leaves it out.
    assert predictor.embedded("context") == {"t0", "t1"}
    rows = pipeline.feature_rows(predictor, predictor.tasks())
    assert [sorted(row.vectors) for row in rows] == [["prompt"], ["context", "prompt"]]


def test_vectors_saved_before_hashes_were_kept_are_trusted_for_the_prompt_and_context(voyage, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no skills installed, so the skills text is the skill's name
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    predictor = _predictor(["prompt", "context", "skills"])
    tasks = [_task(n, f"task {n}", context="we talked", skills="tdd") for n in range(3)]
    predictor.add_tasks(tasks)
    predictor.close()
    # A dataset an older version made: its vectors are in it, with no hash of their text.
    saved_by_an_older_version(predictor)
    db = sqlite3.connect(predictor.dir / "dataset.sqlite")
    db.execute("CREATE TABLE embeddings (task_id, field, vector, PRIMARY KEY (task_id, field))")
    vector = np.zeros(256, dtype=np.float32).tobytes()
    db.executemany(
        "INSERT INTO embeddings VALUES (?, ?, ?)",
        [(task.id, name, vector) for task in tasks for name in ("prompt", "context", "skills")],
    )
    db.commit()
    db.close()
    upgraded = Predictor("default")
    assert upgraded.embedded("prompt").hashes["t0"] == {embeddings.text_hash("task 0")}
    plan = pipeline.plan_embeddings(upgraded, KEY)
    # The skills feature changes what it embeds, so its old vectors are made again. Its text is shared, so once.
    assert sorted({name for _, name, _ in plan.items}) == ["skills"] and plan.texts == ["tdd"]


def test_voyages_own_words_are_kept_but_never_a_code_a_terminal_acts_on():
    error = embeddings.VoyageError(400, "Bad \x1b]0;a title\x07request\x1b[2J")
    assert error.detail == "Bad request" and str(error) == "Voyage said 400: Bad request"
