from __future__ import annotations

import subprocess
import sys

import pytest
from conftest import KEY, fake_vector

from yardstick_predict import embeddings


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


def test_a_tokens_limit_splits_a_batch_at_once_and_later_batches_start_smaller(voyage, monkeypatch):
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", slept.append)
    voyage.most_tokens = 10  # one text of 8 tokens fits, two don't
    texts = [f"task number {i} with a few words" for i in range(6)]
    waits = []
    found = list(embeddings.embed(texts, "voyage-4-large", KEY, counts=[50_000] * 6, on_wait=waits.append))
    assert sorted(p for positions, _ in found for p in positions) == list(range(6))
    assert slept == [] and not any("waits" in w for w in waits)  # no waiting: smaller requests go through now
    assert [len(r["input"]) for r in voyage.requests] == [2, 1, 1, 1, 1, 1, 1]  # later batches hold one text


def test_a_requests_limit_or_a_single_text_still_waits_and_tries_again(voyage, monkeypatch):
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", slept.append)
    voyage.fail_next = [(429, "You have exceeded your RPM rate limit.")]
    list(embeddings.embed(["first text", "second text"], "voyage-4", KEY))
    assert [r["input"] for r in voyage.requests] == [["first text", "second text"]] * 2  # the same batch again
    voyage.fail_next = [(429, "You have exceeded your TPM rate limit.")]
    list(embeddings.embed(["one text"], "voyage-4", KEY))
    assert [r["input"] for r in voyage.requests[2:]] == [["one text"]] * 2
    assert slept == [1.0, 1.0]


def test_a_wrong_key_is_refused_at_once(voyage):
    with pytest.raises(embeddings.VoyageError) as caught:
        embeddings.check_key("wrong-key", "voyage-4")
    assert caught.value.status == 401
    assert len(voyage.requests) == 1


def test_a_busy_voyage_doesnt_fail_the_key_check(voyage):
    voyage.fail_next = [(503, "Busy")]
    embeddings.check_key(KEY, "voyage-4")
    assert len(voyage.requests) == 2


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
    monkeypatch.setenv("VOYAGE_API_BASE", "http://evil.example.com/v1")
    with pytest.raises(embeddings.VoyageError, match="https"):
        embeddings.check_key(KEY, "voyage-4")


def test_the_yardstick_folder_is_private(home):
    from yardstick_predict.store import private_home

    folder = private_home()
    if __import__("os").name == "posix":
        assert folder.stat().st_mode & 0o077 == 0
