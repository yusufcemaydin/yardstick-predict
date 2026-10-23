"""The feature store: embeddings and LLM answers, made once and shared by every predictor that uses them."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from builders import history
from conftest import KEY, fake_vector
from test_cli import QuickModel, fake_analysis, quick_predictor, quick_train
from test_embeddings import _predictor, _task, saved_by_an_older_version

from yardstick_predict import embeddings, feature_store, features, llm, locks, model_types, pipeline, store
from yardstick_predict.cli import main
from yardstick_predict.store import Predictor

START = 1_780_000_000.0
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROMPTS = feature_store.embedding("prompt", "voyage-4", 4)
"""Vectors of four numbers keep these tests small. The store takes any size its feature names."""
ANSWERS = feature_store.answers("codex", "codex-default")


@pytest.fixture
def quickly(tmp_path, voyage, monkeypatch) -> list[str]:
    """The flags for a quick setup: 40 Claude Code tasks on one AI model, and a model type that trains at once.
    A Voyage key and the stand-in Voyage server come with it."""
    records = tmp_path / "records"
    history(records, tasks=40, start=START)
    monkeypatch.setenv("VOYAGE_API_KEY", KEY)
    load = lambda folder: QuickModel(json.loads((folder / "model.json").read_text()))  # noqa: E731
    monkeypatch.setitem(model_types._TYPES, "quick", (quick_train, load))
    return ["--model-type", "quick", "--archive", f"claude-code={records}"]


@pytest.fixture
def saved_by_0_3() -> Predictor:
    """The predictor 0.3.0 saved, in this test's own yardstick folder. fixtures/README.md says how it was made."""
    shutil.copytree(FIXTURES / "predictor-0.3.0", store.predictor_folder("default"))
    return Predictor("default")


def vector(*numbers: float) -> np.ndarray:
    return np.array(numbers, dtype=np.float32)


def parts(feature: feature_store.Feature) -> list[str]:
    return sorted(path.name for path in (feature_store.folder() / feature.id).glob("part-*.parquet"))


# Saving and reading ---------------------------------------------------------------------------------------


def test_vectors_and_answers_read_back_as_they_were_saved():
    assert feature_store.save_vectors(PROMPTS, [("t1", "h1", [1, 2, 3, 4]), ("t2", "h2", vector(0.5, 0, 0, 1))]) == 2
    found = feature_store.vectors(PROMPTS)
    assert sorted(found) == ["t1", "t2"] and np.array_equal(found["t2"]["h2"], vector(0.5, 0, 0, 1))
    assert found["t1"]["h1"].dtype == np.float32
    assert feature_store.vector_hashes(PROMPTS, {"t1"}) == {"t1": {"h1"}}  # only the tasks asked for
    said = {"t1": {"kind": "feature", "by": "codex:codex-default", "fields": ["kind"]}, "t2": {"failed": True}}
    assert feature_store.save_answers(ANSWERS, said) == 2
    assert feature_store.answers_of(ANSWERS) == said
    assert feature_store.answers_of(ANSWERS, {"t2"}) == {"t2": {"failed": True}}
    # The index says what's stored, with the names it was made with.
    entries = feature_store.index()
    assert entries[PROMPTS.id]["made_with"] == {"field": "prompt", "model": "voyage-4", "dimensions": 4}
    assert entries[ANSWERS.id]["kind"] == "llm" and entries[ANSWERS.id]["changed_at"] > 0


def test_a_later_save_wins_and_a_task_keeps_a_vector_for_each_of_its_texts():
    feature_store.save_vectors(PROMPTS, [("t1", "first text", vector(1, 0, 0, 0))])
    feature_store.save_vectors(
        PROMPTS, [("t1", "first text", vector(0, 1, 0, 0)), ("t1", "other text", vector(0, 0, 1, 0))]
    )
    found = feature_store.vectors(PROMPTS)["t1"]
    assert sorted(found) == ["first text", "other text"]
    assert np.array_equal(found["first text"], vector(0, 1, 0, 0))  # the later save
    feature_store.save_answers(ANSWERS, {"t1": {"kind": "feature"}})
    feature_store.save_answers(ANSWERS, {"t1": {"kind": "bug_fix"}})
    assert feature_store.answers_of(ANSWERS) == {"t1": {"kind": "bug_fix"}}


def test_a_vector_of_another_size_is_refused():
    with pytest.raises(ValueError, match="vector of 3 numbers"):
        feature_store.save_vectors(PROMPTS, [("t1", "h", [1, 2, 3])])


def test_compacting_keeps_every_row_and_removes_the_parts():
    for n in range(5):
        feature_store.save_vectors(PROMPTS, [(f"t{n}", "h", vector(n, 0, 0, 0)), ("t0", f"h{n}", vector(0, n, 0, 0))])
    feature_store.save_vectors(PROMPTS, [("t3", "h", vector(9, 9, 9, 9))])
    before = feature_store.vectors(PROMPTS)
    assert len(parts(PROMPTS)) == 6
    feature_store.compact(PROMPTS)
    assert parts(PROMPTS) == [] and (feature_store.folder() / PROMPTS.id / "data.parquet").is_file()
    after = feature_store.vectors(PROMPTS)
    assert {t: sorted(h) for t, h in after.items()} == {t: sorted(h) for t, h in before.items()}
    assert all(np.array_equal(after[t][h], before[t][h]) for t in before for h in before[t])
    assert np.array_equal(after["t3"]["h"], vector(9, 9, 9, 9))
    # Saves after a compaction go on from the parts it merged, and win over the data file.
    feature_store.save_vectors(PROMPTS, [("t3", "h", vector(1, 1, 1, 1))])
    assert parts(PROMPTS) == ["part-7.parquet"]
    assert np.array_equal(feature_store.vectors(PROMPTS)["t3"]["h"], vector(1, 1, 1, 1))


def test_a_feature_with_many_saves_is_compacted_by_itself():
    for n in range(feature_store.MAX_PARTS + 1):
        feature_store.save_answers(ANSWERS, {f"t{n}": {"kind": "feature"}})
    assert parts(ANSWERS) == []  # the save that made one too many compacted them all
    assert len(feature_store.answers_of(ANSWERS)) == feature_store.MAX_PARTS + 1


def test_each_name_gets_its_own_id_of_plain_characters():
    ids = [
        feature_store.embedding("prompt", "voyage-4", 256).id,
        feature_store.embedding("summary", "Voyage-4", 256).id,
        feature_store.answers("anthropic", "claude-opus-5-5").id,
        feature_store.answers("openai", "gpt-6.1-sol").id,
        feature_store.answers("anthropic", "Claude/Opus").id,
        feature_store.answers("anthropic", "claude/opus").id,
        feature_store.answers("codex", "").id,
    ]
    assert ids[0] == "embedding.prompt.voyage-4.256"
    assert re.fullmatch(r"embedding\.summary\.voyage-4-[0-9a-f]{8}\.256", ids[1])  # a capital letter adds a hash
    assert ids[2:4] == ["llm.anthropic.claude-opus-5-5", "llm.openai.gpt-6.1-sol"]
    assert len(set(ids)) == len(ids) and all(feature_store._ID.fullmatch(found) for found in ids)
    assert not any(found.endswith(".") for found in ids)  # Windows can't name a folder so


@pytest.mark.skipif(os.name != "posix", reason="Windows keeps the folder private with icacls")
def test_the_store_is_readable_only_by_you():
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    assert feature_store.folder().stat().st_mode & 0o777 == 0o700
    assert (feature_store.folder() / PROMPTS.id / "part-1.parquet").stat().st_mode & 0o777 == 0o600


def test_each_file_names_its_feature_so_the_store_can_be_rebuilt_from_its_files():
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    meta = json.loads(pq.read_schema(feature_store.folder() / PROMPTS.id / "part-1.parquet").metadata[b"yardstick"])
    assert meta == {
        "format": feature_store.FORMAT,
        "id": PROMPTS.id,
        "kind": "embedding",
        "made_with": PROMPTS.made_with,
    }


# Files deleted or broken by hand --------------------------------------------------------------------------


def test_a_missing_or_broken_index_is_rebuilt_from_the_files_with_a_note(capsys):
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    feature_store.save_answers(ANSWERS, {"t1": {"kind": "feature"}})
    feature_store.compact(ANSWERS)  # one rebuilt from a data file, one from a part
    before = {key: {**entry, "changed_at": None} for key, entry in feature_store.index().items()}
    capsys.readouterr()
    path = feature_store.folder() / "index.json"
    for broken, note in ((None, "was missing"), ("{not json", "was broken"), ('{"features": []}', "")):
        path.unlink() if broken is None else path.write_text(broken)
        assert {key: {**entry, "changed_at": None} for key, entry in feature_store.index().items()} == before
        assert json.loads(path.read_text())["features"].keys() == before.keys()  # saved again
        said = capsys.readouterr().out
        assert (f"{note}, so it was rebuilt from the files" in said) if note else not said  # each note once
    assert sorted(feature_store.vectors(PROMPTS)) == ["t1"]


def test_a_store_saved_to_for_the_first_time_says_nothing_about_its_index(capsys):
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    assert capsys.readouterr().out == ""


def test_a_feature_whose_folder_is_gone_has_no_rows_until_the_next_save():
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    shutil.rmtree(feature_store.folder() / PROMPTS.id)
    assert feature_store.vectors(PROMPTS) == {} and feature_store.vector_hashes(PROMPTS) == {}
    [listed] = feature_store.stored()
    assert listed.missing and listed.tasks == 0
    feature_store.save_vectors(PROMPTS, [("t2", "h", vector(1, 2, 3, 4))])
    assert sorted(feature_store.vectors(PROMPTS)) == ["t2"]
    assert not feature_store.stored()[0].missing


def test_an_unreadable_file_is_skipped_with_a_note_and_the_listing_names_it(capsys):
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    feature_store.save_vectors(PROMPTS, [("t2", "h", vector(1, 2, 3, 4))])
    broken = feature_store.folder() / PROMPTS.id / "part-2.parquet"
    broken.write_bytes(b"not a parquet file")
    capsys.readouterr()
    assert sorted(feature_store.vectors(PROMPTS)) == ["t1"]
    assert sorted(feature_store.vector_hashes(PROMPTS)) == ["t1"]
    said = capsys.readouterr().out
    assert said.count(f"Skipped {broken}") == 1 and "isn't a Parquet file" in said  # said once
    [listed] = feature_store.stored()
    assert listed.unreadable == ["part-2.parquet"] and listed.tasks == 1
    # A compaction keeps every row it can read, and the broken part goes.
    feature_store.compact(PROMPTS)
    assert parts(PROMPTS) == [] and sorted(feature_store.vectors(PROMPTS)) == ["t1"]
    assert feature_store.stored()[0].unreadable == []


def damage_the_vectors(path: Path) -> None:
    """Change one byte inside the last page of a file's vectors, as a failing disk might. The rest of the file, its
    task ids and text hashes too, still reads."""
    raw = bytearray(path.read_bytes())
    vectors = pq.ParquetFile(path).metadata.row_group(0).column(2)
    first = vectors.dictionary_page_offset if vectors.has_dictionary_page else vectors.data_page_offset
    raw[first + vectors.total_compressed_size - 10] ^= 0xFF  # inside the page's numbers, past its header
    path.write_bytes(bytes(raw))


def test_a_vector_that_cant_be_read_doesnt_count_as_stored(capsys):
    feature_store.save_vectors(PROMPTS, [("t1", "h1", vector(1, 2, 3, 4))])
    feature_store.save_vectors(PROMPTS, [("t2", "h2", vector(5, 6, 7, 8))])
    # A part saved as before files had checksums, damaged where its vectors can't be read at all.
    table = pa.table({"task_id": ["t3"], "text_hash": ["h3"], "vector": [[1.0, 0.0, 0.0, 0.0]], "saved_at": [1.0]})
    meta = {"format": feature_store.FORMAT, "id": PROMPTS.id, "kind": "embedding", "made_with": PROMPTS.made_with}
    unchecked = feature_store.folder() / PROMPTS.id / "part-3.parquet"
    pq.write_table(
        table.cast(feature_store._schema(PROMPTS)).replace_schema_metadata({b"yardstick": json.dumps(meta)}),
        unchecked,
        compression="zstd",
    )
    damaged = feature_store.folder() / PROMPTS.id / "part-2.parquet"
    for path in (damaged, unchecked):
        damage_the_vectors(path)
        assert len(pq.read_table(path, columns=["task_id", "text_hash"])) == 1  # its ids and hashes still read
    capsys.readouterr()
    # A task counts as having a vector only when its vector reads, so whatever needs it makes it again.
    assert feature_store.vector_hashes(PROMPTS) == {"t1": {"h1"}}
    assert sorted(feature_store.vectors(PROMPTS)) == ["t1"]
    said = capsys.readouterr().out
    assert said.count(f"Skipped {damaged}") == 1 and said.count(f"Skipped {unchecked}") == 1
    [listed] = feature_store.stored()
    assert listed.tasks == 1 and listed.unreadable == ["part-2.parquet", "part-3.parquet"]


def test_a_task_whose_vector_cant_be_read_is_embedded_again_from_the_cache(voyage):
    predictor = _predictor(["prompt"])
    predictor.add_tasks([_task(n, f"task {n}") for n in range(3)])
    pipeline.run_embeddings(predictor, pipeline.plan_embeddings(predictor, KEY), KEY)
    sent = len(voyage.requests)
    damage_the_vectors(feature_store.folder() / predictor.embedding_feature("prompt").id / "data.parquet")
    plan = pipeline.plan_embeddings(predictor, KEY)
    assert sorted(task for task, _, _ in plan.items) == ["t0", "t1", "t2"] and plan.cost.tokens == 0
    pipeline.run_embeddings(predictor, plan, KEY)
    assert len(voyage.requests) == sent and sorted(predictor.get_embeddings("prompt")) == ["t0", "t1", "t2"]


def test_a_file_of_another_feature_or_shape_is_skipped(capsys):
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    other = feature_store.embedding("context", "voyage-4", 4)
    feature_store.save_vectors(other, [("t9", "h", vector(1, 2, 3, 4))])
    shutil.copy(
        feature_store.folder() / other.id / "part-1.parquet", feature_store.folder() / PROMPTS.id / "part-2.parquet"
    )
    wider = feature_store.embedding("prompt", "voyage-4", 8)  # its id is another, but say someone renamed its folder
    feature_store.save_vectors(wider, [("t8", "h", np.ones(8))])
    folder = feature_store.folder() / PROMPTS.id
    shutil.copy(feature_store.folder() / wider.id / "part-1.parquet", folder / "part-3.parquet")
    capsys.readouterr()
    assert sorted(feature_store.vectors(PROMPTS)) == ["t1"]
    said = capsys.readouterr().out
    assert "part-2.parquet: it holds no rows of this feature" in said
    assert "part-3.parquet: it holds no rows of this feature" in said


def test_a_file_a_newer_version_saved_is_skipped_and_never_compacted_away(capsys):
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    table = pa.table({"task_id": ["t2"], "text_hash": ["h"], "vector": [[1.0, 2.0, 3.0, 4.0]], "saved_at": [1.0]})
    meta = {"format": feature_store.FORMAT + 1, "id": PROMPTS.id, "kind": "embedding", "made_with": {}}
    newer = feature_store.folder() / PROMPTS.id / "part-2.parquet"
    pq.write_table(table.replace_schema_metadata({b"yardstick": json.dumps(meta)}), newer)
    assert sorted(feature_store.vectors(PROMPTS)) == ["t1"]
    assert "a newer version of yardstick-predict saved it" in capsys.readouterr().out
    feature_store.compact(PROMPTS)
    assert parts(PROMPTS) == ["part-1.parquet", "part-2.parquet"]  # left as they are, for the newer version


def test_a_part_a_compaction_merges_during_a_read_is_read_again(monkeypatch):
    for n in range(3):
        feature_store.save_vectors(PROMPTS, [(f"t{n}", "h", vector(n, 0, 0, 0))])
    real = feature_store._load
    compacted = []

    def compaction_meanwhile(feature, path, *args):
        if not compacted:
            compacted.append(path.name)
            feature_store.compact(feature)  # another process compacts as this one starts to read
        return real(feature, path, *args)

    monkeypatch.setattr(feature_store, "_load", compaction_meanwhile)
    assert sorted(feature_store.vector_hashes(PROMPTS)) == ["t0", "t1", "t2"]
    assert compacted == ["part-1.parquet"] and parts(PROMPTS) == []


def test_other_files_in_the_store_folder_are_left_alone():
    feature_store.save_answers(ANSWERS, {"t1": {"kind": "feature"}})
    (feature_store.folder() / "checks.json").write_text("{}")
    (feature_store.folder() / "notes").mkdir()
    feature_store.delete(ANSWERS.id)
    (feature_store.folder() / "index.json").unlink()
    assert feature_store.index() == {} and feature_store.stored() == []
    assert (feature_store.folder() / "checks.json").read_text() == "{}" and (feature_store.folder() / "notes").is_dir()


# Writers at once ------------------------------------------------------------------------------------------


def test_two_writers_at_once_lose_nothing():
    def write(worker: int) -> None:
        for n in range(30):  # enough saves for compactions to run in between
            feature_store.save_vectors(PROMPTS, [(f"w{worker}-t{n}", "h", vector(worker, n, 0, 0))])
            feature_store.save_answers(ANSWERS, {f"w{worker}-t{n}": {"kind": "feature", "n": n}})

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    found = feature_store.vectors(PROMPTS)
    assert len(found) == 90 and np.array_equal(found["w2-t29"]["h"], vector(2, 29, 0, 0))
    assert len(feature_store.answers_of(ANSWERS)) == 90


def test_writers_merging_answers_about_the_same_tasks_at_once_keep_every_part():
    by = "codex:codex-default"
    parts = {"kind": {"kind": "feature"}, "description": {"description": "Add the page."}}

    def write(name: str) -> None:
        for n in range(25):  # enough saves for compactions to run in between
            answer = {**parts[name], "by": by, "fields": [name]}
            feature_store.save_answers(ANSWERS, {f"t{n}": answer}, merge=llm.merge)

    threads = [threading.Thread(target=write, args=(name,)) for name in parts]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    found = feature_store.answers_of(ANSWERS)
    assert len(found) == 25
    assert all(found[task]["kind"] == "feature" and found[task]["description"] == "Add the page." for task in found)


def test_writers_in_separate_processes_lose_nothing():
    script = textwrap.dedent(
        """
        import sys
        from yardstick_predict import feature_store
        feature = feature_store.embedding("prompt", "voyage-4", 4)
        print("ready", flush=True)
        sys.stdin.readline()
        for n in range(15):
            feature_store.save_vectors(feature, [(f"{sys.argv[1]}-{n}", "h", [n, 0, 0, 0])])
        """
    )
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", script, f"p{worker}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        for worker in range(3)
    ]
    assert [worker.stdout.readline() for worker in workers] == ["ready\n"] * 3
    for worker in workers:  # they start together
        worker.stdin.write("go\n")
        worker.stdin.flush()
    assert [worker.wait(timeout=120) for worker in workers] == [0, 0, 0]
    for worker in workers:
        worker.stdin.close()
        worker.stdout.close()
    assert len(feature_store.vector_hashes(PROMPTS)) == 45


# Deleting a feature ---------------------------------------------------------------------------------------


def test_deleting_a_feature_removes_its_folder_and_its_index_line():
    feature_store.save_vectors(PROMPTS, [("t1", "h", vector(1, 2, 3, 4))])
    feature_store.save_answers(ANSWERS, {"t1": {"kind": "feature"}})
    assert feature_store.delete(PROMPTS.id)
    assert not (feature_store.folder() / PROMPTS.id).exists() and PROMPTS.id not in feature_store.index()
    assert feature_store.vectors(PROMPTS) == {} and feature_store.answers_of(ANSWERS) == {"t1": {"kind": "feature"}}
    assert not feature_store.delete(PROMPTS.id)  # nothing left to delete
    for bad in ("../predictors", "index.json", "embedding/../x", ""):
        with pytest.raises(ValueError):
            feature_store.delete(bad)


def test_after_a_feature_is_deleted_its_texts_are_planned_again_and_the_cache_makes_them_free(voyage):
    predictor = _predictor(["prompt", "context"])
    predictor.add_tasks([_task(n, f"task {n}", context="we talked") for n in range(3)])
    pipeline.run_embeddings(predictor, pipeline.plan_embeddings(predictor, KEY), KEY)
    sent = len(voyage.requests)
    assert pipeline.plan_embeddings(predictor, KEY).items == []
    assert feature_store.delete(predictor.embedding_feature("prompt").id)
    plan = pipeline.plan_embeddings(predictor, KEY)
    assert sorted(item[:2] for item in plan.items) == [("t0", "prompt"), ("t1", "prompt"), ("t2", "prompt")]
    assert plan.texts == [] and plan.cost.tokens == 0  # the cache of vectors still holds every text
    pipeline.run_embeddings(predictor, plan, KEY)
    assert len(voyage.requests) == sent and predictor.embedded("prompt") == {"t0", "t1", "t2"}


# Moving an older version's vectors and answers in ---------------------------------------------------------


def old_predictor(name: str = "default", tasks: int = 4) -> Predictor:
    """A predictor as a version before the feature store saved it: its vectors and LLM answers are in its dataset."""
    predictor = Predictor.create(
        name, [], {"model": "voyage-4", "dimensions": 256}, features=["prompt", "context", "llm"]
    )
    predictor.config.update(llm={"provider": "codex", "model": "codex-default"})
    predictor.save_config()
    found = [_task(n, f"task {n}", context="we talked" if n else "") for n in range(tasks)]
    predictor.add_tasks(found)
    predictor.close()
    saved_by_an_older_version(predictor)
    db = sqlite3.connect(predictor.dir / "dataset.sqlite")
    db.execute("CREATE TABLE embeddings (task_id, field, vector, text_hash, PRIMARY KEY (task_id, field))")
    db.execute("CREATE TABLE llm_labels (task_id PRIMARY KEY, labels)")
    for task in found:
        for field, text in (("prompt", task.prompt), ("context", task.context)):
            if text:
                blob = np.array(fake_vector(text), dtype=np.float32).tobytes()
                db.execute(
                    "INSERT INTO embeddings VALUES (?, ?, ?, ?)", (task.id, field, blob, embeddings.text_hash(text))
                )
        labels = {"kind": "feature", "minutes": 3.0, "by": "codex:codex-default", "fields": ["kind", "minutes"]}
        db.execute("INSERT INTO llm_labels VALUES (?, ?)", (task.id, json.dumps(labels)))
    db.execute("INSERT INTO embeddings VALUES ('t0', 'skills', ?, NULL)", (np.ones(256, np.float32).tobytes(),))
    db.execute("INSERT INTO llm_labels VALUES ('elsewhere', ?)", (json.dumps({"kind": "review", "by": "claude:x"}),))
    db.execute("INSERT INTO llm_labels VALUES ('unsigned', ?)", (json.dumps({"kind": "design"}),))
    db.execute("INSERT INTO llm_labels VALUES ('broken', 'not json')")
    db.commit()
    db.close()
    return Predictor(name)


def test_an_older_predictors_vectors_and_answers_move_into_the_store_once():
    predictor = old_predictor()
    assert not predictor.config.get("feature_store")
    found = predictor.get_embeddings("prompt")  # the first read moves them in
    assert sorted(found) == ["t0", "t1", "t2", "t3"]
    assert np.array_equal(found["t2"][embeddings.text_hash("task 2")], np.array(fake_vector("task 2"), np.float32))
    assert sorted(predictor.embedded("context")) == ["t1", "t2", "t3"]
    labels = predictor.llm_labels()
    assert labels["t3"] == {
        "kind": "feature",
        "minutes": 3.0,
        "by": "codex:codex-default",
        "fields": ["kind", "minutes"],
    }
    assert labels["unsigned"] == {"kind": "design"}  # an answer that doesn't say who gave it goes with the predictor's
    assert "elsewhere" not in labels and "broken" not in labels
    assert feature_store.answers_of(feature_store.answers("claude", "x")) == {
        "elsewhere": {"kind": "review", "by": "claude:x"}
    }
    # A vector without its text's hash was never used, so it stays behind.
    assert predictor.embedded("skills") == set()
    assert json.loads((predictor.dir / "predictor.json").read_text())["feature_store"] == feature_store.FORMAT
    # The old tables stay, unread, and the moved features are compacted.
    db = sqlite3.connect(predictor.dir / "dataset.sqlite")
    assert db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 8
    db.close()
    assert parts(predictor.embedding_feature("prompt")) == [] and parts(ANSWERS) == []


def test_moving_in_twice_changes_nothing():
    old_predictor().move_in()
    before = (
        feature_store.vectors(feature_store.embedding("prompt", "voyage-4", 256)),
        feature_store.answers_of(ANSWERS),
    )
    changed = {path: path.stat().st_mtime_ns for path in feature_store.folder().rglob("*.parquet")}
    predictor = Predictor("default")
    saved_by_an_older_version(predictor)  # as if the mark were never saved, so it runs again
    Predictor("default").move_in()
    after = feature_store.vectors(feature_store.embedding("prompt", "voyage-4", 256)), feature_store.answers_of(ANSWERS)
    assert after[1] == before[1] and before[0].keys() == after[0].keys()
    assert {path: path.stat().st_mtime_ns for path in feature_store.folder().rglob("*.parquet")} == changed


def test_a_move_stopped_halfway_finishes_the_next_time(monkeypatch):
    predictor = old_predictor()
    real = feature_store.save_answers

    def stops(*args, **kwargs):
        raise KeyboardInterrupt  # as when you press Ctrl+C

    monkeypatch.setattr(feature_store, "save_answers", stops)
    with pytest.raises(KeyboardInterrupt):
        predictor.move_in()
    assert sorted(feature_store.vector_hashes(predictor.embedding_feature("prompt"))) == ["t0", "t1", "t2", "t3"]
    assert "feature_store" not in json.loads((predictor.dir / "predictor.json").read_text())
    monkeypatch.setattr(feature_store, "save_answers", real)
    again = Predictor("default")
    assert len(again.llm_labels()) == 5 and sorted(again.embedded("prompt")) == ["t0", "t1", "t2", "t3"]
    assert json.loads((again.dir / "predictor.json").read_text())["feature_store"] == feature_store.FORMAT


def test_two_predictors_move_in_at_once():
    predictors = [old_predictor(name, tasks=20) for name in ("first", "second")]
    errors = []

    def move(predictor: Predictor) -> None:
        try:
            predictor.move_in()
        except Exception as error:  # pragma: no cover - only when the test fails
            errors.append(error)

    threads = [threading.Thread(target=move, args=(predictor,)) for predictor in predictors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    for name in ("first", "second"):
        predictor = Predictor(name)
        assert predictor.config["feature_store"] == feature_store.FORMAT
        assert len(predictor.embedded("prompt")) == 20 and len(predictor.llm_labels()) == 21  # with "unsigned"


BY = "codex:codex-default"
KIND = llm.parts(["kind"])
WORDS = llm.parts(["description"])
SIZED = {"kind": "feature", "files": 2.0, "steps": 2.0, "runs_code": True, "looks_up": False, "continues": False,
         "clarity": "clear", "minutes": 5.0, "by": BY, "fields": KIND}  # fmt: skip
"""An answer about a task's kind and size."""
DESCRIBED = {"description": "Add the page.", "by": BY, "fields": WORDS}


def refused(fields: list[str], tries: int) -> dict:
    """A note, as an older version saved it, that the AI model didn't answer, with how many times it was asked."""
    return {"by": BY, "fields": fields, "failed": True, "tries": tries}


def older_with(name: str, answers: dict[str, dict], context: str) -> Predictor:
    """An older version's predictor over tasks t0 to t3, with these LLM answers, and its tasks' context as it read
    them: with another text, a context has another vector."""
    predictor = old_predictor(name)
    db = sqlite3.connect(predictor.dir / "dataset.sqlite")
    db.execute("DELETE FROM llm_labels")
    db.executemany("INSERT INTO llm_labels VALUES (?, ?)", [(task, json.dumps(said)) for task, said in answers.items()])
    db.execute("UPDATE tasks SET context = ? WHERE context != ''", (context,))
    vector = np.array(fake_vector(context), dtype=np.float32).tobytes()
    made = embeddings.text_hash(context)
    db.execute("UPDATE embeddings SET vector = ?, text_hash = ? WHERE field = 'context'", (vector, made))
    db.commit()
    db.close()
    return Predictor(name)


@pytest.mark.parametrize("first", ["kinds", "words"])
def test_two_older_predictors_moving_in_keep_every_part_of_their_answers_in_either_order(first):
    # One asked the AI model for the kind and size, the other for a description, of the same tasks. Each was refused
    # some of them. The other read t3 when it asked for both.
    kinds = {"t0": SIZED, "t1": refused(KIND, 2), "t2": SIZED, "t3": refused(KIND, 1)}
    both = {**SIZED, **DESCRIBED, "fields": KIND + WORDS}
    words = {"t0": DESCRIBED, "t1": DESCRIBED, "t2": refused(WORDS, 1), "t3": both}
    older = {"kinds": older_with("kinds", kinds, "we talked"), "words": older_with("words", words, "we talked more")}
    for name in (first, "words" if first == "kinds" else "kinds"):
        older[name].move_in()
    found = feature_store.answers_of(ANSWERS)
    how = {"provider": "codex", "model": "codex-default"}
    for task in ("t0", "t3"):  # each part from either, and an answer kept over a refusal
        assert found[task]["kind"] == "feature" and found[task]["description"] == "Add the page.", task
        assert set(found[task]["fields"]) == set(KIND + WORDS) and not found[task].get("failed"), task
    assert found["t1"]["description"] == "Add the page." and "kind" not in found["t1"]
    assert not llm.needs(found["t1"], how, KIND)  # refused twice: not asked again
    assert found["t2"]["kind"] == "feature" and "description" not in found["t2"]
    assert llm.needs(found["t2"], how, WORDS) and not llm.needs(found["t2"], how, KIND)  # refused once: once more
    # Each reads the fields it asks for, and its own context's vectors. The other's stay in the store too.
    for name, text in (("kinds", "we talked"), ("words", "we talked more")):
        predictor = Predictor(name)
        assert predictor.llm_labels(every_field=True) == found
        assert sorted(predictor.embedded("prompt")) == ["t0", "t1", "t2", "t3"]
        assert all(predictor.embedded("context").made_from(task, text) for task in ("t1", "t2", "t3"))
    context = feature_store.vector_hashes(Predictor("kinds").embedding_feature("context"))
    assert all(len(context[task]) == 2 for task in ("t1", "t2", "t3"))
    # Moving in again changes nothing.
    changed = {path: path.stat().st_mtime_ns for path in feature_store.folder().rglob("*.parquet")}
    for name in older:
        saved_by_an_older_version(Predictor(name))
        Predictor(name).move_in()
    assert {path: path.stat().st_mtime_ns for path in feature_store.folder().rglob("*.parquet")} == changed


def test_answers_by_different_ai_models_moving_in_are_never_mixed():
    unsigned = {"kind": "design"}  # an answer saved before answers said who gave them
    assert llm.combine(SIZED, unsigned) == SIZED and llm.combine(None, unsigned) == unsigned
    assert llm.combine(refused(KIND, 1), unsigned) == unsigned  # an answer over a refusal
    assert llm.combine(unsigned, refused(KIND, 1)) == unsigned
    both = llm.combine(refused(WORDS, 1), refused(WORDS, 2))
    assert both["unanswered"] == {"description": 2} and llm.combine(both, refused(WORDS, 2)) == both


def test_training_reads_the_same_rows_from_the_store_as_it_did_from_the_dataset(saved_by_0_3):
    # What the version before the feature store read: each task's one vector for each feature, from the
    # dataset, and what LLM analysis said about it.
    db = sqlite3.connect(saved_by_0_3.dir / "dataset.sqlite")
    rows = "SELECT task_id, field, vector, text_hash FROM embeddings"
    kept = {(task_id, name): (np.frombuffer(blob, np.float32), made) for task_id, name, blob, made in db.execute(rows)}
    said = {task_id: json.loads(text) for task_id, text in db.execute("SELECT task_id, labels FROM llm_labels")}
    db.close()
    tasks = saved_by_0_3.tasks()
    found = pipeline.feature_rows(saved_by_0_3, tasks)
    assert saved_by_0_3.llm_labels() == said
    names = [name for name in pipeline.chosen_features(saved_by_0_3) if features.is_text(name)]
    for task, row in zip(tasks, found):
        info = pipeline.task_info(task, saved_by_0_3.config, said)
        expected = {}
        for name in names:
            # Then, a task's vector counted when it was made from the task's text as it is now.
            stored, made = kept.get((task.id, name), (None, None))
            if stored is not None and made == embeddings.text_hash(features.text(name, info)):
                expected[name] = stored
        assert sorted(row.vectors) == sorted(expected), task.id
        assert all(np.array_equal(row.vectors[name], expected[name]) for name in expected), task.id
        assert {n: v for n, v in row.numbers.items() if n.startswith("llm_")} == llm.numbers(said.get(task.id) or {})
    assert sum(len(row.vectors) for row in found) == 512  # every vector the dataset had for its tasks' texts


# Sharing between predictors -------------------------------------------------------------------------------


def embedded_texts(voyage) -> list[str]:
    """The task texts sent to Voyage, without the tiny one each setup sends to check the key."""
    return [text for body in voyage.requests for text in body["input"] if text != "yardstick-predict key check"]


def test_a_second_predictor_over_the_same_records_pays_for_nothing(capsys, quickly, voyage, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(seen))
    flags = [*quickly, "--features", "context,llm", "--llm-provider", "codex"]
    assert main(["create", "--yes", *flags]) == 0
    sent, asked = len(embedded_texts(voyage)), list(seen)
    assert sent and asked == [40]
    capsys.readouterr()
    assert main(["create", "work", "--yes", *flags]) == 0
    out = capsys.readouterr().out
    assert len(embedded_texts(voyage)) == sent and seen == asked  # no text went to Voyage, no task to the AI model
    assert "LLM analysis:" not in out
    first, second = Predictor("default"), Predictor("work")
    assert second.llm_labels() == first.llm_labels() and len(second.llm_labels()) == 40
    for name in ("prompt", "context", "llm"):
        assert second.embedded(name) == first.embedded(name) and second.embedded(name)
    rows = [pipeline.feature_rows(p, p.tasks()) for p in (first, second)]
    assert all(
        sorted(a.vectors) == sorted(b.vectors) and all(np.array_equal(a.vectors[n], b.vectors[n]) for n in a.vectors)
        for a, b in zip(*rows)
    )


def test_predictors_asking_one_ai_model_for_other_fields_share_its_answers(capsys, quickly, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(llm, "analyze", fake_analysis(seen))
    flags = [*quickly, "--llm-provider", "codex"]
    assert main(["create", "--yes", *flags, "--features", "llm:kind"]) == 0
    assert main(["create", "words", "--yes", *flags, "--features", "llm:description"]) == 0
    assert seen == [40, 40]  # the second asked for the descriptions
    capsys.readouterr()
    for name in ("default", "words"):
        assert main(["refresh", "--yes", "--predictor", name]) == 0
    assert seen == [40, 40] and capsys.readouterr().out.count("Nothing new") == 2
    # Each reads only the fields it asked for, though the store keeps both.
    kind_only, described = Predictor("default").llm_labels(), Predictor("words").llm_labels()
    one = next(iter(kind_only))
    assert "kind" in kind_only[one] and "description" not in kind_only[one]
    assert "description" in described[one] and "kind" not in described[one]
    assert {"kind", "description"} <= set(Predictor("default").llm_labels(every_field=True)[one])
    # The descriptions' vectors are in the store, for the predictor that asked for them. The other never reads one.
    kind_rows = pipeline.feature_rows(Predictor("default"), Predictor("default").tasks())
    words_rows = pipeline.feature_rows(Predictor("words"), Predictor("words").tasks())
    assert not any("llm" in row.vectors for row in kind_rows) and all("llm" in row.vectors for row in words_rows)
    assert any("llm_kind_feature" in row.numbers for row in kind_rows)
    assert not any(name.startswith("llm_") for row in words_rows for name in row.numbers)


def asking(name: str, option: str) -> Predictor:
    """A predictor over three tasks whose LLM analysis asks Codex's own model for one of its options."""
    predictor = Predictor.create(
        name,
        [],
        {"model": "voyage-4", "dimensions": 256},
        features=["prompt", "llm"],
        options={"llm": [option]},
        llm={"provider": "codex", "model": "codex-default"},
    )
    predictor.add_tasks([_task(n, f"task {n}") for n in range(3)])
    return predictor


def test_two_predictors_analyzing_the_same_tasks_at_once_keep_each_others_answers(monkeypatch):
    kinds, words = asking("kinds", "kind"), asking("words", "description")
    said = {"kind": "feature", "files": 2.0, "steps": 2.0, "runs_code": True, "looks_up": False, "continues": False,
            "clarity": "clear", "minutes": 5.0, "description": "Add the page."}  # fmt: skip

    def analyze(asks, how, fields, on_batch=None):
        if "kind" in fields:
            # The other predictor reads the same tasks and saves its answers while this one waits for the AI model.
            pipeline.run_analysis(words, pipeline.analysis_plan(words.tasks(), words.config, words.llm_labels()))
        found = {ask.id: {**{name: said[name] for name in fields}, "by": "codex:codex-default", "fields": fields}
                 for ask in asks}  # fmt: skip
        on_batch(found, len(asks))
        return llm.Analysis(found)

    monkeypatch.setattr(llm, "analyze", analyze)
    assert pipeline.run_analysis(kinds, pipeline.analysis_plan(kinds.tasks(), kinds.config, kinds.llm_labels()))[0] == 3
    stored = feature_store.answers_of(ANSWERS)
    assert sorted(stored) == ["t0", "t1", "t2"]
    assert all(stored[task]["kind"] == "feature" and stored[task]["description"] == "Add the page." for task in stored)
    assert all(set(stored[task]["fields"]) == set(llm.parts(["kind", "description"])) for task in stored)
    assert pipeline.analysis_plan(words.tasks(), words.config, words.llm_labels()) == []  # nothing to ask again


def test_an_answer_keeps_what_the_same_ai_model_said_before_in_other_fields():
    kind = {"kind": "feature", "files": 2.0, "by": "codex:codex-default", "fields": ["kind", "files"]}
    words = {"description": "Add the page.", "by": "codex:codex-default", "fields": ["description"]}
    merged = llm.merge(kind, words)
    assert merged == {"kind": "feature", "files": 2.0, "description": "Add the page.", "by": "codex:codex-default",
                      "fields": ["kind", "files", "description"]}  # fmt: skip
    how = {"provider": "codex", "model": "codex-default"}
    assert not llm.needs(merged, how, ["kind", "files"]) and not llm.needs(merged, how, ["description"])
    assert llm.only(merged, ["description"]) == {**words, "fields": merged["fields"]}
    # Another AI model's answer replaces it, as before.
    other = {"kind": "review", "by": "claude:claude-opus-5-5", "fields": ["kind"]}
    assert llm.merge(merged, other) == other


def test_a_refusal_counts_its_tries_and_keeps_the_answers_before_it():
    how = {"provider": "codex", "model": "codex-default"}
    refused = {"by": "codex:codex-default", "fields": ["description"], "failed": True}
    first = llm.merge(None, dict(refused))
    assert first["unanswered"] == {"description": 1} and llm.needs(first, how, ["description"])
    second = llm.merge(first, dict(refused))
    assert second["unanswered"] == {"description": 2} and not llm.needs(second, how, ["description"])  # MAX_TRIES
    # A refusal of the description keeps the kind another predictor got, and that one isn't asked again.
    kind = {"kind": "feature", "by": "codex:codex-default", "fields": ["kind"]}
    both = llm.merge(kind, dict(refused))
    assert both["kind"] == "feature" and both["failed"] and both["unanswered"] == {"description": 1}
    assert not llm.needs(both, how, ["kind"]) and llm.needs(both, how, ["description"])
    # When the kind is asked again and answered, the refusal of the description still stands.
    again = llm.merge(both, {"kind": "bug_fix", "by": "codex:codex-default", "fields": ["kind"]})
    assert again["kind"] == "bug_fix" and again["failed"] and llm.needs(again, how, ["description"])


# Forecasts ------------------------------------------------------------------------------------------------


@pytest.mark.usefixtures("quickly")
def test_forecasts_never_touch_the_store(capsys, monkeypatch):
    quick_predictor()

    def untouchable(*args, **kwargs):
        raise AssertionError("a forecast touched the feature store")

    for name in ("_read", "_save", "_compact", "_entries", "index", "stored", "delete"):
        monkeypatch.setattr(feature_store, name, untouchable)
    monkeypatch.setattr(Predictor, "move_in", untouchable)
    assert main(["predict", "--model", "claude-opus-5-5", "--prompt", "fix the api"]) == 0
    assert json.loads(capsys.readouterr().out)["forecasts"][0]["seconds"] == 60.0


def test_forecasting_doesnt_load_the_store():
    # The status line forecasts in a process of its own each time, so what it loads adds to its wait.
    script = "import sys, yardstick_predict.cli, yardstick_predict.claude_code, yardstick_predict.forecast\n"
    script += "print(sorted(name for name in ('pyarrow', 'yardstick_predict.feature_store') if name in sys.modules))"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


# The listing ----------------------------------------------------------------------------------------------


def listing(capsys, *flags: str) -> tuple[int, str]:
    code = main(["features", *flags])
    return code, capsys.readouterr().out


def test_the_listing_shows_each_feature_with_its_coverage(capsys, quickly):
    assert main(["create", "--yes", *quickly, "--features", "conversation,context,time"]) == 0
    capsys.readouterr()
    code, out = listing(capsys)
    assert code == 0, out
    assert "Stored features" in out and str(feature_store.folder()) in out
    assert "prompt embeddings by voyage-4, 256 numbers" in out and "40 tasks" in out and "used by default" in out
    assert "context embeddings by voyage-4, 256 numbers" in out
    predictor = Predictor("default")
    tasks = pipeline.learnable(predictor.tasks())
    with_context = sum(1 for task in tasks if task.context[-features.CONTEXT_CHARS :].strip())
    assert 0 < with_context < 40  # the first task of each conversation has none
    lines = {line.split()[0]: line for line in out.splitlines() if line.strip()}
    assert "It learns from 40 tasks." in out
    for title, share in (("Prompt", "100%"), ("Task", "100%"), ("Similar", "100%"), ("Time", "100%")):
        assert share in lines[title], lines[title]
    assert f"{with_context / 40:.0%}" in lines["Context"] and "stored: context embeddings" in lines["Context"]
    assert "worked out at training, free" in lines["Time"] and "worked out at training, free" in lines["Conversation"]


def test_the_listing_in_json(capsys, quickly):
    assert main(["create", "--yes", *quickly, "--features", "context"]) == 0
    capsys.readouterr()
    code, out = listing(capsys, "--json")
    assert code == 0
    found = json.loads(out)
    assert found["version"] == 1 and found["folder"] == str(feature_store.folder()) and found["notes"] == []
    stored = {entry["id"]: entry for entry in found["stored"]}
    prompts = stored["embedding.prompt.voyage-4.256"]
    assert prompts["kind"] == "embedding" and prompts["made_with"] == {
        "field": "prompt",
        "model": "voyage-4",
        "dimensions": 256,
    }
    assert prompts["tasks"] == 40 and prompts["rows"] == 40 and prompts["bytes"] > 0 and prompts["changed_at"] > 0
    assert prompts["used_by"] == ["default"] and not prompts["missing"] and prompts["unreadable"] == []
    [predictor] = found["predictors"]
    assert predictor["name"] == "default" and predictor["default"] and predictor["tasks"] == 40
    by_name = {feature["name"]: feature for feature in predictor["features"]}
    assert list(by_name) == ["prompt", "task", "similar", "history", "context"]
    assert by_name["prompt"] == {
        "name": "prompt",
        "title": "Prompt",
        "says": "what the prompt means, read by Voyage",
        "kept": "stored",
        "stored": ["embedding.prompt.voyage-4.256"],
        "tasks_with_value": 40,
        "coverage": 1.0,
    }
    assert by_name["history"]["kept"] == "training" and by_name["history"]["stored"] == []
    context = by_name["context"]
    assert context["coverage"] == round(context["tasks_with_value"] / 40, 4) and context["kept"] == "stored"


def test_the_listing_shows_what_no_predictor_uses_and_what_cant_be_read(capsys, quickly):
    assert main(["create", "--yes", *quickly, "--features", ""]) == 0
    feature_store.save_answers(ANSWERS, {"t1": {"kind": "feature"}})  # made for a predictor deleted since
    broken = store.predictor_folder("broken")
    broken.mkdir(parents=True)
    (broken / "predictor.json").write_text("{")
    unknown = Predictor("default")
    unknown.config["features"] = ["prompt", "weather"]
    unknown.save_config()
    (feature_store.folder() / "index.json").write_text("{")
    capsys.readouterr()
    code, out = listing(capsys)
    assert code == 1  # a predictor it shows can't be used
    assert (
        "LLM answers by Codex's own model, through your ChatGPT subscription" in out and "used by no predictor" in out
    )
    assert "Predictor 'broken' can't be used" in out
    assert "weather" in out and "this version doesn't know it" in out
    assert "was broken, so it was rebuilt from the files" in out
    code, out = listing(capsys, "--json", "--predictor", "default")
    found = json.loads(out)
    assert code == 0 and [entry["name"] for entry in found["predictors"]] == ["default"]
    assert found["predictors"][0]["features"][-1] == {
        "name": "weather",
        "title": "weather",
        "says": "a feature this version doesn't know, so it's skipped",
        "kept": "unknown",
        "stored": [],
        "tasks_with_value": None,
        "coverage": None,
    }
    assert listing(capsys, "--predictor", "nobody")[0] == 1


def test_the_listing_says_when_theres_nothing_to_list(capsys):
    code, out = listing(capsys)
    assert code == 0 and "Nothing is stored yet." in out and "There's no predictor yet." in out
    assert not feature_store.folder().exists()  # listing made nothing
    code, out = listing(capsys, "--json")
    assert json.loads(out) == {
        "version": 1,
        "folder": str(feature_store.folder()),
        "stored": [],
        "predictors": [],
        "notes": [],
    }


def test_the_listing_moves_an_older_predictor_in_unless_a_refresh_is_running(capsys):
    predictor = old_predictor()
    with locks.held(predictor.dir / "refresh.lock"):
        code, out = listing(capsys)
        assert code == 1 and "being set up or refreshed by another command" in out
    assert not Predictor("default").config.get("feature_store")
    code, out = listing(capsys, "--json")
    assert code == 0 and Predictor("default").config["feature_store"] == feature_store.FORMAT
    found = json.loads(out)
    assert {entry["id"] for entry in found["stored"]} >= {"embedding.prompt.voyage-4.256", ANSWERS.id}


def test_the_listing_is_quick_for_thousands_of_tasks(capsys):
    predictor = Predictor.create(
        "default", [], {"model": "voyage-4", "dimensions": 256}, features=["prompt", "context"]
    )
    tasks = [_task(n, f"task {n} " + "word " * (n % 50), context=f"we talked {n % 7}") for n in range(3000)]
    predictor.add_tasks(tasks)
    rng = np.random.default_rng(0)
    for name, text in (("prompt", lambda t: t.prompt), ("context", lambda t: t.context)):
        vectors = {task.id: rng.random(256, dtype=np.float32) for task in tasks}
        predictor.put_embeddings(name, vectors, {task.id: embeddings.text_hash(text(task)) for task in tasks})
    started = time.perf_counter()
    code, out = listing(capsys)
    assert code == 0 and "It learns from 3,000 tasks." in out
    assert time.perf_counter() - started < 15
