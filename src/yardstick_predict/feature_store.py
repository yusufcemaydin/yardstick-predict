"""The feature store: the features that cost money or time to make, kept once for all your predictors.

Two kinds are kept. Embeddings are the vectors Voyage makes from a task's texts, such as its prompt. LLM
answers are what an AI model said about a task when LLM analysis read it. Each is made once and used by every
predictor that uses it, so a second predictor over the same records never pays for them again.

Features that cost nothing, such as the prompt's layout, the time or the project's recent tasks, aren't kept.
They depend on each predictor's own tasks, so training works them out from the tasks themselves, and nothing
can go stale.

    features/
      index.json            the features that should exist, saved in one step
      store.lock            held while anything here is written
      users.lock            held, shared, by each command that may save here while it runs, and alone by a delete
      <feature-id>/
        data.parquet        the feature's rows, compacted
        part-<n>.parquet    rows saved since, one file for each save, each written in one step

Parquet is a compact file format for tables. Each file names its feature in its own metadata, so the index can
always be rebuilt from the files. Compacting merges the part files into the data file, in one step, then
removes them. Reading needs no lock: a file only ever appears whole, and a reader that finds one gone, because
a compaction merged it meanwhile, starts over once.

Files deleted or broken by hand never crash anything. A missing or broken index is rebuilt from the files,
with a note. A feature whose folder is gone has no rows. A file that can't be read is skipped, with a note
naming it. Each file carries checksums of its data, so one changed on disk, as by a failing disk, is skipped
too, rather than read wrong. Whatever is missing is then made again by the step that made it: an embedding
comes back from the cache of vectors for free, and LLM analysis asks about the task again, with the cost
shown first.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import files, locks, store

FORMAT = 1
"""The store's format, in the index and in every file. A predictor's settings say "feature_store": FORMAT once
its vectors and LLM answers moved in. A file a newer version saved in a higher format is skipped, and never
compacted away."""
MAX_PARTS = 20
"""A save compacts a feature with more part files than this, so even a long run never leaves hundreds."""
EMBEDDING = "embedding"
LLM = "llm"
_KEY = b"yardstick"
"""The key in each file's Parquet metadata that names the feature its rows belong to."""
_PART = re.compile(r"part-(\d+)\.parquet")
_ID = re.compile(r"(?:embedding|llm)\.[a-z0-9._-]+")


@dataclass
class Feature:
    """One stored feature. Its id names its folder. `made_with` keeps the names it was made with, as they were:
    for an embedding the text feature, the Voyage model and the vector's size, and for LLM answers the way
    they're paid for and the AI model."""

    id: str
    kind: str
    made_with: dict = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        """How many numbers each vector has. 0 for LLM answers."""
        size = self.made_with.get("dimensions")
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else 0


def embedding(name: str, model: str, dimensions: int) -> Feature:
    """The vectors of one text feature, such as the prompt, made by one Voyage model at one size."""
    made_with = {"field": name, "model": model, "dimensions": int(dimensions)}
    return Feature(f"embedding.{_safe(name)}.{_safe(model)}.{int(dimensions)}", EMBEDDING, made_with)


def answers(provider: str, model: str) -> Feature:
    """What one AI model said about tasks, paid for one way, such as Codex's own model through a subscription."""
    return Feature(f"llm.{_safe(provider)}.{_safe(model)}", LLM, {"provider": provider, "model": model})


def _safe(name: str) -> str:
    """A name as a feature's id holds it: lowercase letters, digits, dots, dashes and underscores. A name with
    anything else, such as a capital letter or a slash, gets a short hash of itself, so no two names share an id."""
    plain = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip(".-")[:60]
    if plain and plain == name:
        return plain
    digest = hashlib.sha256(name.encode("utf-8", "surrogatepass")).hexdigest()[:8]
    return f"{plain}-{digest}" if plain else digest


def folder() -> Path:
    """Where the store lives: in the yardstick folder, so YARDSTICK_HOME moves it too."""
    return store.home() / "features"


def _private_folder() -> Path:
    """The store's folder, made readable only by you, as the rest of the yardstick folder is."""
    found = store.private_home() / "features"
    found.mkdir(exist_ok=True)
    if os.name == "posix":
        os.chmod(found, 0o700)
    return found


# Notes ----------------------------------------------------------------------------------------------------

tell: Callable[[str], None] | None = None
"""Where notes about the store go, such as a file that can't be read. None shows them in the terminal."""
_said: set[str] = set()


def _note(text: str) -> None:
    """Say something about the store, once in this process."""
    if text in _said:
        return
    _said.add(text)
    if tell is not None:
        tell(text)
        return
    from . import ui

    ui.note(text)


# The index ------------------------------------------------------------------------------------------------


def _index_path() -> Path:
    return folder() / "index.json"


def _parsed(text: str) -> dict[str, dict] | None:
    """The features an index lists, by id. None when it isn't an index."""
    try:
        found = json.loads(text)
    except ValueError:
        return None
    if not isinstance(found, dict) or not isinstance(found.get("features"), dict):
        return None
    return {key: entry for key, entry in found["features"].items() if isinstance(key, str) and isinstance(entry, dict)}


def _saved_index() -> tuple[dict[str, dict] | None, str]:
    """The index as saved, and why not when it can't be used: missing or broken."""
    try:
        text = _index_path().read_text()
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError):
        return None, "broken"
    found = _parsed(text)
    return found, "" if found is not None else "broken"


def _entries() -> dict[str, dict]:
    """The index, rebuilt from the files when it's missing or broken. Hold the lock: a rebuilt index is saved."""
    found, why = _saved_index()
    if found is not None:
        return found
    rebuilt = _rebuilt()
    if why == "missing" and not rebuilt:
        return {}  # nothing stored yet
    _note(f"The feature store's index, {_index_path()}, was {why}, so it was rebuilt from the files.")
    _write_index(rebuilt)
    return rebuilt


def index() -> dict[str, dict]:
    """The features that should exist, by id: each one's kind, what it was made with and when it last changed.

    A missing or broken index is rebuilt from the files' own metadata, with a note.
    """
    found, _ = _saved_index()
    if found is not None:
        return found
    if not folder().is_dir():
        return {}
    with _held():
        return _entries()


def _write_index(entries: dict[str, dict]) -> None:
    """Save the index in one step. What a newer version added to it is kept."""
    path = _index_path()
    try:
        kept = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        kept = {}
    kept = kept if isinstance(kept, dict) else {}
    saved = kept.get("format")
    newer = isinstance(saved, int) and not isinstance(saved, bool) and saved > FORMAT
    try:
        files.write_text(path, json.dumps({**kept, "format": saved if newer else FORMAT, "features": entries}) + "\n")
    except OSError as error:
        # The rows are saved all the same. The next save, or the listing, tries again.
        _note(f"Couldn't save the feature store's index, {path} ({error.strerror or error}).")


def _rebuilt() -> dict[str, dict]:
    """The index, as the files' own metadata tell it. A folder whose files name another feature is left out."""
    found: dict[str, dict] = {}
    try:
        folders = sorted(path for path in folder().iterdir() if path.is_dir())
    except OSError:
        return found
    for where in folders:
        data, parts = _files(where)
        for path in ([data] if data else []) + [path for _, path in parts]:
            try:
                meta = _meta(pq.read_schema(path))
            except Exception:  # a broken file: another one may tell
                continue
            if meta is not None and meta.get("id") == where.name and meta.get("kind") in (EMBEDDING, LLM):
                made_with = meta.get("made_with") if isinstance(meta.get("made_with"), dict) else {}
                found[where.name] = {"kind": meta["kind"], "made_with": made_with, "changed_at": _changed(where)}
                break
    return found


def _changed(where: Path) -> float | None:
    """When a feature's files last changed. None when it has none."""
    data, parts = _files(where)
    times = []
    for path in ([data] if data else []) + [path for _, path in parts]:
        try:
            times.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


# Writing --------------------------------------------------------------------------------------------------


@contextmanager
def _held() -> Iterator[Path]:
    """Hold the store's lock. Saves, compactions and deletes take turns, in threads and processes alike."""
    where = _private_folder()
    with locks.held(where / "store.lock"):
        yield where


@contextmanager
def writing(wait: bool = True) -> Iterator[bool]:
    """Hold the lock of the store's users, shared, while a command that may save to the store runs, and yield True.

    Setup and refresh hold it from before they first read the store, to plan what to make, until they're done. The
    feature check and the listings hold it while they move a predictor an older version made into the store. Many
    commands hold it at once. A features delete holds it alone, so it never deletes what a running command planned
    with or saves. A command that finds a delete running waits for it, with a note, or without `wait` yields False
    at once and holds nothing. Windows has no shared locks, so there the commands that hold it take turns.
    """
    path = _private_folder() / "users.lock"
    with locks.held(path, wait=False, shared=True) as mine:
        if mine or not wait:
            yield mine
            return
    _note("Another command is using the feature store. Waiting for it to finish.")
    with locks.held(path, shared=True) as mine:
        yield mine


@contextmanager
def deleting() -> Iterator[bool]:
    """Hold the lock of the store's users alone, without waiting, while a feature is deleted, and yield True. It
    yields False, and holds nothing, while a command that may save to the store runs, as writing() says."""
    with locks.held(_private_folder() / "users.lock", wait=False) as mine:
        yield mine


def _schema(feature: Feature) -> pa.Schema:
    if feature.kind == EMBEDDING:
        return pa.schema(
            [
                ("task_id", pa.string()),
                ("text_hash", pa.string()),
                ("vector", pa.list_(pa.float32(), feature.dimensions)),
                ("saved_at", pa.float64()),
            ]
        )
    return pa.schema([("task_id", pa.string()), ("labels", pa.string()), ("saved_at", pa.float64())])


def _keys(feature: Feature) -> list[str]:
    """The columns that name a row. A later row with the same ones replaces it."""
    return ["task_id", "text_hash"] if feature.kind == EMBEDDING else ["task_id"]


def _encode(feature: Feature, table: pa.Table, last_part: int | None = None) -> bytes:
    """The rows as a Parquet file that names its feature. A data file also says up to which part it merged.

    Each page of the file carries a checksum of its data, so a reader notices a byte changed on disk, as by a
    failing disk, instead of reading a wrong vector.
    """
    meta = {"format": FORMAT, "id": feature.id, "kind": feature.kind, "made_with": feature.made_with}
    if last_part is not None:
        meta["last_part"] = last_part
    sink = pa.BufferOutputStream()
    pq.write_table(
        table.replace_schema_metadata({_KEY: json.dumps(meta)}), sink, compression="zstd", write_page_checksum=True
    )
    return sink.getvalue().to_pybytes()


def _save(feature: Feature, table: pa.Table) -> None:
    """Add rows to a feature in one new part file, then note the feature in the index as changed now. Hold the
    lock."""
    # The index is read first, so a store saved to for the first time isn't taken for one whose index was lost.
    entries = _entries()
    where = folder() / feature.id
    where.mkdir(parents=True, exist_ok=True)
    data, parts = _files(where)
    number = max([_last_part(data), *(n for n, _ in parts)]) + 1
    files.write_bytes(where / f"part-{number}.parquet", _encode(feature, table))
    entry = entries.get(feature.id) or {}
    entries[feature.id] = {**entry, "kind": feature.kind, "made_with": feature.made_with, "changed_at": time.time()}
    _write_index(entries)
    if len(parts) + 1 > MAX_PARTS:
        _compact(feature)


def _last_part(data: Path | None) -> int:
    """The newest part a data file merged. 0 without one, or when it can't be read."""
    if data is None:
        return 0
    try:
        meta = _meta(pq.read_schema(data))
    except Exception:
        return 0
    found = (meta or {}).get("last_part")
    return found if isinstance(found, int) and not isinstance(found, bool) and found > 0 else 0


def save_vectors(feature: Feature, rows: list[tuple[str, str, object]], only_new: bool = False) -> int:
    """Save vectors, each with its task's id and the hash of the text it was made from. Returns how many were saved.

    A vector replaces one saved before for the same task and text. With `only_new`, it's saved only when the
    store has none for that task and text yet, as when an older version's vectors move in.
    """
    if not rows:
        return 0
    size = feature.dimensions
    vectors = []
    for _, _, vector in rows:
        found = np.asarray(vector, dtype=np.float32).reshape(-1)
        if found.size != size:
            raise ValueError(f"A vector of {found.size} numbers can't be kept with vectors of {size}.")
        vectors.append(found)
    with _held():
        kept = list(zip(rows, vectors))
        if only_new:
            have = vector_hashes(feature, {task_id for task_id, _, _ in rows})
            kept = [(row, vector) for row, vector in kept if row[1] not in have.get(row[0], ())]
        if not kept:
            return 0
        values = pa.array(np.concatenate([vector for _, vector in kept]), type=pa.float32())
        table = pa.table(
            {
                "task_id": pa.array([str(row[0]) for row, _ in kept], type=pa.string()),
                "text_hash": pa.array([str(row[1]) for row, _ in kept], type=pa.string()),
                "vector": pa.FixedSizeListArray.from_arrays(values, size),
                "saved_at": pa.array([time.time()] * len(kept), type=pa.float64()),
            },
            schema=_schema(feature),
        )
        _save(feature, table)
    return len(kept)


def save_answers(
    feature: Feature, found: dict[str, dict], merge: Callable[[dict | None, dict], dict] | None = None
) -> int:
    """Save what an AI model said about tasks, by task id. Returns how many tasks' answers changed.

    An answer replaces one saved before for the same task. With `merge`, it's merged with the one the store holds
    for the task: `merge` gets that one, or None, and the new one, and gives what to keep. The store's answers are
    read, merged and saved while its lock is held, so two predictors that save answers about the same tasks at
    once, such as one that asked for the kind of task and one that asked for a description, keep each other's.
    An answer the same as the stored one isn't saved again.
    """
    rows = {str(task_id): labels for task_id, labels in found.items() if isinstance(labels, dict)}
    if not rows:
        return 0
    with _held():
        have = answers_of(feature, set(rows))
        if merge is not None:
            rows = {task_id: merge(have.get(task_id), labels) for task_id, labels in rows.items()}
        rows = {task_id: labels for task_id, labels in rows.items() if labels != have.get(task_id)}
        if not rows:
            return 0
        table = pa.table(
            {
                "task_id": pa.array(list(rows), type=pa.string()),
                "labels": pa.array([json.dumps(labels) for labels in rows.values()], type=pa.string()),
                "saved_at": pa.array([time.time()] * len(rows), type=pa.float64()),
            },
            schema=_schema(feature),
        )
        _save(feature, table)
    return len(rows)


def compact(feature: Feature) -> None:
    """Merge a feature's part files into its data file, in one step, then remove the parts.

    A run that saves many times, such as embedding or LLM analysis, compacts at its end.
    """
    with _held():
        _compact(feature)


def _compact(feature: Feature) -> None:
    """Compact a feature. Hold the lock. A feature with a file this version can't open, or one a newer version
    saved, is left as it is: its rows may still be needed."""
    where = folder() / feature.id
    data, parts = _files(where)
    if not parts:
        return
    found = _collect(feature, None, None, lenient=True)
    if found.kept_back:
        return
    last = max([found.last_part, *(n for n, _ in parts)])
    merged = found.table if found.table is not None else _schema(feature).empty_table()
    files.write_bytes(where / "data.parquet", _encode(feature, merged, last_part=last))
    for path in [path for _, path in parts] + found.broken_parts + list(where.glob(".*.partial")):
        try:
            path.unlink()
        except OSError:
            pass  # on Windows, a file a reader has open stays. It's merged already, so readers skip it.


def delete(feature_id: str) -> bool:
    """Delete a stored feature: its folder and its line in the index. False when there was nothing to delete.

    Whatever used it makes it again when it needs it: embeddings come back from the cache of vectors for free,
    and LLM analysis asks about the tasks again, with the cost shown first.
    """
    if not _ID.fullmatch(feature_id):
        raise ValueError(f"{feature_id!r} isn't the id of a stored feature.")
    with _held():
        where = folder() / feature_id
        entries = _entries()
        existed = where.is_dir() or feature_id in entries
        if where.is_dir():
            shutil.rmtree(where)
        if feature_id in entries:
            del entries[feature_id]
            _write_index(entries)
    return existed


# Reading --------------------------------------------------------------------------------------------------


def _files(where: Path) -> tuple[Path | None, list[tuple[int, Path]]]:
    """A feature's data file, when it has one, and its part files in the order they were saved."""
    try:
        names = os.listdir(where)
    except OSError:  # no folder, as for a feature never saved, or one deleted by hand
        return None, []
    data = where / "data.parquet" if "data.parquet" in names else None
    parts = sorted((int(match.group(1)), where / name) for name in names if (match := _PART.fullmatch(name)))
    return data, parts


def _meta(schema: pa.Schema) -> dict | None:
    """What a file says about the feature its rows belong to. None when it says nothing."""
    try:
        found = json.loads((schema.metadata or {})[_KEY])
    except (KeyError, TypeError, ValueError):
        return None
    return found if isinstance(found, dict) else None


class _Gone(Exception):
    """A file listed a moment ago is gone, as when a compaction merged it meanwhile."""


@dataclass
class _Found:
    """What reading a feature's files found."""

    table: pa.Table | None = None
    last_part: int = 0
    unreadable: list[Path] = field(default_factory=list)
    """Files skipped because they can't be read."""
    broken_parts: list[Path] = field(default_factory=list)
    """Part files that open but hold no rows of this feature. A compaction removes them."""
    kept_back: bool = False
    """A file that couldn't be opened, or one a newer version saved: a compaction leaves the feature alone."""


def _problem(feature: Feature, schema: pa.Schema, meta: dict | None) -> str | None:
    """Why a file's rows can't be read as this feature's, in plain words. None when they can."""
    if meta is None or meta.get("id") != feature.id or meta.get("kind") != feature.kind:
        return "it holds no rows of this feature"
    saved = meta.get("format")
    if isinstance(saved, int) and not isinstance(saved, bool) and saved > FORMAT:
        return "a newer version of yardstick-predict saved it"
    for column in _schema(feature):
        at = schema.get_field_index(column.name)
        if at < 0 or schema.field(at).type != column.type:
            return "its columns aren't the ones this version reads"
    return None


def _load(feature: Feature, path: Path, columns, task_ids, lenient: bool, found: _Found) -> pa.Table | None:
    """One file's rows, or None when they can't be read, with a note naming the file.

    Every column this version knows is read, whichever `columns` are asked for, and each page's checksum is
    checked when the file has them. So a file whose vectors can't be read, or changed since they were saved,
    has none of its rows counted, even when only its task ids are asked for. A row without a vector counts as
    missing too. Whatever needs them makes them again.
    """
    try:
        raw = path.read_bytes()  # read whole and closed at once, so a compaction on Windows can replace it
    except FileNotFoundError:
        if lenient:
            return None
        raise _Gone from None
    except OSError as error:
        _note(f"Skipped {path}: it can't be opened ({error.strerror or error}).")
        found.unreadable.append(path)
        found.kept_back = True
        return None
    try:
        parquet = pq.ParquetFile(pa.BufferReader(raw), page_checksum_verification=True)
        meta = _meta(parquet.schema_arrow)
        problem = _problem(feature, parquet.schema_arrow, meta)
        if problem is None:
            # Only the columns this version knows, so a file with one more, from a newer version, still fits.
            table = parquet.read(columns=[column.name for column in _schema(feature)])
    except Exception:  # anything a broken file can make Parquet raise
        meta, problem = None, "it isn't a Parquet file this version can read"
    if problem is not None:
        _note(f"Skipped {path}: {problem}.")
        found.unreadable.append(path)
        if problem.startswith("a newer version"):
            found.kept_back = True
        elif path.name != "data.parquet":
            found.broken_parts.append(path)
        return None
    if path.name == "data.parquet":
        merged = (meta or {}).get("last_part")
        found.last_part = merged if isinstance(merged, int) and not isinstance(merged, bool) else 0
    if feature.kind == EMBEDDING:
        table = table.filter(pc.is_valid(table["vector"]))
    if task_ids is not None:
        table = table.filter(pc.is_in(table["task_id"], value_set=task_ids))
    return table.select(columns) if columns else table


def _collect(feature: Feature, columns, task_ids, lenient: bool) -> _Found:
    """Read a feature's data file and then its part files, oldest first, keeping the newest row for each key."""
    found = _Found()
    data, parts = _files(folder() / feature.id)
    wanted = None if task_ids is None else pa.array(sorted(task_ids), type=pa.string())
    tables = []
    if data is not None:
        table = _load(feature, data, columns, wanted, lenient, found)
        if table is not None:
            tables.append(table)
    for number, path in parts:
        if number <= found.last_part:
            continue  # merged into the data file already, by a compaction that couldn't remove it
        table = _load(feature, path, columns, wanted, lenient, found)
        if table is not None:
            tables.append(table)
    if tables:
        found.table = _latest(pa.concat_tables(tables), _keys(feature))
    return found


def _latest(table: pa.Table, keys: list[str]) -> pa.Table:
    """One row for each key: the one saved last."""
    if table.num_rows == 0:
        return table
    numbered = table.append_column("_row", pa.array(np.arange(table.num_rows, dtype=np.int64)))
    last = numbered.group_by(keys, use_threads=False).aggregate([("_row", "max")])
    return table.take(pa.array(np.sort(last["_row_max"].to_numpy())))


def _read(feature: Feature, columns: list[str] | None = None, task_ids=None) -> _Found:
    """A feature's rows, with only `columns` and the tasks `task_ids` names, when given.

    A file a compaction merged while this read it is gone, so it reads the files again, once.
    """
    try:
        return _collect(feature, columns, task_ids, lenient=False)
    except _Gone:
        return _collect(feature, columns, task_ids, lenient=True)


def vector_hashes(feature: Feature, task_ids=None) -> dict[str, set[str]]:
    """For each task, the hashes of the texts it has a vector for: only the tasks `task_ids` names, when given.

    A task counts only when its vector reads, as vectors() reads it, so a vector a damaged file holds is made again.
    """
    table = _read(feature, ["task_id", "text_hash"], task_ids).table
    found: dict[str, set[str]] = {}
    if table is None:
        return found
    for task_id, digest in zip(table["task_id"].to_pylist(), table["text_hash"].to_pylist()):
        found.setdefault(task_id, set()).add(digest)
    return found


def vectors(feature: Feature, task_ids=None) -> dict[str, dict[str, np.ndarray]]:
    """For each task, its vectors, by the hash of the text each was made from: only the tasks `task_ids` names,
    when given."""
    table = _read(feature, None, task_ids).table
    found: dict[str, dict[str, np.ndarray]] = {}
    if table is None or table.num_rows == 0:
        return found
    values = table["vector"].combine_chunks().flatten().to_numpy(zero_copy_only=False)
    values = values.reshape(-1, feature.dimensions)
    for row, (task_id, digest) in enumerate(zip(table["task_id"].to_pylist(), table["text_hash"].to_pylist())):
        found.setdefault(task_id, {})[digest] = values[row]
    return found


def answers_of(feature: Feature, task_ids=None) -> dict[str, dict]:
    """What the AI model said about each task, by task id: only the tasks `task_ids` names, when given."""
    table = _read(feature, None, task_ids).table
    found: dict[str, dict] = {}
    if table is None:
        return found
    for task_id, text in zip(table["task_id"].to_pylist(), table["labels"].to_pylist()):
        try:
            labels = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(labels, dict):
            found[task_id] = labels
    return found


# The listing ----------------------------------------------------------------------------------------------


@dataclass
class Stored:
    """One stored feature as the listing shows it."""

    feature: Feature
    missing: bool
    """The index lists it, but its folder is gone. It has no rows until a step that uses it makes them again."""
    tasks: int
    """How many tasks have a value, for any predictor."""
    rows: int
    size: int
    """Bytes on disk."""
    changed_at: float | None
    unreadable: list[str]
    """The files skipped because they can't be read."""


def stored() -> list[Stored]:
    """Every stored feature: the ones the index lists, and any folder whose files name a feature the index lacks,
    as when a save stopped before it could note itself."""
    entries = index()
    for key, entry in _rebuilt().items():
        entries.setdefault(key, entry)
    found = []
    for key, entry in sorted(entries.items()):
        if entry.get("kind") not in (EMBEDDING, LLM) or not _ID.fullmatch(key):
            continue  # a kind a newer version keeps
        made_with = entry.get("made_with") if isinstance(entry.get("made_with"), dict) else {}
        feature = Feature(key, entry["kind"], made_with)
        where = folder() / key
        read = _read(feature, _keys(feature)) if where.is_dir() else _Found()
        table = read.table
        size = 0
        for path in where.iterdir() if where.is_dir() else []:
            try:
                size += path.stat().st_size if path.is_file() else 0
            except OSError:
                continue
        changed = entry.get("changed_at")
        found.append(
            Stored(
                feature=feature,
                missing=not where.is_dir(),
                tasks=0 if table is None else len(set(table["task_id"].to_pylist())),
                rows=0 if table is None else table.num_rows,
                size=size,
                changed_at=changed if isinstance(changed, (int, float)) and not isinstance(changed, bool) else None,
                unreadable=[path.name for path in read.unreadable],
            )
        )
    return found
