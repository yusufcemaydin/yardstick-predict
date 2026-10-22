"""Where each predictor keeps its settings, its dataset and its model.

Everything lives in one folder per predictor, under ~/.yardstick/predictors by default.
Set YARDSTICK_HOME to keep it somewhere else.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import astuple, fields
from pathlib import Path

import numpy as np

from .archives import Task

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_COLUMNS = [f.name for f in fields(Task)]
_TASK_ROW = f"({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})"
"""A task's values, with each column named. A column added to an older dataset sits at the end, not in Task's order."""


def home() -> Path:
    return Path(os.environ.get("YARDSTICK_HOME") or "~/.yardstick").expanduser()


def private_home() -> Path:
    """The yardstick folder, readable only by you. It holds your prompts."""
    folder = home()
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(folder, 0o700)
    return folder


def predictor_names() -> list[str]:
    root = home() / "predictors"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "predictor.json").is_file())


def default_name() -> str | None:
    """The predictor a command uses when it names none: the one you chose, else 'default', else the only one."""
    names = predictor_names()
    try:
        chosen = json.loads((home() / "settings.json").read_text()).get("default")
    except (OSError, ValueError, AttributeError):
        chosen = None
    if chosen in names:
        return chosen
    if "default" in names:
        return "default"
    return names[0] if len(names) == 1 else None


def set_default(name: str) -> None:
    path = private_home() / "settings.json"
    try:
        settings = json.loads(path.read_text())
    except (OSError, ValueError):
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    settings["default"] = name
    path.write_text(json.dumps(settings, indent=2) + "\n")


class Predictor:
    """One predictor: its archives, its embedding model, its dataset and its trained model."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = home() / "predictors" / name
        self.config: dict = json.loads((self.dir / "predictor.json").read_text())
        self._db: sqlite3.Connection | None = None

    @classmethod
    def create(cls, name: str, archives: list[dict], embedding: dict) -> Predictor:
        if not NAME_PATTERN.match(name):
            raise ValueError("Use 1 to 40 lowercase letters, digits, dashes or underscores for the name.")
        folder = private_home() / "predictors" / name
        if (folder / "predictor.json").exists():
            raise ValueError(f"A predictor called {name!r} already exists.")
        folder.mkdir(parents=True, exist_ok=True)
        config = {
            "format": 1,
            "name": name,
            "created_at": time.time(),
            "embedding": embedding,
            "archives": archives,
            "refreshed_at": None,
        }
        (folder / "predictor.json").write_text(json.dumps(config, indent=2) + "\n")
        return cls(name)

    def save_config(self) -> None:
        (self.dir / "predictor.json").write_text(json.dumps(self.config, indent=2) + "\n")

    @property
    def embedding_model(self) -> str:
        return self.config["embedding"]["model"]

    @property
    def model_dir(self) -> Path:
        return self.dir / "model"

    # The dataset ------------------------------------------------------------------------------------

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._db = sqlite3.connect(self.dir / "dataset.sqlite")
            self._db.execute(f"CREATE TABLE IF NOT EXISTS tasks ({', '.join(_COLUMNS)}, PRIMARY KEY (id))")
            # A dataset made by an older version may lack newer columns. Add them, empty.
            existing = {row[1] for row in self._db.execute("PRAGMA table_info(tasks)")}
            for column in _COLUMNS:
                if column not in existing:
                    self._db.execute(f"ALTER TABLE tasks ADD COLUMN {column} DEFAULT ''")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (task_id, field, vector, PRIMARY KEY (task_id, field))"
            )
            self._db.execute("CREATE TABLE IF NOT EXISTS files (path PRIMARY KEY, size, mtime)")
            self._db.execute("CREATE TABLE IF NOT EXISTS llm_labels (task_id PRIMARY KEY, labels)")
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def add_tasks(self, tasks: list[Task]) -> int:
        """Add tasks the dataset doesn't have yet. A task already there, such as a copy, is skipped."""
        before = self.count_tasks()
        self.db.executemany(f"INSERT OR IGNORE INTO tasks {_TASK_ROW}", [astuple(t) for t in tasks])
        self.db.commit()
        return self.count_tasks() - before

    def replace_tasks(self, tasks: list[Task]) -> None:
        """Store newer copies of tasks the dataset already has, such as a parent whose helper finished later."""
        self.db.executemany(f"INSERT OR REPLACE INTO tasks {_TASK_ROW}", [astuple(t) for t in tasks])
        self.db.commit()

    def known_calls(self) -> dict[str, int]:
        """How many model calls each stored task has."""
        return dict(self.db.execute("SELECT id, calls FROM tasks"))

    def stopped_ids(self) -> set[str]:
        """The stored tasks marked stopped."""
        return {row[0] for row in self.db.execute("SELECT id FROM tasks WHERE stopped")}

    def count_tasks(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    def effort_counts(self) -> dict[str, list[tuple[str, int]]]:
        """How many finished tasks each AI model ran at each effort level, the most used first."""
        rows = self.db.execute(
            "SELECT model, effort, COUNT(*) FROM tasks WHERE NOT stopped AND NOT idle "
            "GROUP BY model, effort ORDER BY COUNT(*) DESC, effort"
        )
        found: dict[str, list[tuple[str, int]]] = {}
        for model, effort, count in rows:
            found.setdefault(model, []).append((effort, count))
        return found

    def task_ids(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT id FROM tasks")}

    def tasks(self) -> list[Task]:
        """Every task, oldest first."""
        rows = self.db.execute(f"SELECT {', '.join(_COLUMNS)} FROM tasks ORDER BY started_at, id")
        found = []
        for row in rows:
            values = dict(zip(_COLUMNS, row))
            for flag in ("direct", "stopped", "idle"):
                values[flag] = bool(values[flag])
            # A task stored by an older version has an empty finish time: it isn't known.
            if values["finished_at"] == "":
                values["finished_at"] = None
            found.append(Task(**values))
        return found

    # Which record files were read, so a refresh reads only what changed ---------------------------------

    def known_files(self) -> dict[str, tuple[int, float]]:
        return {path: (size, mtime) for path, size, mtime in self.db.execute("SELECT path, size, mtime FROM files")}

    def remember_files(self, files: dict[str, tuple[int, float]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", [(p, s, m) for p, (s, m) in files.items()])
        self.db.commit()

    # Embeddings -------------------------------------------------------------------------------------

    def embedded(self, field: str) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT task_id FROM embeddings WHERE field = ?", (field,))}

    def put_embeddings(self, field: str, vectors: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
            [(task_id, field, np.asarray(v, dtype=np.float32).tobytes()) for task_id, v in vectors.items()],
        )
        self.db.commit()

    def get_embeddings(self, field: str) -> dict[str, np.ndarray]:
        rows = self.db.execute("SELECT task_id, vector FROM embeddings WHERE field = ?", (field,))
        return {task_id: np.frombuffer(blob, dtype=np.float32) for task_id, blob in rows}

    def forget_embeddings(self, field: str) -> None:
        """Drop every embedding of one feature, such as one whose text changed. The next refresh embeds it again."""
        self.db.execute("DELETE FROM embeddings WHERE field = ?", (field,))
        self.db.commit()

    # What LLM analysis said about each task -------------------------------------------------------------

    def llm_labels(self) -> dict[str, dict]:
        found = {}
        for task_id, text in self.db.execute("SELECT task_id, labels FROM llm_labels"):
            try:
                labels = json.loads(text)
            except (TypeError, ValueError):
                continue
            if isinstance(labels, dict):
                found[task_id] = labels
        return found

    def put_llm_labels(self, labels: dict[str, dict]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO llm_labels VALUES (?, ?)", [(i, json.dumps(found)) for i, found in labels.items()]
        )
        self.db.commit()
