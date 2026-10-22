"""Where each predictor keeps its settings, its dataset and its model.

Everything lives in one folder per predictor, under ~/.yardstick/predictors by default.
Set YARDSTICK_HOME to keep it somewhere else.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import astuple, fields
from pathlib import Path

import numpy as np

from . import files
from .archives import Task

NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
"""A predictor's name. Check it with fullmatch: match and $ would let a name with a newline at its end through."""
DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", *(f"com{n}" for n in range(10)), *(f"lpt{n}" for n in range(10))})
"""Names Windows keeps for devices. No folder can have one there, so no predictor can, on any computer."""
_COLUMNS = [f.name for f in fields(Task)]
_TASK_ROW = f"({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})"
"""A task's values, with each column named. A column added to an older dataset sits at the end, not in Task's order."""
_TEXTS = ("prompt", "context", "skills", "prior_skills")
"""A task's texts. They take most of the room."""


def home() -> Path:
    return Path(os.environ.get("YARDSTICK_HOME") or "~/.yardstick").expanduser()


def private_home() -> Path:
    """The yardstick folder, readable only by you. It holds your prompts."""
    folder = home()
    folder.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(folder, 0o700)
    elif os.name == "nt":
        private_on_windows(folder)
    return folder


def private_on_windows(folder: Path, run=subprocess.run) -> bool:
    """Let only you open the folder on Windows, with Windows' own icacls. It never stops a command.

    It takes away the access the folder inherits from the folders above it and gives you full access, to the
    folder and everything in it. A marker file says it worked, so later commands don't run icacls again.
    """
    marker = folder / ".private"
    if marker.exists():
        return True
    user = os.environ.get("USERNAME")
    if not user:
        return False
    domain = os.environ.get("USERDOMAIN")
    who = f"{domain}\\{user}" if domain else user
    command = ["icacls", str(folder), "/inheritance:r", "/grant:r", f"{who}:(OI)(CI)F"]
    try:
        # 0x08000000 opens no console window, as when a forecast runs in the background for the status line.
        done = run(command, capture_output=True, timeout=30, creationflags=0x08000000)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if done.returncode != 0:
        return False
    try:
        marker.touch()
    except OSError:
        pass
    return True


def name_problem(name: str) -> str | None:
    """Why a predictor can't have this name, in plain words. None when it can."""
    if not NAME_PATTERN.fullmatch(name):
        return "Use 1 to 40 lowercase letters, digits, dashes or underscores for the name."
    if name in DEVICE_NAMES:
        return f"Windows keeps the name {name} for a device, so no folder can have it. Pick another name."
    return None


def predictor_folder(name: str) -> Path:
    """Where a predictor with this name keeps everything, whether or not it exists yet."""
    return home() / "predictors" / name


def predictor_names() -> list[str]:
    root = home() / "predictors"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "predictor.json").is_file())


def settings() -> dict:
    """Your yardstick settings, such as the default predictor. Empty when there are none or they can't be read."""
    try:
        found = json.loads((home() / "settings.json").read_text())
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def save_setting(name: str, value) -> None:
    """Change one setting, and keep the others. The file is saved in one step."""
    path = private_home() / "settings.json"
    files.write_text(path, json.dumps({**settings(), name: value}, indent=2) + "\n")


def default_name() -> str | None:
    """The predictor a command uses when it names none: the one you chose, else 'default', else the only one."""
    names = predictor_names()
    chosen = settings().get("default")
    if chosen in names:
        return chosen
    if "default" in names:
        return "default"
    return names[0] if len(names) == 1 else None


def set_default(name: str) -> None:
    save_setting("default", name)


class BrokenPredictor(Exception):
    """A predictor whose settings file can't be read, or is empty or broken, so it can't be used."""


def _read_config(name: str, path: Path) -> dict:
    """A predictor's settings. Raises BrokenPredictor with a plain reason when they can't be used.

    An older version saved them in place, so one stopped halfway through a save could leave them empty.
    """
    try:
        text = path.read_text()
    except OSError as error:
        why = f"couldn't be read ({error.strerror or error})"
    else:
        try:
            config = json.loads(text)
        except ValueError:
            config = None
        embedding = config.get("embedding") if isinstance(config, dict) else None
        if isinstance(embedding, dict) and isinstance(config.get("archives"), list) and embedding.get("model"):
            return config
        why = "are empty or broken"
    raise BrokenPredictor(
        f"Predictor {name!r} can't be used: its settings in {path} {why}. Fix that file, "
        f"or delete the folder {path.parent} and create the predictor again."
    )


class Predictor:
    """One predictor: its archives, its embedding model, its dataset and its trained model."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.dir = predictor_folder(name)
        self.config: dict = _read_config(name, self.dir / "predictor.json")
        self._db: sqlite3.Connection | None = None

    @classmethod
    def create(cls, name: str, archives: list[dict], embedding: dict, **settings) -> Predictor:
        """Make a new predictor. Its whole config, `settings` included, is saved in one step.

        So a command that finds the new predictor, such as a refresh, never sees it half made.
        """
        problem = name_problem(name)
        if problem:
            raise ValueError(problem)
        private_home()
        folder = predictor_folder(name)
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
            **settings,
        }
        files.write_text(folder / "predictor.json", json.dumps(config, indent=2) + "\n")
        return cls(name)

    def save_config(self) -> None:
        files.write_text(self.dir / "predictor.json", json.dumps(self.config, indent=2) + "\n")

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
            self._embeddings_table()
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
        """How many tasks each AI model ran at each effort level, the most used first.

        It counts the tasks the model learns from, as pipeline.learnable picks them: finished tasks that took
        some time and tokens, and with max_tasks only the newest that many.
        """
        limit = int(self.config.get("max_tasks") or 0) or -1  # -1 is no limit
        rows = self.db.execute(
            "SELECT model, effort, COUNT(*) FROM ("
            "SELECT model, effort FROM tasks WHERE NOT stopped AND NOT idle AND seconds > 0 AND model != '' "
            "AND input_tokens + cache_write_tokens + cache_read_tokens + output_tokens > 0 "
            "ORDER BY started_at DESC, id DESC LIMIT ?"
            ") GROUP BY model, effort ORDER BY COUNT(*) DESC, effort",
            (limit,),
        )
        found: dict[str, list[tuple[str, int]]] = {}
        for model, effort, count in rows:
            found.setdefault(model, []).append((effort, count))
        return found

    def task_ids(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT id FROM tasks")}

    def tasks(self, texts: bool = True) -> list[Task]:
        """Every task, oldest first. Without `texts`, its prompt, context and skills are left empty: quicker,
        when only what it took is needed, as for what came before a forecast."""
        columns = [name if texts or name not in _TEXTS else f"'' AS {name}" for name in _COLUMNS]
        rows = self.db.execute(f"SELECT {', '.join(columns)} FROM tasks ORDER BY started_at, id")
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

    def _embeddings_table(self) -> None:
        """Make the embeddings table, or bring one made by an older version up to date.

        Each vector is kept with the hash of the text it was made from, so a vector whose text changed is
        made again. An older version kept no hash. For the prompt and the context, whose texts are built as
        that version built them, the hash of each task's text as stored now is filled in, in one step and
        before this process replaces any task. Another feature's old vectors have no hash, so they are made
        again once. A text the cache of vectors holds costs nothing.
        """
        db = self._db
        db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (task_id, field, vector, text_hash, PRIMARY KEY (task_id, field))"
        )
        if "text_hash" in {row[1] for row in db.execute("PRAGMA table_info(embeddings)")}:
            return
        hashes = self._old_text_hashes()
        db.execute("SAVEPOINT text_hashes")  # the new column and its hashes arrive together, or not at all
        try:
            db.execute("ALTER TABLE embeddings ADD COLUMN text_hash")
            db.executemany(
                "UPDATE embeddings SET text_hash = ? WHERE task_id = ? AND field = ? AND text_hash IS NULL", hashes
            )
        except sqlite3.OperationalError:
            db.execute("ROLLBACK TO text_hashes")  # another process brought it up to date first, or it can't change
        db.execute("RELEASE text_hashes")

    def _old_text_hashes(self) -> list[tuple[str, str, str]]:
        """The hash of each stored task's prompt and context, as a vector made from them was made."""
        from . import embeddings, features

        rows = []
        try:
            for task in self.tasks():
                info = features.TaskInfo.of(task)
                for name in ("prompt", "context"):
                    text = features.text(name, info)
                    if text.strip():
                        rows.append((embeddings.text_hash(text), task.id, name))
        except Exception:
            return []  # then the old vectors keep no hash, and are made again
        return rows

    def embedded(self, field: str) -> Embedded:
        """The tasks that have a vector for a feature, with the hash of the text each was made from."""
        rows = self.db.execute("SELECT task_id, text_hash FROM embeddings WHERE field = ?", (field,))
        return Embedded(dict(rows))

    def put_embeddings(self, field: str, vectors: dict[str, np.ndarray], hashes: dict[str, str] | None = None) -> None:
        """Save each task's vector for a feature, with the hash of the text it was made from."""
        hashes = hashes or {}
        self.db.executemany(
            "INSERT OR REPLACE INTO embeddings (task_id, field, vector, text_hash) VALUES (?, ?, ?, ?)",
            [
                (task_id, field, np.asarray(v, dtype=np.float32).tobytes(), hashes.get(task_id))
                for task_id, v in vectors.items()
            ],
        )
        self.db.commit()

    def forget_embeddings(self, field: str, task_ids) -> None:
        """Delete these tasks' vectors for a feature, as when a task no longer has a text for it."""
        self.db.executemany("DELETE FROM embeddings WHERE task_id = ? AND field = ?", [(t, field) for t in task_ids])
        self.db.commit()

    def forget_task_vectors(self, task_ids) -> None:
        """Delete every vector of these tasks, as when the tasks themselves are deleted."""
        self.db.executemany("DELETE FROM embeddings WHERE task_id = ?", [(t,) for t in task_ids])
        self.db.commit()

    def remove_tasks(self, task_ids) -> None:
        """Delete tasks and their vectors, as when a re-read of their records no longer finds them."""
        task_ids = list(task_ids)
        if not task_ids:
            return
        self.db.executemany("DELETE FROM tasks WHERE id = ?", [(t,) for t in task_ids])
        self.db.commit()
        self.forget_task_vectors(task_ids)

    def get_embeddings(self, field: str) -> dict[str, np.ndarray]:
        rows = self.db.execute("SELECT task_id, vector FROM embeddings WHERE field = ?", (field,))
        return {task_id: np.frombuffer(blob, dtype=np.float32) for task_id, blob in rows}

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


class Embedded(set):
    """The tasks that have a vector for one feature. `hashes` says which text each task's vector was made from.

    It is a set of task ids, so code that only asks whether a task has a vector works as before. A vector
    an older version saved has no hash: its value is None.
    """

    def __init__(self, hashes: dict[str, str | None]) -> None:
        super().__init__(hashes)
        self.hashes = hashes
