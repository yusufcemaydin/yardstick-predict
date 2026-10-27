"""Where each predictor keeps its settings, its dataset and its model.

Everything lives in one folder per predictor, under ~/.yardstick/predictors by default.
Set YARDSTICK_HOME to keep it somewhere else. The features that cost money or time to make, the embeddings
and LLM answers, live in the feature store next to it, shared by every predictor: see feature_store.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from . import files, locks
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
FORMAT = 2
"""The format a predictor's settings are saved in. Settings in an older format are brought up to date when read.
Settings in a newer format are read as far as they can be, and never saved."""
PREPARED = "prepared.parquet"
"""The prepared dataset's file, in the predictor's folder."""
LOCKS = ("refresh.lock", "model.lock")
"""The lock files in a predictor's folder. A setup or refresh holds the first, and a model being saved or read
holds the second."""


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


def name_lock(name: str, wait: bool = True):
    """Hold the lock of a predictor's name while the block runs, and yield True. Without `wait`, it yields False at
    once when another command holds it.

    A delete holds it until the predictor's folder is gone, and setup while it makes the folder and saves the
    settings. So a setup of a new predictor with the same name waits for the delete, and the delete never removes
    what setup makes. It sits in the locks folder, outside the predictors folder and readable only by you, so it
    stays when the predictor's folder goes, whatever the predictor is called. Its file is named by a hash of the
    name, so any name gives a short file name. Names a folder can't tell apart, such as Work and work where letter
    case doesn't count, share one lock. It's never deleted: a command waiting for it would then hold a lock no other
    sees.
    """
    root = private_home() / "locks"
    root.mkdir(exist_ok=True)
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", name).casefold())
    digest = hashlib.sha256(folded.encode("utf-8", "surrogatepass")).hexdigest()[:32]
    return locks.held(root / f"predictor-{digest}.lock", wait=wait)


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
    with _settings_lock():
        path = private_home() / "settings.json"
        files.write_text(path, json.dumps({**settings(), name: value}, indent=2) + "\n")


def _settings_lock():
    """The lock each change of the settings holds, from reading them to saving them. Two commands changing
    settings at once, such as one choosing the default while another notes something, then both keep their change.
    It sits in the locks folder, readable only by you."""
    root = private_home() / "locks"
    root.mkdir(exist_ok=True)
    return locks.held(root / "settings.lock")


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


def forget_setting(name: str) -> None:
    """Take one setting out, and keep the others. The file is saved in one step."""
    with _settings_lock():
        found = settings()
        if name in found:
            del found[name]
            files.write_text(private_home() / "settings.json", json.dumps(found, indent=2) + "\n")


# Deleting a predictor -------------------------------------------------------------------------------------


class Busy(Exception):
    """Another command is setting up or refreshing the predictor, so it can't be deleted now."""


@dataclass
class Contents:
    """What a predictor's folder holds: how many tasks its dataset has, and the bytes each part takes."""

    folder: Path
    settings: bool
    """Whether its settings file is there. It may still be broken."""
    tasks: int | None
    """How many tasks its dataset holds: 0 without one, and None when it can't be read."""
    dataset: int
    prepared: int
    model: int
    other: int
    """Everything else, such as its settings and the model facts it downloaded."""
    link_to: Path | None = None
    """Where the folder leads, when it's a link to another folder rather than a folder of its own, as one made by
    hand to give a predictor a second name. Deleting it deletes only the link, so nothing else counts then."""

    @property
    def size(self) -> int:
        return self.dataset + self.prepared + self.model + self.other


def contents(name: str) -> Contents:
    """What a predictor's folder holds, read without changing anything, even when its settings are broken."""
    folder = predictor_folder(name)
    found = (folder / "predictor.json").is_file()
    if is_link(folder):
        return Contents(folder, found, 0, 0, 0, 0, 0, link_to=folder.resolve())
    sizes = {"dataset": 0, "prepared": 0, "model": 0, "other": 0}
    named = {PREPARED: "prepared", "model": "model"}
    for path in _entries(folder):
        # The dataset comes with the files SQLite keeps beside it while it writes.
        part = "dataset" if path.name.startswith("dataset.sqlite") else named.get(path.name, "other")
        sizes[part] += _bytes(path)
    return Contents(folder, found, _count_tasks(folder / "dataset.sqlite"), **sizes)


def delete_predictor(name: str) -> bool:
    """Delete a predictor's folder: its settings, its dataset of tasks, its prepared dataset and its model. False
    when the folder was gone already.

    Nothing outside the folder changes, except that a default setting naming the predictor is forgotten, and the
    name's lock, an empty file kept apart from the predictors' folders, stays: your records, the feature store and
    the other predictors stay as they are. Nothing is ever deleted through a link:
    a folder that is itself a link to another folder has only the link deleted, and a link inside the folder goes
    without what it links to. It raises Busy while a setup or refresh of the predictor runs, and waits for a
    forecast that is reading its model. The settings go last, so a delete that stops halfway, as on a file another
    program has open, leaves the predictor listed, to delete again. The name's lock is held until the folder is
    gone, so a setup of a new predictor with the same name waits, and keeps everything it makes.
    """
    folder = predictor_folder(name)
    with name_lock(name):
        link = is_link(folder)
        if not link and not folder.is_dir():
            return False
        refresh_lock, model_lock = (folder / lock for lock in LOCKS)
        # A link's locks are in the folder it leads to, where a command using the predictor through the link holds
        # them. No lock file is made there, so nothing where the link leads changes.
        with _lock(refresh_lock, wait=False, make=not link) as mine:
            if not mine:
                raise Busy(f"{name} is being set up or refreshed by another command. Try again when it's done.")
            with _lock(model_lock, make=not link):  # a forecast holds it, shared, while it reads the model
                if link:
                    folder.unlink()  # only the link: the folder it leads to stays as it is
                else:
                    for path in sorted(_entries(folder), key=lambda path: path.name == "predictor.json"):
                        if path.name not in LOCKS:
                            _remove(path)
        if not link:
            # The lock files go once no command holds them. On Windows one that another command opened meanwhile
            # stays, with the folder: no predictor shows without its settings.
            shutil.rmtree(folder, ignore_errors=True)
        if settings().get("default") == name:
            forget_setting("default")  # before a setup of the same name can make it the default again
    return True


def rename_predictor(old: str, new: str) -> None:
    """Give a predictor another name: its folder moves to the new name, with everything in it. The default stays
    the default, also when it was found only by being called default or by being the only one.

    It raises ValueError with a plain reason when the new name can't be used or is taken, or when the predictor's
    settings can't be read, and Busy while a setup or refresh of the predictor runs. Each is said before anything
    moves. It waits for a forecast that is reading its model. Both names' locks are held, so no setup of either
    name and no delete runs meanwhile. A predictor that is a link to another folder has only the link renamed:
    nothing in the folder it leads to changes. When the settings can't be saved after the move, the folder moves
    back, and the OSError says why.
    """
    problem = name_problem(new)
    if problem:
        raise ValueError(problem)
    if new == old:
        raise ValueError(f"It's called {old} already.")
    source, target = predictor_folder(old), predictor_folder(new)
    first, second = sorted([old, new])
    with name_lock(first), name_lock(second):
        if not (source / "predictor.json").is_file():
            raise ValueError(f"There's no model called {old!r}.")
        if target.exists() or is_link(target):
            raise ValueError(f"There's already a model called {new}. Pick another name.")
        was_default = default_name() == old
        link = is_link(source)
        try:
            Predictor(old)  # settings that can't be read are said before anything moves
        except BrokenPredictor as error:
            raise ValueError(f"{error} It wasn't renamed.") from None
        refresh_lock, model_lock = (source / lock for lock in LOCKS)
        with _lock(refresh_lock, wait=False, make=not link) as mine:
            if not mine:
                raise Busy(f"{old} is being set up or refreshed by another command. Try again when it's done.")
            with _lock(model_lock, make=not link):  # a forecast holds it, shared, while it reads the model
                if os.name != "nt":
                    source.rename(target)
        if os.name == "nt":
            # Windows can't move a folder while a file in it is open, such as a lock that's held. So it moves once
            # they're let go. A command that opened one meanwhile makes the move fail, and nothing changes.
            source.rename(target)
        if not link:  # a link's settings are the folder's it leads to, which keep their own name
            try:
                predictor = Predictor(new)
                predictor.config["name"] = new
                predictor.save_config()
            except OSError:
                target.rename(source)  # so nothing changed
                raise
        if was_default:
            set_default(new)


def being_refreshed(name: str) -> bool:
    """Whether another command is setting up or refreshing the predictor: it holds the predictor's refresh lock.
    Where the predictor's folder is a link, no lock file is made where it leads."""
    folder = predictor_folder(name)
    with _lock(folder / LOCKS[0], wait=False, make=not is_link(folder)) as free:
        return not free


@contextmanager
def _lock(path: Path, wait: bool = True, make: bool = True) -> Iterator[bool]:
    """Hold a lock, as locks.held does. Without `make`, a lock file that isn't there isn't made: no command can
    hold it then, so it yields True and holds nothing."""
    if not make and not path.exists():
        yield True
        return
    with locks.held(path, wait=wait) as mine:
        yield mine


def is_link(path: Path) -> bool:
    """Whether a path is a link that leads somewhere else, rather than a file or folder of its own: a symbolic
    link, or on Windows a junction, which leads from one folder to another."""
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return False


def _entries(folder: Path) -> list[Path]:
    try:
        return list(folder.iterdir())
    except OSError:
        return []


def _bytes(path: Path) -> int:
    """The bytes a file takes, or all the files in a folder. A link counts as itself: it's never followed."""
    try:
        if path.is_dir() and not is_link(path):
            return sum(found.lstat().st_size for found in path.rglob("*") if not found.is_dir())
        return path.lstat().st_size
    except OSError:
        return 0


def _remove(path: Path) -> None:
    """Delete a file or a folder with what's in it. A link is deleted itself, and what it links to stays. Inside a
    folder, shutil.rmtree deletes each link itself too."""
    if path.is_dir() and not is_link(path):
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _count_tasks(dataset: Path) -> int | None:
    """How many tasks a dataset holds, read only. 0 without a dataset, None when it can't be read."""
    if not dataset.is_file():
        return 0
    try:
        db = sqlite3.connect(f"{dataset.resolve().as_uri()}?mode=ro", uri=True)
    except (sqlite3.Error, ValueError):
        return None
    try:
        return int(db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        db.close()


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
            return _current(config)
        if isinstance(config, dict) and _newer(config):
            raise BrokenPredictor(
                f"Model {name!r} was saved by a newer version of yardstick-predict, and this version can't "
                "read its settings. Upgrade with: uv tool upgrade yardstick-predict"
            )
        why = "are empty or broken"
    raise BrokenPredictor(
        f"Model {name!r} can't be used: its settings in {path} {why}. Fix that file, "
        f"or delete the model with yardstick-predict delete {name} and create it again."
    )


def _newer(config: dict) -> bool:
    """Whether a newer version of yardstick-predict saved these settings, in a format this version doesn't know."""
    saved = config.get("format")
    return isinstance(saved, int) and not isinstance(saved, bool) and saved > FORMAT


def _current(config: dict) -> dict:
    """Settings saved in an older format, as this version keeps them. The next save writes them so.

    In format 1, the conversation so far wasn't a feature to choose. Every predictor read it, so one saved
    then has the conversation feature. One that listed no features had the defaults, which now have it too.
    Settings a newer version saved stay as they are.
    """
    if isinstance(config.get("format"), int) and config["format"] >= FORMAT:
        return config
    listed = config.get("features")
    if isinstance(listed, list) and listed and "conversation" not in listed:
        at = 1 if listed[:1] == ["prompt"] else 0
        config["features"] = [*listed[:at], "conversation", *listed[at:]]
    config["format"] = FORMAT
    return config


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

        So a command that finds the new predictor, such as a refresh, never sees it half made. Setup holds the
        name's lock while it calls this, as name_lock says, so a delete of an older predictor with the same name
        never takes away what it makes.
        """
        problem = name_problem(name)
        if problem:
            raise ValueError(problem)
        private_home()
        folder = predictor_folder(name)
        if (folder / "predictor.json").exists():
            raise ValueError(f"A model called {name!r} already exists.")
        folder.mkdir(parents=True, exist_ok=True)
        from . import feature_store

        config = {
            "format": FORMAT,
            "name": name,
            "created_at": time.time(),
            "embedding": embedding,
            "archives": archives,
            "refreshed_at": None,
            "feature_store": feature_store.FORMAT,  # its vectors and LLM answers are kept there from the start
            **settings,
        }
        files.write_text(folder / "predictor.json", json.dumps(config, indent=2) + "\n")
        return cls(name)

    @property
    def newer(self) -> bool:
        """Whether a newer version of yardstick-predict saved the settings. This version reads what it can of them,
        and never saves them: it would write them in its older format."""
        return _newer(self.config)

    def save_config(self) -> None:
        """Save the settings in one step. Settings a newer version saved are left as they are."""
        if self.newer:
            return
        files.write_text(self.dir / "predictor.json", json.dumps(self.config, indent=2) + "\n")

    @property
    def embedding_model(self) -> str:
        return self.config["embedding"]["model"]

    @property
    def embedding_dimensions(self) -> int:
        """How many numbers each of the predictor's vectors has."""
        from .embeddings import DIMENSIONS

        size = self.config["embedding"].get("dimensions")
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else DIMENSIONS

    @property
    def model_dir(self) -> Path:
        return self.dir / "model"

    @property
    def prepared_path(self) -> Path:
        """The prepared dataset: what went into the last training, one row for each task the model learned from.
        It holds no prompt or other text. pipeline.train writes it."""
        return self.dir / PREPARED

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
            self._old_vectors_table()
            self._db.execute("CREATE TABLE IF NOT EXISTS files (path PRIMARY KEY, size, mtime)")
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

    def remove_tasks(self, task_ids) -> None:
        """Delete tasks, as when a re-read of their records no longer finds them.

        Their vectors and LLM answers stay in the feature store, where another predictor may use them. This one
        no longer has the tasks, so it never reads them.
        """
        task_ids = list(task_ids)
        if not task_ids:
            return
        self.db.executemany("DELETE FROM tasks WHERE id = ?", [(t,) for t in task_ids])
        self.db.commit()

    # Embeddings and LLM answers, in the feature store ------------------------------------------------------

    def embedding_feature(self, field: str):
        """Where the feature store keeps the vectors of one text feature: by the predictor's Voyage model and size."""
        from . import feature_store

        return feature_store.embedding(field, self.embedding_model, self.embedding_dimensions)

    def llm_feature(self):
        """Where the feature store keeps what the predictor's LLM analysis says. None without LLM analysis."""
        return _llm_feature(self.config)

    def embedded(self, field: str, task_ids=None) -> Embedded:
        """The tasks that have a vector for a feature, with the hashes of the texts their vectors were made from.

        Only the predictor's own tasks count, or the tasks `task_ids` names, such as ones a refresh is about to
        add. The feature store may hold vectors of other tasks, for other predictors.
        """
        from . import feature_store

        self.move_in()
        wanted = self.task_ids() if task_ids is None else set(task_ids)
        return Embedded(feature_store.vector_hashes(self.embedding_feature(field), wanted))

    def get_embeddings(self, field: str, task_ids=None) -> dict[str, dict[str, np.ndarray]]:
        """The vectors of a feature, by task and then by the hash of the text each was made from.

        A task can have several: one for each text it had, such as the skills texts of two predictors with
        different skills options. Only the predictor's own tasks count, or the tasks `task_ids` names.
        """
        from . import feature_store

        self.move_in()
        wanted = self.task_ids() if task_ids is None else set(task_ids)
        return feature_store.vectors(self.embedding_feature(field), wanted)

    def put_embeddings(self, field: str, vectors: dict[str, np.ndarray], hashes: dict[str, str] | None = None) -> None:
        """Save each task's vector for a feature, with the hash of the text it was made from, in the feature
        store. A vector saved without its text's hash is never used: no text matches it."""
        from . import feature_store

        self.move_in()
        hashes = hashes or {}
        rows = [(task_id, hashes.get(task_id) or "", vector) for task_id, vector in vectors.items()]
        feature_store.save_vectors(self.embedding_feature(field), rows)

    def llm_labels(self, every_field: bool = False) -> dict[str, dict]:
        """What LLM analysis said about each task, by the predictor's LLM provider and model.

        Each answer keeps only the fields the predictor's options ask for, unless `every_field`: another
        predictor may have asked the same AI model for more about the same task.
        """
        self.move_in()
        return llm_answers(self.config, every_field)

    def put_llm_labels(self, labels: dict[str, dict]) -> None:
        """Save what LLM analysis said about tasks, in the feature store, each merged with what the same AI model
        said about the task before, as llm.merge does. The store merges with what it holds as it saves, so
        another predictor's answers saved meanwhile stay."""
        from . import feature_store, llm

        feature = self.llm_feature()
        if feature is None or not labels:
            return
        self.move_in()
        feature_store.save_answers(feature, labels, merge=llm.merge)

    def compact_features(self, fields=(), answers: bool = False) -> None:
        """Merge the rows saved one batch at a time into one file for each feature, as at the end of a run that
        embedded or analyzed tasks. `fields` names the text features, and `answers` adds the LLM answers."""
        from . import feature_store

        found = [self.embedding_feature(field) for field in dict.fromkeys(fields)]
        if answers and self.llm_feature() is not None:
            found.append(self.llm_feature())
        for feature in found:
            feature_store.compact(feature)

    # Moving an older version's vectors and LLM answers into the feature store ---------------------------------

    def _old_vectors_table(self) -> None:
        """Bring the vectors an older version kept in the dataset up to date, before they move to the feature store.

        Each vector is kept with the hash of the text it was made from, so a vector whose text changed is made
        again. An older version kept no hash. For the prompt and the context, whose texts are built as that
        version built them, the hash of each task's text as stored now is filled in, in one step and before
        this process replaces any task. Another feature's old vectors have no hash, so they are made again once.
        A text the cache of vectors holds costs nothing. A dataset this version made has no such table.
        """
        db = self._db
        columns = {row[1] for row in db.execute("PRAGMA table_info(embeddings)")}
        if not columns or "text_hash" in columns:
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

    def _old_rows(self, query: str) -> list[tuple]:
        """The rows of a table an older version kept in the dataset. Empty when the dataset never had the table."""
        # A dataset this version made never has the embeddings and llm_labels tables. One an older version made
        # keeps them, unread once their rows moved to the feature store. A later release can drop them.
        try:
            return list(self.db.execute(query))
        except sqlite3.OperationalError:
            return []

    def move_in(self) -> None:
        """Copy the vectors and LLM answers an older version kept in the dataset into the feature store, once.

        It runs the first time this version reads or saves them, and never for a forecast. Only vectors the store
        lacks are copied, and each answer is combined with the one the store holds for its task, as llm.combine
        does, so another predictor's answers about the same tasks keep every field either has. Rows are keyed and
        combining again changes nothing, so a move stopped halfway runs again safely, and two predictors can move
        in at once: the store's lock keeps their saves apart. A vector without its text's hash was never used, so
        it stays behind. The old tables stay in the dataset, unread: a later release can drop them. Then the
        settings say the move is done.
        """
        if self.config.get("feature_store"):
            return
        from . import feature_store, llm

        size = self.embedding_dimensions
        vectors: dict[str, list] = {}
        old = self._old_rows("SELECT task_id, field, vector, text_hash FROM embeddings")
        for task_id, field, blob, text_hash in old:
            if isinstance(field, str) and field and text_hash and isinstance(blob, bytes) and len(blob) == 4 * size:
                vectors.setdefault(field, []).append((str(task_id), str(text_hash), np.frombuffer(blob, np.float32)))
        moved = []
        for field, rows in vectors.items():
            feature = self.embedding_feature(field)
            feature_store.save_vectors(feature, rows, only_new=True)
            moved.append(feature)
        # Each answer goes with the AI model that gave it. One that doesn't say goes with the predictor's own.
        answers: dict[str, tuple] = {}
        own = self.llm_feature()
        for task_id, text in self._old_rows("SELECT task_id, labels FROM llm_labels"):
            try:
                labels = json.loads(text)
            except (TypeError, ValueError):
                continue
            if not isinstance(labels, dict):
                continue
            by = labels.get("by")
            provider, _, model = by.partition(":") if isinstance(by, str) else ("", "", "")
            feature = feature_store.answers(provider, model) if provider and model else own
            if feature is not None:
                answers.setdefault(feature.id, (feature, {}))[1][str(task_id)] = labels
        for feature, found in answers.values():
            feature_store.save_answers(feature, found, merge=llm.combine)
            moved.append(feature)
        for feature in moved:
            feature_store.compact(feature)
        self.config["feature_store"] = feature_store.FORMAT
        self._mark_moved_in()

    def _mark_moved_in(self) -> None:
        """Say in the saved settings that the vectors and LLM answers moved to the feature store.

        The settings are read again first, so a change another command saved meanwhile stays. Settings a newer
        version saved stay as they are: this version never saves them.
        """
        from . import feature_store

        path = self.dir / "predictor.json"
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        if not isinstance(saved, dict) or _newer(saved):
            return
        saved["feature_store"] = feature_store.FORMAT
        files.write_text(path, json.dumps(saved, indent=2) + "\n")


def _llm_feature(config: dict):
    """Where the feature store keeps the answers of the LLM provider and model a predictor's settings name."""
    from . import feature_store

    how = config.get("llm")
    if not isinstance(how, dict) or not isinstance(how.get("provider"), str) or not isinstance(how.get("model"), str):
        return None
    return feature_store.answers(how["provider"], how["model"])


def llm_answers(config: dict, every_field: bool = False) -> dict[str, dict]:
    """What LLM analysis said about each task, as the feature store keeps it, by the LLM provider and model a
    predictor's settings name. Empty for settings without LLM analysis.

    Each answer keeps only the fields the settings' options ask for, unless `every_field`. So a predictor that
    didn't ask for the description never reads one that another predictor asked the same AI model for.
    """
    from . import feature_store, features, llm

    feature = _llm_feature(config)
    if feature is None:
        return {}
    found = feature_store.answers_of(feature)
    if every_field:
        return found
    fields = llm.parts(features.options(config, "llm"))
    return {task_id: llm.only(labels, fields) for task_id, labels in found.items()}


class Embedded(set):
    """The tasks that have a vector for one feature. `hashes` holds, for each, the hashes of the texts its vectors
    were made from: one for each text it had, as when two predictors build its skills text differently.

    It is a set of task ids, so code that only asks whether a task has a vector works as before.
    """

    def __init__(self, hashes: dict[str, set[str]]) -> None:
        super().__init__(hashes)
        self.hashes = hashes

    def made_from(self, task_id: str, text: str) -> bool:
        """Whether the task has a vector made from this text."""
        from .embeddings import text_hash

        return text_hash(text) in self.hashes.get(task_id, ())
