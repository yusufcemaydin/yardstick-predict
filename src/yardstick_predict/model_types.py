"""Model types: the kind of model a predictor trains. XGBoost is built in. Others can be added.

A model type is two functions:

- train(tasks, rows, settings) returns a trained model. `rows` holds each task's features.
- load(folder) returns a model saved with model.save(folder).

A trained model has forecast(rows), served(model_name), save(folder) and a meta dict whose "report"
says how it did on its test. served may also take an effort level, as served(model_name, effort), to
say no for a level the model never ran at.

save(folder) must write the meta dict, as JSON, to model.json in the folder. The command reads only
that file to show a predictor's status and to pick the AI models to forecast, so it works without
loading the whole model.

Add one with register_model_type(), or name it in a plugin module, as "module:object", where the
object has train and load.

A plugin is a module that registers features or model types. It's named by its module name, or by
the path of its .py file. A module in the current folder, such as my_features for my_features.py, is
found there too. A predictor saves where it found each plugin, so it loads the same from any folder.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable
from pathlib import Path

BUILT_IN_FORMAT = 4
"""The format the built-in model saves in. A model in a newer format needs a newer version of yardstick-predict.
It's here so the command can check a saved model without loading XGBoost."""
PREVIOUS_FORMAT = 3
"""The format the previous release, 0.3, saved the built-in model in. Such a model still forecasts, as that
release did, until a refresh trains it again. A model in an older format has to be trained again first."""

_TYPES: dict[str, tuple[Callable, Callable]] = {}


class PluginError(ImportError):
    """A plugin that couldn't be found or loaded. The message says which one, and why."""


def register_model_type(name: str, train: Callable, load: Callable) -> None:
    """Add a model type a predictor can use."""
    if not name or name == "xgboost":
        raise ValueError("Pick a new name for your model type.")
    _TYPES[name] = (train, load)


def get(name: str) -> tuple[Callable, Callable]:
    """The train and load functions of a model type."""
    if name == "xgboost":
        from . import model

        return model.train, model.load
    if name in _TYPES:
        return _TYPES[name]
    if ":" in name:
        module, attribute = name.split(":", 1)
        found = getattr(importlib.import_module(module), attribute)
        return found.train, found.load
    raise ValueError(f"Unknown model type {name!r}. Built in: xgboost. Added: {', '.join(_TYPES) or 'none'}.")


def _is_path(plugin: str) -> bool:
    """Whether a plugin is named by its file: a name with a slash, or one that ends in .py."""
    return plugin.endswith(".py") or "/" in plugin or "\\" in plugin


def find_plugin(plugin: str, folder: Path | None = None) -> str:
    """What a predictor saves for a plugin, so it loads the same from any folder later.

    The path of a .py file is saved in full. So is a module in `folder`, the current folder by default,
    such as my_features for my_features.py there, or a package's __init__.py. Any other name is saved as it
    is, for Python to import. Raises PluginError when a path names no file.
    """
    folder = Path.cwd() if folder is None else folder
    if _is_path(plugin):
        path = Path(plugin).expanduser()
        path = path if path.is_absolute() else folder / path
        if path.is_dir() and (path / "__init__.py").is_file():
            path = path / "__init__.py"  # a package's folder
        if not path.is_file():
            raise PluginError(f"Couldn't load the plugin {plugin}: there's no file at {path}.")
        return str(path.resolve())
    if "." not in plugin:
        for candidate in (folder / f"{plugin}.py", folder / plugin / "__init__.py"):
            if candidate.is_file():
                return str(candidate.resolve())
    return plugin


def load_plugins(plugins: list[str]) -> None:
    """Import the plugins a predictor names, so their features and model types are registered.

    Each is a module name or the full path of a .py file, as find_plugin gives. Raises PluginError.
    """
    for plugin in plugins:
        try:
            if _is_path(plugin):
                _load_file(Path(plugin))
            else:
                importlib.import_module(plugin)
        except PluginError:
            raise
        except (Exception, SystemExit) as error:  # a plugin's own code can fail in any way, even by exiting
            reason = str(error) if isinstance(error, ImportError) else f"{type(error).__name__}: {error}"
            raise PluginError(f"Couldn't load the plugin {plugin}: {reason}") from error


def _load_file(path: Path) -> None:
    """Run a plugin's .py file once, as the module its name gives: my_features for my_features.py.

    A package's __init__.py becomes the package, so its own relative imports work. A name Python already
    has for another module, such as json, would hide that module, so it's refused.
    """
    path = path.resolve()
    package = path.name == "__init__.py"
    name = path.parent.name if package else path.stem
    loaded = sys.modules.get(name)
    if loaded is not None:
        if _same_file(getattr(loaded, "__file__", None), path):
            return
        raise PluginError(f"Couldn't load the plugin {path}: Python already has a module called {name}. Rename it.")
    try:
        other = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        other = None
    if other is not None and other.origin and not _same_file(other.origin, path):
        raise PluginError(f"Couldn't load the plugin {path}: Python already has a module called {name}. Rename it.")
    if not path.is_file():
        raise PluginError(f"Couldn't load the plugin {path}: the file isn't there anymore.")
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    if spec is None or spec.loader is None:
        raise PluginError(f"Couldn't load the plugin {path}: Python can't load it as a module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise


def _same_file(one: str | None, other: Path) -> bool:
    if not one:
        return False
    try:
        return os.path.samefile(one, other)
    except OSError:
        return False
