"""Model types: the kind of model a predictor trains. XGBoost is built in. Others can be added.

A model type is two functions:

- train(tasks, rows, settings) returns a trained model. `rows` holds each task's features.
- load(folder) returns a model saved with model.save(folder).

A trained model has forecast(rows), served(model_name), save(folder) and a meta dict whose "report"
says how it did on its test. See yardstick_predict.model for the built-in one.

Add one with register_model_type(), or name it in a plugin module, as "module:object", where the
object has train and load.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

_TYPES: dict[str, tuple[Callable, Callable]] = {}


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


def load_plugins(modules: list[str]) -> None:
    """Import the plugin modules a predictor names, so their features and model types are registered."""
    for module in modules:
        importlib.import_module(module)
