from __future__ import annotations

import builtins
import json
import subprocess
import sys

import jsonschema
import pytest
from test_forecast import schema

import yardstick_predict
from yardstick_predict import cli
from yardstick_predict.model_types import BUILT_IN_FORMAT
from yardstick_predict.store import Predictor


@pytest.fixture
def no_xgboost(monkeypatch):
    """XGBoost can't load, as on a Mac without OpenMP. The command has already restarted once, so it reports."""
    real_import = builtins.__import__

    def without_xgboost(name, *args, **kwargs):
        if name == "xgboost":
            raise RuntimeError("XGBoost Library (libxgboost.dylib) could not be loaded. libomp.dylib for Mac OSX")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_xgboost)
    # Forget the modules loaded so far, so anything that needs XGBoost has to load it again, and can't.
    for name in [name for name in sys.modules if name == "xgboost" or name.startswith("xgboost.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "yardstick_predict.model", raising=False)
    monkeypatch.delattr(yardstick_predict, "model", raising=False)
    monkeypatch.setenv(cli.RELAUNCHED, "1")


def test_a_missing_mac_library_gets_the_exact_fix(no_xgboost, capsys):
    assert cli.main(["refresh"]) == 1
    assert "brew install libomp" in capsys.readouterr().err
    assert "Reinstall yardstick-predict" in cli.xgboost_problem(["refresh"])

    assert cli.main(["predict", "--model", "m", "--prompt", "x"]) == 1
    answer = json.loads(capsys.readouterr().out)
    assert answer["error"]["code"] == "missing_library"
    jsonschema.validate(answer, schema("forecast-response.v1.json"))


def test_commands_that_dont_train_or_forecast_work_without_xgboost(no_xgboost, capsys, tmp_path, monkeypatch):
    predictor = Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    predictor.model_dir.mkdir()
    report = {
        "tested_on": 40,
        "models": {
            "claude-opus-5-5": {
                "served": False,
                "reason": "not_accurate_enough",
                "tasks_tested": 40,
                "heads": ["seconds"],
                "seconds": {"better_than_simple_rule": -0.1, "in_90_range": 0.5, "expected_over_real_total": 1.0},
                "failed_checks": ["seconds.simple_rule"],
            }
        },
    }
    saved = {"format": BUILT_IN_FORMAT, "tasks": 140, "report": report}
    (predictor.model_dir / "model.json").write_text(json.dumps(saved))
    capsys.readouterr()

    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    squashed = " ".join(out.split())  # long lines are broken to fit the terminal
    assert "Trained on 140 tasks" in out and "10% worse than the simple rule. They may be at most 2% worse." in squashed
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["predictors"][0]["report"] == report
    assert cli.main(["default"]) == 0 and "Your default predictor is default" in capsys.readouterr().out
    monkeypatch.chdir(tmp_path)  # claude-code with --here reads only this folder's settings
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert "Nothing to remove" in capsys.readouterr().out
    assert "yardstick_predict.model" not in sys.modules  # nothing loaded the model, so nothing needed XGBoost

    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0 and yardstick_predict.__version__ in capsys.readouterr().out


def test_on_a_mac_the_command_restarts_once_with_the_bundled_openmp(monkeypatch, tmp_path):
    folder = tmp_path / "sklearn" / ".dylibs"
    folder.mkdir(parents=True)
    (folder / "libomp.dylib").write_bytes(b"")
    spec = type("Spec", (), {"origin": str(tmp_path / "sklearn" / "__init__.py")})()
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: spec if name == "sklearn" else None)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    launched = {}

    def execve(path, args, env):
        launched.update(path=path, args=args, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(cli.os, "execve", execve)
    try:
        cli.relaunch_with_openmp(["predict", "--model", "m"])
    except SystemExit:
        pass
    # -P keeps the current folder off the import path, so a json.py there can't stand in for Python's own.
    assert launched["args"][1:] == ["-P", "-m", "yardstick_predict", "predict", "--model", "m"]
    assert launched["env"]["DYLD_FALLBACK_LIBRARY_PATH"].startswith(str(folder) + ":")
    assert launched["env"][cli.RELAUNCHED] == "1"

    # After one restart it never restarts again.
    launched.clear()
    monkeypatch.setenv(cli.RELAUNCHED, "1")
    cli.relaunch_with_openmp(["status"])
    assert launched == {}


def test_the_restart_never_runs_code_from_the_current_folder(tmp_path):
    (tmp_path / "json.py").write_text("raise SystemExit('the json.py in this folder ran')\n")
    done = subprocess.run(
        [sys.executable, "-P", "-m", "yardstick_predict", "--version"], cwd=tmp_path, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert "ran" not in done.stderr and yardstick_predict.__version__ in done.stdout
