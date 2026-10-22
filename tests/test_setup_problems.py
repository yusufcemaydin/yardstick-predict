from __future__ import annotations

import builtins
import json

import jsonschema
from test_forecast import schema

from yardstick_predict import cli


def test_a_missing_mac_library_gets_the_exact_fix(monkeypatch, capsys):
    real_import = builtins.__import__

    def no_xgboost(name, *args, **kwargs):
        if name == "xgboost":
            raise RuntimeError("XGBoost Library (libxgboost.dylib) could not be loaded. libomp.dylib for Mac OSX")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_xgboost)
    monkeypatch.setenv(cli.RELAUNCHED, "1")  # as if it had already restarted once, so it reports instead
    assert cli.main(["status"]) == 1
    assert "brew install libomp" in capsys.readouterr().err
    assert "Reinstall yardstick-predict" in cli.xgboost_problem(["status"])

    assert cli.main(["predict", "--model", "m", "--prompt", "x"]) == 1
    answer = json.loads(capsys.readouterr().out)
    assert answer["error"]["code"] == "missing_library"
    jsonschema.validate(answer, schema("forecast-response.v1.json"))


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
    assert launched["args"][1:] == ["-m", "yardstick_predict", "predict", "--model", "m"]
    assert launched["env"]["DYLD_FALLBACK_LIBRARY_PATH"].startswith(str(folder) + ":")
    assert launched["env"][cli.RELAUNCHED] == "1"

    # After one restart it never restarts again.
    launched.clear()
    monkeypatch.setenv(cli.RELAUNCHED, "1")
    cli.relaunch_with_openmp(["status"])
    assert launched == {}
