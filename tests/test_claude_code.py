from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from builders import ClaudeSession, usage

from yardstick_predict import claude_code as cc
from yardstick_predict import cli, embeddings, locks
from yardstick_predict.archives import read_archive
from yardstick_predict.screens import version
from yardstick_predict.store import Predictor, predictor_names

THEIRS = {
    "model": "opus",
    "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "say done"}]}]},
    "permissions": {"allow": ["Bash(ls:*)"]},
}
NOTIFICATION = "<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n</task-notification>"


@pytest.fixture(autouse=True)
def no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")


@pytest.fixture(autouse=True)
def own_home(tmp_path, monkeypatch):
    """Claude Code's settings for all sessions are in a home folder of the test's own, and the test runs as
    an ordinary Claude Code session would, whatever started the tests."""
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
    return home


class Clock:
    """A stand-in for the time module. Time moves when a test moves it, by `step` at each look, and when
    the code waits."""

    def __init__(self, now: float, step: float = 0.0):
        self.now, self.step = now, step

    def time(self) -> float:
        now = self.now
        self.now += self.step
        return now

    def monotonic(self) -> float:
        return self.time()

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def send(monkeypatch, event: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert cc.hook() == 0


def prompt_event(prompt: str, session: str = "s1", **more) -> dict:
    return {"session_id": session, "hook_event_name": "UserPromptSubmit", "prompt": prompt, **more}


def forecasts_by_hand(monkeypatch, clock: Clock | None = None) -> list[list[str]]:
    """Keep each background job the hook would start, to run it here by hand with cc.main.

    The forecasts use a stand-in predictor that forecasts model m. Unless a test brings its own clock,
    the clock moves 10 seconds at each look, so two prompts never share a time.
    """
    commands: list[list[str]] = []
    monkeypatch.setattr(
        cc.subprocess, "Popen", lambda command, **options: commands.append(command[command.index(cc.MARK) + 1 :])
    )
    monkeypatch.setattr(cc, "time", clock or Clock(1_780_000_000, step=10))
    monkeypatch.setattr(cli, "request_predictor", lambda chosen: SimpleNamespace(name="default"))
    monkeypatch.setattr("yardstick_predict.forecast.load", lambda predictor: SimpleNamespace(meta={}))
    monkeypatch.setattr(cli, "forecast_models", lambda predictor, meta=None: ["m"])
    monkeypatch.setattr(
        cli, "effort_pairs", lambda predictor, models, effort, meta=None: [(model, None) for model in models]
    )
    return commands


def answers(asked: list, seconds: float = 120.0):
    """A stand-in for forecast.answer that keeps the tasks it's asked about."""

    def answer(predictor, tasks, distribution=False, model=None, recent=None):
        asked.extend(tasks)
        return {"forecasts": [{"model": "m", "seconds": seconds, "tokens": 9, "usd": None}]}

    return answer


def plain(text: str) -> str:
    return text.replace("\033[2m", "").replace("\033[22m", "")


def state(session: str = "s1") -> dict:
    return json.loads(cc._state_path(session).read_text())


# Settings ----------------------------------------------------------------------------------------------


def test_adding_keeps_your_own_settings_and_removing_puts_them_back():
    added = cc.added(THEIRS, "/py")
    assert added["model"] == "opus" and added["permissions"] == THEIRS["permissions"]
    stop = [entry["command"] for group in added["hooks"]["Stop"] for entry in group["hooks"]]
    assert stop[0] == "say done" and stop[1].endswith("-P -m yardstick_predict.claude_code hook")
    assert added["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == (
        f"{cc._quoted('/py')} -P -m yardstick_predict.claude_code hook"  # in double quotes on Windows
    )
    assert added["statusLine"]["command"].endswith("-m yardstick_predict.claude_code status")
    assert added["statusLine"]["refreshInterval"] == 1  # the clock ticks once a second
    assert "StopFailure" in added["hooks"]  # a failed request to the model ends the task too
    assert cc.added(added, "/py") == added  # adding twice changes nothing
    assert cc.removed(added) == THEIRS
    assert cc.other_status_line({"statusLine": {"command": "my-line"}}) == "my-line"
    assert cc.other_status_line(added) is None


def test_odd_shapes_in_your_settings_are_kept_and_never_crash():
    odd_hooks = {
        "UserPromptSubmit": None,
        "Stop": ["say done", {"hooks": None}, {"hooks": ["x", {"command": 5}]}],
        "Notification": "notify",
    }
    added = cc.added({"hooks": odd_hooks}, "/py")
    assert added["hooks"]["UserPromptSubmit"] is None and added["hooks"]["Notification"] == "notify"
    assert added["hooks"]["Stop"][:3] == odd_hooks["Stop"] and len(added["hooks"]["Stop"]) == 4
    assert cc.removed(added) == {"hooks": odd_hooks}
    for theirs in ({"hooks": None, "statusLine": None}, {"hooks": "x", "statusLine": "my-line"}, {"hooks": []}):
        assert cc.removed(theirs) == theirs
        assert cc.removed(cc.added(theirs, "/py")) == {k: v for k, v in theirs.items() if k != "statusLine"}
    for theirs in ({"hooks": {}}, {"hooks": {"Stop": []}}, [], "text", None):  # nothing of ours to take out
        assert cc.removed(theirs) == theirs and cc.other_status_line(theirs) is None
        assert cc.installed_home(theirs) is None
    mixed = {"hooks": {"Stop": [{"hooks": [{"command": "say hi"}, {"command": cc.commands("/py")[0]}]}]}}
    assert cc.removed(mixed) == {"hooks": {"Stop": [{"hooks": [{"command": "say hi"}]}]}}  # only ours goes
    assert cc.other_status_line({"statusLine": "my-line"}) == "my-line"
    assert cc.other_status_line({"statusLine": ["my-line"]}) is None


def test_installing_again_replaces_the_commands_from_before_which_lacked_P():
    before = '"/py" -m yardstick_predict.claude_code'  # how version 0.2.1 wrote them
    settings = {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": f"{before} hook"}]}],
            "Stop": [
                {
                    "hooks": [
                        {"type": "command", "command": "say done"},
                        {"type": "command", "command": f"{before} hook"},
                    ]
                }
            ],
        },
        "statusLine": {"type": "command", "command": f"{before} status"},
    }
    added = cc.added(settings, "/py")
    hooks = [entry["command"] for groups in added["hooks"].values() for group in groups for entry in group["hooks"]]
    ours = [command for command in hooks if cc.MARK in command] + [added["statusLine"]["command"]]
    assert hooks.count("say done") == 1 and len(ours) == 4  # one each on prompts, stops, failed stops, and the line
    assert all(command.startswith(f"{cc._quoted('/py')} -P -m ") for command in ours)


def test_the_commands_never_run_code_from_the_session_folder(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for name in ("json.py", "re.py", "math.py", "subprocess.py"):
        (project / name).write_text("open(__file__ + '.ran', 'w').close()\n")  # notes that it ran
    hook, status = cc.commands(sys.executable, str(tmp_path / "home"))
    for command in (hook, status):
        done = subprocess.run(shlex.split(command), cwd=project, input=b"{}", capture_output=True, timeout=60)
        assert done.returncode == 0 and not list(project.glob("*.ran")), done.stderr
    assert done.stdout == b"yardstick\n"
    # A command an older version installed, without -P, would run the folder's own json.py instead of Python's.
    # The package now keeps the folder out for it too, and the line asks for the command to be updated.
    old = subprocess.run(
        shlex.split(status.replace(" -P", "")), cwd=project, input=b"{}", capture_output=True, timeout=60
    )
    assert old.returncode == 0 and not list(project.glob("*.ran")), old.stderr
    assert b"update it: yardstick-predict claude-code install" in old.stdout


def test_only_a_command_an_older_version_installed_asks_to_be_updated():
    assert cc.old_command(["/py", "-m", cc.MARK, "status"], safe=False)
    assert not cc.old_command(["/py", "-P", "-m", cc.MARK, "status"], safe=True)
    assert not cc.old_command(["/py", "-m", "pytest"], safe=False)  # not one of our commands


def test_the_package_keeps_the_folder_python_started_in_off_the_import_path_only_for_its_own_modules(monkeypatch):
    import os

    import yardstick_predict

    here = os.getcwd()
    monkeypatch.setattr(sys, "path", [here, "/somewhere/else"])
    yardstick_predict._keep_the_current_folder_out(["/py", "-m", "pytest"])
    assert sys.path == [here, "/somewhere/else"]  # another program's -m is left alone
    yardstick_predict._keep_the_current_folder_out(["/py", "-m", f"{cc.MARK}", "status"])
    assert sys.path == ["/somewhere/else"]


def test_the_background_job_keeps_the_session_folder_out_too(monkeypatch):
    started = []
    monkeypatch.setattr(cc.subprocess, "Popen", lambda command, **options: started.append((command, options)))
    send(monkeypatch, prompt_event("Add a login page"))
    [(command, options)] = started
    assert command[1:] == ["-P", "-m", cc.MARK, "work", "s1"] and options["cwd"] == str(cc.folder())


def test_paths_with_spaces_stay_one_word_for_the_shell():
    if os.name != "nt":  # a POSIX shell's quotes. Windows has no " in its paths, and its quotes are below.
        python, home = "/Users/Jo Smith/my $env/bin/python", '/Users/Jo Smith/it\'s "here"'
        hook, status = cc.commands(python, home)
        assert shlex.split(hook) == [python, "-P", "-m", cc.MARK, "--home", home, "hook"]
        assert cc.installed_home({"statusLine": {"command": status}}) == home
    # On Windows, cmd and Git Bash both read double quotes, and forward slashes, the same way.
    assert (
        cc._quoted(r"C:\Program Files\Python 3.12\python.exe", windows=True)
        == '"C:/Program Files/Python 3.12/python.exe"'
    )
    assert cc._quoted("D:\\", windows=True) == '"D:/"'  # a backslash before the quote would escape it in both
    python, home = (cc._quoted(path, windows=True) for path in (r"C:\Py 3\python.exe", r"C:\My Home"))
    windows = f"{python} -P -m {cc.MARK} --home {home} status"
    assert cc.installed_home({"statusLine": {"command": windows}}) == "C:/My Home"


def test_a_windows_path_a_shell_would_read_as_more_than_text_is_refused(tmp_path, monkeypatch, capsys):
    for path in (r"C:\Tools\$work\python.exe", r"C:\100%\python.exe", 'C:\\a"b\\python.exe', r"C:\Hi!\py"):
        assert cc.unsafe_path(path, windows=True) is not None
        assert cc.unsafe_path(path, windows=False) is None  # elsewhere single quotes keep every character
    assert cc.unsafe_path(r"C:\Program Files\Python 3.12\python.exe", windows=True) is None
    assert cc.unsafe_path(r"C:\$a\b`c", windows=True).startswith(r"The path C:\$a\b`c has $ and ` in it.")
    # The install refuses such a yardstick folder before it changes anything.
    monkeypatch.setenv("YARDSTICK_HOME", str(tmp_path / "100%"))
    Predictor.create("main", [], {"model": "voyage-4", "dimensions": 256})
    real = cc.unsafe_path
    monkeypatch.setattr(cc, "unsafe_path", lambda path, windows=None: real(path, windows=True))  # as on Windows
    monkeypatch.chdir(tmp_path)
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 1
    assert "Nothing was added." in capsys.readouterr().err and not (tmp_path / ".claude").exists()


def test_install_and_remove_in_this_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps(THEIRS))
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 1
    assert "There's no model yet." in capsys.readouterr().err
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert cli.main(["claude-code", "install", "--here"]) == 2  # a program must say yes
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert "statusLine" in json.loads(settings.read_text())
    assert json.loads(settings.with_name("settings.local.json.before-yardstick").read_text()) == THEIRS
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert json.loads(settings.read_text()) == THEIRS
    assert "Removed" in capsys.readouterr().out


@pytest.mark.parametrize("settings", [["not", "an", "object"], {"hooks": None}, {"hooks": {"Stop": "say done"}}])
def test_install_leaves_settings_it_cant_add_to_alone(tmp_path, monkeypatch, capsys, settings):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    path.write_text(json.dumps(settings))
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 1
    error = capsys.readouterr().err
    assert "left alone" in error or "nothing was added" in error
    assert json.loads(path.read_text()) == settings


def test_removing_puts_back_the_status_line_that_installing_replaced(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    mine = {"statusLine": {"type": "command", "command": "my-line", "padding": 2}}
    path.write_text(json.dumps(mine))
    for _ in range(2):  # installing twice still remembers yours
        assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert "Removing yardstick puts yours back." in capsys.readouterr().out
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert json.loads(path.read_text()) == mine
    assert "Your own status line is back: my-line" in capsys.readouterr().out
    # A status line you set after installing stays when you remove yardstick.
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    newer = {"type": "command", "command": "newer-line"}
    path.write_text(json.dumps({**json.loads(path.read_text()), "statusLine": newer}))
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert json.loads(path.read_text()) == {"statusLine": newer}


def test_each_install_and_remove_keeps_a_copy_of_the_file_as_it_was(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    path.write_text(json.dumps(THEIRS))
    path.chmod(0o640)
    backup = path.with_name("settings.local.json.before-yardstick")
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert json.loads(backup.read_text()) == THEIRS
    edited = {**json.loads(path.read_text()), "model": "sonnet"}  # you change something after installing
    path.write_text(json.dumps(edited))
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert json.loads(backup.read_text()) == edited  # the copy is of the file just before each change
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert json.loads(backup.read_text()) == json.loads(
        json.dumps(cc.added(edited, sys.executable, cc.installed_home(edited)))
    )
    assert "A copy of the file as it was is kept next to it" in capsys.readouterr().out
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o640  # the settings keep their permissions


@pytest.mark.skipif(os.name != "posix", reason="making a link needs extra rights on Windows")
def test_a_settings_file_that_links_elsewhere_stays_a_link(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    real = tmp_path / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"model": "opus", "note": "café ✓"}))
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    path.symlink_to(real)
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert path.is_symlink() and "statusLine" in json.loads(real.read_text())
    assert "café ✓" in real.read_text()  # your text stays as you wrote it
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert path.is_symlink() and json.loads(real.read_text()) == {"model": "opus", "note": "café ✓"}


def test_installing_keeps_a_note_of_each_settings_file_and_removing_drops_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert cc.added_places() == [str(tmp_path / ".claude" / "settings.local.json")]
    cc.clean()  # what the hook kept for each session goes, and the note stays
    assert cc.added_places() == [str(tmp_path / ".claude" / "settings.local.json")]
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert cc.added_places() == []


def test_uninstall_takes_the_forecasts_out_of_every_place_then_deletes_what_yardstick_made(
    home, own_home, tmp_path, monkeypatch, capsys, wide
):
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    everywhere = own_home / ".claude" / "settings.json"
    everywhere.parent.mkdir()
    everywhere.write_text(json.dumps(THEIRS))
    assert cli.main(["claude-code", "install", "--yes"]) == 0
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    # A folder an earlier version added them to without keeping a note: Claude Code's own list of folders has it.
    older = tmp_path / "older"
    (older / ".claude").mkdir(parents=True)
    (older / ".claude" / "settings.local.json").write_text(json.dumps(cc.added({}, sys.executable)))
    (own_home / ".claude.json").write_text(json.dumps({"projects": {str(older): {}, str(tmp_path / "gone"): {}}}))
    monkeypatch.chdir(tmp_path)
    capsys.readouterr()

    assert cli.main(["uninstall"]) == 2  # a program must say yes
    assert cc.in_use(everywhere) and home.is_dir()
    assert cli.main(["uninstall", "--yes"]) == 0
    assert json.loads(everywhere.read_text()) == THEIRS
    assert json.loads((project / ".claude" / "settings.local.json").read_text()) == {}
    assert json.loads((older / ".claude" / "settings.local.json").read_text()) == {}
    assert not home.exists()
    out = capsys.readouterr().out
    assert "Took the forecasts out of all your Claude Code sessions." in out
    assert f"Took the forecasts out of Claude Code sessions in {project}." in out
    assert f"Took the forecasts out of Claude Code sessions in {older}." in out
    assert f"Deleted {home}." in out
    assert f"To finish, remove the program itself:\n    {version.uninstall_command()}" in out
    # Run again, it finds nothing left to take out.
    assert cli.main(["uninstall", "--yes"]) == 0
    assert "Claude Code's settings hold no forecasts." in capsys.readouterr().out


def test_uninstall_can_keep_your_models_and_the_voyage_key(home, tmp_path, monkeypatch, capsys, wide):
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    monkeypatch.chdir(tmp_path)
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    deleted = []
    monkeypatch.setattr("keyring.delete_password", lambda service, user: deleted.append(user))
    assert cli.main(["uninstall", "--keep-data", "--yes"]) == 0
    assert not cc.in_use(tmp_path / ".claude" / "settings.local.json")
    assert predictor_names() == ["default"] and not deleted
    assert f"Your models and the Voyage key stay. Your models are in {home}." in capsys.readouterr().out


def test_uninstall_deletes_the_voyage_key_and_never_a_folder_holding_files_it_didnt_make(
    home, monkeypatch, capsys, wide
):
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    (home / "notes.txt").write_text("mine")
    kept = {("yardstick-predict", "voyage-api-key"): "a key"}
    monkeypatch.setattr("keyring.delete_password", lambda service, user: kept.pop((service, user)))
    assert cli.main(["uninstall", "--yes"]) == 0
    assert not kept
    assert predictor_names() == ["default"] and (home / "notes.txt").read_text() == "mine"
    out = capsys.readouterr().out
    assert "files yardstick-predict didn't make. Nothing in it is deleted." in out
    assert "Deleted the Voyage key from your computer's password store." in out
    assert f"{home} stays as it is. It also holds notes.txt." in out


@pytest.mark.skipif(os.name != "posix", reason="making a link needs extra rights on Windows")
def test_uninstall_deletes_a_link_in_the_yardstick_folder_but_never_what_it_links_to(home, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "kept.txt").write_text("still here")
    (home / "predictors").mkdir(parents=True)
    (home / "predictors" / "linked").symlink_to(elsewhere)
    assert cli.main(["uninstall", "--yes"]) == 0
    assert not home.exists() and (elsewhere / "kept.txt").read_text() == "still here"


def test_uninstall_leaves_settings_it_cant_read_alone_and_says_so(tmp_path, monkeypatch, capsys, wide):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".claude" / "settings.local.json"
    path.parent.mkdir()
    broken = '{"statusLine": {"command": "python -m yardstick_predict.claude_code status"'
    path.write_text(broken)
    assert cli.main(["uninstall", "--yes"]) == 1
    assert path.read_text() == broken
    assert f"{path} still names yardstick." in capsys.readouterr().out


def test_deleting_the_voyage_key_waits_only_so_long_for_the_password_store(monkeypatch):
    assert embeddings.delete_key(1) == "no_store"  # the tests' password store is one that's never there
    release = threading.Event()
    monkeypatch.setattr("keyring.delete_password", lambda service, user: release.wait(10))
    try:
        assert embeddings.delete_key(0.05) == embeddings.NO_ANSWER
    finally:
        release.set()


def test_install_puts_the_yardstick_folder_into_the_commands_and_the_hook_uses_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YARDSTICK_HOME", "my yardstick")  # a folder named from here
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    home = str(tmp_path / "my yardstick")
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    settings = json.loads((tmp_path / ".claude" / "settings.local.json").read_text())
    found = shlex.split(settings["statusLine"]["command"])[-3:]
    # On Windows the commands have forward slashes, which mean the same as backslashes there.
    assert found[0] == "--home" and Path(found[1]) == Path(home) and found[2] == "status"
    assert Path(cc.installed_home(settings)) == Path(home)
    # The hook keeps its forecasts there, whatever folder Claude Code's own environment names.
    monkeypatch.setenv("YARDSTICK_HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(prompt_event("Add a login page"))))
    assert cc.main(["--home", home, "hook"]) == 0
    assert (Path(home) / "claude-code" / "s1.json").is_file() and not (tmp_path / "elsewhere").exists()


def test_install_offers_to_save_a_voyage_key_that_is_only_in_the_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    store: dict = {}
    monkeypatch.setattr("keyring.get_password", lambda service, user: store.get((service, user)))
    monkeypatch.setattr("keyring.set_password", lambda service, user, key: store.__setitem__((service, user), key))
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key-never-real")
    # A program adds it with --yes. Nothing is saved, and it says how to save the key.
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "only in the VOYAGE_API_KEY environment variable" in out and "With --yes it isn't saved" in out
    assert not store
    # In a terminal, it asks.
    monkeypatch.setattr(cli, "interactive", lambda: True)
    replies = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    assert cli.main(["claude-code", "install", "--here"]) == 0
    assert store == {("yardstick-predict", "voyage-api-key"): "test-key-never-real"}
    assert "Saved in your computer's password store." in capsys.readouterr().out
    # With the key in the password store, there's nothing to say.
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert "VOYAGE_API_KEY" not in capsys.readouterr().out


def test_install_says_so_when_no_voyage_key_can_be_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert "No Voyage key was found" in capsys.readouterr().out
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key-never-real")  # and this computer has no password store
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    assert "This computer has no password store to keep it in" in capsys.readouterr().out


def test_removing_deletes_what_the_hook_kept_once_no_settings_use_it(tmp_path, monkeypatch, own_home):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    Predictor.create("default", [], {"model": "voyage-4", "dimensions": 256})
    kept = Path(os.environ["YARDSTICK_HOME"]) / "claude-code"
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    send(monkeypatch, prompt_event("Add a login page"))
    assert list(kept.glob("s1.*"))
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert not kept.exists()
    # While yardstick is on for all your sessions, removing it from this folder keeps what the hook kept.
    everywhere = own_home / ".claude" / "settings.json"
    everywhere.parent.mkdir()
    everywhere.write_text(json.dumps(cc.added({}, sys.executable)))
    assert cli.main(["claude-code", "install", "--here", "--yes"]) == 0
    send(monkeypatch, prompt_event("Add a login page"))
    assert cli.main(["claude-code", "remove", "--here"]) == 0
    assert list(kept.glob("s1.*"))


# The status line ---------------------------------------------------------------------------------------


def test_the_status_line_follows_a_task_from_prompt_to_stop(monkeypatch):
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    send(monkeypatch, prompt_event("Add a login page"))
    started = state()["prompt_at"]
    assert json.loads(cc._prompt_path("s1", started).read_text())["prompt"] == "Add a login page"
    assert cc.status_line("s1", "claude-opus-5-5", started + 1).endswith("· forecasting…")

    forecasts = [
        {"model": "claude-opus-5-5", "effort": "max", "seconds": 240.0, "tokens": 9, "usd": 1.234},
        {"model": "claude-opus-5-5", "effort": "high", "seconds": 120.0, "tokens": 9, "usd": 0.5},
        {"model": "claude-sonnet-5", "seconds": None, "tokens": None, "usd": None, "reason": "not_enough_data"},
    ]
    cc._forecast_path("s1", started).write_text(json.dumps({"prompt_at": started, "forecasts": forecasts}))
    line = plain(cc.status_line("s1", "claude-opus-5-5[1m]", started + 72))
    assert line == "yardstick · 1:12 of ~4 min · ~$1.23"
    at_high = plain(cc.status_line("s1", "claude-opus-5-5", started + 72, effort="high"))
    assert at_high == "yardstick · 1:12 of ~2 min · ~$0.50"  # Claude Code says which effort level is on
    missing = plain(cc.status_line("s1", "claude-sonnet-5", started + 5))
    assert missing == "yardstick · no forecast for claude-sonnet-5 yet"

    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop"})
    done = state()["done_at"]
    line = plain(cc.status_line("s1", "claude-opus-5-5", done + 999))
    assert line.startswith("yardstick · took 0:0") and line.endswith("· forecast ~4 min · ~$1.23")


def test_the_clock_runs_again_when_the_agent_goes_on_after_a_stop(tmp_path, monkeypatch):
    clock = Clock(100.0)
    monkeypatch.setattr(cc, "time", clock)
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    record = tmp_path / "continued.jsonl"
    record.write_text("")
    send(monkeypatch, prompt_event("Fix and test it", "s", transcript_path=str(record)))
    cc._write(cc._forecast_path("s", 100.0), {"prompt_at": 100.0, "forecasts": [{"model": "m", "seconds": 120}]})
    clock.now = 130.0
    # Another Stop hook blocks this stop and tells Claude to run the tests, so Claude goes on.
    send(monkeypatch, {"session_id": "s", "hook_event_name": "Stop", "stop_hook_active": False})
    reply = {"type": "assistant", "timestamp": "1970-01-01T00:02:25Z", "message": {"model": "m", "content": []}}
    record.write_text(json.dumps(reply) + "\n")
    assert plain(cc.status_line("s", "m", 160.0, transcript=str(record))) == "yardstick · 1:00 of ~2 min"
    clock.now = 190.0
    send(monkeypatch, {"session_id": "s", "hook_event_name": "Stop", "stop_hook_active": True})
    done = plain(cc.status_line("s", "m", 200.0, transcript=str(record)))
    assert done == "yardstick · took 1:30 · forecast ~2 min"  # the final stop sets the final time


def test_took_leaves_out_time_spent_waiting_for_you(tmp_path, monkeypatch):
    clock = Clock(1_780_000_000.5)
    commands = forecasts_by_hand(monkeypatch, clock)
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers([], seconds=60.0))
    session = ClaudeSession(tmp_path, "s1", 1_780_000_000)
    line = session.prompt("Deploy it")
    tool = session.reply(10, tool="Bash")
    session.tool_result(tool, 60, line["promptId"], source="user_temporary")  # you allow the command a minute later
    session.reply(10, text="Deployed.")
    record = session.save()
    [trained] = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=2e9)
    assert trained.seconds == 20  # training leaves the wait out

    send(monkeypatch, prompt_event("Deploy it", prompt_id=line["promptId"], transcript_path=str(record)))
    assert cc.main(commands[-1]) == 0
    started = state()["prompt_at"]
    running = plain(cc.status_line("s1", "m", started + 75, transcript=str(record)))
    assert running == "yardstick · 0:15 of ~60 s"  # while it runs, a wait that's over is left out already
    clock.now = started + 81
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop", "transcript_path": str(record)})
    assert cc.main(commands[-1]) == 0  # works out the agent's time in the background
    assert state()["took"] == trained.seconds
    assert (
        plain(cc.status_line("s1", "m", started + 999, transcript=str(record)))
        == "yardstick · took 0:20 · forecast ~60 s"
    )
    assert not list(cc.folder().glob("*.task.jsonl"))  # the copy of the task's lines is gone


def test_pressing_esc_stops_the_clock(tmp_path):
    record = tmp_path / "session.jsonl"
    stopped = {
        "type": "user",
        "timestamp": "2026-10-22T10:00:30Z",
        "message": {"content": [{"type": "text", "text": "[Request interrupted by user]"}]},
    }
    lines = [{"type": "assistant", "message": {"model": "m"}}, stopped]
    record.write_text("\n".join(json.dumps(line) for line in lines))
    started = datetime.fromisoformat("2026-10-22T10:00:00+00:00").timestamp()
    assert cc.interrupted_at(str(record), started) == started + 30
    assert cc.interrupted_at(str(record), started + 60) is None  # an older stop doesn't count
    not_replies = [
        {"type": "assistant", "message": {"model": "<synthetic>"}},  # an error notice from Claude Code
        {"type": "assistant", "isSidechain": True, "message": {"model": "m"}},  # a helper's reply
    ]
    for later in not_replies:
        record.write_text("\n".join(json.dumps(line) for line in [*lines, later]))
        assert cc.interrupted_at(str(record), started) == started + 30
    replied = {"type": "assistant", "message": {"model": "m"}}
    record.write_text("\n".join(json.dumps(line) for line in [*lines, replied]))
    assert cc.interrupted_at(str(record), started) is None  # the agent went on after it
    helper = {**stopped, "isSidechain": True}
    mention = {
        "type": "user",
        "timestamp": "2026-10-22T10:00:40Z",
        "message": {"content": [{"type": "tool_result", "content": "[Request interrupted by user]"}]},
    }
    for line in (helper, mention):
        record.write_text(json.dumps(line))
        assert cc.interrupted_at(str(record), started) is None  # a helper's stop, or a tool's output saying it


def test_a_model_matches_its_dated_name_both_ways_and_never_another_model():
    dated = [{"model": "claude-opus-4-5-20251101", "effort": "high", "seconds": 60.0}]
    assert cc._pick(dated, "claude-opus-4-5", "high") == dated[0]  # Claude Code may name the model without a date
    alias = [{"model": "claude-opus-4-5", "seconds": 30.0}]
    assert cc._pick(alias, "claude-opus-4-5-20251101", "") == alias[0]
    assert cc._pick([{"model": "claude-opus-4", "seconds": 9.0}], "claude-opus-4-5", "") is None
    assert cc._pick([{"model": "claude-opus-4-5-20250101", "seconds": 8.0}], "claude-opus-4-5-20251101", "") is None
    cc._state_path("s1").write_text(json.dumps({"prompt_at": 100.0}))
    cc._forecast_path("s1", 100.0).write_text(json.dumps({"prompt_at": 100.0, "forecasts": dated}))
    assert plain(cc.status_line("s1", "claude-opus-4-5[1m]", 172.0)) == "yardstick · 1:12 of ~60 s"
    assert plain(cc.status_line("s1", "claude-opus-4", 172.0)) == "yardstick · no forecast for claude-opus-4 yet"


def test_the_session_effort_level_picks_the_forecast(tmp_path):
    record = tmp_path / "session.jsonl"
    lines = [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant", "effort": "high", "message": {"model": "claude-opus-5-5"}},
        {"type": "assistant", "effort": "max", "message": {"model": "claude-opus-5-5"}},
    ]
    record.write_text("\n".join(json.dumps(line) for line in lines))
    assert cc.last_model_and_effort(str(record)) == ("claude-opus-5-5", "max")
    assert cc.last_model_and_effort(str(tmp_path / "missing.jsonl")) == ("", "")
    later = [
        {"type": "assistant", "message": {"model": "<synthetic>"}},  # an error notice from Claude Code
        {"type": "assistant", "message": {"model": ""}},
        {"type": "assistant", "isSidechain": True, "message": {"model": "claude-haiku-5-5"}},  # a helper's reply
        {"type": "assistant", "message": "assistant"},
        ["assistant"],
    ]
    record.write_text("\n".join(json.dumps(line) for line in [*lines, *later]))
    assert cc.last_model_and_effort(str(record)) == ("claude-opus-5-5", "max")
    forecasts = [{"model": "m", "effort": "high", "seconds": 1}, {"model": "m", "effort": "max", "seconds": 2}]
    assert cc._pick(forecasts, "m", "max")["seconds"] == 2
    assert cc._pick(forecasts, "m-20260101", "")["seconds"] == 1  # a dated name, and the usual level


def test_the_status_line_takes_the_model_and_effort_as_objects_or_as_text(monkeypatch, capsys):
    cc._state_path("s1").write_text(json.dumps({"prompt_at": 100.0}))
    forecasts = [
        {"model": "claude-opus-5-5", "effort": "max", "seconds": 240.0},
        {"model": "claude-opus-5-5", "effort": "high", "seconds": 120.0},
    ]
    cc._forecast_path("s1", 100.0).write_text(json.dumps({"prompt_at": 100.0, "forecasts": forecasts}))
    monkeypatch.setattr(cc, "time", SimpleNamespace(time=lambda: 172.0))
    events = [
        {"session_id": "s1", "model": {"id": "claude-opus-5-5"}, "effort": {"level": "max"}},
        {"session_id": "s1", "model": "claude-opus-5-5", "effort": "max"},
        {"session_id": "s1", "model": ["odd"], "effort": 5, "transcript_path": {"odd": True}},
        ["not", "an", "object"],
    ]
    lines = []
    for event in events:
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
        assert cc.status() == 0
        lines.append(plain(capsys.readouterr().out.strip()))
    assert lines[:2] == ["yardstick · 1:12 of ~4 min"] * 2
    assert lines[2:] == ["yardstick · no forecast for this AI model yet", "yardstick"]


@pytest.mark.parametrize(
    "raw",
    [b"", b"not json", b"[1, 2]", b'{"session_id": 5}', b"\xff\xfe{", b"[" * 100_000],
    # Short names: pytest keeps the test's name in a variable, which Windows allows only 32,767 characters.
    ids=["empty", "not-json", "a-list", "a-number-for-the-session", "not-utf-8", "deeply-nested"],
)
def test_nothing_it_is_sent_breaks_the_hook_or_the_status_line(monkeypatch, capsys, raw):
    for run in (cc.hook, cc.status):
        monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(raw)))
        assert run() == 0
    assert capsys.readouterr().out == "yardstick\n"


def test_damaged_files_never_break_the_hook_or_the_status_line(monkeypatch):
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    states = [
        {"prompt_at": "soon"},
        {"prompt_at": 100.0, "done_at": [1], "took": "long", "took_for": 5, "earlier": "x", "transcript": 7},
        {"prompt_at": True, "earlier": [5, None, {"prompt_at": "x"}]},
    ]
    forecasts = [{"prompt_at": 100.0, "forecasts": "x"}, {"prompt_at": 100.0, "forecasts": [5, {"model": 5}]}]
    for broken in states:
        for found in forecasts:
            cc._state_path("s1").write_text(json.dumps(broken))
            cc._forecast_path("s1", 100.0).write_text(json.dumps(found))
            assert plain(cc.status_line("s1", "m", 172.0)).startswith("yardstick")
            send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop"})
    cc._state_path("s1").write_text("{not json")
    send(monkeypatch, prompt_event("Add a login page"))
    assert state()["earlier"] == []


def test_half_an_emoji_in_a_prompt_never_breaks_the_hook(monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    # Claude Code can send half of an emoji, as JSON's \ud83d. No file can hold it alone.
    raw = '{"session_id": "s1", "hook_event_name": "UserPromptSubmit", "prompt": "Fix the \\ud83d bug"}'
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    assert cc.hook() == 0
    assert cc.main(commands[-1]) == 0
    assert [task.prompt for task in asked] == ["Fix the \ufffd bug"]
    assert plain(cc.status_line("s1", "m", state()["prompt_at"] + 1)) == "yardstick · 0:01 of ~2 min"


# Which prompts start a task ----------------------------------------------------------------------------


def test_the_forecast_says_a_person_typed_the_prompt_after_the_conversation_so_far(tmp_path, monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    sent = 1_780_000_000  # when the hook runs, by the clock of forecasts_by_hand
    session = ClaudeSession(tmp_path, "s1", sent - 600)
    session.prompt("Add a login page")
    session.reply(5, text="Added the login page.")
    session.prompt("continue")
    session.reply(5, text="Styled the form too.")
    session.clock = sent - 1
    session.prompt("continue")  # the prompt being forecast, recorded just before the hook ran
    session.reply(60, text="Checking the tests.")  # the agent's answer, after it
    record = session.save()
    tasks = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=time.time() + 86400)
    trained = max(tasks, key=lambda t: t.started_at).context  # the context training gives this prompt
    assert trained.endswith("User: continue\n\nAssistant: Styled the form too.")

    for lines in (session.lines, session.lines[:-2]):  # with the prompt already in the record, and not yet
        record.write_text("".join(json.dumps(line) + "\n" for line in lines))
        send(monkeypatch, prompt_event("continue", transcript_path=str(record)))
        assert cc.main(commands[-1]) == 0
    assert [(t.origin, t.agent, t.context) for t in asked] == [("person", "claude-code", trained)] * 2


def test_the_forecast_says_which_conversation_and_project_the_prompt_is_in_and_how_big_it_is(tmp_path, monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    sent = 1_780_000_000  # when the hook runs, by the clock of forecasts_by_hand
    session = ClaudeSession(tmp_path, "s1", sent - 600, project="shop")
    session.prompt("Add a login page")
    session.reply(5, calls_usage=usage(new_input=300, write=1_000, read=40_000, output=700))
    session.clock = sent - 1
    session.prompt("Now add tests")  # the prompt being forecast, recorded just before the hook ran
    record = session.save()
    send(monkeypatch, prompt_event("Now add tests", transcript_path=str(record), cwd="/work/shop"))
    assert cc.main(commands[-1]) == 0
    [task] = asked
    assert (task.session, task.project) == ("s1", "shop")  # as training names them
    # As in training, the size at the first call: all the last reply read and wrote, and the new prompt.
    assert task.conversation_tokens == 42_000 + len("Now add tests") // 4


def test_the_forecast_sends_the_skills_known_before_the_prompt_as_training_does(tmp_path, monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    sent = 1_780_000_000  # when the hook runs, by the clock of forecasts_by_hand
    session = ClaudeSession(tmp_path, "s1", sent - 600)
    session.prompt("<command-name>/deploy</command-name>\n<command-args>staging</command-args>")  # a typed /deploy
    session.reply(5, tool="Skill", tool_input={"skill": "changelog"})  # the agent loads another skill
    session.reply(5, text="Deployed.")
    session.clock = sent - 1
    # Claude Code records the /review you type as a command. The hook gets it as you typed it.
    line = session.prompt("<command-name>/review</command-name>\n<command-args>the change</command-args>")
    session.reply(30, text="Reviewed.")
    record = session.save()
    tasks = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=2e9)
    trained = max(tasks, key=lambda t: t.started_at).prior_skills.split("\n")
    assert trained == ["changelog", "deploy", "review"]
    for more in ({"prompt_id": line["promptId"]}, {}):  # found by its id, and by when it was sent
        send(monkeypatch, prompt_event("/review the change", transcript_path=str(record), **more))
        assert cc.main(commands[-1]) == 0
    assert [task.skills for task in asked] == [trained, trained]
    # A helper's skill, in a line older versions of Claude Code wrote into the main record, isn't the agent's.
    helper = {**session.lines[1], "uuid": "helper-line", "isSidechain": True}
    helper["message"] = {
        **helper["message"],
        "content": [{"type": "tool_use", "id": "t", "name": "Skill", "input": {"skill": "lint"}}],
    }
    record.write_text("".join(json.dumps(line) + "\n" for line in [*session.lines[:2], helper, *session.lines[2:]]))
    assert cc.before_prompt(str(record), line["message"]["content"], sent, line["uuid"])[1] == trained


def test_lines_the_record_holds_twice_count_once_as_in_training(tmp_path):
    sent = 1_780_000_000
    session = ClaudeSession(tmp_path, "s1", sent - 600)
    session.prompt("<command-name>/deploy</command-name>")
    session.reply(5, tool="Skill", tool_input={"skill": "changelog"})
    session.reply(5, text="Deployed it.")
    session.lines += [dict(line) for line in session.lines]  # Claude Code wrote the history out again
    session.clock = sent - 1
    line = session.prompt("continue")
    session.reply(10, text="Checked it.")
    record = session.save()
    trained = max(
        read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path)}, now=2e9), key=lambda t: t.started_at
    )
    context, skills = cc.before_prompt(str(record), "continue", sent, line["uuid"])
    assert (context, skills) == (trained.context, trained.prior_skills.split("\n"))
    assert context.count("Deployed it.") == 1 and skills == ["changelog", "deploy"]


def test_a_scheduled_prompt_is_forecast_as_a_programs_as_in_training(tmp_path, monkeypatch):
    clock = Clock(101.0)
    commands = forecasts_by_hand(monkeypatch, clock)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    session = ClaudeSession(tmp_path / "records", "s", 100.0)
    line = session.prompt("Check the build", origin="scheduled")
    session.reply(20)
    record = session.save()
    tasks = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path / "records")}, now=2e9)
    assert [task.origin for task in tasks] == ["program"]
    # Claude Code sends the prompt's id. Without one, the prompt's text finds its line.
    for more in ({"prompt_id": line["promptId"]}, {}):
        send(monkeypatch, prompt_event("Check the build", "s", transcript_path=str(record), **more))
        assert cc.main(commands[-1]) == 0
        clock.now += 1
    assert [task.origin for task in asked] == ["program", "program"]


def test_a_message_from_another_session_is_counted_as_training_counts_it(tmp_path, monkeypatch):
    commands = forecasts_by_hand(monkeypatch, Clock(101.0))
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    session = ClaudeSession(tmp_path / "records", "s", 100.0)
    text = 'Another Claude session sent a message:\n<cross-session-message from="a">Done.</cross-session-message>'
    line = session.prompt(text, origin="peer", meta=True)  # as Claude Code writes it
    session.reply(20)
    record = session.save()
    trained = read_archive({"name": "c", "format": "claude-code", "path": str(tmp_path / "records")}, now=2e9)
    assert [task.origin for task in trained] == ["program"]
    send(monkeypatch, prompt_event(text, "s", prompt_id=line["promptId"], transcript_path=str(record)))
    assert cc.main(commands[-1]) == 0
    assert [task.origin for task in asked] == ["program"] and state("s")["prompt_at"] == 101.0


def test_a_background_job_reporting_back_starts_no_task_and_restarts_the_clock(monkeypatch):
    clock = Clock(100.0)
    commands = forecasts_by_hand(monkeypatch, clock)
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers([], seconds=300.0))
    send(monkeypatch, prompt_event("Build it in the background"))
    cc._write(cc._forecast_path("s1", 100.0), {"prompt_at": 100.0, "forecasts": [{"model": "m", "seconds": 300}]})
    clock.now = 130.0
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop"})
    assert plain(cc.status_line("s1", "m", 150.0)) == "yardstick · took 0:30 · forecast ~5 min"
    started = len(commands)
    clock.now = 200.0
    send(monkeypatch, prompt_event(NOTIFICATION))  # the build reports back, and the agent goes on
    assert state()["prompt_at"] == 100.0 and "done_at" not in state()
    assert len(commands) == started and not cc._prompt_path("s1", 200.0).exists()  # no forecast for it
    assert plain(cc.status_line("s1", "m", 210.0)) == "yardstick · 1:50 of ~5 min"
    # A monitor's event reports no finished job. Training counts it as no task, and as no part of one.
    clock.now = 300.0
    send(
        monkeypatch, prompt_event("<task-notification><task-id>m1</task-id><event>A line.</event></task-notification>")
    )
    assert cc.main(commands[-1]) == 0  # its line isn't in the record, so its text decides
    assert state()["prompt_at"] == 100.0 and not cc._forecast_path("s1", 300.0).exists()


def test_a_prompt_that_training_counts_as_no_task_brings_back_the_task_before_it(tmp_path, monkeypatch):
    clock = Clock(1_780_000_000.0)
    commands = forecasts_by_hand(monkeypatch, clock)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    session = ClaudeSession(tmp_path, "s1", 1_780_000_000 - 1)
    first = session.prompt("Add a login page")
    session.reply(5)
    record = session.save()
    sent = {"transcript_path": str(record)}
    send(monkeypatch, prompt_event("Add a login page", prompt_id=first["promptId"], **sent))
    assert cc.main(commands[-1]) == 0
    clock.now += 6
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop", "prompt_id": first["promptId"], **sent})
    assert cc.main(commands[-1]) == 0
    took = "yardstick · took 0:05 · forecast ~2 min"  # by training's rules: from its prompt to its last reply
    assert plain(cc.status_line("s1", "m", clock.now + 60, transcript=str(record))) == took
    # Claude Code tells the agent about messages it delivered. Training counts that as no task, and no part of one.
    clock.now += 24
    session.clock = clock.now - 1
    notice = session.prompt("[Cross-session delivery notice] 3 of your messages were delivered.", origin="system")
    notice["isMeta"] = True
    session.save()
    send(monkeypatch, prompt_event(notice["message"]["content"], prompt_id=notice["promptId"], **sent))
    assert state()["prompt_at"] == clock.now
    assert cc.main(commands[-1]) == 0
    assert state()["prompt_at"] == 1_780_000_000.0 and [task.prompt for task in asked] == ["Add a login page"]
    session.reply(3, text="Noted.")  # the agent answers the notice, and stops again
    session.save()
    clock.now += 4
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop", "prompt_id": notice["promptId"], **sent})
    assert plain(cc.status_line("s1", "m", clock.now + 60, transcript=str(record))) == took


def test_runs_with_no_status_line_make_no_forecast(tmp_path, monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")  # claude -p
    send(monkeypatch, prompt_event("Add a login page"))
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop"})
    assert commands == [] and not list(cc.folder().glob("*.prompt"))
    assert state()["done_at"] > state()["prompt_at"]  # the clock alone is kept
    # Without the variable, the session's record says how Claude Code was started.
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT")
    record = tmp_path / "session.jsonl"
    for entrypoint, started in (("sdk-ts", 0), ("cli", 1)):
        record.write_text(json.dumps({"type": "user", "entrypoint": entrypoint, "message": {"content": "hi"}}) + "\n")
        send(monkeypatch, prompt_event("Add a login page", transcript_path=str(record)))
        assert len(commands) == started


# The work in the background ----------------------------------------------------------------------------


def test_an_older_forecast_never_replaces_a_newer_one(monkeypatch):
    commands = forecasts_by_hand(monkeypatch)

    def answer(predictor, tasks, distribution=False, model=None, recent=None):
        if tasks[0].prompt == "First prompt":
            # While the first forecast runs, a second prompt is sent. The job that's running picks it up.
            send(monkeypatch, prompt_event("Second prompt"))
            assert cc.main(commands[-1]) == 0  # a second job for the session ends at once
        seconds = 60.0 if tasks[0].prompt == "First prompt" else 120.0
        return {"forecasts": [{"model": "m", "seconds": seconds, "tokens": 9, "usd": None}]}

    monkeypatch.setattr("yardstick_predict.forecast.answer", answer)
    send(monkeypatch, prompt_event("First prompt"))
    assert cc.main(commands[0]) == 0 and len(commands) == 1
    latest = state()["prompt_at"]
    assert plain(cc.status_line("s1", "m", latest + 1)) == "yardstick · 0:01 of ~2 min"  # the second prompt's
    assert len(list(cc.folder().glob("s1.*.forecast.json"))) == 1  # the first one's is no longer needed


def test_a_newer_prompt_takes_the_place_of_one_whose_forecast_never_started(monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    send(monkeypatch, prompt_event("First prompt"))
    send(monkeypatch, prompt_event("Second prompt"))
    for command in commands:
        assert cc.main(command) == 0
    assert [task.prompt for task in asked] == ["Second prompt"]
    assert not list(cc.folder().glob("*.prompt"))
    assert plain(cc.status_line("s1", "m", state()["prompt_at"] + 1)) == "yardstick · 0:01 of ~2 min"


def test_a_forecast_stops_waiting_for_its_prompts_line_once_a_newer_prompt_comes(tmp_path, monkeypatch):
    clock = Clock(1_780_000_000.0)
    commands = forecasts_by_hand(monkeypatch, clock)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    record = tmp_path / "session.jsonl"
    record.write_text("")  # Claude Code hasn't written the prompts' lines yet
    send(monkeypatch, prompt_event("First prompt", prompt_id="p1", transcript_path=str(record)))
    waited: list = []

    def sleep(seconds):
        waited.append(seconds)
        clock.now += seconds
        if len(waited) == 1:  # while the first prompt's forecast waits, a second prompt is sent
            send(monkeypatch, prompt_event("Second prompt", prompt_id="p2", transcript_path=str(record)))

    clock.sleep = sleep
    assert cc.main(commands[0]) == 0 and len(commands) == 1
    assert [task.prompt for task in asked] == ["Second prompt"]
    assert sum(waited) < 1.5 * cc.FIND_SECONDS  # the first gave up at once, and only the second waited


def test_one_session_runs_one_background_job_at_a_time(monkeypatch):
    commands = forecasts_by_hand(monkeypatch)
    asked: list = []
    monkeypatch.setattr("yardstick_predict.forecast.answer", answers(asked))
    with locks.held(cc._lock_path("s1", "worker")):  # a job for this session is running
        send(monkeypatch, prompt_event("First prompt"))
        send(monkeypatch, prompt_event("Second prompt"))
        assert commands == []  # the running job picks the prompts up, so none starts
        assert cc.main(["work", "s1"]) == 0 and asked == []  # and one started anyway ends at once
        send(monkeypatch, prompt_event("Another session's prompt", "s2"))
        assert commands == [["work", "s2"]]  # other sessions don't wait
    assert cc.main(["work", "s1"]) == 0
    assert [task.prompt for task in asked] == ["Second prompt"]


def test_a_password_store_that_never_answers_holds_a_forecast_up_for_seconds_at_most(monkeypatch):
    from types import SimpleNamespace

    from yardstick_predict import embeddings, forecast

    release = threading.Event()
    monkeypatch.setattr("keyring.get_password", lambda service, user: release.wait(10))
    monkeypatch.setattr(forecast, "KEY_WAIT", 0.05)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    predictor = SimpleNamespace(embedding_model="voyage-4")
    try:
        began = time.monotonic()
        with pytest.raises(forecast.RequestError) as stopped:
            forecast._embed(predictor, [(0, "prompt", "Add a login page")])
        assert time.monotonic() - began < 5
        assert stopped.value.code == "no_embedding_key" and "didn't answer in time" in stopped.value.message
        # A forecast whose texts are all in the cache needs no key, so it never asks the store.
        embeddings.remember("voyage-4", {"Add a login page": [0.5] * embeddings.DIMENSIONS})
        monkeypatch.setattr("keyring.get_password", lambda service, user: pytest.fail("asked the password store"))
        assert forecast._embed(predictor, [(0, "prompt", "Add a login page")])["prompt"][0][0] == 0.5
    finally:
        release.set()


def test_tidying_forgets_old_sessions_and_prompts_whose_forecast_never_started():
    root = cc.folder()
    now = time.time()
    ages = {
        "old.json": 8 * 86400,
        "old.100.0.forecast.json": 8 * 86400,
        "old.state.lock": 8 * 86400,
        "new.json": 3600,
        "new.100.0.forecast.json": 8 * 86400,  # an active session keeps what it has
        "new.200.0.prompt": 2 * 3600,  # its forecast never started
        "new.300.0.prompt": 60,  # its forecast may still start
        ".new.json.x1.partial": 2 * 3600,  # left by a save that stopped halfway
        cc.REPLACED: 30 * 86400,
    }
    for name, age in ages.items():
        (root / name).write_text("{}")
        os.utime(root / name, (now - age, now - age))
    cc._tidy()
    kept = {"new.json", "new.100.0.forecast.json", "new.300.0.prompt", cc.REPLACED}
    assert {path.name for path in root.iterdir()} == kept | {".tidied"}
    (root / "new.400.0.prompt").write_text("{}")
    os.utime(root / "new.400.0.prompt", (now - 7200, now - 7200))
    cc._tidy()  # once a day at most
    assert (root / "new.400.0.prompt").exists()


def test_a_forecast_error_in_the_status_line_never_holds_a_code_that_acts():
    from yardstick_predict.forecast import RequestError

    error = RequestError("embedding_failed", "Voyage said 500: \x1b]0;a title\x07down for now")
    assert cc._no_forecast(error, None) == "Voyage said 500: down for now"


# The weekly limit ---------------------------------------------------------------------------------------

NOW = 1_780_000_000.0  # in the past, so tidying never takes the files it just saved for a week old
RESETS = NOW + 2 * 86400  # the week began five days ago


def week_reading(percent: float = 30.0, resets: float = RESETS) -> dict:
    """What Claude Code sends the status line on a Claude subscription."""
    return {"rate_limits": {"seven_day": {"used_percentage": percent, "resets_at": resets}}}


BIG = usage(new_input=100_000, read=0, output=20_000)
"""A reply's tokens: enough that a week of a few replies is worth dividing by."""


def one_call() -> float:
    """What one reply with BIG tokens costs on claude-opus-5-5, at list prices."""
    from yardstick_predict import prices

    return prices.call_cost("claude-opus-5-5", 100_000, 0, 0, 0, 20_000)


def a_week_of_work(root: Path, monkeypatch) -> tuple[Path, float]:
    """Another session's work this week and last, then a task in session s1. Returns s1's record and its prompt time.

    This week holds six replies: three in the other session, one before the task and two in it. One of the
    task's replies is written over two lines, as Claude Code does, and counts once.
    """
    other = ClaudeSession(root, "e1", NOW - 6 * 86400, project="other")
    other.prompt("Last week's work")
    other.reply(5, calls_usage=BIG)  # before the week began
    other.clock = NOW - 86400
    other.prompt("This week's work")
    for _ in range(3):
        other.reply(5, calls_usage=BIG)
    other.save(mtime=NOW)
    session = ClaudeSession(root, "s1", NOW - 3600)
    session.prompt("Earlier")
    session.reply(5, calls_usage=BIG)
    record = session.save(mtime=NOW)
    monkeypatch.setattr(cc, "time", Clock(NOW - 600))
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    send(monkeypatch, prompt_event("Add a login page", transcript_path=str(record)))
    session.clock = NOW - 600
    session.prompt("Add a login page")
    session.reply(10, calls_usage=BIG, repeat=2)
    session.reply(10, calls_usage=BIG)
    session.save(mtime=NOW)
    return record, state()["prompt_at"]


def test_on_a_subscription_the_status_line_shows_shares_of_the_weekly_limit_in_place_of_dollars(tmp_path, monkeypatch):
    from yardstick_predict import weekly

    record, started = a_week_of_work(tmp_path, monkeypatch)
    forecast = {"model": "claude-opus-5-5", "effort": "high", "seconds": 240.0, "tokens": 9, "usd": one_call()}
    cc._forecast_path("s1", started).write_text(json.dumps({"prompt_at": started, "forecasts": [forecast]}))
    week = weekly.note(week_reading())
    weekly.calibrate(tmp_path, now=NOW)
    # 30% of the week went on six replies, so each reply is 5% of it. The task's two replies are 10%.
    assert weekly.rate() == pytest.approx(5 / one_call())
    line = plain(cc.status_line("s1", "claude-opus-5-5", started + 72, "high", str(record), week))
    assert line == "yardstick · 1:12 · 10.0000% (~4 min · ~5.0000%)"

    cc.time.now = started + 30  # the agent stops after its last reply
    send(monkeypatch, {"session_id": "s1", "hook_event_name": "Stop"})
    done = plain(cc.status_line("s1", "claude-opus-5-5", started + 999, "high", str(record), week))
    assert done == "yardstick · took 0:30 · 10.0000% (~4 min · ~5.0000%)"

    # With an API key, Claude Code sends no reading, and the line shows dollars as before.
    no_week = plain(cc.status_line("s1", "claude-opus-5-5", started + 999, "high", str(record)))
    assert no_week.endswith(f"· forecast ~4 min · ~${one_call():.2f}")


def test_before_the_share_of_a_dollar_is_known_the_brackets_show_the_time_alone(tmp_path, monkeypatch):
    from yardstick_predict import weekly

    record, started = a_week_of_work(tmp_path, monkeypatch)
    forecast = {"model": "claude-opus-5-5", "effort": "high", "seconds": 240.0, "tokens": 9, "usd": 1.5}
    cc._forecast_path("s1", started).write_text(json.dumps({"prompt_at": started, "forecasts": [forecast]}))
    week = weekly.note(week_reading())
    line = plain(cc.status_line("s1", "claude-opus-5-5", started + 72, "high", str(record), week))
    assert line == "yardstick · 1:12 (~4 min)"  # no dollars on a subscription


def test_the_status_line_keeps_the_weekly_reading_claude_code_sends(monkeypatch, capsys):
    from yardstick_predict import weekly

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": "s1", **week_reading(12.5)})))
    assert cc.status() == 0
    assert json.loads((cc.folder() / weekly.READING).read_text()) == {"percent": 12.5, "resets_at": RESETS}
    for odd in ({}, {"rate_limits": None}, {"rate_limits": {"seven_day": {"used_percentage": "a lot"}}}):
        assert weekly.note(odd) is None


def test_a_task_share_counts_its_helpers_work(tmp_path, monkeypatch):
    from yardstick_predict import prices, weekly

    session = ClaudeSession(tmp_path, "s1", NOW - 600)
    record = session.save(mtime=NOW)
    monkeypatch.setattr(cc, "time", Clock(NOW - 600))
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    send(monkeypatch, prompt_event("Look into it", transcript_path=str(record)))
    line = session.prompt("Look into it")
    session.reply(10, tool="Agent", calls_usage=BIG)
    session.helper(line["promptId"], calls=2)
    session.save(mtime=NOW)
    helper = prices.call_cost("claude-haiku-5-5", 10, 0, 0, 5_000, 50)
    assert weekly.task_spent("s1", state()) == pytest.approx(one_call() + 2 * helper)


def test_a_task_share_follows_its_record_when_claude_code_moves_it(tmp_path, monkeypatch):
    from yardstick_predict import weekly

    record, _ = a_week_of_work(tmp_path, monkeypatch)
    # The session moved into a worktree. Claude Code moved its record and its helpers to that folder's records.
    moved = tmp_path / "-work-demo-worktree" / record.name
    moved.parent.mkdir()
    record.rename(moved)
    assert weekly.task_spent("s1", state()) == 0.0  # where the record was at the prompt, there's nothing now
    assert weekly.task_spent("s1", state(), str(moved)) == pytest.approx(2 * one_call())
    assert weekly.task_spent("s1", state(), str(moved)) == pytest.approx(2 * one_call())  # each reply counts once


def test_a_helper_an_earlier_task_started_counts_toward_that_task_as_in_training(tmp_path, monkeypatch):
    from yardstick_predict import prices, weekly

    session = ClaudeSession(tmp_path, "s1", NOW - 1200)
    first = session.prompt("Look into the logs")
    session.reply(10, tool="Agent", calls_usage=BIG)
    old_helper = session.helper(first["promptId"], calls=1)
    session.reply(10, text="It's looking.", calls_usage=BIG)
    record = session.save(mtime=NOW)
    monkeypatch.setattr(cc, "time", Clock(NOW - 600))
    monkeypatch.setattr(cc, "_start_worker", lambda session: None)
    send(monkeypatch, prompt_event("Add a login page", transcript_path=str(record)))
    session.clock = NOW - 600
    second = session.prompt("Add a login page")
    session.reply(10, tool="Agent", calls_usage=BIG)
    session.helper_more(old_helper, None, calls=2)  # the first task's helper is still at work
    session.helper(second["promptId"], calls=1)  # this task's own helper
    session.reply(10, text="Done.", calls_usage=BIG)
    session.save(mtime=NOW)
    helper = prices.call_cost("claude-haiku-5-5", 10, 0, 0, 5_000, 50)
    assert weekly.task_spent("s1", state()) == pytest.approx(2 * one_call() + helper)
    archive = {"name": "c", "format": "claude-code", "path": str(tmp_path)}
    trained = sorted(read_archive(archive, now=2e9), key=lambda task: task.started_at)
    assert trained[-1].usd == pytest.approx(2 * one_call() + helper)  # training counts it the same way


def test_moving_a_sessions_records_never_counts_its_replies_twice(tmp_path):
    from yardstick_predict import weekly

    session = ClaudeSession(tmp_path, "s1", NOW - 3600)
    first = session.prompt("Fix it")
    for _ in range(2500):  # many replies: none of them may count again
        session.reply(1)
    session.helper(first["promptId"], calls=3)
    record = session.save(mtime=NOW)
    total = weekly.spent_since(tmp_path, NOW - 86400, NOW)
    assert total > 0
    # The session moved into a worktree. Claude Code moved its record and its helpers to that folder's records.
    moved = tmp_path / "-work-demo-worktree"
    moved.mkdir()
    record.rename(moved / record.name)
    record.with_suffix("").rename(moved / "s1")
    assert weekly.spent_since(tmp_path, NOW - 86400, NOW) == pytest.approx(total)
    shutil.copy(moved / record.name, record)  # a copy left in both folders counts once too
    assert weekly.spent_since(tmp_path, NOW - 86400, NOW) == pytest.approx(total)


def test_a_resumed_sessions_copy_of_the_conversation_counts_once(tmp_path):
    from yardstick_predict import weekly

    session = ClaudeSession(tmp_path, "s1", NOW - 3600)
    session.prompt("Fix it")
    for _ in range(2):
        session.reply(5, calls_usage=BIG)
    session.save(mtime=NOW)
    # Resuming it starts a new session, whose record begins with a copy of the conversation so far.
    resumed = ClaudeSession(tmp_path, "s2", session.clock)
    resumed.lines = [{**line, "sessionId": "s2"} for line in session.lines]
    resumed.prompt("And test it")
    resumed.reply(5, calls_usage=BIG)
    resumed.save(mtime=NOW)
    assert weekly.spent_since(tmp_path, NOW - 86400, NOW) == pytest.approx(3 * one_call())


def test_a_record_written_again_from_the_start_is_counted_afresh(tmp_path):
    from yardstick_predict import weekly

    session = ClaudeSession(tmp_path, "s1", NOW - 3600)
    session.prompt("Fix it")
    for _ in range(3):
        session.reply(5, calls_usage=BIG)
    session.save(mtime=NOW)
    assert weekly.spent_since(tmp_path, NOW - 86400, NOW) == pytest.approx(3 * one_call())
    session.lines = session.lines[:-1]
    session.save(mtime=NOW)
    assert weekly.spent_since(tmp_path, NOW - 86400, NOW) == pytest.approx(2 * one_call())


def test_the_week_spend_reads_only_what_was_added_and_counts_each_reply_once(tmp_path):
    from yardstick_predict import weekly

    session = ClaudeSession(tmp_path, "s1", NOW - 3600)
    session.prompt("Fix it")
    session.reply(5, calls_usage=BIG)
    session.save(mtime=NOW)
    since = NOW - 86400
    assert weekly.spent_since(tmp_path, since, NOW) == pytest.approx(one_call())
    session.reply(5, calls_usage=BIG, repeat=2)
    lines = session.lines
    session.lines = lines[:-1]  # Claude Code has written the reply's first line only
    session.save(mtime=NOW)
    assert weekly.spent_since(tmp_path, since, NOW) == pytest.approx(2 * one_call())
    session.lines = lines
    session.save(mtime=NOW)
    assert weekly.spent_since(tmp_path, since, NOW) == pytest.approx(2 * one_call())  # its second line adds nothing
    assert weekly.spent_since(tmp_path, NOW - 60, NOW) == 0.0


def test_the_share_of_a_dollar_waits_for_a_reading_it_can_divide(tmp_path, monkeypatch):
    from yardstick_predict import weekly

    a_week_of_work(tmp_path, monkeypatch)
    weekly.note(week_reading(0.5))  # too little of the week used: the reading has one decimal
    weekly.calibrate(tmp_path, now=NOW)
    assert weekly.rate() is None
    weekly.note(week_reading(30.0, resets=NOW - 1))  # the week is over
    weekly.calibrate(tmp_path, now=NOW)
    assert weekly.rate() is None
    weekly.note(week_reading(30.0))
    weekly.calibrate(tmp_path, now=NOW)
    first = weekly.rate()
    weekly.note(week_reading(60.0))
    weekly.calibrate(tmp_path, now=NOW + 60)  # worked out a minute ago
    assert weekly.rate() == first
    weekly.calibrate(tmp_path, now=NOW + weekly.CALIBRATE_SECONDS)
    assert weekly.rate() == pytest.approx(2 * first)


def test_the_background_job_works_out_the_share_of_a_dollar_after_the_forecast(tmp_path, monkeypatch):
    from yardstick_predict import weekly

    record, _ = a_week_of_work(tmp_path, monkeypatch)
    worked: list[Path] = []
    monkeypatch.setattr(weekly, "calibrate", lambda records: worked.append(records))
    cc._calibrate(str(record))
    assert worked == [tmp_path]
