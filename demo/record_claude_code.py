"""Record the Claude Code demo: a real session that shows yardstick's forecast in the status line.

Claude Code runs on your own Claude account, in a settings folder of its own. Your usual settings, plugins and chats
stay out of it. Sign in there once, in a terminal:

    CLAUDE_CONFIG_DIR=FOLDER claude

Then run this from the repository's folder, with the same FOLDER:

    uv run python demo/record_claude_code.py FOLDER

It copies the made-up shop in demo/storefront to ~/storefront. Claude Code starts there on Opus at max effort, with
yardstick's hook and status line. yardstick forecasts from the made-up history, through the stand-in for Voyage. VHS
types the task and films Claude as it really works. That takes a few minutes and some of your Claude usage. Then
ffmpeg speeds up the middle and says so on screen. Claude's commands can show your user name in a path. macOS's own
text recognition finds each place. A box covers it. That needs swift, from Apple's command line tools.

On a Claude subscription the status line shows shares of your weekly limit, worked out from your own usage. The demo
shows dollars instead, as with an API key: status_in_dollars.py leaves the weekly reading out.

It writes demo/claude-code.gif for the README and demo/claude-code.mp4 for posts, with three stills. The film as it
was shot stays in demo/claude-code-raw.mp4. With --edit-only, it speeds that up again and films nothing. Last, it
removes ~/storefront and takes yardstick's hook and status line out of FOLDER.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import record  # demo/record.py, next to this file

HERE = record.HERE
PROJECT = Path.home() / "storefront"
"""Where Claude works: a copy of demo/storefront, removed afterwards. Claude Code shows it as ~/storefront."""
RAW = HERE / "claude-code-raw.mp4"
TASK_SENT = 3.8
"""When the task is sent, in seconds into the film: the pauses and the typing in claude-code.tape before Enter."""
AFTER_TASK = 6.0
"""How long the film plays at normal speed after the task is sent. So the forecast is seen arriving."""
FAST = 12.0
"""How many seconds the sped-up middle lasts."""
END = 7.0
"""How long the end plays at normal speed, with the status line saying how long Claude took. The tape waits as long."""
FRAMES_A_SECOND = 12
LINE = 26
"""How tall a box is that covers one line of text, in pixels: about a line of the terminal."""
BACKGROUND = "0x1e1e2e"
"""The terminal's background in the Catppuccin Mocha theme. Boxes this color cover what has to be hidden."""
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def commands(env: dict[str, str]) -> tuple[str, str]:
    """yardstick's hook and status line commands, for the made-up model and the stand-in for Voyage.

    They run with the made-up home folder. So yardstick never looks into yours, such as for skills.
    """
    names = ("HOME", "VOYAGE_API_KEY", "VOYAGE_API_BASE", "YARDSTICK_NO_UPDATE_CHECK", "PYTHON_KEYRING_BACKEND")
    settings = " ".join(f"{name}={shlex.quote(env[name])}" for name in (*names, "HF_HUB_OFFLINE"))
    python, home = shlex.quote(sys.executable), shlex.quote(env["YARDSTICK_HOME"])
    hook = f"env {settings} {python} -P -m yardstick_predict.claude_code --home {home} hook"
    status = f"env {settings} {python} -P {shlex.quote(str(HERE / 'status_in_dollars.py'))} --home {home}"
    return hook, status


def add_yardstick(config: Path, hook: str, status: str) -> str:
    """Put yardstick's hook and status line into the Claude Code settings in `config`. Returns the settings before."""
    path = config / "settings.json"
    before = path.read_text(encoding="utf-8") if path.exists() else "{}\n"
    settings = json.loads(before)
    group = [{"hooks": [{"type": "command", "command": hook}]}]
    settings["hooks"] = {"UserPromptSubmit": group, "Stop": group, "StopFailure": group}
    settings["statusLine"] = {"type": "command", "command": status, "padding": 0, "refreshInterval": 1}
    settings["theme"] = "dark"
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    return before


def quiet_start(config: Path) -> None:
    """Trust ~/storefront and mark Claude Code's offers as seen. Then nothing asks a question or shows an offer."""
    path = config / ".claude.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    project = data.setdefault("projects", {}).setdefault(os.path.realpath(PROJECT), {})
    project.update(hasTrustDialogAccepted=True, hasCompletedProjectOnboarding=True)
    data.update(hasVisitedPasses=True, passesUpsellSeenCount=3)
    data["fullscreenUpsellSeenCount"] = max(3, data.get("fullscreenUpsellSeenCount", 0))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def filming_environment(config: Path) -> dict[str, str]:
    """What the filmed terminal sees: your home and programs, the demo's settings folder and the fullscreen mode.

    Nothing else gets in from where this runs. A Claude Code session that runs this would pass on its own markers,
    and one of them turns transcripts off. The fullscreen mode redraws in place. So VHS can tell when Claude stops.
    """
    env = {name: os.environ[name] for name in ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR") if name in os.environ}
    env.update(SHELL="/bin/bash", LANG="en_US.UTF-8", CLAUDE_CONFIG_DIR=str(config))
    env.update(CLAUDE_CODE_NO_FLICKER="1", DISABLE_AUTOUPDATER="1")
    return env


def speed_label(folder: Path, speed: int) -> Path | None:
    """A small picture that says how much faster the middle plays. Chrome draws it. None without Chrome."""
    chrome = shutil.which("google-chrome") or (CHROME if Path(CHROME).exists() else None)
    if chrome is None:
        return None
    page, shot = folder / "label.html", folder / "label.png"
    style = "font:600 20px Menlo,monospace;color:#1e1e2e;background:#f9e2af;border-radius:8px;padding:6px 14px"
    page.write_text(
        f'<body style="margin:0;background:transparent"><span style="{style}">▶▶ {speed}× speed</span></body>',
        encoding="utf-8",
    )
    browser = [chrome, "--headless=new", f"--user-data-dir={folder / 'chrome'}", "--hide-scrollbars"]
    browser += ["--default-background-color=00000000", "--window-size=210,44", f"--screenshot={shot}", page.as_uri()]
    subprocess.run(browser, capture_output=True, timeout=120)
    return shot if shot.exists() else None


def find(word: str, pictures: list[Path]) -> list[tuple[int, int, int, int, int]]:
    """Each place `word` shows in the pictures: the picture's number from 0 and the box around it in pixels.

    macOS's own text recognition reads them, through find_text.swift.
    """
    reading = ["swift", str(HERE / "find_text.swift"), word, *map(str, pictures)]
    found = subprocess.run(reading, capture_output=True, text=True, check=True).stdout
    number = {str(picture): n for n, picture in enumerate(pictures)}
    return [(number[path], *map(int, box)) for path, *box in (line.split("\t") for line in found.splitlines())]


def frames(video: Path, folder: Path) -> list[Path]:
    """Every frame of a film, as pictures in a new folder."""
    folder.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), str(folder / "%05d.png")], check=True)
    return sorted(folder.glob("*.png"))


def edit(raw: Path, folder: Path) -> None:
    """Speed up the middle of the film as shot, and cover your user name wherever it shows.

    Claude's commands can show a path with your user name in it, such as ~/storefront written out in full. Each place
    gets a box the color of the background. Then the finished film is read again, and nothing is written if any
    place is left. Writes the MP4, the GIF and three stills in demo/.
    """
    probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(raw)]
    length = float(subprocess.run(probe, capture_output=True, text=True, check=True).stdout)
    start, end = TASK_SENT + AFTER_TASK, length - END
    speed = max((end - start) / FAST, 1.0)
    shown = (end - start) / speed  # how long the middle lasts once sped up
    film = (
        f"[0:v]trim=0:{start},setpts=PTS-STARTPTS[a];"
        f"[0:v]trim={start}:{end},setpts=(PTS-STARTPTS)/{speed}[b];"
        f"[0:v]trim={end},setpts=PTS-STARTPTS[c];"
        f"[a][b][c]concat=n=3:v=1:a=0,fps={FRAMES_A_SECOND}[v]"
    )
    inputs, last = ["-i", str(raw)], "[v]"
    label = speed_label(folder, round(speed)) if speed > 1 else None
    if label is not None:
        inputs += ["-i", str(label)]
        film += f";[v][1:v]overlay=x=W-w-24:y=48:enable='between(t,{start},{start + shown})'[out]"
        last = "[out]"
    cut = folder / "cut"  # the edited film, frame by frame, without loss
    cut.mkdir()
    subprocess.run(
        ["ffmpeg", "-v", "error", *inputs, "-filter_complex", film, "-map", last, str(cut / "%05d.png")], check=True
    )
    name = Path.home().name
    covers = [  # one line tall, around the middle of what was found
        f"drawbox=x={x - 4}:y={y + h // 2 - LINE // 2}:w={w + 8}:h={LINE}:color={BACKGROUND}:t=fill"
        f":enable='eq(n\\,{n})'"
        for n, x, y, w, h in find(name, sorted(cut.glob("*.png")))
    ]
    mp4 = HERE / "claude-code.mp4"
    encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart"]
    pictures = ["-framerate", str(FRAMES_A_SECOND), "-i", str(cut / "%05d.png")]
    cover = ["-vf", ",".join(covers)] if covers else []
    subprocess.run(["ffmpeg", "-y", "-v", "error", *pictures, *cover, *encode, str(mp4)], check=True)
    left = find(name, frames(mp4, folder / "check"))
    if left:
        mp4.unlink()
        sys.exit(f"The film still shows your user name in {len(left)} places. Nothing was written.")
    # In the GIF, the sped-up middle keeps every other frame. It's a blur of work anyway. The GIF stays small.
    keep = f"select='lt(t\\,{start})+gt(t\\,{start + shown})+not(mod(n\\,2))'"
    palette = "split[s0][s1];[s0]palettegen=stats_mode=diff:max_colors=96[p];[s1][p]paletteuse=dither=none"
    gif = ["-i", str(mp4), "-vf", f"fps=10,{keep},{palette}", str(HERE / "claude-code.gif")]
    subprocess.run(["ffmpeg", "-y", "-v", "error", *gif], check=True)
    for number, moment in enumerate((start - 0.5, start + shown / 3, start + shown + END - 0.5), 1):
        still = ["-ss", f"{moment:.2f}", "-i", str(mp4), "-frames:v", "1", str(HERE / f"claude-code-{number}.png")]
        subprocess.run(["ffmpeg", "-y", "-v", "error", *still], check=True)


def main(argv: list[str]) -> int:
    if argv == ["--edit-only"]:
        with tempfile.TemporaryDirectory(prefix="yardstick-demo-") as folder:
            edit(RAW, Path(folder))
        return 0
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    config = Path(argv[0]).expanduser().resolve()
    if not (config / ".claude.json").exists():
        print(f"Sign in to Claude Code there first: CLAUDE_CONFIG_DIR={config} claude", file=sys.stderr)
        return 1
    missing = [name for name in ("vhs", "claude", "ffmpeg", "ffprobe", "swift") if shutil.which(name) is None]
    if missing:
        print(f"Install {', '.join(missing)} first.", file=sys.stderr)
        return 1
    if PROJECT.exists():
        print(f"{PROJECT} is there already. Move it away first.", file=sys.stderr)
        return 1
    with record.stand_in_for_voyage() as voyage, tempfile.TemporaryDirectory(prefix="yardstick-demo-") as folder:
        print("Training a model on the made-up history...")
        env = record.made_up_model(Path(folder), voyage)
        before = add_yardstick(config, *commands(env))
        try:
            shutil.copytree(HERE / "storefront", PROJECT)
            quiet_start(config)
            print("Filming Claude Code at work. This takes a few minutes...")
            tape = [shutil.which("vhs"), str(HERE / "claude-code.tape")]
            subprocess.run(tape, env=filming_environment(config), cwd=folder, check=True)
            shutil.copyfile(Path(folder) / "raw.mp4", RAW)
            edit(RAW, Path(folder))
        finally:
            (config / "settings.json").write_text(before, encoding="utf-8")
            shutil.rmtree(PROJECT, ignore_errors=True)
    print("Wrote demo/claude-code.gif, demo/claude-code.mp4 and three stills")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
