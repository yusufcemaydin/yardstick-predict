# The demo clips

The [README](../README.md) has three short clips. Each runs the real yardstick-predict on a made-up history, not on anyone's own runs. So their forecasts are an example.

- `claude-code.gif`, at the top: a real Claude Code session, with the forecast in its status line. The middle is sped up. The clip says so.
- `demo.gif`, in Quick start: the home screen forecasts a task on two AI models.
- `cli.gif`, in Use it from a program: the command line names the AI model and the thinking level, then asks for each forecast's curve in JSON.

## Make them again

You need [VHS](https://github.com/charmbracelet/vhs), a program that films a terminal. On a Mac, `brew install vhs` adds it. The command-line clip also uses jq, which macOS has. Run this from the repository's folder:

```bash
uv run python demo/record.py
```

It takes about a minute. It writes the home screen and command-line clips, each as a GIF for the README and an MP4 for posts. Git leaves the MP4s out.

The Claude Code clip also needs Claude Code, ffmpeg and your Claude account. Claude really works on a task. That takes a few minutes and some of your usage. It gets a settings folder of its own. So your usual settings, plugins and chats stay out. Pick a folder, such as `~/yardstick-demo-claude`. Sign in there once:

```bash
CLAUDE_CONFIG_DIR=~/yardstick-demo-claude claude
```

Then film it:

```bash
uv run python demo/record_claude_code.py ~/yardstick-demo-claude
```

## What they do

1. `history.py` writes made-up Claude Code and Codex records into a home folder made for the demo. They read like eight weeks of work on three small projects with two AI models, claude-opus-5-5 and gpt-6.1-sol.
2. `record.py` starts the stand-in for Voyage that the tests use. It turns each text into numbers on your computer. Then it trains a model with `yardstick-predict create --express`.
3. VHS follows `demo.tape` and `cli.tape`. It types the keys and films the forecasts.
4. `record_claude_code.py` copies the made-up shop in `storefront/` to `~/storefront` and adds yardstick's hook and status line to the demo's settings folder. VHS follows `claude-code.tape`: it starts Claude Code, types the task and films until Claude stops. Then ffmpeg speeds up the middle and puts a label on it. Last, the script removes `~/storefront` and takes the hook and status line out again.

Nothing reads your own records, your keys or your password store. Nothing is sent to Voyage. The update check is off. It doesn't download Voyage's tokenizer, which counts tokens, either. yardstick's hook and status line run with the made-up home folder. So they never look into yours.

On a Claude subscription, the status line shows shares of your weekly limit, worked out from your own usage. The Claude Code clip shows dollars instead, as with an API key. `status_in_dollars.py` leaves the weekly reading out.

To change what a clip shows, edit its tape. [VHS's own guide](https://github.com/charmbracelet/vhs#vhs-command-reference) lists its commands.
