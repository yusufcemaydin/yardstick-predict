# yardstick-predict

**Know how long a coding-agent task will take and what it will cost, before it runs.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/yusufcemaydin/yardstick-predict/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://github.com/yusufcemaydin/yardstick-predict/blob/main/pyproject.toml)

![A real Claude Code session on Opus 5.5 at max effort. The task "Add a login page with tests" is sent. Under the input box, yardstick's status line forecasts about 7 minutes and $1.26, then counts up while Claude works. The middle is sped up. When Claude stops, the line says how long it took, next to the forecast.](https://raw.githubusercontent.com/yusufcemaydin/yardstick-predict/main/demo/claude-code.gif)

A real Claude Code session, sped up in the middle. The line under the input box is yardstick's forecast, counting up while Claude works. The forecast comes from made-up past runs. So its numbers are an example.

yardstick-predict learns from your own past Claude Code and Codex runs. Give it a task and it forecasts the time, tokens and dollars on each AI model you use. It's built for orchestrators: programs that run many coding agents and pick an AI model, a time limit or a budget for each task.

```console
$ yardstick-predict predict "Add a login page with tests"

  Add a login page with tests

  AI model            effort             time  tokens   cost
  claude-opus-5-5     max     expected  4 min    1.8M  $0.92
                              25%       3 min    1.1M  $0.60
                              median    4 min    1.6M  $0.84
                              75%       5 min    2.2M  $1.19
  gpt-6.1-sol      ⚠  xhigh   expected  5 min    950k  $0.41
                              25%       3 min    531k  $0.23
                              median    5 min    862k  $0.37
                              75%       7 min    1.4M  $0.60

  ⚠ Not enough data: gpt-6.1-sol.
  Those are a rough guide. Plan with a wider margin there.
  The expected line is what to expect on average. A quarter of real results
  come in under the 25% line, half under the median and three quarters under
  the 75% line. Costs are at API prices.
  Learned from your own runs · model main, updated 2 days ago · add --json for
  details
```

The numbers above are an example. Yours come from your own history.

- **Nothing to label.** Claude Code and Codex already keep a record of every run. yardstick-predict learns from those.
- **It checks itself first.** It replays your newest tasks as if they hadn't happened yet. Programs get forecasts only for the AI models it forecast well.
- **Your prompts go to Voyage.** So does the end of the conversation before each one, as the Context feature is on by default. Voyage turns them into numbers that capture what they mean. With LLM analysis on, both also go to the AI model you pick. Before anything is sent, text shaped like a key, token or password is replaced with `[hidden]`. Training and forecasts run on your computer.
- **It shows the cost before it spends anything.**
- **Programs get JSON** in a fixed, versioned format.

## Install

```bash
uv tool install yardstick-predict
```

You need [uv](https://docs.astral.sh/uv/), a tool that installs Python programs. It works on macOS, Linux and Windows. It needs Python 3.12 or newer, which uv fetches if you don't have it. To update later, run `uv tool upgrade yardstick-predict`.

## Quick start

1. Get a [Voyage API key](https://dashboard.voyageai.com). Voyage turns each prompt into numbers that capture what it means. Every account gets 200 million free tokens.
2. Run `yardstick-predict` and choose Add New. It finds your past runs, asks for your key when it has none, shows the cost and trains a model, which makes your forecasts. A key you paste is kept in your computer's password store. To take every default and agree to the cost in one line instead:

   ```bash
   export VOYAGE_API_KEY=your-key
   yardstick-predict create --express
   ```

3. Before you start a task, ask what it will take:

   ```bash
   yardstick-predict predict "Add a login page with tests"
   ```

4. Every week or so, run `yardstick-predict` and choose Update. It reads your new runs and trains again.

Run `yardstick-predict` with nothing after it for the home screen. Its menus do all of this. Here it forecasts a task on two AI models:

![yardstick-predict in a terminal. On the home screen, Estimate is chosen. The task "Add a login page with tests" is typed and two AI models are ticked. A table then shows each AI model's expected time, tokens and cost, with the range real results fall in.](https://raw.githubusercontent.com/yusufcemaydin/yardstick-predict/main/demo/demo.gif)

The clip runs the real program on made-up past runs. Its numbers are an example.

## See forecasts in Claude Code

```bash
yardstick-predict claude-code install
```

Claude Code's status line then shows a forecast for each prompt you send. It counts up while the agent works:

```text
yardstick · 1:12 of ~4 min · ~$1.23
```

The clip at the top of this page shows it in a real session.

On a Claude subscription, it shows shares of your weekly limit in place of dollars. First comes what the task has used so far. Then the brackets hold the forecast's time and its share of the week:

```text
yardstick · 1:12 · 0.0236% (~4 min · ~0.0789%)
```

Each prompt goes to Voyage to be embedded, with the end of the conversation before it when Context is on. That costs a tiny fraction of a cent each time. `yardstick-predict claude-code remove` turns it off.

## Use it from a program

`predict` reads a JSON request on standard input and writes a JSON answer. It never asks questions.

```bash
echo '{"version": 1, "tasks": [
  {"id": "a", "prompt": "Fix the flaky test", "models": ["claude-opus-5-5", "gpt-6.1-sol"]}
]}' | yardstick-predict predict
```

Each forecast has `seconds`, `tokens` and `usd`. They are averages. Ten tasks' forecasts add up to their expected total. When a forecast isn't given, its numbers are `null` and `reason` says why. [The format](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/format.md) explains every field. Setup and refresh answer in JSON too, with `--json`.

Flags do the same in one line. `--model` and `--effort` choose the AI model and the thinking level. `--distribution` adds each forecast's whole curve. It gives the chance that the agent answers at once, without tools. For time, tokens and cost it gives a center and a width. From those a program can work out any range, such as a time limit that 9 runs in 10 fit in. [The format](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/format.md#the-distribution) says how.

![yardstick-predict on the command line. First it forecasts "Add a login page with tests" on claude-opus-5-5 at max effort and prints a table. Then the same command with --distribution, piped to jq, prints the forecast as JSON with its curve. The curve has the chance of a direct answer, then a center and a width for time, tokens and cost.](https://raw.githubusercontent.com/yusufcemaydin/yardstick-predict/main/demo/cli.gif)

The clip runs the real program on made-up past runs. Its numbers are an example.

## Commands

| Command | What it does |
|---|---|
| `yardstick-predict` | Opens the home screen: Estimate, Update, Models and Add New. |
| `yardstick-predict create [NAME]` | Sets up a new model. |
| `yardstick-predict refresh` | Reads your new runs and trains again. |
| `yardstick-predict predict "TASK"` | Forecasts a task. |
| `yardstick-predict status` | Shows how close each model came on your newest tasks. |
| `yardstick-predict list` | Lists your models. |
| `yardstick-predict check [NAME]` | Says which features help a model's forecasts. |
| `yardstick-predict features` | Lists the features each model uses and what's stored. |
| `yardstick-predict default [NAME]` | Shows or sets the model commands use when they name none. |
| `yardstick-predict rename OLD NEW` | Renames a model. |
| `yardstick-predict delete NAME` | Deletes a model. Your records stay. |
| `yardstick-predict claude-code install` | Shows forecasts in Claude Code's status line. |
| `yardstick-predict uninstall` | Removes yardstick-predict: takes it out of Claude Code and deletes your models. |

Most commands take `--json` for programs. `yardstick-predict COMMAND --help` lists every flag of a command. [The guide](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/guide.md#commands) says more about each one.

## Remove it

```bash
yardstick-predict uninstall
```

It takes the forecasts out of Claude Code, everywhere you added them. Then it deletes the Voyage key it kept and your models. It says what goes and asks first. Add `--keep-data` to keep your models and the key. Last, it shows the command that removes the program itself, such as `uv tool uninstall yardstick-predict`.

## Learn more

- [The guide](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/guide.md): the home screen, the features a model learns from, LLM analysis, costs, settings for experts, plugins, privacy and how to update or remove it.
- [How it works](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/how-it-works.md): the data, the model and how it tests itself.
- [The format](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/format.md): every field programs send and get back.
- [What's next](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/features.md): what's planned.

## Contributing

Bug reports, ideas and pull requests are welcome. See the [contributing guide](https://github.com/yusufcemaydin/yardstick-predict/blob/main/CONTRIBUTING.md) and the [security policy](https://github.com/yusufcemaydin/yardstick-predict/blob/main/SECURITY.md).

## License

[Apache License 2.0](https://github.com/yusufcemaydin/yardstick-predict/blob/main/LICENSE).
