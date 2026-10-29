# Open Runs

Open Runs is a set of public coding-agent runs that comes with yardstick-predict. Each run is a coding task someone published: its prompt, the AI model that did it, the agent, how long it took, the tokens it used and what it cost.

It does three things:

- **A first model on day one.** A new model can learn from Open Runs alone. You get forecasts before you have any runs of your own.
- **Help where you have little data.** A model made from your own runs also learns from Open Runs, unless you leave it out. It helps most for AI models you've used little.
- **An honest accuracy number.** Your own runs are private. Open Runs is public. Anyone can check how well a model made from it forecasts.

## What's in it

14,433 runs of 119 different prompts, by 42 AI models and 17 agents. They ran between October 2025 and September 2026. 89 prompts are Terminal-Bench tasks, such as building a program or fixing a server. 30 are SWE-bench Lite tasks: fixes to real issues in open-source Python projects. They come from three public collections:

| Collection | What it is | License |
|---|---|---|
| [Terminal-Bench 2.0 runs](https://huggingface.co/datasets/yoonholee/terminalbench-trajectories), gathered by Yoonho Lee | Hard terminal tasks, run by many agents and AI models | Apache 2.0 |
| [Terminal-Bench 2.0 leaderboard](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard) | The same tasks, run by the agents on the public leaderboard | Apache 2.0 |
| [SWE-bench Lite runs](https://huggingface.co/datasets/melissapan/swe-bench-lite-agent-traces-v14), gathered by Melissa Pan | Fixes to real GitHub issues, run by Claude Code and Pi | CC BY 4.0 |

The Terminal-Bench prompts are the tasks' own instructions, from [Terminal-Bench 2.0](https://huggingface.co/datasets/harborframework/terminal-bench-2.0) and [2.1](https://huggingface.co/datasets/harborframework/terminal-bench-2.1) (Apache 2.0).

Each run keeps only its prompt, AI model, agent, start time, seconds, tokens and cost. Runs whose time or tokens weren't measured are left out. So are Codex's runs in the SWE-bench Lite collection: their records don't say how long they took. At most two runs of one prompt by one AI model and agent are kept. More of the same would add little and make training slower. Email addresses, home folder names and anything shaped like a key are cleaned out of the prompts. [`SOURCES.md`](../src/yardstick_predict/open_runs/SOURCES.md) lists each collection with its license.

Open Runs also comes with the Voyage embeddings of every prompt, for each Voyage 4 model. An embedding is a list of numbers that captures what a text means. So learning from Open Runs costs no Voyage tokens. It comes with LLM analysis answers too, from Codex's default AI model. LLM analysis is the optional feature where an AI model reads each prompt and reports the kind of task and its size. A model that uses LLM analysis through a ChatGPT subscription pays nothing for Open Runs. With another AI model, LLM analysis reads each different Open Runs prompt once, with the cost shown first.

## How a model uses it

- **Start from Open Runs.** On a computer with no Claude Code or Codex records, Add New offers to start from Open Runs. `yardstick-predict create --archive open-runs` does the same from a program.
- **Also learn from Open Runs.** With records of your own, Add New asks whether to learn from Open Runs too. It's on by default. `--no-open-runs` leaves it out. A program that makes a model from 1,000 or more runs of its own leaves Open Runs out, unless it names it with `--archive open-runs`. With that many runs, Open Runs adds little and makes training slower.

A model that also learns from your own runs keeps Open Runs in its place:

1. **Your own tasks decide the test.** The test replays only your own newest tasks. Open Runs only adds to what each test window learns from. So the test says how well the model forecasts your work, not public tasks.
2. **Your own tasks decide when to stop adding trees.** Training adds trees only while they help on your own newest tasks.
3. **Newer tasks count more.** A task 30 days older than your newest counts half as much. Open Runs is months old. It counts far less than your own runs.

A model made from Open Runs alone counts every run the same, however old. The public runs were gathered months apart. A newer one says no more about your tasks. Its test replays each AI model's newest runs in one window, not three. Setup then takes minutes, not a quarter of an hour.

**What it can't do yet.** Open Runs holds the AI models people published runs of. Your newest AI models may not be in it. A model learns nothing from Open Runs about an AI model that isn't there. Forecasting AI models it has never seen is on the [list of what's next](features.md).

## How good it is

Measured on 29 October 2026, on Open Runs as it ships. Every number here comes from public data.

The test is strict. The 119 prompts are split into five groups. For each group, a model learns from the runs of the other four and forecasts this group's runs. So every forecast is of a prompt the model never saw, as a new task of yours would be. The forecasts are scored as the tool's own test scores them: against the simple rule, which gives every task on an AI model the same forecast, from that AI model's past spread. A score 10% better than the simple rule means the forecasts sat 10% closer to what really happened, by the standard score for forecasts with ranges.

| | Time | Tokens | Cost |
|---|---|---|---|
| Better than the simple rule, with the prompt alone | 0% | 5% | 1% |
| Better than the simple rule, with LLM analysis | 7% | 6% | 11% |
| Real results inside the 90% range, with the prompt alone | 93% | 93% | 67% |
| Real results inside the 90% range, with LLM analysis | 95% | 96% | 73% |

With LLM analysis, 6 of the 42 AI models pass every check of the tool's test. With the prompt alone, 1 does. The time and token ranges are honest. The cost ranges are too narrow. Treat cost forecasts from Open Runs as rough.

**What this means.** From 119 prompts, the model barely learns which new prompts take longer. Yet the prompt matters a lot. Within each AI model, the prompt explains 63% of the spread in time and 50% of the spread in tokens. The rest is chance: the same prompt on the same AI model takes longer one run than the next. So a forecast that understood prompts well could do far better than these numbers. That takes many more different prompts than Open Runs has today. LLM analysis already helps, because an AI model's reading of a prompt carries over to prompts the model never saw.

Your own runs are a different case. Your tasks repeat kinds of work you do often. A model made from them learns from close matches. The test of your own model, on your own newest tasks, says how well it forecasts your work.

## Rebuild it

Three scripts in [`scripts/`](../scripts) make Open Runs from the public collections. Anyone can run them:

1. `uv run --with duckdb python scripts/build_open_runs.py --cache FOLDER` downloads the runs and packs them.
2. `uv run python scripts/open_runs_extras.py` makes the embeddings and LLM answers. It needs a Voyage key and a signed-in Codex.
3. `uv run python scripts/open_runs_accuracy.py` measures the numbers above.
