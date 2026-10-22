# yardstick-predict

**Know how long a coding-agent task will take, and what it will cost, before it runs.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

yardstick-predict learns from your own past Claude Code and Codex runs. Give it a prompt and an AI model, and it forecasts the expected seconds, tokens and dollars. It's built for orchestrators: programs that run many coding agents and decide which model runs each task, what runs side by side, and what fits a budget or a usage limit.

```console
$ yardstick-predict predict --model claude-opus-5-5 --prompt "Add a login page with tests"
{"version": 1, "predictor": "default", "embedding_model": "voyage-4", "forecasts": [
  {"task": "task", "model": "claude-opus-5-5", "seconds": 245.0, "tokens": 1830000, "usd": 0.92, "reason": null}]}
```

The numbers above are an example. Yours come from your own history.

## Features

- **Expected time, tokens and cost** for any task on any AI model you've used, with the full probability curve on request.
- **Learns from records you already have.** Claude Code and Codex keep a record of every run. There's nothing to label.
- **Tests itself before it forecasts.** It replays your newest tasks and gives forecasts only for AI models that pass.
- **One fixed, versioned JSON format,** with published schemas, so any program in any language can call it.
- **Runs on your computer.** Only the text you choose to embed is sent, and only to Voyage.
- **Shows the exact embedding cost before spending anything,** and counts Voyage's free tokens.
- **You choose the features:** the prompt, plus the conversation's context, the skills used, public model benchmarks, or your own.
- **Advanced settings** for experts: output heads, curve shape, loss, dropout, data split, and other model types.

## Install

```bash
uv tool install git+https://github.com/yusufcemaydin/yardstick-predict
```

You need [uv](https://docs.astral.sh/uv/), a fast tool that installs Python programs. yardstick-predict runs on macOS, Linux and Windows, with Python 3.12 or newer. uv fetches that Python if you don't have it, and everything else comes with the install.

## Quick start

You need a [Voyage API key](https://dashboard.voyageai.com). Voyage turns each prompt into an embedding: a list of numbers that captures what the text means.

**The express way, in one line.** It takes every default and agrees to the cost:

```bash
export VOYAGE_API_KEY=your-key
yardstick-predict create --express
```

**The guided way.** Run it and answer a few questions:

```bash
yardstick-predict
```

It finds your past runs, asks which features to use, shows the exact cost of each Voyage 4 model, checks your key and keeps it in your computer's password store. Then it trains and shows how well it forecast your newest tasks.

**Then forecast:**

```bash
yardstick-predict predict --model claude-opus-5-5 --prompt "Add a login page with tests"
```

Run `yardstick-predict` again later. It tells you how many new tasks it found and what embedding them costs, and asks before it refreshes.

## Features the model learns from

The prompt is always used. You can add more, and each one is tested on your data like everything else.

| Feature | What it is | Default |
|---|---|---|
| `prompt` | The task's prompt, embedded. | Always on |
| `context` | The last 1,500 characters of the conversation before the prompt, embedded. | On |
| `skills` | The skills the agent used, embedded by their descriptions. | Off |
| `benchmarks` | Public facts about each AI model from [OpenRouter's model list](https://openrouter.ai/models): prices, context size, longest answer and release date. They are the same for old and new models. | Off |
| your own | Any text or number you compute from a task. See [Extend it](#extend-it). | — |

Every task also brings its AI model, thinking level, agent, who started it and the length of its prompt.

Choose features at setup, or with a flag:

```bash
yardstick-predict create --yes --features context,skills,benchmarks
```

Each embedded feature becomes what the most similar past tasks really took. A predictor keeps its features and its embedding model for good. To change them, create a new predictor.

## Use it from a program

`predict` reads a JSON request on standard input and writes a JSON answer. It never asks questions.

```bash
echo '{"version": 1, "tasks": [
  {"id": "a", "prompt": "Fix the flaky test", "models": ["claude-opus-5-5", "gpt-6.1-sol"]},
  {"id": "b", "prompt": "Write the release notes", "models": ["claude-opus-5-5"], "context": "We ship on Fridays."}
]}' | yardstick-predict predict
```

- **`seconds`, `tokens` and `usd`** are expected values: averages, not middles. They add up, so the expected total of ten tasks is the sum of their ten forecasts.
- **`tokens`** counts every kind: new input, cache writes, cache reads and output, including helper agents.
- **`usd`** is the cost at the model's list price.
- **`reason`** is `null` when a forecast is given. Otherwise the numbers are `null` and it says why: `unknown_model`, `not_enough_data` or `not_accurate_enough`.
- **`"distribution": true`** adds each forecast's full curve, so a program can work out any chance or any range.

One request can hold up to 1,000 tasks, each on up to 50 models. The format is fixed and versioned: within version 1, fields may be added but are never renamed or removed. Schemas for the [request](schemas/forecast-request.v1.json) and the [answer](schemas/forecast-response.v1.json) let a program check every answer. [The format](docs/format.md) explains every field.

## How it tests itself

A forecast you can't trust is worse than none, because a program acts on it. So before it gives any forecast, a predictor replays your newest 30% of tasks in three windows. Each window is forecast by a model trained only on the tasks before it. Then each AI model must pass three checks:

1. **Beat the simple rule.** The simple rule gives every task on a model the same forecast: the spread of that model's past tasks. The forecasts must score at least as well on CRPS, the standard score for probability forecasts.
2. **Keep its ranges honest.** At least 75% of real results must fall inside the 90% range.
3. **Add up.** The expected values must add up to between half and twice the real total.

A model that fails a check gets `null` and a reason. `yardstick-predict status` shows the scores and which check failed. Much of how long an agent takes is chance, so the predictor's job is to be honest about that spread, and to narrow it where your data allows.

## Advanced settings

These are for experts and have sensible defaults. The guided setup never asks about them.

| Flag | What it sets | Default |
|---|---|---|
| `--model-type` | The kind of model: `xgboost`, or one a plugin adds. | `xgboost` |
| `--heads` | What to forecast: any of `seconds`, `tokens`, `usd`. | all three |
| `--head` | `distribution` for a full curve, or `expected` for the expected value alone. | `distribution` |
| `--shape` | The curve's shape: `lognormal` or `weibull`. | `lognormal` |
| `--loss` | What training minimizes: `log_score` or `crps` for curves, `gamma` or `tweedie` for expected values. | `log_score` |
| `--dropout` | The share of trees dropped each round, from 0 to 0.9. | `0` |
| `--test-share` | The share of newest tasks held back as the test. | `0.3` |
| `--test-windows` | How many windows the test is split into. | `3` |
| `--validation-share` | The share of training tasks that decides when to stop adding trees. | `0.15` |
| `--half-life` | Days until a task counts half as much as the newest. `0` counts every task alike. | `30` |

```bash
yardstick-predict create --yes --heads seconds,usd --shape weibull --test-windows 5
```

## Extend it

New features may help forecasts more than anything else. Add your own in a plugin: a Python module that registers them.

```python
# my_features.py
from yardstick_predict.features import register_feature

register_feature("words", "number", lambda info: len(info.prompt.split()))
register_feature("project", "text", lambda info: info.project)
```

```bash
yardstick-predict create --yes --plugin my_features --features context,words,project
```

A text feature is embedded like the prompt. A number feature goes to the model as it is. Your function gets the task's prompt, context, skills, project, model, thinking level and agent, from past records when training and from the request when forecasting. A request can also carry an `"extra"` object for your features, which reaches them as `info.extra`.

A plugin can also add a model type with `yardstick_predict.model_types.register_model_type(name, train, load)`, then use it with `--model-type`. [`model_types.py`](src/yardstick_predict/model_types.py) describes what a model type provides.

## Commands

| Command | What it does |
|---|---|
| `yardstick-predict` | Sets up your first predictor, or finds new runs and offers to refresh. |
| `yardstick-predict create [NAME]` | Creates a predictor. `--express` takes every default. `--archive codex=PATH` adds a folder of records. |
| `yardstick-predict refresh` | Adds new runs, embeds them and retrains. |
| `yardstick-predict predict` | Forecasts tasks. JSON in, JSON out. |
| `yardstick-predict status` | Shows each predictor, its features and how it did on its test. `--json` for programs. |

## Privacy and safety

- **Your records stay on your computer.** The dataset and the model live in `~/.yardstick`, or wherever `YARDSTICK_HOME` points. The folder is readable only by you.
- **Only the text you choose to embed is sent, and only to Voyage,** over an encrypted connection.
- **Your Voyage key** is read from `VOYAGE_API_KEY` or kept in your computer's password store. yardstick-predict never writes it to a file of its own.
- **Other downloads** are public data, made only when needed: Voyage's tokenizer from Hugging Face, at one fixed version, and OpenRouter's model list if you use benchmarks.
- **Saved models are plain JSON and plain numbers.** Loading one runs no code.
- **Plugins are code you choose to run.** yardstick-predict only imports the modules you name.

## Cost

Voyage gives every account 200 million free tokens. yardstick-predict can't see your balance, so it keeps its own count and asks once whether your account is new.

Before it embeds anything, it shows the exact token count, counted with Voyage's own tokenizer, and the price. If a big history won't fit in your free tokens, it offers to embed only your newest tasks. Each refresh shows the cost of just the new tasks.

## Dependencies

| Package | Why |
|---|---|
| [XGBoost](https://xgboost.ai) | The gradient-boosted trees. Linux and Windows get `xgboost-cpu`, the same library without the large GPU part. |
| [NumPy](https://numpy.org) | The numbers. |
| [tokenizers](https://github.com/huggingface/tokenizers) | Counts tokens exactly with Voyage's own tokenizer, so you see the cost before paying it. |
| [keyring](https://github.com/jaraco/keyring) | Keeps your Voyage key in your computer's password store. |
| [scikit-learn](https://scikit-learn.org), on macOS only | Brings the OpenMP library XGBoost needs on a Mac, so there's nothing else to install. |

## Documentation

- [How it works](docs/how-it-works.md): the data, the features, the model and the test.
- [The format](docs/format.md): every field of the request and the answer.
- [What's next](docs/features.md): what's planned.

## Contributing

Bug reports, ideas and pull requests are welcome. See the [contributing guide](CONTRIBUTING.md) and the [security policy](SECURITY.md).

## License

[Apache License 2.0](LICENSE).
