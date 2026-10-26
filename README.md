# yardstick-predict

**Know how long a coding-agent task will take, and what it will cost, before it runs.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/yusufcemaydin/yardstick-predict/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://github.com/yusufcemaydin/yardstick-predict/blob/main/pyproject.toml)

yardstick-predict learns from your own past Claude Code and Codex runs. Give it a prompt and an AI model, and it forecasts the expected seconds, tokens and dollars. It's built for orchestrators: programs that run many coding agents and decide which model runs each task, what runs side by side, and what fits a budget or a usage limit.

```console
$ yardstick-predict predict "Add a login page with tests"

  Add a login page with tests

  AI model          effort             time         tokens             cost
  claude-opus-5-5   max               4 min           1.8M            $0.92
                                 2 to 8 min   703k to 3.6M   $0.37 to $1.93
                    high              3 min           1.2M            $0.61   ⚠ only 12 runs
                              72 s to 6 min   483k to 2.5M   $0.25 to $1.27
  gpt-6.1-sol       xhigh             5 min           950k            $0.41   ⚠ failed the test
                             85 s to 14 min   269k to 2.7M   $0.12 to $1.17
  Too few of your tasks to forecast: claude-haiku-5-5 (6).

  Under each forecast is the range it lands in 9 times out of 10. Costs are at API prices.
  ⚠ marks a rough guide. Plan with a wider margin there.
  Learned from your own runs · predictor default, updated 2 days ago · add --json for details
```

The numbers above are an example. Yours come from your own history.

## Features

- **Expected time, tokens and cost** for any task on any AI model and effort level you've used, with the full probability curve on request.
- **Learns from records you already have.** Claude Code and Codex keep a record of every run. There's nothing to label.
- **Tests itself before it forecasts.** It replays your newest tasks and gives programs forecasts only for AI models that pass. In a terminal, the others show too, marked ⚠ as a rough guide.
- **One fixed, versioned JSON format,** with published schemas, so any program in any language can call it.
- **Runs on your computer.** Only the text you choose is sent: to Voyage to embed it and, if you turn on LLM analysis, to the AI model you pick.
- **Shows the exact cost before spending anything,** and counts Voyage's free tokens.
- **You choose the features,** and training uses exactly those. The task, its similar past tasks and the project's recent tasks are always used. You can add the conversation so far, the end of the conversation, skills, public model facts, when the task started, an AI model's read of each task, or your own. A feature check tells you which of them really help your forecasts.
- **Advanced settings** for experts: output heads, curve shape, loss, dropout, data split, and other model types.

## Install

```bash
uv tool install yardstick-predict
```

This installs the newest release from PyPI, the Python package index. Between releases, the main branch on GitHub may hold work in progress.

You need [uv](https://docs.astral.sh/uv/), a fast tool that installs Python programs. yardstick-predict runs on macOS, Linux and Windows, with Python 3.12 or newer. uv fetches that Python if you don't have it, and everything else comes with the install.

## Quick start

You need a [Voyage API key](https://dashboard.voyageai.com). Voyage turns each prompt into an embedding: a list of numbers that captures what the text means.

**The express way, in one line.** It takes every default and agrees to the cost:

```bash
export VOYAGE_API_KEY=your-key
yardstick-predict create --express
```

**The guided way.** Run it with nothing after it:

```bash
yardstick-predict
```

The home screen opens. It greets you, lists your models and offers Estimate, Update, Models, Add New and Quit. When a newer version of yardstick-predict is out, it says so under the box, with the command that updates it. Move with the arrow keys and press Enter. A choice's number picks it at once. Esc goes back. Estimate asks for a task. Then it asks which AI models: the ones that passed the test or any you tick from a list, 6 to a page. After the forecasts you choose what's next. Choose Add New and answer a few questions. Setup finds your past runs, asks which features to use, shows the exact cost of each Voyage 4 model, checks your key and keeps it in your computer's password store. Before it goes ahead, it says about how long the rest will take and how much disk space your tasks will need. Then it trains and shows how well it forecast your newest tasks. Each long step shows a progress bar with the time left.

When you pick features beyond the three that are always on, setup offers to check which of them actually help. Some features carry no signal for your kind of work, and with few tasks a feature can't show anything. The check tests the model with each feature and without it, shows what helped for each AI model, and turns on only the features that helped. You confirm or change them before it trains. LLM analysis first reads only a sample of your newest tasks. If you keep it after the check, setup shows what reading the rest costs and asks. The check is advice: say no, and setup trains with the features you picked.

**Then, before you start a task, ask what it will take:**

```bash
yardstick-predict predict "Add a login page with tests"
```

It shows the expected time, tokens and cost on every AI model you've used, at each effort level you often use it at. Under each forecast is the range it lands in 9 times out of 10. A forecast that failed the test or rests on few runs is marked ⚠ as a rough guide. `--model` and `--effort` name them yourself, `--predictor` picks another predictor, and `--json` prints the answer programs get. Use it to pick a model, set a time limit, or see what a batch of tasks will cost before you start.

Every week or so, run `yardstick-predict` and choose Update. Before it reads anything, it says where your new work is and how much there is to read. It reads with a labelled bar that Esc stops. Then it shows what each model will analyze, embed and train, with every cost and time. It asks once. Cancel spends nothing and saves nothing. `yardstick-predict refresh` in a terminal shows the same summary. With `--yes` it asks nothing.

Models lists your models with `(default)` after the default one. Choose one to update it, make it the default or rename it.

## Features the model learns from

Setup lists each feature with its options under it. Type a number to turn a feature on or off, or a number and a letter for one option. The first three are always on: they cost nothing and send nothing anywhere.

| Feature | What it is | Default |
|---|---|---|
| Task | What the prompt means, how it's laid out (list items, file names, links, lines, whether it asks a question), the AI model, the effort level as a scale from low to max, the agent and who started it. | Always on |
| Similar past tasks | What the past tasks closest in meaning really took: the closest one, the closest ten and the closest hundred, how much they varied, quick replies and real work apart, and closer, newer tasks counting more. Plus a first guess from the prompt's whole embedding. | Always on |
| Project history | The project's last finished tasks, from any conversation: what they took, and how long ago the last one finished. | Always on |
| `conversation`, Conversation | The conversation so far, such as how long its last task took, and how big it has grown. With `context`, also how close the prompt is to the end of the conversation. It costs nothing and sends nothing anywhere. | On |
| `context`, Context | The last 1,500 characters of the conversation before the prompt, embedded. | On |
| `skills`, Skills | The skills known before the task runs, embedded by their descriptions. Options: `named`, the skills the prompt invokes, such as a typed `/review`, and `earlier`, the skills used earlier in the conversation. | Off |
| `benchmarks`, Model facts | Public facts about each AI model from [OpenRouter's model list](https://openrouter.ai/models): prices, context size, longest answer and release date. They are the same for old and new models. | Off |
| `time`, Time | When the task started. Options: `trend`, the days since 1 January 2020, and `clock`, the hour of the day and the day of the week, in your computer's time zone. A forecast's task starts the moment you ask. It costs nothing and sends nothing anywhere. | Off |
| `llm`, LLM analysis | An AI model reads each task before it starts and describes the work. Options: `kind`, the kind of task and its expected size, and `description`, a few sentences on the work, embedded. | Off |
| your own | Any text or number you compute from a task. See [Extend it](#extend-it). | — |

Training uses exactly the features you choose. The conversation tells a lot about a person's sessions and can mislead about a program's one-off runs, so a predictor that mostly forecasts programs' runs may forecast better without `conversation`. With LLM analysis, training also makes the same model without it, with a test of its own, for the tasks that LLM analysis can't read.

To find out which features help your own forecasts, run the feature check: setup offers it, and `yardstick-predict check` runs it on a predictor you already have. The Features screen shows each feature's latest verdicts. [How it works](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/how-it-works.md#5-the-feature-check) explains how to read them.

Choose features at setup, or with flags. A feature named alone gets all its options:

```bash
yardstick-predict create --yes --features conversation,context,skills:named,llm --llm-provider claude
```

A predictor keeps its embedding model for good. To change its features, create a new predictor.

Embeddings and LLM answers cost money or time, so each is made once and kept in a feature store that all your predictors share. A second predictor over the same records pays for nothing it already has. `yardstick-predict features` lists every feature of every predictor: what's stored, and its coverage, the share of the predictor's tasks that have a value for it. `yardstick-predict status` shows each predictor's coverage too.

`yardstick-predict features delete ID` deletes a stored feature, by the id the listing shows. First it says which predictors use it and what making it again would cost: an embedding usually comes back from the cache of vectors for free, and LLM answers have to be asked for again. It asks before it deletes anything.

### LLM analysis

It sends each task's prompt and the end of the conversation before it to the AI model you pick, and never what the task went on to do. Pay for it one of four ways:

| `--llm-provider` | How it's paid | Models |
|---|---|---|
| `anthropic` | An Anthropic API key, per use. Setup and refresh show the cost first. | Claude Opus 5.5 (the default), Sonnet 5.5, Haiku 5.5 |
| `openai` | An OpenAI API key, per use. Setup and refresh show the cost first. | GPT 6.1 Sol (the default), GPT 6 Astra, GPT 6 Luna |
| `claude` | Your Claude subscription, through the Claude Code command. No extra bill, but it counts toward your plan's usage limits. | Claude Opus 5.5 (the default), Sonnet 5.5, Haiku 5.5 |
| `codex` | Your ChatGPT subscription, through the Codex command. No extra bill, but it counts toward your plan's usage limits. | Codex's own model |

With Claude Opus 5.5, it costs about $8 per 1,000 tasks at list price; with Claude Haiku 5.5, about $0.20. Each forecast reads its own task once more. A subscription runs Claude Code or Codex with no tools, none of your settings or hooks, and no saved session, so these calls never show up in the records your predictor learns from. When Claude Code or Codex isn't signed in, setup and refresh say how to sign in, stop asking, and train without LLM analysis for now. The next refresh asks again. An API key needs the providers' own libraries:

```bash
uv tool install 'yardstick-predict[llm]@latest'
```

## Forecasts in Claude Code

`yardstick-predict claude-code install` puts the forecast in Claude Code's status line. When you send a prompt, it shows the forecast for the model and effort level you're on, counts up while the agent works, then shows how long the task really took:

```text
yardstick · 1:12 of ~4 min · ~$1.23
```

"Took" is the agent's working time, as training counts it: waits for your answers and approvals are left out. When another hook keeps the agent working after it stops, the clock runs on. A background job reporting back continues its task. Scheduled tasks and messages from other sessions are forecast as a program's prompts, as training counts them. Scripted runs, such as `claude -p`, get no forecast, since they show no status line.

While it's on, every new prompt you send in Claude Code goes to Voyage to be embedded, a few tokens each. Add `--here` to turn it on only for sessions in the current folder. `yardstick-predict claude-code remove` turns it off and puts back the status line it replaced. Both keep a copy of Claude Code's settings file next to it, as it was before.

If your Voyage key is only in the `VOYAGE_API_KEY` environment variable, Claude Code may not see it. Install says so, and offers to keep the key in your computer's password store.

If you installed it with version 0.2.1 or earlier, run `yardstick-predict claude-code install` again. The older commands let Python load code from the folder you work in, and the status line asks you to update them until you do.

## Use it from a program

`predict` reads a JSON request on standard input and writes a JSON answer. It never asks questions.

```bash
echo '{"version": 1, "tasks": [
  {"id": "a", "prompt": "Fix the flaky test", "models": ["claude-opus-5-5", "gpt-6.1-sol"]},
  {"id": "b", "prompt": "Write the release notes", "models": ["claude-opus-5-5"], "context": "We ship on Fridays."}
]}' | yardstick-predict predict
```

From Python, an orchestrator can pick the cheapest model for a task like this:

```python
import json
import subprocess

request = {
    "version": 1,
    "tasks": [{"id": "login", "prompt": "Add a login page with tests", "models": ["claude-opus-5-5", "gpt-6.1-sol"]}],
}
done = subprocess.run(["yardstick-predict", "predict"], input=json.dumps(request), capture_output=True, text=True)
answer = json.loads(done.stdout)
if "error" in answer:
    raise SystemExit(answer["error"]["message"])

priced = [f for f in answer["forecasts"] if f["usd"] is not None]
cheapest = min(priced, key=lambda f: f["usd"])
print(f"Run it on {cheapest['model']}: about {cheapest['seconds'] / 60:.0f} minutes and ${cheapest['usd']:.2f}.")
```

- **`effort`** is the effort level the forecast is for: the request's, or the level the model ran at most often.
- **`seconds`, `tokens` and `usd`** are expected values: averages, not middles. They add up, so the expected total of ten tasks is the sum of their ten forecasts.
- **`tokens`** counts every kind: new input, cache writes, cache reads and output, including helper agents.
- **`usd`** is the cost at the model's list price.
- **`reason`** appears only when a forecast isn't given. Then the numbers are `null` and it says why: `unknown_model`, `not_enough_data` or `not_accurate_enough`.
- **`"distribution": true`** adds each forecast's full curve, so a program can work out any chance or any range.

One request can hold up to 1,000 tasks, each on up to 50 models. The format is fixed and versioned: within version 1, fields may be added but are never renamed or removed. Schemas for the [request](https://github.com/yusufcemaydin/yardstick-predict/blob/main/schemas/forecast-request.v1.json) and the [answer](https://github.com/yusufcemaydin/yardstick-predict/blob/main/schemas/forecast-response.v1.json) let a program check every answer. [The format](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/format.md) explains every field.

## How it tests itself

A forecast you can't trust is worse than none, because a program acts on it. So before it gives any forecast, a predictor replays each AI model's newest 30% of tasks in three windows. Each window is forecast by a model trained only on the tasks that finished before it, on every AI model. So a model you used less lately is still tested on its own newest tasks. Then each AI model must pass three checks:

1. **Beat the simple rule.** The simple rule gives every task on a model the same forecast: the spread of that model's past tasks. The forecasts must score at least as well on CRPS, the standard score for probability forecasts, within 2%.
2. **Keep its ranges honest.** At least 75% of real results must fall inside the 90% range.
3. **Add up.** The expected values must add up to between half and twice the real total.

A model that fails a check gets `null` and a reason. So does an effort level the model never ran at. `yardstick-predict status` shows the scores and which check failed. Each AI model needs 60 tasks of its own: at least 30 to test on and 30 before them. Much of how long an agent takes is chance, so the predictor's job is to be honest about that spread, and to narrow it where your data allows.

## Advanced settings

These are for experts and have sensible defaults. The guided setup never asks about them.

| Flag | What it sets | Default |
|---|---|---|
| `--model-type` | The kind of model: `xgboost`, or one a plugin adds. | `xgboost` |
| `--heads` | What to forecast: any of `seconds`, `tokens`, `usd`. | all three |
| `--head` | `distribution` for a full curve, or `expected` for the expected value alone. | `distribution` |
| `--shape` | The curve's shape: `lognormal` or `weibull`. Only curves have one. | `lognormal` |
| `--loss` | What training minimizes: `log_score` or `crps` for curves, `gamma` or `tweedie` for expected values. | `log_score` |
| `--dropout` | The share of trees dropped each round, from 0 to 0.9. | `0` |
| `--test-share` | The share of each AI model's newest tasks held back as the test, and at least 30. | `0.3` |
| `--test-windows` | How many windows each AI model's test is split into. | `3` |
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
yardstick-predict create --yes --plugin my_features --features conversation,context,words,project
```

`--plugin` takes a module in the current folder, as here, or the path of a `.py` file. The predictor remembers where the file is, so the plugin also loads when a forecast runs from another folder, as the status line's do.

A text feature is embedded like the prompt. A number feature goes to the model as it is. Your function gets the task's prompt, context, skills, project, model, thinking level and agent, from past records when training and from the request when forecasting. It also gets when the task started, as `info.started_at`, in seconds since 1970: a forecast's task starts the moment you ask. A request can also carry an `"extra"` object for your features, which reaches them as `info.extra`. Past tasks have no extra, so a feature should also work its value out from the task itself. Training says so when one of your features had no value for any past task.

A plugin can also add a model type with `yardstick_predict.model_types.register_model_type(name, train, load)`, then use it with `--model-type`. [`model_types.py`](https://github.com/yusufcemaydin/yardstick-predict/blob/main/src/yardstick_predict/model_types.py) describes what a model type provides.

## Commands

| Command | What it does |
|---|---|
| `yardstick-predict` | In a terminal, the home screen: Estimate, Update, Models and Add New. A program gets one line per predictor. |
| `yardstick-predict create [NAME]` | Creates a predictor. In a terminal it asks for a name first. `--express`, or `--yes`, takes every default, asks nothing and agrees to the cost. `--archive codex=PATH` adds a folder of records. |
| `yardstick-predict refresh` | Adds new runs, embeds them and retrains. |
| `yardstick-predict predict "TASK"` | Forecasts a task: a table in a terminal, JSON for programs. |
| `yardstick-predict status` | Shows each predictor, its features with their coverage, the size of its prepared dataset and how it did on its test. `--json` for programs. |
| `yardstick-predict list` | Lists your predictors, one line each: features, how many tasks each learned from, when it was last trained and which AI models get forecasts. `--json` for programs. |
| `yardstick-predict delete NAME` | Deletes a predictor: its settings, its dataset of tasks, its prepared dataset and its model. Your records and the feature store stay. In a terminal it says what goes and asks first. `--yes` for programs, with `--json` for an answer in JSON. |
| `yardstick-predict check [NAME]` | Checks which of a predictor's features help its forecasts: it tests the model with each feature and without it. It's advice only and changes nothing. `--json` for programs. |
| `yardstick-predict features` | Lists every feature of every predictor: whether the feature store keeps it, with what made it, how big it is and when it changed, or training works it out for free. Shows the share of each predictor's tasks it covers. `--json` for programs. |
| `yardstick-predict features delete ID` | Deletes a stored feature. It says which predictors use it and what making it again costs, and asks first. `--yes` for programs. |
| `yardstick-predict default [NAME]` | Shows or sets your default predictor: the one commands use when they name none. `--predictor NAME` picks another for one command. |
| `yardstick-predict rename OLD NEW` | Gives a predictor another name. The default stays the default. |
| `yardstick-predict claude-code install` | Shows forecasts in Claude Code's status line. `--here` for one folder, `remove` to turn it off. |

## Update or remove it

To get the newest version:

```bash
uv tool upgrade yardstick-predict
```

It keeps the `llm` extra if you added it. If you installed 0.3.0 or an earlier version from GitHub, switch to PyPI once with `uv tool install 'yardstick-predict@latest'`, or `uv tool install 'yardstick-predict[llm]@latest'` if you use LLM analysis with an API key. After that, the upgrade command above gets each new release.

Upgrading keeps your predictors. When a new version reads records more accurately, the next refresh reads them all again, once, and embeds only the texts that changed. When a new version reads new features or trains the model another way, the next refresh trains it again. Until then, a model the previous version trained keeps forecasting as it did. A model from an older version gives no forecasts until it's trained again.

From 0.4 on, going back to an older version is safe too. It reads what it can of a predictor that a newer version saved, and never changes it: it skips any feature it doesn't know, with a note, and it won't refresh that predictor.

To remove one predictor, run `yardstick-predict delete NAME`. Your records, the feature store and your other predictors stay.

To remove everything, uninstall the program, delete its folder and delete the Voyage key it kept:

```bash
uv tool uninstall yardstick-predict
rm -rf ~/.yardstick
uvx keyring del yardstick-predict voyage-api-key
```

If you set `YARDSTICK_HOME`, delete that folder instead. On Windows, delete the `.yardstick` folder in your user folder. If you only ever used `VOYAGE_API_KEY`, the last line reports that it found no key. That's fine.

## Privacy and safety

- **Your records stay on your computer.** The dataset, the feature store and the model live in `~/.yardstick`, or wherever `YARDSTICK_HOME` points. The folder is readable only by you, on Windows too.
- **The cache of embeddings holds no text.** For each text it keeps a one-way hash and the text's vector, in the same private folder.
- **Only the text you choose is sent,** over an encrypted connection: to Voyage to embed it, and with LLM analysis, each task's prompt and the end of its conversation to the AI model you pick.
- **Your Voyage key** is read from `VOYAGE_API_KEY` or kept in your computer's password store. yardstick-predict never writes it to a file of its own.
- **Other downloads** are public data, made only when needed: Voyage's tokenizer from Hugging Face, at one fixed version, and OpenRouter's model list if you use benchmarks. The home screen also reads yardstick-predict's public page on PyPI at most once a day, to say when a newer version is out. Set `YARDSTICK_NO_UPDATE_CHECK` to turn that off.
- **Saved models are plain JSON and plain numbers.** Loading one runs no code.
- **Plugins are code you choose to run.** yardstick-predict only imports the modules you name.

## Cost

Voyage gives every account 200 million free tokens. yardstick-predict can't see your balance, so it keeps its own count and asks once whether your account is new.

Before it embeds anything, it shows the exact token count, counted with Voyage's own tokenizer, and the price. If a big history won't fit in your free tokens, it offers to embed only your newest tasks. Each refresh shows the cost of just the new tasks.

A text is paid for once. The same text in many tasks, such as a prompt that says only "continue", is sent once. A text embedded before with the same Voyage model, for any predictor or for a forecast, comes from a cache on your computer and costs nothing.

An account without a payment method gets low limits, such as a few requests and 10,000 tokens a minute. When Voyage says a limit is reached, yardstick-predict waits for it to come back, so a big first setup on such an account can take a while.

## Dependencies

| Package | Why |
|---|---|
| [XGBoost](https://xgboost.ai) | The gradient-boosted trees. Linux and Windows get `xgboost-cpu`, the same library without the large GPU part. |
| [NumPy](https://numpy.org) | The numbers. |
| [tokenizers](https://github.com/huggingface/tokenizers) | Counts tokens exactly with Voyage's own tokenizer, so you see the cost before paying it. |
| [keyring](https://github.com/jaraco/keyring) | Keeps your Voyage key in your computer's password store. |
| [PyArrow](https://arrow.apache.org/docs/python/) | Reads and writes the feature store's files, in Parquet, a compact file format for tables. |
| [scikit-learn](https://scikit-learn.org), on macOS only | Brings the OpenMP library XGBoost needs on a Mac, so there's nothing else to install. |
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) and [openai](https://github.com/openai/openai-python), only with the `llm` extra | The providers' own libraries, for LLM analysis paid with an API key. Not installed unless you ask for them. |

## Documentation

- [How it works](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/how-it-works.md): the data, the features, the model and the test.
- [The format](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/format.md): every field of the request and the answer.
- [What's next](https://github.com/yusufcemaydin/yardstick-predict/blob/main/docs/features.md): what's planned.

## Contributing

Bug reports, ideas and pull requests are welcome. See the [contributing guide](https://github.com/yusufcemaydin/yardstick-predict/blob/main/CONTRIBUTING.md) and the [security policy](https://github.com/yusufcemaydin/yardstick-predict/blob/main/SECURITY.md).

## License

[Apache License 2.0](https://github.com/yusufcemaydin/yardstick-predict/blob/main/LICENSE).
