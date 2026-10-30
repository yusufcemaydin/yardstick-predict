# The guide

This guide covers what the [README](../README.md) leaves out. [How it works](how-it-works.md) explains the model behind it.

A model is what yardstick-predict trains from your runs. It makes your forecasts. The `--predictor` option and the names in the JSON call it a predictor, for the programs that use them. The models a forecast is about, such as claude-opus-5-5, are AI models.

## The home screen

Run `yardstick-predict` with nothing after it. The home screen greets you, lists your models and offers Estimate, Update, Models, Add New and Quit. Move with the arrow keys and press Enter. A choice's number picks it at once. Esc goes back.

When a newer version of yardstick-predict is out, it says so under the box, with the command that updates it the way you installed it: with uv, pipx or pip.

### Estimate

Estimate asks for a task. With more than one model, it asks which of your models should forecast it. Each one says how accurate it is next to your default model, such as 12% more accurate than main. Then it asks which AI models: the ones with enough data or any from a list, 6 to a page, numbered on from page to page. Enter forecasts the line you're on. To forecast several, tick them with space first. After the forecasts you choose what's next.

### Add New

Choose Add New and answer a few questions. Setup finds your past runs, asks which features to use, shows the exact cost of each Voyage 4 model, checks your key and keeps it in your computer's password store. Before it goes ahead, it says about how long the rest will take and how much disk space your tasks will need. Then it trains and shows how well it forecast your newest tasks. Each long step shows a progress bar with the time left.

When you pick features beyond the three that are always on, setup offers to check which of them actually help. Some features carry no signal for your kind of work. With few tasks, a feature can't show anything. The check compares forecasts with each feature and without it. It shows what helped for each AI model and keeps on only the features that helped. Then it asks once whether to go on with them. Saying no opens the Features screen, to change them before it trains. LLM analysis first reads only a sample of your newest tasks. If you keep it after the check, setup shows what reading the rest costs and asks. The check is advice. When you say no, setup trains with the features you picked.

### Update

Every week or so, choose Update. Before it reads anything, it says where your new work is and how much there is to read. It reads with a labelled bar that Esc stops. Then it shows what each model will analyze, embed and train, with every cost and time. It asks once. Cancel spends nothing and saves nothing. `yardstick-predict refresh` in a terminal shows the same summary. With `--yes` it asks nothing.

### Models

Models lists your models with `(default)` after the default one. Choose one to update it, make it the default, rename it or delete it. Delete says what goes and what stays, then asks. Your records stay.

### With a screen reader

Set `YARDSTICK_NUMBERED_MENUS=1` or add `"numbered_menus": true` to `settings.json` in the yardstick folder. Every menu then lists its choices with numbers. You type the number of the one you want. Nothing on screen redraws itself: there are no progress bars.

## Forecasts in a terminal

```bash
yardstick-predict predict "Add a login page with tests"
```

It shows the expected time, tokens and cost on every AI model you've used, at each effort level you often use it at. A column after the effort level names each line of a forecast. The expected line comes first. It's what to expect on average. Under it are the 25%, median and 75% lines. A quarter of real results come in under the 25% line, half under the median and three quarters under the 75% line. The effort level shows once, on the expected line. A forecast with not enough data behind it is marked ⚠ as a rough guide. The default effort level, which no one picks, isn't shown.

`--model` and `--effort` name the AI models and effort levels yourself. `--predictor` picks another of your models. `--json` prints the answer programs get. Use it to pick an AI model, set a time limit or see what a batch of tasks will cost before you start.

## Features the model learns from

Setup lists each feature with its options under it. Type a number to turn a feature on or off. Type a number and a letter for one option. Type all to turn on every feature with all its options. The first three are always on. For them, each prompt goes to Voyage, which works out what it means.

| Feature | What it is | Default |
|---|---|---|
| Task | What the prompt means, how it's laid out (list items, file names, links, lines, whether it asks a question), the AI model, the effort level as a scale from low to max, the agent and who started it. | Always on |
| Similar past tasks | What the past tasks closest in meaning really took: the closest one, the closest ten and the closest hundred. How much they varied, with quick replies and real work kept apart. Closer and newer tasks count more. Plus a first guess from the prompt's whole embedding. | Always on |
| Project history | The project's last finished tasks, from any conversation: what they took and how long ago the last one finished. | Always on |
| `conversation`, Conversation | The conversation so far, such as how long its last task took and how big it has grown. With `context`, also how close the prompt is to the end of the conversation. It costs nothing and sends nothing anywhere. | On |
| `context`, Context | The last 1,500 characters of the conversation before the prompt, embedded. | On |
| `skills`, Skills | The skills known before the task runs, embedded by their descriptions. Option `named` reads the skills the prompt invokes, such as a typed `/review`. Option `earlier` reads the skills used earlier in the conversation. | Off |
| `benchmarks`, Model facts | Public facts about each AI model from [OpenRouter's model list](https://openrouter.ai/models): prices, context size, longest answer and release date. They are the same for old and new AI models. | Off |
| `time`, Time | When the task started. Option `trend` is the days since 1 January 2020. Option `clock` is the hour of the day and the day of the week, in your computer's time zone. A forecast's task starts the moment you ask. It costs nothing and sends nothing anywhere. | Off |
| `llm`, LLM analysis | An AI model reads each task before it starts and describes the work. Option `kind` is the kind of task and its expected size. Option `description` is a few sentences on the work, embedded. | Off |
| your own | Any text or number you compute from a task. See [Plugins](#plugins). | Off |

Training uses exactly the features you choose. The conversation tells a lot about a person's sessions and can mislead about a program's one-off runs. So a model that mostly forecasts programs' runs may forecast better without `conversation`. With LLM analysis, training also makes a second version of the model without it, with a test of its own, for the tasks that LLM analysis can't read.

To find out which features help your own forecasts, run the feature check. Setup offers it. `yardstick-predict check` runs it on a model you already have. The Features screen shows each feature's latest verdicts. [How it works](how-it-works.md#5-the-feature-check) explains how to read them.

Choose features at setup or with flags. A feature named alone gets all its options:

```bash
yardstick-predict create --yes --features conversation,context,skills:named,llm --llm-provider claude
```

A model always embeds with the Voyage model it was made with. To change its features, create a new model.

Embeddings and LLM answers cost money or time. So each is made once and kept in a feature store that all your models share. A second model over the same records pays for nothing it already has. `yardstick-predict features` lists every feature of every model: what's stored and its coverage. Coverage is the share of the model's tasks that have a value for it. `yardstick-predict status` shows each model's coverage too.

`yardstick-predict features delete ID` deletes a stored feature, by the id the listing shows. First it says which models use it and what making it again would cost. An embedding usually comes back from the cache of vectors for free. LLM answers have to be asked for again. It asks before it deletes anything.

### LLM analysis

It sends each task's prompt and the end of the conversation before it to the AI model you pick. It never sends what the task went on to do. Pay for it one of four ways:

| `--llm-provider` | How it's paid | AI models |
|---|---|---|
| `anthropic` | An Anthropic API key, per use. Setup and refresh show the cost first. | Claude Opus 5.5 (the default), Sonnet 5.5, Haiku 5.5 |
| `openai` | An OpenAI API key, per use. Setup and refresh show the cost first. | GPT 6.1 Sol (the default), GPT 6 Astra, GPT 6 Luna |
| `claude` | Your Claude subscription, through the Claude Code command. No extra bill. It counts toward your plan's usage limits. | Claude Opus 5.5 (the default), Sonnet 5.5, Haiku 5.5 |
| `codex` | Your ChatGPT subscription, through the Codex command. No extra bill. It counts toward your plan's usage limits. | Codex's own model |

With Claude Opus 5.5, it costs about $8 per 1,000 tasks at list price. With Claude Haiku 5.5, it's about $0.20. Each forecast reads its own task once more. A subscription runs Claude Code or Codex with no tools, none of your settings or hooks and no saved session. So these calls never show up in the records your model learns from. When Claude Code or Codex isn't signed in, setup and refresh say how to sign in and stop asking. They train without LLM analysis for now. The next refresh asks again.

## Forecasts in Claude Code

`yardstick-predict claude-code install` puts the forecast in Claude Code's status line. When you send a prompt, it shows the forecast for the AI model and effort level you're on, counts up while the agent works, then shows how long the task really took:

```text
yardstick · 1:12 of ~4 min · ~$1.23
```

"Took" is the agent's working time, as training counts it: waits for your answers and approvals are left out. When another hook keeps the agent working after it stops, the clock runs on. A background job reporting back continues its task. Scheduled tasks and messages from other sessions are forecast as a program's prompts, as training counts them. Scripted runs, such as `claude -p`, get no forecast, since they show no status line.

### Your weekly limit on a Claude subscription

On a Claude subscription, the line shows shares of your weekly limit in place of dollars. First comes what the task has used so far, with four decimals. Then the brackets hold the forecast: its time and its share of the week.

```text
yardstick · 1:12 · 0.0236% (~4 min · ~0.0789%)
```

Claude Code tells the status line how much of the week's limit you've used. It doesn't say what each task used. So yardstick works out how much of the limit a dollar of work uses. It divides the share you've used by what all your Claude Code sessions on this computer spent since the week began, at list prices. A task's share is what it has spent so far times that. The task's spend counts its helper agents too. Nothing is sent anywhere for this. It all comes from Claude Code's records on your computer.

The shares start once a forecast has worked this out. That waits until at least 1% of the week is used. Until then, the brackets show the forecast's time alone. Chats on claude.ai count toward your limit too. They leave no record here. While you use them, the shares come out a little high. With an API key, Claude Code sends no weekly limit. The line then shows dollars.

While it's on, every new prompt you send in Claude Code goes to Voyage to be embedded. So does the end of the conversation before it when Context is on. Each costs a tiny fraction of a cent. Add `--here` to turn it on only for sessions in the current folder. `yardstick-predict claude-code remove` turns it off and puts back the status line it replaced. Both keep a copy of Claude Code's settings file next to it, as it was before.

If your Voyage key is only in the `VOYAGE_API_KEY` environment variable, Claude Code may not see it. Install says so and offers to keep the key in your computer's password store.

If you installed it with version 0.2.1 or earlier, run `yardstick-predict claude-code install` again. The older commands let Python load code from the folder you work in. The status line asks you to update them until you do.

## Use it from a program

`predict` reads a JSON request on standard input and writes a JSON answer. It never asks questions. `--request FILE` reads the request from a file instead, which a program can do without a pipe: `yardstick-predict predict --request tasks.json`.

```bash
echo '{"version": 1, "tasks": [
  {"id": "a", "prompt": "Fix the flaky test", "models": ["claude-opus-5-5", "gpt-6.1-sol"]},
  {"id": "b", "prompt": "Write the release notes", "models": ["claude-opus-5-5"], "context": "We ship on Fridays."}
]}' | yardstick-predict predict
```

From Python, an orchestrator can pick the cheapest AI model for a task like this:

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
if priced:
    cheapest = min(priced, key=lambda f: f["usd"])
    print(f"Run it on {cheapest['model']}: about {cheapest['seconds'] / 60:.0f} minutes and ${cheapest['usd']:.2f}.")
else:  # no AI model has enough data yet. Each forecast says why.
    print("No forecast yet:", ", ".join(f["reason"] for f in answer["forecasts"]))
```

- **`effort`** is the effort level the forecast is for: the request's or the level the AI model ran at most often.
- **`seconds`, `tokens` and `usd`** are expected values: averages, not middles. They add up. So the expected total of ten tasks is the sum of their ten forecasts.
- **`tokens`** counts every kind: new input, cache writes, cache reads and output, including helper agents.
- **`usd`** is the cost at the AI model's list price.
- **`reason`** appears only when a forecast isn't given. Then the numbers are `null` and it says why: `unknown_model`, `not_enough_data` or `not_accurate_enough`.
- **`"distribution": true`** adds each forecast's full curve. A program can then work out any chance or any range.

One request can hold up to 1,000 tasks, each on up to 50 AI models. The format is fixed and versioned: within version 1, fields may be added but are never renamed or removed. Schemas for the [request](../schemas/forecast-request.v1.json) and the [answer](../schemas/forecast-response.v1.json) let a program check every answer. [The format](format.md) explains every field.

### Set up and refresh from a program

`create` and `refresh` answer in JSON with `--json`. They never ask anything then. Without `--yes`, they only say what they'd do and what it costs. Then they exit with code 2. With `--yes`, they do it:

```bash
yardstick-predict refresh --yes --json
```

```json
{
  "version": 1,
  "predictor": "main",
  "done": true,
  "up_to_date": false,
  "new_tasks": 3,
  "changed_tasks": 0,
  "removed_tasks": 0,
  "cost": {
    "usd": 0.024028,
    "analysis": {"tasks": 3, "model": "claude-opus-5-5", "paid_with": "anthropic", "usd": 0.024},
    "embedding": {"texts": 6, "tokens": 462, "exact": false, "model": "voyage-4", "usd": 2.8e-05}
  },
  "retrain_reason": "new_work",
  "analyzed": 3,
  "embedded": 6,
  "trained": true,
  "forecasts_for": ["claude-opus-5-5"],
  "max_cost": null,
  "spent": null,
  "notes": []
}
```

- **`done`** is false when it only said what it would do.
- **`cost`** is what it costs before anything is spent, at list price, in US dollars. Voyage's free tokens aren't taken off. LLM analysis paid by a subscription counts as 0.
- **`retrain_reason`** says why a refresh trains the model. It's `new_work` for new or changed tasks. With nothing new, it's `no_model`, `unfinished`, `outdated`, `previous` or `damaged`. It's `null` when the model is up to date.
- **`analyzed`** and **`embedded`** say how many tasks LLM analysis read and how many texts went to Voyage.
- **`forecasts_for`** lists the AI models that get forecasts now.
- **`notes`** holds anything worth knowing that didn't stop it.

When it can't go on, the answer is an `error` with a `code` and a `message`. The exit code is 1 then. [The format](format.md#setup-and-refresh) lists every field and code. It also says what `create --json` adds.

## Cost

Voyage gives every account 200 million free tokens. yardstick-predict can't see your balance. So it keeps its own count and asks once whether your account is new.

Before it embeds anything, it shows the exact token count and the price. The count comes from Voyage's own tokenizer. If a big history won't fit in your free tokens, it offers to embed only your newest tasks. Each refresh shows the cost of just the new tasks.

A text is paid for once. The same text in many tasks, such as a prompt that says only "continue", is sent once. A text embedded before with the same Voyage model, for any model or for a forecast, comes from a cache on your computer and costs nothing.

An account without a payment method gets low limits, such as a few requests and 10,000 tokens a minute. When Voyage says a limit is reached, yardstick-predict waits for it to come back. So a big first setup on such an account can take a while.

**A spending limit.** `--max-cost 1.50` keeps a setup or a refresh to $1.50 at list price. It goes ahead only when what it would cost fits. While LLM analysis runs, no request starts unless the most it can cost fits in what's left. That counts every byte of its text as a token and its answer at the longest it may be. Under a limit, a request the AI model declines isn't tried again on another one. So no request can take the spending past the limit. Before each step that embeds, it checks what's left. When a step doesn't fit, it stops there and keeps what was done. With the limit, the JSON answer also says what was spent, in `spent`.

## How it tests itself

A forecast you can't trust is worse than none, because a program acts on it. Much of how long an agent takes is chance. A model's job is to be honest about that spread and to narrow it where your data allows.

Before it gives any forecast, a model replays each AI model's newest tasks and checks three things. Its forecasts must beat a simple rule that gives every task the same forecast. Its ranges must hold the real results often enough. Its expected values must add up to about the real total. An AI model that fails gets no forecast. Each AI model needs 60 tasks of its own before it can pass. `yardstick-predict status` shows the scores and which check failed. [How it works](how-it-works.md#4-the-test) gives the exact rules.

## Settings for experts

These have sensible defaults. The guided setup never asks about them.

| Flag | What it sets | Default |
|---|---|---|
| `--model-type` | The kind of model: `xgboost` or one a plugin adds. | `xgboost` |
| `--heads` | What to forecast: any of `seconds`, `tokens`, `usd`. | all three |
| `--head` | `distribution` for a full curve or `expected` for the expected value alone. | `distribution` |
| `--shape` | The curve's shape: `lognormal` or `weibull`. Only curves have one. | `lognormal` |
| `--loss` | What training minimizes: `log_score` or `crps` for curves, `gamma` or `tweedie` for expected values. | `log_score` |
| `--dropout` | The share of trees dropped each round, from 0 to 0.9. | `0` |
| `--test-share` | The share of each AI model's newest tasks held back as the test. It's at least 30 tasks. | `0.3` |
| `--test-windows` | How many windows each AI model's test is split into. | `3` |
| `--validation-share` | The share of training tasks that decides when to stop adding trees. | `0.15` |
| `--half-life` | Days until a task counts half as much as the newest. `0` counts every task alike. | `30` |

```bash
yardstick-predict create --yes --heads seconds,usd --shape weibull --test-windows 5
```

## Plugins

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

`--plugin` takes a module in the current folder, as here. It also takes the path of a `.py` file. The model remembers where the file is. So the plugin also loads when a forecast runs from another folder, as the status line's do.

A text feature is embedded like the prompt. A number feature goes to the model as it is. Your function gets the task's prompt, context, skills, project, AI model, thinking level and agent, from past records when training and from the request when forecasting. It also gets when the task started, as `info.started_at`, in seconds since 1970: a forecast's task starts the moment you ask. A request can also carry an `"extra"` object for your features, which reaches them as `info.extra`. Past tasks have no extra. So a feature should also work its value out from the task itself. Training says so when one of your features had no value for any past task.

A plugin can also add a model type with `yardstick_predict.model_types.register_model_type(name, train, load)`, then use it with `--model-type`. [`model_types.py`](../src/yardstick_predict/model_types.py) describes what a model type provides.

## Commands

| Command | What it does |
|---|---|
| `yardstick-predict` | In a terminal, the home screen: Estimate, Update, Models and Add New. A program gets one line per model. |
| `yardstick-predict create [NAME]` | Creates a model. In a terminal it asks for a name first. `--express` or `--yes` takes every default, asks nothing and agrees to the cost. `--archive codex=PATH` adds a folder of records. A folder that isn't there stops it before anything is read. `--json` answers a program in JSON. `--max-cost USD` keeps it to a spending limit. |
| `yardstick-predict refresh` | Adds new runs, embeds them and retrains. It says when a folder of records isn't there. The model keeps the tasks it read from there. `--yes` for programs, with `--json` for an answer in JSON. `--max-cost USD` keeps it to a spending limit. |
| `yardstick-predict predict "TASK"` | Forecasts a task: a table in a terminal, JSON for programs. |
| `yardstick-predict status` | Shows each model, its features with their coverage, the size of its prepared dataset and how close its forecasts came on your newest tasks. `--json` for programs. |
| `yardstick-predict list` | Lists your models, one line each: features, how many tasks each learned from, when it was last trained and which AI models get forecasts. `--json` for programs. |
| `yardstick-predict delete NAME` | Deletes a model: its settings, its dataset of tasks, its prepared dataset and what it learned. Your records and the feature store stay. In a terminal it says what goes and asks first. `--yes` for programs, with `--json` for an answer in JSON. |
| `yardstick-predict check [NAME]` | Checks which of a model's features help its forecasts: it compares forecasts with each feature and without it. It's advice only and changes nothing. `--json` for programs. |
| `yardstick-predict features` | Lists every feature of every model. A stored one shows what made it, how big it is and when it changed. Training works out the others for free. It shows the share of each model's tasks each feature covers. `--json` for programs. |
| `yardstick-predict features delete ID` | Deletes a stored feature. It says which models use it and what making it again costs. It asks before it deletes. `--yes` for programs. |
| `yardstick-predict default [NAME]` | Shows or sets your default model: the one commands use when they name none. `--predictor NAME` picks another for one command. |
| `yardstick-predict rename OLD NEW` | Gives a model another name. The default stays the default. |
| `yardstick-predict claude-code install` | Shows forecasts in Claude Code's status line. `--here` for one folder, `remove` to turn it off. |
| `yardstick-predict uninstall` | Removes yardstick-predict: takes the forecasts out of Claude Code everywhere you added them, then deletes the Voyage key it kept and your models. It ends with the command that removes the program itself. In a terminal it says what goes and asks first. `--keep-data` keeps your models and the key. `--yes` for programs. |

## Update or remove it

To get the newest version:

```bash
uv tool upgrade yardstick-predict
```

This gets the newest release from PyPI, the Python package index. Between releases, the main branch on GitHub may hold work in progress.

If you installed 0.3.0 or an earlier version from GitHub, switch to PyPI once with `uv tool install 'yardstick-predict@latest'`. After that, the upgrade command above gets each new release.

If you added the `llm` extra before 0.5.0, the upgrade may warn that the package has no extra named `llm`. That's fine: its libraries come with every install now. To stop the warning, install once more with `uv tool install --force yardstick-predict`.

Upgrading keeps your models. When a new version reads records more accurately, the next refresh reads them all again, once. It embeds only the texts that changed. When a new version reads new features or trains another way, the next refresh trains each model again. Until then, a model the previous version trained keeps forecasting as it did. A model an older version trained gives no forecasts until it's trained again.

From 0.4 on, going back to an older version is safe too. It reads what it can of a model that a newer version saved and never changes it. It skips any feature it doesn't know, with a note. It won't refresh that model.

To remove one model, run `yardstick-predict delete NAME`. Your records, the feature store and your other models stay.

To remove everything, run:

```bash
yardstick-predict uninstall
```

It says what goes and asks first. Then it does it all in one go:

- It takes the forecasts out of Claude Code: for all your sessions and for each folder you added them to with `--here`. It also finds folders that an earlier version added them to, in Claude Code's own list of the folders it has opened. Where yardstick replaced your status line, yours comes back. A copy of each settings file as it was is kept next to it.
- It deletes the Voyage key it kept in your computer's password store.
- It deletes the yardstick folder, with your models, their datasets and the feature store. It never deletes a folder that also holds files it didn't make, as one that `YARDSTICK_HOME` names might.
- Last, it shows the command that removes the program itself, the way you installed it, such as `uv tool uninstall yardstick-predict`.

Add `--keep-data` to keep your models and the Voyage key. Then it only takes the forecasts out of Claude Code. `--yes` goes ahead without asking, for programs. If you set `VOYAGE_API_KEY` in your shell's startup file, take that line out too.

Removed the program first? Claude Code then can't run the forecasts' hooks. It may say so after each prompt. `uvx yardstick-predict uninstall` takes them out. It runs the newest version once, without installing it.

## Privacy and safety

- **Your records stay on your computer.** Your models, their datasets and the feature store live in `~/.yardstick` or wherever `YARDSTICK_HOME` points. The folder is readable only by you, on Windows too.
- **The cache of embeddings holds no text.** For each text it keeps a one-way hash and the text's vector, in the same private folder.
- **Your prompts are sent to Voyage** over an encrypted connection, to be embedded. So is the end of the conversation before each one while the Context feature is on. With LLM analysis, both also go to the AI model you pick.
- **Keys and passwords are hidden first.** Before any text is sent, each part of it shaped like a key, token or password is replaced with `[hidden]`. That covers keys from AI and cloud services, tokens from GitHub, Slack and Stripe, private keys, web tokens, passwords in web addresses and a value given to a name such as `password` or `api_key`. A value in quotes is hidden whole. A secret with no known shape gets through. So keep secrets out of your prompts.
- **Your Voyage key** is read from `VOYAGE_API_KEY` or kept in your computer's password store. yardstick-predict never writes it to a file of its own.
- **Other downloads** are public data, made only when needed: Voyage's tokenizer from Hugging Face at one fixed version and OpenRouter's model list if you use benchmarks. The home screen also reads yardstick-predict's public page on PyPI at most once a day, to say when a newer version is out. To turn that off, set `YARDSTICK_NO_UPDATE_CHECK` or `DO_NOT_TRACK`. Adding `"update_check": false` to `settings.json` in the yardstick folder works too.
- **What a model learned is saved as plain JSON and plain numbers.** Loading it runs no code.
- **Plugins are code you choose to run.** yardstick-predict only imports the modules you name.
- **What other services say is cleaned before it's shown.** An error from Voyage, Anthropic, OpenAI, Claude Code, Codex, OpenRouter or PyPI keeps its words but loses any hidden codes. Such codes could make your terminal show something else, change its title or write over a line.

## Libraries it uses

| Package | Why |
|---|---|
| [XGBoost](https://xgboost.ai) | The gradient-boosted trees. Linux and Windows get `xgboost-cpu`, the same library without the large GPU part. |
| [NumPy](https://numpy.org) | The numbers. |
| [tokenizers](https://github.com/huggingface/tokenizers) | Counts tokens exactly with Voyage's own tokenizer. So you see the cost before paying it. |
| [keyring](https://github.com/jaraco/keyring) | Keeps your Voyage key in your computer's password store. |
| [PyArrow](https://arrow.apache.org/docs/python/) | Reads and writes the feature store's files, in Parquet, a compact file format for tables. |
| [scikit-learn](https://scikit-learn.org), on macOS only | Brings the OpenMP library XGBoost needs on a Mac. So there's nothing else to install. |
| [anthropic](https://github.com/anthropics/anthropic-sdk-python) and [openai](https://github.com/openai/openai-python) | The providers' own libraries, for LLM analysis paid with an API key. |
