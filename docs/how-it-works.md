# How it works

## 1. The data: your past runs

Claude Code and Codex keep a record of every run on your computer. Claude Code keeps them in `~/.claude/projects`, and Codex keeps them in `~/.codex/sessions`. yardstick-predict reads those records and turns each task into one row of a dataset. You can add folders of records in either format, copied from another computer or a server.

A **task** is one turn that a person or a program started. It runs from the prompt until the agent stops and hands control back. A person started it when they typed it, including a `!` shell command the agent then works on. A program started it when it came from a script or headless run, a scheduled task or another session. Turns started by an automatic event, such as a background job finishing, are not tasks of their own.

Each row holds:

- the prompt, the conversation before it, the skills known before it and the skills the agent used
- the AI model, the thinking level and the agent
- whether a person or a program started the task
- how long the task took, in seconds
- the tokens it used, by kind: new input, cache writes, cache reads and output
- its cost at the model's list price
- whether the agent answered directly, without tools

The records need care, and these rules matter more than the model:

1. **Each model reply counts once.** Claude Code writes one reply across several lines and repeats its token counts on each. Adding them all up would roughly double the tokens.
2. **Helper agents count toward the task that started them.** Their tokens and cost are added to it, even when they finish after a refresh or work on after your next prompt. A helper belongs to the prompt that gave it its first instruction, or to a later prompt that sent it more work. Helpers that a workflow starts count the same way. A task counts as finished only once its helpers are done. While one may still be running, the task waits for a later refresh. A helper that goes quiet without ever reporting back leaves its task stopped.
3. **Work after a background job reports back counts toward the task that started the job.** When a background command or helper finishes, Claude Code tells the agent in a turn of its own, and the agent often works on. That work's tokens, cost and working time count toward the task that started the job. The time the job itself ran between turns doesn't. Events from a monitor or from CI aren't any task's work, and aren't counted.
4. **Copies count once.** A resumed session copies its earlier messages into a new record file, and real histories hold many such copies. A session and its copies are read together, and each line counts once, so a copy is never counted twice and a copy that lost a line can't mix two tasks together. A Codex helper's work is matched the same way, and a Codex session spread over several files keeps the work in each. Without this, a copy could land in both the training data and the test data, and the test would look better than it really is.
5. **Waiting for a person isn't agent time.** When a Claude Code agent asks a question or waits for approval, that wait is taken out of the task's time. When the tool's result says how long the tool ran after it was approved, only the wait before that is taken out. Shell commands and MCP tools don't say, so their whole wait is taken out. Codex records don't mark these waits yet.
6. **Tasks left waiting are kept but not learned from.** A task with a silence of more than 30 minutes while the agent worked was left waiting, not working.
7. **Stopped tasks are kept but not learned from.** A task is stopped when you press Esc, when its last reply is an error from Claude Code, such as a server error or a usage limit, or when its last tool call never got a result. All you know about a stopped task is that it would have taken at least this long.
8. **A task that may still be running waits.** The last task in a file written to in the last 15 minutes is read on the next refresh.

A refresh reads only the record files that changed since the last one, each with the others of its session. A task read again from its own record replaces the saved copy, even when it now turns out to be stopped. A saved task that its record no longer holds is taken out. A task whose record is gone, as when Claude Code deletes old sessions, is kept.

When a new version reads records more accurately, the next refresh reads all of them again, once, by itself. It embeds only the texts that changed.

When a new version trains the model another way, the next refresh trains it again, even when it finds no new tasks. Until then, a model the previous version trained keeps forecasting, exactly as that version did. A model from an older version gives no forecasts until it's trained again.

## 2. Features

Every feature describes what's known before a task starts. Three are always used. They cost nothing, and they send nothing anywhere:

- **The task:** the prompt's meaning and its layout (how many list items, file names, links and lines it has, and whether it ends in a question), the AI model, the agent, who started it, and the thinking level, both as a label and as a step on one scale from low to max, so a level with few tasks can learn from the levels next to it.
- **Similar past tasks:** see below.
- **Project history:** the project's five most recent finished tasks, from any conversation: their typical time, tokens and cost, how often they were answered directly, how many came before and how long ago the last one finished.

These can be added. The first two are on by default:

- **`conversation`:** the conversation so far: how many tasks it had, what its last finished task took and how long ago that was, and the conversation's size in tokens when the task started. A person's conversation and a program's one-off run grow differently, so each has its own size number. It costs nothing and sends nothing anywhere. With `context` too, the model also learns how close the prompt's meaning is to the conversation's, and finds past tasks similar in both. The conversation tells a lot about a person's sessions and can mislead about a program's one-off runs, so a predictor that mostly forecasts programs' runs may forecast better without it.
- **`context`:** the last 1,500 characters of the conversation before the prompt. A longer cut made forecasts worse on real runs, and costs more.
- **`skills`:** the skills known before the task runs: the ones its prompt invokes, such as a typed `/review`, and the ones used earlier in its conversation. Each comes with its description from its own `SKILL.md`. Its options can keep only one of the two kinds. The skills a task went on to use are in the records too, but a forecast can't know them, so the model doesn't learn from them.
- **`benchmarks`:** public facts about each AI model from OpenRouter's model list: input, output and cache-read prices, context size, longest answer, release date, and whether it reasons. They're described the same way for old and new models. The list is downloaded at setup and at each refresh, and saved with the predictor.
- **`time`:** when the task started. It costs nothing and sends nothing anywhere. The `trend` option is the number of days since 1 January 2020, with the time of day as a fraction, so the model can follow slow changes, such as tasks growing over the months. That date never changes, so a saved model always reads the number as it learned it. The `clock` option is the hour of the day, minutes included, and the day of the week, in your computer's time zone. A forecast's task starts the moment you ask. The model can't look past the newest day it trained on, so it treats a new task like the tasks of its most recent weeks.
- **`llm`, LLM analysis:** an AI model reads each task's prompt and the end of the conversation before it, and reports the kind of task, how many files and steps it likely takes, whether it must run code or look things up, whether it continues earlier work, how clear it is, and its own guess of the minutes. With the description option it also writes a few sentences on the work, which are embedded like the prompt. It never sees how a task went. Each task is read once and the answer is saved, for every predictor that asks the same AI model.
- **Your own:** text or number features registered by a plugin.

An embedding is a list of numbers that captures what a text means: similar texts get similar numbers. Each embedded feature is embedded once with the predictor's Voyage 4 model, at 256 numbers, and saved for every predictor that uses the same model. For a new task, the predictor finds the 100 most similar past tasks by that feature. It averages what they really took: their seconds, tokens and cost on a log scale, and how often they were answered directly. It also records how similar they were. For the prompt it adds more: what the closest task took, the same averages over the closest 10, how much the 100 varied, the typical direct answer and the typical piece of work kept apart, and an average where closer and newer tasks count more. On real runs, the closest ten told more than the hundred, and the spread said how sure to be.

This works better than giving the trees the embedding itself. 256 weak numbers let trees find patterns that don't hold for new tasks. On real runs, the raw numbers made forecasts worse, and averages over similar tasks made them better. The whole embedding still gets one say: a weighted sum of all its numbers, with the weights kept small so it can't chase coincidences, makes a first guess of the time, tokens and cost. In training, each task's guess comes from weights fit only on tasks that had finished before it, as a forecast's guess is. The trees get that guess as one more input.

## 3. The model: one curve per task

For each task on each AI model, the model forecasts a curve for each head: seconds, tokens and dollars. An agent either answers directly, without tools, or works with tools, so each curve has two parts:

1. the chance of a direct answer
2. a curve for a direct answer
3. a curve for work with tools

By default each part is a log-normal curve: a bell curve on a log scale. It fits times and token counts, which are never negative and run from a few seconds to many hours. On a log scale, task times form two humps, quick direct answers and longer work, and the two parts give each hump its own curve.

**Gradient-boosted trees** learn every part, using XGBoost, the standard library for this kind of data. For each part, the trees learn two numbers per task: the curve's center and its width. They learn them by maximum likelihood: finding the curves under which what really happened is as likely as possible.

Three choices keep the model honest when there's little data:

1. **The trees start from each AI model's own curve.** That's the simple rule: the spread of the model's past tasks. A model with few tasks has its starting curve pulled toward the curve of all models, as if it had 10 more tasks that looked like the average. Each effort level the model ran at gets its own starting curve, pulled toward the model's in the same way. Old tasks count less here too, so a model with many old tasks and few new ones is pulled as hard as one with few tasks.
2. **Trees are added only while they help.** The newest 15% of training tasks are held back. Trees are added only while they beat the starting curves on those tasks, and those starting curves are fitted without them. With little data, the model stays close to the simple rule. With a lot of data, it gets sharper.
3. **Newer tasks count more.** How people use agents changes over time. A task 30 days older than the newest one counts half as much.

A last step widens or narrows each AI model's curves by one factor, picked on the held-back tasks before the trees have seen them, by the same score the trees learn by. Then the starting curves are fitted again and every part is trained again, on all the tasks.

**One model, with the features you chose.** Training makes one model, which reads every feature the predictor has, and the test below judges it. With LLM analysis, training also makes the same model without it, and tests that one too. A task that LLM analysis can't read, such as when its AI model can't be reached, is forecast by the model without it, and only if that model passed the test too.

### Advanced settings

Experts can change the model with flags on `create`:

- **Heads:** forecast any of seconds, tokens and dollars.
- **Expected values only:** instead of curves, trees learn each task's expected value directly, with a gamma or Tweedie loss.
- **The curve's shape:** log-normal or Weibull, a curve common for durations.
- **The loss:** the log score, or CRPS for log-normal curves.
- **Dropout:** drop a share of trees each round, as XGBoost's DART does.
- **The data split:** the test's share and windows, the share that decides when to stop, and how fast older tasks count less.
- **The model type:** XGBoost, or one a plugin adds. A model type is a `train` and a `load` function. See [`model_types.py`](../src/yardstick_predict/model_types.py).

## 4. The test

Before it gives any forecast, the predictor tests itself on tasks it hasn't seen.

It replays each AI model's own newest 30% of tasks, and at least 30 of them, in three windows, oldest first. Each window is forecast by a model trained only on the tasks, on every AI model, that had finished before it opened, waits for you and helpers included. Testing each AI model on its own newest tasks means a burst of tasks on one model can't push another model's tasks out of the test. Using three windows means one unusual week can't decide the verdict alone.

Each AI model's forecasts are then scored against **the simple rule**: the same forecast for every task on that model, the spread of its past tasks. To be given, a model's forecasts must pass every check, for every head:

1. **Beat the simple rule.** They must score at least as well as the simple rule, within 2%. Curves are scored with CRPS, the standard score for probability forecasts. It measures how far the whole curve sat from the real result, so it rewards curves that are both honest and narrow. Expected values are scored with their loss.
2. **Keep the ranges honest.** For curves, at least 75% of real results must fall inside the 90% range.
3. **Add up.** The expected values must add up to between half and twice the real total.

A model also needs 30 earlier tasks and 30 test tasks, so it needs 60 tasks of its own before the test can judge it. A window that has too few finished tasks before it doesn't run, and its tasks don't count as tested. A model that fails gets `null` forecasts and a reason. So does an effort level the model never ran at: nothing was learned about it. `yardstick-predict status` shows the scores and which check failed.

## 5. The feature check

Some features carry no signal for your kind of work, and with few tasks a feature can't show anything. The feature check finds out which of a predictor's features help its forecasts. Setup offers it when you pick features beyond the three that are always on, and at least one AI model has the 60 tasks the test needs. `yardstick-predict check NAME` runs it on a predictor you already have. It tests the built-in model, not a model type from a plugin.

It runs the test above once with every feature, then once without each optional feature in turn. It only tests: it fits a model for each test window, and no final model. The similar past tasks are worked out once for all the tests, since leaving a feature out only drops what that feature brings. For each feature and each AI model, it gives one of three verdicts:

- **Helped:** the AI model's test scored more than 1% worse without the feature. A lower score is better. When the test scored time, tokens and cost, each counts alike: the check averages how much worse each scored.
- **Didn't help:** the test scored about the same without it, or better.
- **Not enough data:** the AI model had too few tasks to test, or under 20% of its test tasks had the feature, as with skills that few prompts name.

A feature that didn't help costs embedding tokens or an AI model's time for nothing, and adds noise the trees can chase. Not enough data is no verdict either way: check again once you have more tasks. The check is advice. It never changes a predictor by itself.

In setup, the Features screen comes back after the check, with the features that helped at least one AI model turned on. You keep them or change them, and the model trains with what you choose. LLM analysis is tried on a sample first: before the check it reads only your newest 30% of tasks, and at most 500, with its cost shown first. If you keep it on, setup shows what reading the rest costs and asks. With a very large history, 500 tasks may be under 20% of an AI model's test tasks, and then LLM analysis gets not enough data. Turning on a feature the check didn't have shows its cost before anything is embedded.

Each verdict is saved with when it was checked and the predictor's name, in `features/checks.json` in the yardstick folder. A later verdict replaces it, except a verdict of not enough data, which tells nothing new. The Features screen shows the latest verdicts next to each feature. With `--json`, `yardstick-predict check` gives programs every verdict, with `worse_without`, how much worse the test scored without the feature (0.05 is 5% worse), and `share`, the share of the AI model's test tasks that had it.

## 6. On a Mac

XGBoost needs a library called OpenMP. On a Mac, XGBoost looks for it only where Homebrew would put it. scikit-learn, installed with yardstick-predict on Macs, carries its own copy. When XGBoost can't find OpenMP, the command restarts once with scikit-learn's folder on the list of places the Mac searches, so nothing else has to be installed.

## 7. Speed

- **Reading records** takes well under a minute for gigabytes of history. A refresh reads only what changed.
- **Training** takes under a minute for a few thousand tasks, test included, and about twice as long with LLM analysis, which also trains and tests the model without it. On that much data, XGBoost's threads cost more time than they save, so it uses one, and up to four on very large histories.
- **A forecast** takes about half a second from the command line. Most of that is starting Python. Add one short request to Voyage for the embeddings, unless the cache already holds them, and with LLM analysis, one request to the AI model you picked. When Voyage can't be reached, a forecast says so within seconds.
- **LLM analysis** reads past tasks once, in batches: several at a time with an API key, fewer through a subscription.

Before setup or a refresh goes ahead, it says about how long the rest will take, and how much disk space the dataset will take. It works this out from what it knows by then: how many tokens go to Voyage, how many tasks LLM analysis reads, how many tasks training learns from and, in setup, how many features the feature check tests. A few rough constants turn those into time, disk space and memory. Training's and the dataset's were measured on synthetic tasks. Voyage's and the AI models' speeds can't be measured without paying, so theirs are worked out from how the requests go out. Voyage's assume an account with a payment method: one without gets low limits, and embedding takes much longer. The constants are in [`estimate.py`](../src/yardstick_predict/estimate.py), each with how it was found. Nothing is timed while the command runs, and the time is given as a range, from 70% to 150% of the estimate. While each long step runs, its progress bar says how long is left, from how fast it has gone lately.

## 7. Where your data lives

Everything stays in one folder on your computer: `~/.yardstick`, or the folder `YARDSTICK_HOME` names. Only you can read it.

- **`predictors/`** holds a folder for each predictor: its settings, its dataset of tasks, its model and the model facts it downloaded.
- **`features/`** is the feature store. It keeps the features that cost money or time to make: embeddings, and what LLM analysis said about each task. Each is made once and shared by every predictor that uses it, so a second predictor over the same records pays for nothing it already has. Embeddings are kept by Voyage model and size, and LLM answers by the AI model and the way it's paid for.
- **`cache/`** keeps every vector Voyage sent, by a one-way hash of its text, so no text is paid for twice, a forecast's included.

The features that cost nothing aren't stored: the prompt's layout, the project history, the conversation so far, the time, model facts and your own number features. They depend on each predictor's own tasks, so training works them out from the tasks each time, and they can't go stale.

`yardstick-predict features` lists every feature of every predictor. A stored one shows what made it, how many tasks have it, its size on disk and when it last changed. The others are worked out at training, for free. Each feature also shows its coverage: the share of the tasks the predictor learns from that have a value for it. A text feature has one when the task has its embedding, made from its text as it is now. A number feature has one when any of its numbers does. `--json` gives the same to programs.

The store is plain files. Each feature has a folder of Parquet files, a compact file format for tables, and `index.json` lists the features. Each file also names its feature, so nothing depends on the index alone. Deleting a feature's folder, or any file, never breaks a command:

- A missing or broken index is rebuilt from the files, with a note.
- A file that can't be read is skipped, with a note naming it. `yardstick-predict features` names it too.
- Whatever is missing is made again by the next refresh that needs it. An embedding comes back from the cache for free. An LLM answer is asked for again, with the cost shown first.

A predictor made by an older version kept its embeddings and LLM answers in its dataset. The first time this version refreshes or lists it, they move into the store, once. The old copies stay in the dataset, unread. A forecast never reads the store: it uses what the model saved and the cache, so it's as quick as before.
