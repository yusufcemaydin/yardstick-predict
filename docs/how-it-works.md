# How it works

## 1. The data: your past runs

Claude Code and Codex keep a record of every run on your computer. Claude Code keeps them in `~/.claude/projects`, and Codex keeps them in `~/.codex/sessions`. yardstick-predict reads those records and turns each task into one row of a dataset. You can add folders of records in either format, copied from another computer or a server.

A **task** is one turn that a person or a program started. It runs from the prompt until the agent stops and hands control back. Turns started by an automatic event, such as a background job finishing, are not tasks.

Each row holds:

- the prompt, the conversation before it and the skills the agent used
- the AI model, the thinking level and the agent
- whether a person or a program started the task
- how long the task took, in seconds
- the tokens it used, by kind: new input, cache writes, cache reads and output
- its cost at the model's list price
- whether the agent answered directly, without tools

The records need care, and these rules matter more than the model:

1. **Each model reply counts once.** Claude Code writes one reply across several lines and repeats its token counts on each. Adding them all up would roughly double the tokens.
2. **Helper agents count toward the task that started them.** Their tokens and cost are added to it, even when they finish after a refresh or work on after your next prompt. A task counts as finished only once its helpers are done.
3. **Copies count once.** A resumed session copies its earlier messages into a new record file, and real histories hold many such copies. Each task is matched by its own id, so a copy is never counted twice, and the most complete copy is kept. A Codex helper's work is matched the same way, and a Codex session spread over several files keeps the work in each. Without this, a copy could land in both the training data and the test data, and the test would look better than it really is.
4. **Waiting for a person isn't agent time.** When a Claude Code agent asks a question or waits for approval, that wait is taken out of the task's time. Codex records don't mark these waits yet.
5. **Tasks left waiting are kept but not learned from.** A task with a silence of more than 30 minutes while the agent worked was left waiting, not working.
6. **Stopped tasks are kept but not learned from.** A task is stopped when you press Esc, or when its last reply is an error from Claude Code, such as a server error or a usage limit. All you know about a stopped task is that it would have taken at least this long.
7. **A task that may still be running waits.** The last task in a file written to in the last 15 minutes is read on the next refresh.

A refresh reads only the record files that changed since the last one.

## 2. Features

The prompt is always used. These can be added:

- **`context`:** the last 1,500 characters of the conversation before the prompt. A longer cut made forecasts worse on real runs, and costs more.
- **`skills`:** the skills the agent used, found in the records, with each skill's description from its own `SKILL.md`.
- **`benchmarks`:** public facts about each AI model from OpenRouter's model list: input, output and cache-read prices, context size, longest answer, release date, and whether it reasons. They're described the same way for old and new models. The list is downloaded at setup and at each refresh, and saved with the predictor.
- **Your own:** text or number features registered by a plugin.

Every task also brings its AI model, thinking level, agent, who started it and its prompt's length.

An embedding is a list of numbers that captures what a text means: similar texts get similar numbers. Each embedded feature is embedded once with the predictor's Voyage 4 model, at 256 numbers, and saved. For a new task, the predictor finds the 100 most similar past tasks by that feature. It averages what they really took: their seconds, tokens and cost on a log scale, and how often they were answered directly. It also records how similar they were. These five numbers are what the model learns from.

This works better than giving the trees the embedding itself. 256 weak numbers let trees find patterns that don't hold for new tasks. On real runs, the raw numbers made forecasts worse, and averages over similar tasks made them better.

## 3. The model: one curve per task

For each task on each AI model, the model forecasts a curve for each head: seconds, tokens and dollars. An agent either answers directly, without tools, or works with tools, so each curve has two parts:

1. the chance of a direct answer
2. a curve for a direct answer
3. a curve for work with tools

By default each part is a log-normal curve: a bell curve on a log scale. It fits times and token counts, which are never negative and run from a few seconds to many hours. On a log scale, task times form two humps, quick direct answers and longer work, and the two parts give each hump its own curve.

**Gradient-boosted trees** learn every part, using XGBoost, the standard library for this kind of data. For each part, the trees learn two numbers per task: the curve's center and its width. They learn them by maximum likelihood: finding the curves under which what really happened is as likely as possible.

Three choices keep the model honest when there's little data:

1. **The trees start from each AI model's own curve.** That's the simple rule: the spread of the model's past tasks. A model with few tasks has its starting curve pulled toward the curve of all models, as if it had 10 more tasks that looked like the average.
2. **Trees are added only while they help.** The newest 15% of training tasks are held back. Trees are added only while they beat the starting curves on those tasks. With little data, the model stays close to the simple rule. With a lot of data, it gets sharper.
3. **Newer tasks count more.** How people use agents changes over time. A task 30 days older than the newest one counts half as much.

A last step widens or narrows each AI model's curves by one factor, picked on the held-back tasks before the trees have seen them. Then every part is trained again on all the tasks.

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

It replays the newest 30% of tasks in three windows, oldest first. Each window is forecast by a model trained only on the tasks that had finished before it opened, waits for you and helpers included. Using three windows means one unusual week can't decide the verdict alone.

Each AI model's forecasts are then scored against **the simple rule**: the same forecast for every task on that model, the spread of its past tasks. To be given, a model's forecasts must pass every check, for every head:

1. **Beat the simple rule.** They must score at least as well as the simple rule, within 2%. Curves are scored with CRPS, the standard score for probability forecasts. It measures how far the whole curve sat from the real result, so it rewards curves that are both honest and narrow. Expected values are scored with their loss.
2. **Keep the ranges honest.** For curves, at least 75% of real results must fall inside the 90% range.
3. **Add up.** The expected values must add up to between half and twice the real total.

A model also needs 30 earlier tasks and 30 test tasks. A model that fails gets `null` forecasts and a reason. `yardstick-predict status` shows the scores and which check failed.

## 5. On a Mac

XGBoost needs a library called OpenMP. On a Mac, XGBoost looks for it only where Homebrew would put it. scikit-learn, installed with yardstick-predict on Macs, carries its own copy. When XGBoost can't find OpenMP, the command restarts once with scikit-learn's folder on the list of places the Mac searches, so nothing else has to be installed.

## 6. Speed

- **Reading records** takes well under a minute for gigabytes of history. A refresh reads only what changed.
- **Training** takes seconds for thousands of tasks, including the test.
- **A forecast** takes about half a second from the command line. Most of that is starting Python. Add one short request to Voyage for the embeddings.
