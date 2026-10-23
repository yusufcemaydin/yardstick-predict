# The forecast format, version 1

`yardstick-predict predict` reads one JSON request on standard input and writes one JSON answer on standard output. It never asks questions. The answer always says which format version it follows.

In a terminal, `yardstick-predict predict "TASK"` shows the same forecasts as a small table. Add `--json` to get this format there too.

Within version 1, new fields may be added. Fields are never renamed or removed. A change that would break a program reading version 1 gets version 2.

The [request schema](../schemas/forecast-request.v1.json) and the [answer schema](../schemas/forecast-response.v1.json) describe the same rules in a form programs can check automatically.

## The request

```json
{
  "version": 1,
  "distribution": false,
  "tasks": [
    {
      "id": "a",
      "prompt": "Fix the flaky test in the payments module",
      "models": ["claude-opus-5-5", "gpt-6.1-sol"],
      "context": "We moved the payments module to the new queue last week.",
      "effort": "high"
    }
  ]
}
```

| Field | Needed | What it is |
|---|---|---|
| `version` | yes | Always `1`. |
| `tasks` | yes | 1 to 1,000 tasks. |
| `tasks[].id` | yes | Your name for the task. Each id must be unique in the request. |
| `tasks[].prompt` | yes | The prompt the agent will get. It needs some text, not only spaces. As in training, the spaces at its ends are left out and only its first 24,000 characters count. |
| `tasks[].models` | yes | 1 to 50 AI models to forecast the task on, by the model's API name. A model named twice is forecast once. |
| `tasks[].context` | no | The conversation so far, if the task continues one. Used when the predictor has the `context` feature. Only its last 24,000 characters count. |
| `tasks[].skills` | no | The skills known before the task runs, by name: the ones the conversation used before it. Used when the predictor has the `skills` feature. The ones the prompt invokes, such as a typed `/review`, are read from the prompt, so they may be left out. |
| `tasks[].effort` | no | The thinking level, such as `high`, `xhigh` or `max`. Default: the level this model ran at most often, with recent tasks counting more. |
| `tasks[].agent` | no | `claude-code` or `codex`. Default: the agent this model ran in most often. |
| `tasks[].origin` | no | `program` when a program starts the task, `person` when a person types it. Default: `program`. |
| `tasks[].project` | no | The project's folder name. The forecast uses the project's recent tasks from your records, and your own features get it too. |
| `tasks[].session` | no | The conversation the task is in, by its id in the agent's records. Used when the predictor has the `conversation` feature: the forecast uses what the conversation's earlier tasks took. |
| `tasks[].conversation_tokens` | no | How big the conversation is so far, in tokens. 0 or left out for a new one. Used when the predictor has the `conversation` feature. |
| `tasks[].extra` | no | Any values your own features need. They reach them as `info.extra`. Past tasks have no extra, so a feature should also work its value out from the task itself. |
| `distribution` | no | `true` to get each forecast's full curve. Default: `false`. |
| `predictor` | no | Which predictor to use. Default: your default predictor, set with `yardstick-predict default NAME`. |

A field that isn't needed may also be `null`. That means the same as leaving it out. Any other value of the wrong kind, such as `[]` for `extra` or `""` for `skills`, is an `invalid_request`.

When the predictor has the `time` feature, each task is forecast as if it starts the moment the request is read.

## The answer

```json
{
  "version": 1,
  "predictor": "default",
  "embedding_model": "voyage-4",
  "forecasts": [
    {"task": "a", "model": "claude-opus-5-5", "effort": "high", "seconds": 245.0, "tokens": 1830000, "usd": 0.92},
    {"task": "a", "model": "gpt-6.1-sol", "effort": "high", "seconds": null, "tokens": null, "usd": null, "reason": "not_accurate_enough"}
  ]
}
```

These numbers are an example. There is one forecast for each pair of task and model, in the order of the request.

| Field | What it is |
|---|---|
| `effort` | The effort level the forecast is for: the one in the request, or the level this model ran at most often. |
| `seconds` | Expected seconds from the prompt until the agent stops and hands control back. In Claude Code runs, time spent waiting for a person to answer or approve is left out. |
| `tokens` | Expected tokens of every kind: new input, cache writes, cache reads and output, including helper agents. A whole number. |
| `usd` | Expected cost in US dollars at the model's list price. `null` when the model's price isn't in the price list. `0` for a model whose list price is 0 for everything, with no curve for it in `distribution`. |
| `reason` | Only when the forecast isn't given: why the numbers are `null`. |

A predictor can be set to forecast only some of `seconds`, `tokens` and `usd`. The others are always `null`. A value the test couldn't check for an AI model is `null` too, such as `usd` on a model whose cost it couldn't test.

Expected values are averages, not middles. A few long runs pull the average above the typical run. Expected values add up: the expected total of several tasks is the sum of their forecasts.

### Reasons

| Reason | What it means |
|---|---|
| `unknown_model` | The predictor has never seen a task on this model. |
| `not_enough_data` | Too few tasks on this model to test its forecasts. A model needs 30 earlier tasks and 30 test tasks. A request for an effort level the model never ran at gets it too. |
| `not_accurate_enough` | The forecasts on this model didn't pass the test. `yardstick-predict status` shows which check failed. |

### Errors

When the whole request can't be answered, the answer has an `error` instead of `forecasts`, and the command exits with code 1.

```json
{"version": 1, "error": {"code": "invalid_request", "message": "Task 'a' needs a \"prompt\" string."}}
```

| Code | What it means |
|---|---|
| `invalid_request` | The request breaks a rule above. The message says which. A program gets it too when it runs `predict` with a command line it can't read, such as a flag it doesn't know. |
| `no_predictor` | No predictor exists yet. |
| `no_default_predictor` | There are several predictors, none is the default, and the request names none. Name one in `predictor`, or choose a default with `yardstick-predict default NAME`. |
| `unknown_predictor` | The request names a predictor that doesn't exist, or the predictor was deleted while the request was answered. |
| `broken_predictor` | The predictor's settings file can't be read, or is empty or broken. The message says which file. |
| `plugin_failed` | A plugin the predictor uses couldn't be loaded. The message says which, and why. |
| `not_trained` | The predictor has no model yet, or its model was made by a version before the previous release and has to be trained again. `yardstick-predict refresh` trains it. |
| `no_embedding_key` | No Voyage key was found. |
| `embedding_failed` | Voyage couldn't embed the request's text. The message gives Voyage's reason. |
| `missing_library` | A library the model needs is missing. The message says how to add it. |
| `model_too_new` | The predictor's model was saved by a newer version of yardstick-predict. Upgrade with `uv tool upgrade yardstick-predict`. |
| `internal_error` | Something went wrong that the tool didn't expect. The message says what. Please report it. |

Whatever goes wrong, a program that runs `predict` finds only the JSON answer on standard output, never a traceback or other text.

## The distribution

With `"distribution": true`, each forecast that's given also has a `distribution`. It is `null` when the forecast isn't given, or when the predictor forecasts expected values only.

```json
"distribution": {
  "shape": "two_part_lognormal",
  "direct_probability": 0.23,
  "seconds": {"direct": {"mu": 2.71, "sigma": 0.62}, "work": {"mu": 5.12, "sigma": 1.48}},
  "tokens":  {"direct": {"mu": 12.9, "sigma": 0.31}, "work": {"mu": 14.7, "sigma": 1.07}},
  "usd":     {"direct": {"mu": -1.8, "sigma": 0.35}, "work": {"mu": 0.1, "sigma": 1.12}}
}
```

An agent either answers directly, without using tools, or works with tools. So each curve has two parts:

- `direct_probability` is the chance of a direct answer.
- `direct` and `work` are the two parts' curves. On the natural log of the value, each has a center `mu` and a width `sigma`.

The `shape` says which curve each part is. With `p` for `direct_probability`, `x` for a value, `w = (ln x − mu) / sigma` for each part, and each part's chance `F(w)`:

- **The chance the value is at most `x`:** `p · F(w_direct) + (1 − p) · F(w_work)`.
- **A range, such as the 90% range:** the values where that chance is 0.05 and 0.95.

| Shape | Each part | `F(w)` | Expected value of a part |
|---|---|---|---|
| `two_part_lognormal` | log-normal | `Φ(w)`, the standard normal curve's chance | `exp(mu + sigma² / 2)` |
| `two_part_weibull` | Weibull | `1 − exp(−exp(w))` | `exp(mu) · Γ(1 + sigma)` |

The expected value of the whole curve is `p` times the direct part's plus `1 − p` times the work part's. It's the number in `seconds`, `tokens` or `usd`.

Seconds, tokens and dollars are forecast as separate curves. They usually move together: a task that runs long usually uses more tokens too.
