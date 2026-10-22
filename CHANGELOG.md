# Changelog

Every notable change to yardstick-predict is listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-22

Records are read more accurately. A predictor made with 0.1.0 keeps the tasks it already read. To re-read all of yours, create a new predictor with `yardstick-predict create NAME`, then make it the default with `yardstick-predict default NAME`. Setup shows what embedding costs before it spends anything.

### Added

- Forecasts in Claude Code's status line. `yardstick-predict claude-code install` shows the forecast for the model and effort level you're on as soon as you send a prompt, counts up while the agent works, and shows how long the task really took. `--here` turns it on for one folder, and `yardstick-predict claude-code remove` turns it off.

### Fixed

- `predict "TASK" --json` gave each effort level's forecast its own task id. Every forecast for one prompt now has the task id `task`.
- The test before any forecast learned from tasks that were still running when it tested. It now learns only from tasks that had finished, and so do the similar-task features, so its accuracy is no longer too optimistic.
- A forecast value such as cost was given without being tested, or after being tested on one or two tasks. Each value is now given only for an AI model whose test checked it on at least 30 tasks.
- An AI model used only long ago was missing from `yardstick-predict status`. It's now listed, with the reason it has no forecasts.
- Reading records:
  - The conversation before a prompt was stored with each reply before its prompt. It's now in order.
  - Codex helpers' tokens were lost, or charged to the wrong task, when a helper replied after its task finished, was reused, was started by another helper, or changed between refreshes.
  - A task read before it finished stayed stopped. It's now repaired when it finishes.
  - When two archives held one task, the less complete copy could win.
  - A line written while the agent waited for you could mark the task left waiting.
  - Two waits for you at the same time were taken off the task's time twice.
- A refresh after a failed training run said the model was up to date. It now trains again.
- Saving a model removed the working one before the new one was in place.
- `--max-tasks` embedded and trained on different tasks. Setup now stops before spending anything when fewer than 30 tasks can be learned from.
- Voyage:
  - What a split batch had embedded was lost when a later part failed, and it was paid for again.
  - Usage counts were lost when two processes recorded usage at once.
  - A batch over the account's tokens-a-minute limit was retried several times before it was split.
  - A forecast embedded texts the model couldn't use.
  - A wrong `VOYAGE_API_BASE` was retried for about two minutes.
- Skill descriptions written over several lines were read as `|` or `>`.
- Your own text features saw no model, effort level or agent when forecasting.
- A forecast answer or the test report could hold NaN, which strict JSON readers reject.
- In Claude Code's status line:
  - A slow forecast could replace a newer prompt's forecast.
  - The forecast didn't know a person typed the prompt, or what came before it in the conversation.
  - A model could be shown another model's forecast.
  - Odd shapes in Claude Code's settings could crash it.
- Tasks are saved with each column named, so an older dataset keeps every value in its column.
- Two archives with the same name now get their own names.
- Forecasting with a predictor that has no model yet says `not_trained`.

## [0.1.0] - 2026-10-22

The first release.

### Added

- Forecasts of a coding-agent task's expected seconds, tokens and dollars on each AI model, learned from your own past Claude Code and Codex runs.
- A forecast for each effort level: how hard the model thinks, such as `high`, `xhigh` or `max`. The terminal shows a row for each level you often use a model at, and `--effort` names the levels yourself. Each forecast in the JSON answer says which level it's for.
- `yardstick-predict predict "TASK"`, which shows a small table in a terminal. Programs, and `--json`, get the JSON answer. With no `--model`, it forecasts every model the predictor covers.
- A fixed JSON format, version 1, with published schemas for the request and the answer. One request holds up to 1,000 tasks, each on up to 50 models. `"distribution": true` adds each forecast's full probability curve.
- A test before any forecast. A predictor replays your newest 30% of tasks, and each AI model must beat the simple rule, keep its ranges honest and add up. A model that fails gets `null` and a reason.
- Guided setup with `yardstick-predict`, or setup in one line with `yardstick-predict create --express`. Setup shows the exact Voyage cost before it spends anything, and keeps your key in your computer's password store.
- `yardstick-predict refresh`, which adds new runs and retrains, and `yardstick-predict status`, which shows each predictor and how it did on its test.
- Named predictors. `yardstick-predict default NAME` picks the one commands use when they name none, and `--predictor NAME` picks another for one command.
- Optional features: the conversation before the prompt (on by default), the skills the agent used, and public facts about each model from OpenRouter's model list.
- Plugins that add your own features or model types.
- Advanced settings for what to forecast, the curve's shape, the loss, dropout, the data split and how fast old tasks count less.
- Support for macOS, Linux and Windows, with Python 3.12 or newer.

[0.2.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yusufcemaydin/yardstick-predict/releases/tag/v0.1.0
