# Changelog

Every notable change to yardstick-predict is listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/).

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

[0.1.0]: https://github.com/yusufcemaydin/yardstick-predict/releases/tag/v0.1.0
