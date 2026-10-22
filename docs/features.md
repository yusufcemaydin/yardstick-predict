# What's next

These build on the first version. Each one goes in only if it passes the same test as everything else: it must make forecasts better on real runs, without making them less honest.

## Forecasts

- **Models you've never used.** The benchmarks feature already describes every model the same way. Next, pretend each known model is new, forecast it from the others only, and give forecasts for new models only if that check passes.
- **Learn from stopped tasks.** A stopped task took at least as long as it ran. The score can use that instead of leaving the task out.
- **Draw time, tokens and cost together.** A task that runs long usually uses more tokens too. Give orchestrators possible futures where the three move together.
- **More benchmark sources.** Add public quality and speed benchmarks that describe old and new models the same way.
- **An LLM's read of the task.** Let a language model write a short analysis of the task, and use what similar analyses took.
- **Subscription usage limits.** Forecast a task's share of a plan's usage window, not only its dollar cost.

## Data

- **More agents.** Read the records of other coding agents, such as Gemini CLI, OpenCode and Cursor.
- **A simple record format.** Let any program report what a task took, one JSON line per task, so tools that don't use Claude Code or Codex can train a predictor too.
- **Codex approval waits.** Take time spent waiting for a person out of Codex tasks, as for Claude Code.
- **Prices kept current.** Update the price list automatically when providers change their prices.

## Using it

- **A small local server.** Answer forecast requests over HTTP on your own computer, for orchestrators that ask thousands of times.
- **Voyage 4 nano on your computer.** Embed text locally with Voyage's open model, so nothing leaves the computer at all. It shares Voyage 4's embedding space.
- **Other embedding providers.** Let a predictor embed with providers other than Voyage.
- **A forecast with every prompt.** Show the forecast in Claude Code as each prompt is sent.

## Sharing

- **Pooled forecasts.** A hosted service that pools many people's runs, so forecasts improve faster and cover more models. Runs would be shared as noisy totals per group of similar tasks, under a differential privacy guarantee.
