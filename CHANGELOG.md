# Changelog

Every notable change to yardstick-predict is listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-10-22

Records are read more accurately again, and this time your predictors catch up by themselves: the next refresh reads all your records again, once, and embeds only the texts that changed. Training is several times faster, and more so on computers with many processor cores.

### Fixed

- Reading records:
  - A Claude Code helper's tokens and cost were never added to the task that started it, because the reader looked for a mark that helper replies don't have. Helpers that a workflow starts were missed too, and their records were read as if they were sessions.
  - A task whose helper was still running could be learned from with only part of its tokens and cost. It now waits for a later refresh. A helper that goes quiet without reporting back leaves its task stopped.
  - A scheduled task, a message from another session or a headless run wasn't a task of its own. Its work was added to the task before it. Each is now a task, started by a program.
  - Work the agent did after a background job reported back was counted nowhere. It now counts toward the task that started the job.
  - A person's `!` shell command that the agent then worked on was left out. It's now a task.
  - A copy of a session that had lost a line could join two tasks into one and count the second one twice. A session and its copies are now read together, line by line.
  - A refresh never learned that a saved task had stopped, such as on a usage limit, when no new model call came with it. A task read again from its own record now replaces its saved copy. A saved task its record no longer holds is taken out.
  - An approved tool's whole run was taken out of the task's time as a wait for you. When the tool's result says how long it ran, only the wait before that is taken out now.
  - A task whose last tool call never got a result was learned from as finished. It's now stopped.
  - Codex: a turn cut short by the next one was dropped. It's now kept, stopped. Helpers that review approvals were dropped, and a helper started by a helper could lose both their turns.
  - A record with an empty text, or one deleted while it was read, stopped the whole read.
  - Big records took more memory than they needed, and each refresh looked through the record folders twice.
- The model:
  - Training used every processor core on small jobs, where the threads cost more time than they saved. On a computer with many cores, it's now about ten times faster.
  - Weibull curves started from numbers meant for log-normal ones, so their expected values came out far too low.
  - The newest training tasks, held back to decide when to stop adding trees, were also used to build the starting curves those trees were judged against.
  - A model with many old tasks and few new ones got a starting curve that was too narrow.
  - With `--loss crps`, curve widths learned at half speed, and the width correction was picked by another score.
  - The simple rule's narrowest curve was twice as wide as the model's, so on values that barely varied the model could pass on that alone.
  - With dropout, the trees kept were not quite the ones early stopping picked.
  - Weibull with expected values only, and unknown settings, were accepted and then ignored. They're now refused.
  - "Tested on" counted test windows that never ran.
  - A forecast searched for similar tasks once for each AI model and effort level, not once for each prompt.
  - A forecast could read a model while a refresh replaced it, and mix the old model's settings with the new one's trees.
- Embedding with Voyage:
  - On an account with low limits, one rate limit made every later request hold a single text, and a request limit made it send more requests, not fewer. A big setup could take hours. It now waits for the limit to come back, as Voyage asks.
  - The same text in many tasks, such as "continue", was paid for once per task. A prompt the status line forecast was paid for again at the next refresh.
  - A vector made from an older text of a task was kept after the text changed.
  - A request with several texts gave up after a few seconds of server errors.
  - When Voyage's tokenizer couldn't be downloaded, every count waited about 23 seconds again, and every forecast loaded it.
  - A forecast with Voyage out of reach waited about two minutes before it said so. It now says so within seconds.
  - A damaged record of your Voyage usage could offer the 200 million free tokens again. It's now set aside, and costs assume no free tokens.
  - A redirect could send your key to another address. A key with a hidden character in it, or an odd reply, ended in a traceback.
- The command:
  - A program could get a traceback instead of a JSON answer, for example when a plugin was missing or a predictor's settings were broken.
  - Saving a setting, a predictor or the benchmarks wrote the file in place. A crash halfway left it empty, and a reader at the wrong moment saw half of it. Saves now happen in one step.
  - The README's plugin example didn't work with the installed command.
  - Code in the current folder, such as a file called `json.py`, could run in place of Python's own when the command restarted itself on a Mac.
  - A broken XGBoost stopped commands that don't need it, such as `status`, `default` and `claude-code remove`.
  - `--yes` still asked questions in a terminal.
  - After a failed setup, the advice to finish it refreshed your default predictor, not the new one. A first predictor whose setup failed never became the default.
  - A plain `predict` when no AI model had passed the test gave an error, not each model's reason. A repeated `--model` gave repeated forecasts.
  - A record file that was only touched was read again on every refresh.
  - The guided refresh stopped at the first predictor that was busy.
  - Ctrl+D at a question ended in a traceback.
  - Not-a-number and infinite values were accepted for number flags, and so was a negative `--half-life`.
  - Requests the schema refused were accepted, and a prompt of only spaces was refused by one and accepted by the other.
  - `status` loaded every whole model just to print its report.
  - A predictor name could end in a line break, or be a name Windows keeps for a device.
  - The help said the default features were the prompt alone. They're the prompt and the context.
- In Claude Code's status line:
  - Its commands let Python load code from the folder you work in, so a project's own file called `json.py` ran in place of Python's own. The commands now start Python so that it never looks there, and the package keeps that folder out for the older commands too. Run `yardstick-predict claude-code install` again to get the new ones: until you do, the status line asks you to.
  - The clock stopped at the first stop, even when another hook kept the agent working.
  - A background job reporting back, a scheduled task or a message from another session started a new task and a new forecast. A job's report now continues its task. Scheduled tasks and messages from other sessions are forecast as a program's prompts, as training counts them.
  - "Took" counted the time spent waiting for your answers and approvals. It's now the agent's working time, as training counts it.
  - Removing it didn't put back the status line it replaced, and the copy of your settings kept next to them was only made the first time.
  - A prompt with half an emoji in it broke the hook, and the line showed "forecasting…" for good.
  - Prompts whose forecast never started piled up, and nothing was cleaned up after removing it.
  - A locked password store kept each forecast waiting, and each new prompt started another one. Now one forecast runs at a time, and it waits a few seconds at most.
  - Scripted runs, such as `claude -p`, made forecasts no one saw.
  - If you set `YARDSTICK_HOME`, the status line looked for its forecasts in the usual folder. Installing now puts your folder into its commands.
- Prices: Claude Sonnet 4 and 4.5 cost more above 200,000 input tokens, and five Codex models had no price.

### Changed

- The `skills` feature now uses only the skills known before a task runs: the ones its prompt invokes and the ones the conversation used before it. The skills a task went on to use can't be known when it's forecast, so learning from them made the test look better than real forecasts are. A predictor with this feature embeds its skills again once. The status line's forecasts send them too.
- Installing the status line warns when your Voyage key is only in the environment, where Claude Code may not see it, and offers to keep it in your password store.
- Each AI model is tested on its own newest tasks: 30% of them, and at least 30. Before, the test used the newest 30% of all tasks, so a burst of tasks on one model could push another model out of the test, and cost it its forecasts. An AI model now needs 60 tasks of its own to be tested.
- An effort level that an AI model never ran at gets no forecast, with the reason `not_enough_data`. A request with no effort level gets the level the model ran at most often lately.
- Each effort level gets its own starting curve, pulled toward its AI model's.
- A free AI model's cost is `0`.
- `--plugin` also takes the path of a `.py` file. A predictor remembers where it found each plugin, so forecasts from other folders load it too.
- Errors that stop a command go to standard error.

### Added

- A cache of embeddings in the yardstick folder. It holds each text's vector and a one-way hash of the text, never the text.
- New error codes in the JSON answer: `no_default_predictor`, `broken_predictor`, `plugin_failed`, `model_too_new` and `internal_error`.
- Training says when one of your own features had no value for any past task, as when it reads only a request's `extra`.
- The yardstick folder is made private on Windows too.

## [0.2.1] - 2026-10-22

Records are read more accurately. A predictor made with an older version keeps the tasks it already read. To re-read all of yours, create a new predictor with `yardstick-predict create NAME`, then make it the default with `yardstick-predict default NAME`.

### Fixed

- Reading records:
  - The test before any forecast could learn from a task that hadn't finished yet, when the task had waited for you or its helpers worked on. Each task now keeps when it really finished.
  - A Claude Code task that ended on an error, such as a server error or a usage limit, was learned from as if it had finished. It's now stopped, unless a reply came after the error.
  - A Codex helper's work was charged to the wrong task when it went on after your next prompt. It now counts toward the task Codex names for it, even when it was started by another helper.
  - A Codex helper's work counted twice when its record file was copied, and was partly lost when it was spread over several files.
- A forecast asked for while a refresh saved a new model, such as the status line's, could find no model. It now waits a moment for the new one.
- Two refreshes of one predictor could run at once, and the slower one could save an older model over a newer one. A refresh started while another refresh or setup is running now stops and says to try again later.
- Training crashed with a short `--half-life` when an AI model was used only long ago. Old tasks now keep a tiny weight.
- An unknown `--model-type` was refused only after the tasks were read and embedded. It's now refused before anything is read or spent.
- Guided setup offered to embed only the tasks your free tokens cover, even when that was fewer than the 30 training needs. It now offers that only when enough tasks fit.
- `predict "TASK"` sent the prompt as a program's. It's now sent as a person's. A JSON request still says who wrote each task.
- Voyage: after the account's tokens-a-minute limit split a batch, later requests could still be too big, so each was refused and split again. They now hold no more than what got through after the split.
- In Claude Code's status line:
  - A slow forecast could still, rarely, replace a newer prompt's forecast. Each prompt's forecast now has its own file.
  - An error notice from Claude Code, or a helper's reply, could decide which model and effort level's forecast was shown. An unusual line in the record could crash it.
  - Pressing Esc wasn't noticed after an error notice or a helper's reply. A tool's output that mentioned a stop, or a helper being stopped, could stop the clock.

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

[0.3.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yusufcemaydin/yardstick-predict/releases/tag/v0.1.0
