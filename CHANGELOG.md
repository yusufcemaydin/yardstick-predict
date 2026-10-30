# Changelog

Every notable change to yardstick-predict is listed here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.7.0] - 2026-10-30

Keys and passwords in your prompts are hidden before they're sent. One command now removes yardstick-predict.

### Added

- `yardstick-predict uninstall` removes yardstick-predict in one go. First it takes the forecasts out of Claude Code: for all your sessions and for every folder you added them to with `--here`. It also finds folders an earlier version added them to, in Claude Code's own list of the folders it has opened. Then it deletes the Voyage key it kept and the yardstick folder with your models. It says what goes and asks first. If a settings file can't be read or a model is being refreshed, it says so and keeps the rest for the next try. `--keep-data` keeps your models and the key. Last, it shows the command that removes the program itself, the way you installed it. Before, removing yardstick took four commands, plus one more in each folder you'd added it to.

### Fixed

- The front page, the guide, the Update screen and what the command says when run on its own no longer say that only the text you choose leaves your computer. The Update screen no longer says your chats never do. They all say plainly that your prompts go to Voyage. What the command says when run on its own now wraps as one paragraph, with no short line in the middle.
- Small errors in the docs are fixed. The contributing guide points to the guide's Plugins section. The two schemas give web addresses that open. The how-it-works page counts the three parts of a curve. The issue forms no longer offer Cursor, which yardstick-predict doesn't read.

### Security

- Keys and passwords in your prompts are hidden before they're sent. Each part of a text shaped like a key, token or password becomes `[hidden]`. That holds for the prompt and for the end of the conversation before it. It holds for Voyage and, with LLM analysis on, for the AI model that reads them. It covers keys from AI and cloud services, tokens from GitHub, Slack and Stripe, private keys, web tokens, passwords in web addresses and a value given to a name such as `password` or `api_key`. A value in quotes is hidden whole. Keys are hidden before a text is cut to length. So a key the cut splits in two is hidden too. Tasks read from your records from now on keep their prompts with keys hidden. Texts embedded before keep their vectors. Nothing is paid for again.

## [0.6.0] - 2026-10-29

On a Claude subscription, the Claude Code status line shows each task's share of your weekly limit, in place of dollars.

### Added

- On a Claude subscription, the Claude Code status line shows shares of your weekly limit in place of dollars. First comes what the task has used so far, with four decimals. Then the brackets hold the forecast's time and its share of the week: `yardstick · 1:12 · 0.0236% (~4 min · ~0.0789%)`. Claude Code reports how much of the week you've used. yardstick divides that by what your Claude Code sessions on this computer spent this week at list prices. That says how much of the limit a dollar of work uses. Nothing is sent anywhere for it. Chats on claude.ai count toward the limit too. They leave no record here. While you use them, the shares come out a little high. With an API key, the line shows dollars as before.

## [0.5.2] - 2026-10-28

A fix release. Every message, the help and the docs call a predictor a model, as the home screen does.

### Added

- The PyPI page links to the guide and the change log.

### Fixed

- Every message, every command's help and the docs call a predictor a model, as the home screen does. They used to say predictor. Where a message spoke of a predictor's model, it now speaks of the model's training. The status line, for one, says a model isn't trained yet. Where the docs meant an AI model, such as claude-opus-5-5, they now say AI model. The `--predictor` option, the names in a program's JSON and the error codes stay as they were. Programs and scripts that use them keep working.

## [0.5.1] - 2026-10-27

A fix release. The forecast table names each line of a forecast in a column of its own.

### Fixed

- The forecast table names each line of a forecast in a column of its own, after the effort level: expected, then 25%, median and 75%. The effort level shows once, on the expected line. The 25%, median and 75% lines used to be named in the effort column, where they looked like effort levels.

## [0.5.0] - 2026-10-27

A bare `yardstick-predict` in a terminal opens a home screen with clear choices. It no longer refreshes every predictor at once behind a bar with no label.

### Added

- A home screen. Run `yardstick-predict` in a terminal with nothing after it. It greets you, lists your models under Your models with `(default)` after the default one and offers Estimate, Update, Models, Add New and Quit. Move with the arrow keys and press Enter. A choice's number picks it at once. Esc goes back. It reads only your models' settings. It opens at once. The box names 5 models at most, the default first, then says how many more Models lists. A terminal that can't pass single key presses gets the choices numbered, to type one in. So do a dumb terminal and an old Windows console that can't move the cursor. Keys pressed quickly each count, such as an arrow and then Enter.
- The home screen says when a newer version of yardstick-predict is out on PyPI, with the command that updates it the way you installed it: uv, pipx or pip. It looks at most once a day, in the background. It never makes you wait. `YARDSTICK_NO_UPDATE_CHECK`, `DO_NOT_TRACK` or `"update_check": false` in the yardstick folder's `settings.json` turns it off.
- Estimate asks for a task. With more than one model, it asks which of your models should forecast it. Each one says how accurate it is next to the default, such as 12% more accurate than main. Then it asks which AI models to forecast: the ones with enough data or any you've used, from a list. Each opens a list, 6 to a page, numbered on from page to page. Two digits typed quickly pick a number above 9. Enter forecasts the line you're on. To forecast several, tick them with space first. The left and right arrows turn the page. When no AI model has enough data yet, it says so and the full list is still there. The forecasts show 6 to a page too, under the task as a title of 3 lines at most. After them you choose what's next: another page, another task, other AI models for the same task, another of your models for it or home.
- The forecast table in a terminal, in Estimate and in `predict`, has a heading over each column. It shows every AI model you've used, at each effort level you used it at 10 times or more. Each forecast's first line is what to expect on average. Under it are its 25%, median and 75% lines. A quarter of real results come in under the 25% line, half under the median and three quarters under the 75% line. A forecast with not enough data behind it is marked ⚠. The lines under the table name the AI models that have too little. AI models with under 10 tasks are listed by name. The default effort level, which no one picks, isn't shown. A program's JSON is as before: it never gets a forecast with not enough data behind it.
- Update first says where the new work is and how much there is to read, from file sizes alone. It reads with a labelled bar that Esc stops. Then it shows what each model will analyze, embed and train, with every cost and time. A Voyage key is asked for first, only when texts go to Voyage. It asks once. Cancel spends nothing and saves nothing. A second Update in the same session reads nothing again. When Voyage can't be reached, Update all says so once and doesn't try again for the models after it.
- Models shows each model with its tasks, its forecasts and what it learns from. Choose Update, Make default, Rename or Delete. Delete says what goes and what stays. Then it asks, with Keep it first. Your records and the feature store stay.
- Numbered menus on request, for screen readers. Set `YARDSTICK_NUMBERED_MENUS=1` or add `"numbered_menus": true` to `settings.json` in the yardstick folder. Every menu then lists its choices with numbers to type in. Enter on its own goes back from a list of AI models. Nothing on screen redraws itself: there are no progress bars and no box on the home screen.
- `create` and `refresh` answer a program in JSON with `--json`: what was found, what it costs at list price and what was done. They never ask anything then. Without `--yes` they only say what they'd do and what it costs. When they can't go on, the answer is an error with a code, such as `busy` or `embedding_failed`. `refresh` says why it trains the model, in `retrain_reason`. Every command given `--json` answers a wrong command line in JSON too. So does one that needs a library that's missing. [The format](docs/format.md#setup-and-refresh) lists every field. It now describes the JSON answers of `status`, `list`, `delete`, `features` and `check` as well.
- A spending limit for setup and refresh: `--max-cost 1.50` keeps one to $1.50 at list price. It goes ahead only when what it would cost fits, counting every text Voyage gets at the most tokens it can take. While LLM analysis runs, no request starts unless the most it can cost fits in what's left. Every byte of its text counts as a token. Its answer counts at the longest it may be. A request that doesn't fit whole is sent with fewer tasks. Under a limit, a request the AI model declines isn't tried again on another one. Before each step that embeds, it checks again. A step that doesn't fit stops the command, with what was done kept.
- `predict --request FILE` reads the JSON request from a file. A program needs no pipe then. It's easier to allow ahead of time, as a coding agent's settings do. `--request -` reads standard input, as before.
- `yardstick-predict rename OLD NEW` gives a predictor another name. The default stays the default, also when it was the default only because it was called default. A predictor whose settings can't be read isn't renamed. Renaming a link to another predictor's folder leaves that folder as it was.
- On the Features screen, type all to turn on every feature with all its options. The help under the screen says so.
- `AGENTS.md` holds the rules for AI coding agents working on yardstick-predict: how to check a change, how to write what people read and how to keep private data out. Claude Code reads it through `CLAUDE.md`.

### Changed

- `yardstick-predict refresh` in a terminal shows the same summary as Update and asks once. With `--yes` it asks nothing, as before.
- Outside a terminal, a bare `yardstick-predict` prints one line per predictor, as before.
- Setup suggests the name main for your first predictor instead of default. So no name looks like the `(default)` mark. `create --express` with no name makes main too. On the home screen, Add New calls it a model, as every screen does.
- Every line fits the terminal, at 80 columns and at 45. A line or a note too long for it is broken to fit. A command in it stays whole, to be copied as it is. A menu's long names and hints are cut short. So a menu redraws in place. A progress bar gets shorter, then leaves out the time left and cuts its step's words short. Questions, tables, the Features screen and the report after training break to fit too. A menu of more than 9 choices, such as Models with many models, shows 6 at a time. A terminal too narrow for the forecast table gets each forecast as a block of its own, with its time, tokens and cost under its name. `list` gets a few lines for each predictor in a terminal too narrow for its table.
- The report after training, in setup and in `status`, says how close the forecasts came on your newest tasks, in plain words. Not enough data covers an AI model whose forecasts weren't close enough and one with too few tasks to show it. The feature check's report says the same way what helped. `status` lists a predictor's features in a sentence, with no semicolons.
- The Anthropic and OpenAI libraries come with every install. LLM analysis with an API key needs no second install. The `llm` extra is gone. An upgrade of an install that had it may warn that there's no extra named `llm`. That's harmless. `uv tool install --force yardstick-predict` stops it.
- An error yardstick-predict didn't expect is said in one line, with where its details are kept and where to report it. It used to print a page of Python. The details go to `last-crash.txt` in the yardstick folder. `YARDSTICK_DEBUG=1` shows them where it happens too. A program that asked for JSON gets the error as `internal_error`.
- Output sent to a program that stops reading early, such as `head`, ends quietly.
- A new release of a library gets into `uv.lock` only once it has been out for 7 days. The tests and the release builds use only libraries that had that time.
- A security scan checks every locked library against OSV, the public database of known vulnerabilities. It runs on each pull request, each push to main and once a week.
- A release runs the tests on the tagged commit and installs the built wheel to try it before anything is published. The source package holds `CHANGELOG.md` and `AGENTS.md` too.
- On a Mac, a forecast and the status line start faster. XGBoost loads without scikit-learn, which yardstick never uses. scikit-learn stays installed there for its copy of OpenMP.
- Costs under a cent show their first digits, such as $0.0021, instead of $0.00.
- Every progress bar keeps its step's label. Reading your records used to show a bar like `0 KB / 6.2 GB` with no word of what it was.
- The README is a third as long. It says what yardstick-predict does, how to install it and how to start, with a short table of commands. Everything else moved to [the guide](docs/guide.md): the home screen, the features, LLM analysis, costs, settings for experts, plugins, privacy and how to update or remove it.
- After the feature check, setup says which features stay on and which go off, then asks once whether to go on and make the predictor. It used to open the Features screen again, which looked like starting over. Saying no still opens it, to change them first.

### Fixed

- What other services say is cleaned before it's shown. An error from Voyage, Anthropic, OpenAI, Claude Code, Codex, OpenRouter or PyPI used to be printed as it came, hidden codes included. Such codes could make a terminal show something else, change its title or write over a line. The words stay. The codes go.
- The guide's steps for removing everything start by taking the forecasts out of Claude Code. Without that, Claude Code kept running a program that was gone.
- The cost shown before an update now counts embedding the descriptions LLM analysis writes. A refresh used to embed them without having priced them.
- A model with a file missing or damaged no longer breaks every forecast with a request for a bug report. Forecasts say the model is damaged and needs training. The next refresh trains it again, where it used to say the model was up to date.
- One field of an unexpected type in a Codex record no longer stops the whole import. That field is skipped.
- Pressing Ctrl+C while LLM analysis runs stops at once. It used to wait for the requests already sent, which through a subscription can take minutes.
- A damaged entry in the cache of vectors was priced as free, then paid for anyway, every time. It now counts as missing and is replaced.
- A program's setup without `--yes` no longer sends a key check to Voyage before it has agreement to spend. The check comes once it goes ahead. It counts toward a spending limit.
- A refresh no longer prices or analyzes tasks it's about to take out because their records are gone.
- Setup no longer embeds again texts another predictor has in the feature store when the cache of vectors is gone. It counts what the store holds, as a refresh does.
- A refresh asks for a Voyage key only when texts go to Voyage. It warns about a damaged record of Voyage usage before the question, not after.
- A task whose prompt changed is analyzed again by LLM analysis, which kept what it said of the old prompt. Answers from before this version aren't read again. So the upgrade costs nothing.
- A mistyped feature or option with `--features` is refused before any record is read. So are LLM flags without the `llm` feature.
- Setup and the docs said the three features always on send nothing anywhere. For them, each prompt goes to Voyage. They say so now.
- Messages that said to run yardstick-predict to train again now name `yardstick-predict refresh`. A bare `yardstick-predict` opens the home screen now, where Update does it.
- `predict "TASK"` in a terminal says when there's not enough data for any AI model yet, instead of an error meant for programs. An empty task is asked to have some text.
- The first download of Voyage's tokenizer no longer draws its own progress bar over the screen.
- When Voyage can't be reached, the message says so once. It used to show an internal status of 0 and say it twice.
- On Windows, `claude-code install` refuses a Python or yardstick folder whose path has $, `, ", % or ! in it. cmd or Git Bash would read those as more than text when Claude Code runs the hook.
- Two commands changing the settings at the same time no longer lose one change.
- A note too long for one line no longer breaks an AI model's name at a hyphen, such as claude-sonnet-4-6.
- The cost of LLM analysis through a subscription writes its name as it's spelled, such as ChatGPT. It used to say chatgpt.
- Setup told with `--archive` to read a folder of records that isn't there says so at once. Nothing is read or spent. It used to read nothing and then say it found no finished tasks.
- Update and `refresh` say when a model's folder of records isn't there, as when its records moved or its drive isn't plugged in. The model keeps the tasks it read from there. It reads nothing new there until the records are back. Update used to show nothing new for that folder. With `--json`, the warning is in `notes`.

## [0.4.0] - 2026-10-23

Training uses exactly the features you choose, and setup can check which of them really help your forecasts. Embeddings and LLM answers are kept once, in a feature store all your predictors share, so a second predictor never pays for them again. A model trained by 0.3.0 keeps forecasting until you train it again: the next refresh does that, and brings this version's improvements. Releases now come from PyPI, so `uv tool install yardstick-predict` installs it.

### Added

- A feature check, which finds out which of your features really help your forecasts. It tests the model with every feature, then without each one in turn. For each feature and AI model, it says whether the feature helped, didn't help, or whether there wasn't enough data to tell. There isn't enough data when under 20% of the AI model's test tasks have the feature, or under 20% of the earlier tasks the test learns from: a model that never learned from a feature can't show what it does. LLM analysis that read only some of your tasks is judged on those tasks alone, with it and without it. Setup offers it when you pick features beyond the three that are always on. Then the Features screen comes back with the features that helped turned on, for you to keep or change. LLM analysis reads only a sample of your newest tasks before the check, and the rest only if you keep it and agree to the cost. The estimate before setup counts the check's time too.
- `yardstick-predict check NAME` runs the feature check on a predictor you already have. It's advice only: it embeds nothing, asks no AI model and changes nothing about the predictor. `--json` answers programs. The Features screen shows each feature's latest verdicts.
- A feature store, shared by all your predictors, keeps the features that cost money or time to make: embeddings, and what LLM analysis said about each task. Each is made once. A second predictor over the same records pays Voyage for nothing it already has, and LLM analysis asks the same AI model nothing it answered before, even when the two predictors ask for different parts of the answer. The store is in the `features` folder of the yardstick folder.
- `yardstick-predict features` lists every feature of every predictor. A stored one shows what made it, how many tasks have it, its size on disk and when it last changed. The others are worked out at training, for free. Each shows its coverage: the share of the tasks the predictor learns from that have a value for it. `--json` gives the same to programs.
- Files deleted or broken by hand in the feature store never break a command. A missing index is rebuilt from the files, and a file that can't be read is skipped, each with a note. The next refresh makes again what's missing: an embedding comes back from the cache for free, and an LLM answer is asked for again, with the cost shown first.
- A new feature, `time`, off by default: when each task started. Its `trend` option is the number of days since 1 January 2020, so the model can follow how your tasks change over the months. Its `clock` option is the hour of the day and the day of the week, in your computer's time zone. A forecast's task starts the moment you ask. It costs nothing and sends nothing anywhere. Choose it on the Features screen, or with `--features time`, `time:trend` or `time:clock`.
- Your own features get when each task started, as `info.started_at`.
- `yardstick-predict list` lists your predictors, one line each: its features, how many tasks it learned from, when it was last trained and which AI models get forecasts. `--json` gives the same to programs. `status` stays the detailed view, and says the same.
- `yardstick-predict status` shows each feature's coverage: the share of the tasks the predictor can learn from that have a value for it, counted as `yardstick-predict features` counts it. Coverage needs XGBoost, as training does. Without it, status shows the rest.
- Each training saves the prepared dataset, `prepared.parquet` in the predictor's folder: one row for each task the model learned from, with when it started, its AI model, its number features, the hash of each text whose embedding the model read, and what the task took. It never holds a prompt or any other text. `status` shows its size.
- `yardstick-predict delete NAME` deletes a predictor: its settings, its dataset of tasks, its prepared dataset and its model. Your records, the feature store and your other predictors stay. In a terminal it says exactly what goes and asks first. `--yes` skips the question for programs, and `--json` answers in JSON. It refuses while a setup or refresh of that predictor runs, and waits for a forecast that is reading its model. When it was your default predictor, it says how to choose another. A predictor whose settings are broken can be deleted too. Nothing is ever deleted through a link: when a predictor's folder is a link to another folder, only the link goes.
- `yardstick-predict features delete ID` deletes a stored feature. First it says which predictors use it and what making it again costs: an embedding usually comes back from the cache for free, and LLM answers have to be asked for again. It asks before it deletes anything, and `--yes` skips the question for programs. The listing shows each stored feature's id.
- When the Claude Code status line's predictor is deleted, the status line says so in plain words, such as that there's no default predictor now and how to choose one.
- Before setup or a refresh goes ahead, it says about how long the rest will take, as a range such as "about 4 to 8 minutes", and how long LLM analysis, embedding and training each take. It also says how much disk space the dataset will take. With `--yes` it says so once and goes ahead. When training may need more than a quarter of your computer's memory, it says that too.
- Every long step shows a progress bar: reading your records, LLM analysis, embedding and training. After a few seconds, each bar also says about how long is left. Reading and training used to show only a line saying they were working. A model type from a plugin still shows that line while it trains.
- A model trained by 0.3.0 keeps forecasting until you retrain it, exactly as 0.3.0 did. `status` and `refresh` say that the previous version made it and that training it again brings this version's improvements. A model made before 0.3.0 still gives no forecasts until it's trained again.
- Settings saved by a newer version are read as far as this version can, and never saved back in this version's older format. `status` and `predict` say so once. `refresh` leaves such a predictor as it is and asks you to upgrade.
- Releases are published on PyPI, the Python package index. From this release on, `uv tool install yardstick-predict` installs it, and `uv tool upgrade yardstick-predict` updates it. If you installed an earlier version from GitHub, switch once with `uv tool install 'yardstick-predict@latest'`, or `'yardstick-predict[llm]@latest'` if you use LLM analysis with an API key.

### Changed

- The conversation so far is now a feature of its own, `conversation`, on by default. It's what the conversation's earlier tasks took and how big the conversation has grown. With `context`, it also covers how close the prompt is to the end of the conversation, and past tasks found by both together. It costs nothing and sends nothing anywhere. Turn it off on the Features screen, or leave it out of `--features`. A predictor made with an older version keeps it.
- Training makes one model, with the features you chose, and tests it. Before, it made a version with the conversation and one without, four with LLM analysis, and each AI model used whichever did better on the test.
- With LLM analysis, training makes the model with it and the same model without it, for the tasks LLM analysis can't read. Each has its own test, as before.
- The test report no longer says, for each AI model, whether it uses the conversation and LLM analysis. Every AI model uses the features you chose.
- On the Features screen, the end of the conversation is now called Context, after its `context` flag, and History is called Project history.
- Once this version saves a predictor's settings, an older version can't use that predictor: it doesn't know the `conversation` feature.
- A feature this version doesn't know, such as one a newer version added or one from a plugin that no longer adds it, is skipped with a one-line note. Before, forecasts and refreshes stopped with an error. The model still forecasts, without that feature's values. A feature you name with `--features` must still exist.
- A predictor's embeddings and LLM answers move from its dataset into the feature store, once, the first time this version refreshes it or lists its features. The old copies stay in the dataset, unread. Training reads exactly what it read before. Forecasts never read the store, so they're as quick as before.
- An embedding is no longer deleted when its task's text changes or the task is taken out. It stays in the store, for any predictor whose task still has that text. Training uses a task's embedding only when it was made from the task's text as it is now, as before.
- yardstick-predict needs one more library, PyArrow, to read and write the feature store's files.
- A predictor whose settings can't be read says to delete it with `yardstick-predict delete NAME`, in place of deleting its folder by hand.

### Fixed

- LLM analysis through a Claude or ChatGPT subscription that isn't signed in, or whose sign-in expired, says once how to sign in: run `claude` in a terminal and type `/login`, or run `codex login`. It sends no more requests, training goes on without LLM analysis, and the next refresh asks about every task again. Before, every request was sent and failed, and the error showed Claude Code's raw JSON.
- Any other error from Claude Code shows its own words, not its raw JSON.

## [0.3.0] - 2026-10-22

Forecasts learn from much more of what's known before a task starts, and records are read more accurately again. Your predictors catch up by themselves: the next refresh reads all your records again, once, embeds only the texts that changed, and trains the model again. Until then, a predictor made with an older version gives no forecasts. Training tests more versions of the model, so it takes longer: about a minute for a few thousand tasks, and two or three with LLM analysis.

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

- The `skills` feature now uses only the skills known before a task runs: the ones its prompt invokes and the ones the conversation used before it. The skills a task went on to use can't be known when it's forecast, so learning from them made the test look better than real forecasts are. A predictor with this feature embeds its skills again once. The status line's forecasts send them too. Its options can keep only the skills the prompt invokes, or only the ones used before it.
- Installing the status line warns when your Voyage key is only in the environment, where Claude Code may not see it, and offers to keep it in your password store.
- Each AI model is tested on its own newest tasks: 30% of them, and at least 30. Before, the test used the newest 30% of all tasks, so a burst of tasks on one model could push another model out of the test, and cost it its forecasts. An AI model now needs 60 tasks of its own to be tested.
- An effort level that an AI model never ran at gets no forecast, with the reason `not_enough_data`. A request with no effort level gets the level the model ran at most often lately.
- Each effort level gets its own starting curve, pulled toward its AI model's.
- A free AI model's cost is `0`.
- `--plugin` also takes the path of a `.py` file. A predictor remembers where it found each plugin, so forecasts from other folders load it too.
- Errors that stop a command go to standard error.
- `status` lists each predictor's features with their options.

### Added

- Inputs that are always used, cost nothing and send nothing anywhere:
  - Similar past tasks, in more detail: what the closest one took, the closest ten, how much the hundred closest varied, direct answers and real work kept apart, and closer, newer tasks counting more.
  - A first guess from the prompt's whole embedding, which the trees take as one more input.
  - The project's recent tasks, from any conversation.
  - The conversation so far, and how big it has grown.
  - The prompt's layout: list items, file names, links and lines, and whether it asks a question.
  - The effort level as one scale from low to max, so a level with few tasks learns from the levels next to it.
- With the `context` feature, how close the prompt is to the conversation before it, and past tasks found by both together.
- Training makes two versions, with and without the conversation, and each AI model's forecasts use the one that did better on the test. The report says which.
- LLM analysis, a new feature: an AI model reads each task before it starts and reports the kind of task, its expected size and, as an option, a few sentences on the work, which are embedded. It's paid with an Anthropic or OpenAI API key, or with a Claude or ChatGPT subscription through Claude Code or Codex. Each task is read once, never next to another task of its conversation. Training also tests each version without it, and each AI model uses it only when it did better with it. A task it can't read, such as when its AI model can't be reached, is forecast by the version without it. When it can't run, such as with no key, training goes on without it and a refresh tries again; a task it refuses is tried twice at most. The new `llm` extra installs the providers' libraries for the API keys.
- A new Features screen at setup: each feature with its options under it, turned on and off by number. `--features` takes options too, such as `skills:named`, and `--llm-provider` and `--llm-model` set up LLM analysis.
- Forecast requests take `session` and `conversation_tokens`, and use `project` for the project's recent tasks. The status line in Claude Code sends all three.
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

[Unreleased]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yusufcemaydin/yardstick-predict/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yusufcemaydin/yardstick-predict/releases/tag/v0.1.0
