# Home screen and menus

This is the design of yardstick-predict's terminal screens. It was agreed on 24 October 2026 and built for 0.5.0.

## Why

Before 0.5.0, a bare `yardstick-predict` refreshed every predictor at once. The first thing it showed was a bar like `0 KB / 6.2 GB` with no label. That number was the size of your Claude Code and Codex records that it was about to read. It looked like a download. It then asked "Refresh now?" once per predictor and printed a long test report for every AI model. The screens replace that with a greeting and clear choices, the way other coding agents' commands work: hint texts, clearly labelled options and nothing heavy before you choose.

## Words

- Every screen calls a predictor a **model**, as people do. The models a forecast is about, such as claude-opus-5-5, stay **AI models**. Add New calls it a model too, from its first question to its last line.
- Screens never talk about the test. These show **not enough data**: an AI model whose forecasts were often wrong on your past tasks, one with too few tasks to check and one that ran fewer than 30 tasks at an effort level. One never used shows **no data**. One with enough data needs no word.
- The **default** effort level is what the records call a task run without one set. No one picks it. Lists leave it out when the AI model ran at a named level too. An AI model that ran only at the default level shows with no level.
- A long list numbers its choices on from one page to the next. Two digits typed quickly pick a number above 9.
- Models show the names you gave them. The default model has `(default)` after its name in every list. The first model is called `main`, from setup or from `create --express`. So no name looks like the `(default)` mark.
- Command names and every `--json` field stay as they are. Scripts keep working.

## 1. The home screen

It appears only when you run `yardstick-predict` with nothing after it and both input and output are a terminal. It runs nothing heavy. It only reads the models' settings.

```
╭────────────────────────────────────────────╮
│ ◆ yardstick-predict 0.5.0                  │
│   Forecasts a coding-agent task's time and │
│   cost before it runs                      │
│                                            │
│   Your models                              │
│   main (default)  updated 12 hours ago     │
│   llm             updated 12 hours ago     │
╰────────────────────────────────────────────╯

  yardstick-predict 0.6.0 is out. You have 0.5.0.
  Update with: uv tool upgrade yardstick-predict

  What would you like to do?

  › 1. Estimate  Forecast a task's time and cost
    2. Update    Learn from your newest work
    3. Models    Update, rename, delete or pick the default
    4. Add New   Make another model
    5. Quit

  ↑↓ move · enter choose · esc quit
  Every command: yardstick-predict --help
```

- There is no Status choice. The `status` command stays for anyone who wants the long report.
- When a choice finishes, you come back to this screen.
- The box is as wide as its longest line. It names 5 models at most, the default first, then says how many more Models lists.
- With no model yet, the box says No models yet and Add New comes first, with one line on what setup needs.
- When a newer version is out on PyPI, a line under the box says so, with the command that updates it. PyPI is asked at most once a day, in the background. The screen never waits for it. Its answer is kept in `~/.yardstick/update-check.json` and shows from the next screen on. It names the update command for the way yardstick-predict was installed: uv, pipx or pip. `YARDSTICK_NO_UPDATE_CHECK`, `DO_NOT_TRACK` or `"update_check": false` in `settings.json` turns it off.
- The box is drawn with Unicode when the terminal can show it and with plain characters otherwise. It's left out under 50 columns.

## 2. How menus work

- The arrow keys move and Enter chooses. A number key jumps to a choice. Esc goes back or quits.
- Keys pressed quickly each count, such as an arrow and then Enter.
- A menu of more than 9 choices shows 6 at a time. The left and right arrows turn its page.
- Every line fits the terminal, at 80 columns and at 45. A long line or note is broken onto more lines. A command in it always stays whole, to be copied as it is. A table too wide for the terminal breaks its last column under itself. A long name or hint in a menu is cut short, as a menu redraws in place and a line that wrapped on its own would break that.
- A progress bar fits too. In a narrow terminal its bar gets shorter, the time left goes first and then the step's own words are cut short. The counts always show.
- A terminal too narrow for the forecast table gets each forecast as a block: the AI model with its mark and effort level, then its time, tokens and cost, each with its two ranges beside it, one under the other. The `list` command does the same with a few lines for each model.
- It's built with the standard library only: `termios` and `tty` on macOS and Linux, `msvcrt` on Windows. No new dependency.
- When a terminal can't pass single key presses, you type the number instead. So do a dumb terminal (`TERM=dumb`) and an old Windows console where the codes that move the cursor can't be turned on.
- Numbered menus can be asked for, as a screen reader needs: `YARDSTICK_NUMBERED_MENUS=1` or `"numbered_menus": true` in `settings.json`. Every menu is typed into. Enter on its own goes back from a list of AI models. Nothing redraws itself: no progress bar, no box on the home screen. Esc can't stop a long step then. Ctrl+C still does.

## 3. Estimate

```
  Estimate  Forecast a task before you run it

  Describe the task, as you'd ask a coding agent.
  For example: add a dark mode toggle to the settings page
  › _
```

It uses plain `input()`. Enter on its own goes back home. With more than one model, it asks next which of your models should forecast it. The list starts on the one used last, else on the default. With one model it asks nothing. A model that can't forecast yet says it needs training. Then the list comes back. Then, step by step:

```
  Which AI models should it forecast?

  › 1. AI models with enough data  claude-opus-5-5 and claude-fable-5
    2. Choose from a list  every one you've used, 15 in all
    3. Back
```

Each choice opens a list of AI models at their effort levels, 6 to a page, with nothing ticked. Enter forecasts the line you're on. A number picks its line at once, as in every menu. To forecast several, tick them with space first. Enter then forecasts the ticked ones. A number ticks too. The left and right arrows turn the page. Under the full list, it names the AI models with too few tasks to forecast. When no AI model has enough data yet, the first choice says so and the list is still there. A model whose AI models all ran too few tasks says how many each ran.

```
  › 1. ○ claude-opus-5-5   max
    2. ○ claude-opus-5-5   high     ⚠ not enough data
    3. ○ claude-opus-5     max      ⚠ not enough data

  Page 1 of 3
  ↑↓ move · enter forecast this one · space tick several · ←→ page · esc go back
```

The forecasts show 6 to a page, under the task as a title of 3 lines at most. Under each forecast are two ranges, named in the effort column. Half the time the real result lands in the 1 in 2 range. 9 times out of 10 it lands in the 9 in 10 range. A rough one is marked ⚠. The lines under the table say why. Then it asks what's next, never going home by itself: Next page, Previous page, Estimate another task, Choose other AI models for the same task, Use another of your models for the same task or Back to home. Use another of your models shows only when you have more than one.

## 4. Update

Update opens its own menu. Update all comes first, then each model. With only one model the menu is skipped. Choosing one model updates only that model.

```
  Update

  › 1. Update all  main and llm
    2. main (default)  updated 12 hours ago
    3. llm             updated 12 hours ago
    4. Back
```

Then, in this order:

1. **Where the new work is, at once.** From file sizes only, with no reading, it shows each archive's folder, how many chats changed and how much that means to read. Or it says "nothing new". When nothing changed, it says "Nothing new since <time>" and goes back without reading.
2. **Reading, with a labelled bar.**
   ```
     New work since 12 hours ago
       Claude Code  ~/.claude/projects
                    86 chats changed · 6.2 GB to read
       Codex        ~/.codex/sessions
                    nothing new

     Free. Your chats never leave this computer. Esc stops it.
     Reading  ━━━━━━━━━━──────  2.4 GB / 6.2 GB · about 1 min left
   ```
3. **Every cost and time, then one question.** A Voyage key is asked for first, only when texts go to Voyage. The costs keep to a spending limit given with `refresh --max-cost`.
   ```
     Found 44 new tasks and 28 changed ones,
     all in Claude Code

     main  Embed 25 texts with Voyage
           $0.00 · 10,173 tokens
           Train again · under a minute
     llm   Analyze 21 tasks with Codex
           your ChatGPT plan · about 2 requests
           Embed 23 texts with Voyage
           $0.00 · about 1,500 tokens
           Train again · 1 min

     About 3 min in all. Their data grows to 256 MB.

     › 1. Update all  $0.00 · about 3 min
       2. Cancel
   ```
   The summary counts every cost before it asks, the descriptions LLM analysis writes too. Its data counts what the feature store holds for the model already. The figure never goes down from one update to the next.
4. **The work, each step with a labelled bar.** Then one line per model and back home:
   ```
     ✓ main updated · forecasts on for claude-opus-5-5 and claude-fable-5
     ✓ llm updated · forecasts on for claude-opus-5-5 and claude-fable-5
   ```
   One line follows for each AI model whose forecasts turned on or off, with the reason. The long test report isn't printed here.

Cancel or Esc spends nothing and saves nothing. A second Update in the same session reuses what the first one read. When Voyage can't be reached, Update all says so once and stops embedding for the models after it.

## 5. Models

```
  Models

  › 1. main (default)  updated 12 hours ago
    2. llm             updated 12 hours ago
    3. Back
```

Choosing one opens its screen:

```
  llm
  5,454 tasks · updated 12 hours ago
  Forecasts on for claude-opus-5-5 and claude-fable-5
  Learns from: prompt text, conversation, LLM analysis

  › 1. Update        Learn from your newest work
    2. Make default  Use it when you don't pick one
    3. Rename        Give it another name
    4. Delete        Remove it for good
    5. Back
```

- **Update** is the same flow as choosing that model in the Update menu.
- **Make default** shows only on a model that is not the default already. It confirms: "llm is now the default. Estimate and Claude Code's status line use it."
- **Rename** asks for a new name, with the same rules as setup. It renames the model's folder under `~/.yardstick/predictors` with the same locks as every other change. A default model stays the default, also when the old name was `default` and it was the default only because of that name. A model whose settings can't be read isn't renamed. Renaming a link to another model's folder leaves that folder as it was.
- The `rename OLD NEW` command does the same for scripts.
- **Delete** says exactly what goes and what stays, as `delete NAME` does: the model's folder goes. Your records and the feature store stay. Then it asks, with Keep it first. Enter or Esc keeps the model. A model that another command is setting up or updating isn't deleted. After the default is deleted, it says which model is the default now. When that's your only model, it says so. With several left and none the default, it says Claude Code's status line shows no forecasts until you choose one.

## 6. Other changes

- Setup suggests the name `main` for the first model instead of `default`. Then no name clashes with the `(default)` mark.
- A progress bar keeps the current step's words in front of it everywhere, setup's reading too.
- Scripts keep working. A bare run outside a terminal prints one line per model, as before. Every command and `--json` output stays the same. `refresh NAME` in a terminal uses the new summary and question. With `--yes` it asks nothing.

## How it's built

- The screens live in small modules under `screens`, each with one purpose: `home.py`, `estimate.py`, `update.py`, `summary.py`, `models.py`, `listing.py`, `version.py`, `checklist.py` and `keys.py`, which reads single key presses and draws menus. How lines fit the terminal is in `ui.py`.
- The tests cover the menus with a stand-in key reader (arrows, numbers, Enter, Esc and the typed fallback), keys read from a pseudo terminal, the text of each screen at 80 and at 45 columns, the update summary coming before any Voyage or LLM call, Cancel spending nothing, Rename keeping the default, the output outside a terminal staying the same and `refresh --yes` asking nothing.
- CI runs on Linux and Windows. The `termios` import is guarded for Windows.
- Text in screens, docs, the change log and commits uses plain words, with no dashes, no semicolons and no comma right before and, but, or, so or yet.
