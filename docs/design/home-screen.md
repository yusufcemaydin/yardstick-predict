# Home screen and menus

This is the agreed design for yardstick-predict's terminal screens. Cem approved it on 24 October 2026. Nothing here is built yet. Build it on this branch.

## Why

Today a bare `yardstick-predict` refreshes every predictor at once. The first thing it shows is a bar like `0 KB / 6.2 GB` with no label. That number is the size of your Claude Code and Codex records that it is about to read, but it looks like a download. It then asks "Refresh now?" once per predictor and prints a long test report for every AI model. Cem called this unacceptable. He wants a greeting and clear choices, the way the Claude Code, Codex and agy commands work: hint texts, clearly labelled options and nothing heavy before you choose.

The label disappears because `ui.working` prints "Looking for new tasks" and `ui.progress` then starts its line with `\r\033[2K`, which clears it.

## Words

- Every screen calls a predictor a **model**, as Cem does. The models a forecast is about, such as claude-opus-5-5, stay **AI models**.
- Models show the names you gave them. The default model has `(default)` after its name in every list.
- Command names and every `--json` field stay as they are, so scripts keep working.

## 1. The home screen

It appears only when you run `yardstick-predict` with nothing after it and both input and output are a terminal. It runs nothing heavy. It only reads the models' settings.

```
╭──────────────────────────────────────────────╮
│ ◆ yardstick-predict 0.4.0                    │
│   Forecasts a coding-agent task's time and   │
│   cost before it runs                        │
│                                              │
│   main (default)  updated 12 hours ago       │
│   llm             updated 12 hours ago       │
╰──────────────────────────────────────────────╯

  What would you like to do?

  › 1. Estimate  Forecast a task's time and cost
    2. Update    Learn from your newest work
    3. Models    Update, rename or pick the default
    4. Add New   Make another model
    5. Quit

  ↑↓ move · enter choose · esc quit
  Every command: yardstick-predict --help
```

- There is no Status choice. The `status` command stays for anyone who wants the long report.
- When a choice finishes, you come back to this screen.
- With no model yet, the header says so and Add New comes first, with one line on what setup needs.
- Draw the box with Unicode when the terminal supports it and with plain characters otherwise. Leave the box out under 50 columns.

## 2. How menus work

- Arrow keys and Enter, number keys jump to a choice, and Esc goes back or quits.
- Build it with the standard library only: `termios` and `tty` on macOS and Linux, `msvcrt` on Windows. No new dependency.
- When a terminal cannot pass single key presses, fall back to typing the number.

## 3. Estimate

```
  Estimate  Forecast a task before you run it

  Describe the task, as you'd ask a coding agent.
  For example: add a dark mode toggle to the settings page
  › _
```

Use plain `input()`. Then show today's forecast table (`show_forecasts`) for the default model and return home.

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

1. **Where the new work is, at once.** From file sizes only (`record_files` and `families`, no reading), show each archive's folder, how many chats changed and how many GB that means to read, or "nothing new". When nothing changed, say "Nothing new since <time>" and go back without reading.
2. **Reading, with a labelled bar.**
   ```
     New work since 12 hours ago
       Claude Code  ~/.claude/projects
                    86 chats changed · 6.2 GB to read
       Codex        ~/.codex/sessions
                    nothing new

     Reading ━━━━━━━━━━────── 2.4 of 6.2 GB · 40 s left
     Free, and your chats never leave this computer.
     Esc stops it.
   ```
3. **Every cost and time, then one question.**
   ```
     Found 44 new tasks and 28 changed ones,
     all in Claude Code

     main  Embed 25 texts with Voyage
           $0.00 · 10,173 tokens
           Train again · under 1 min
     llm   Analyze 21 tasks with Codex
           your ChatGPT plan · about 2 requests
           Embed 23 texts with Voyage
           $0.00 · about 1,500 tokens
           Train again · about 1 min

     About 3 min in all. Their data grows to 256 MB.

     › 1. Update all  $0.00 · about 3 min
       2. Cancel
   ```
   The summary must count every cost before it asks. Today one is missing. After LLM analysis, `_refresh` in cli.py embeds the descriptions the analysis wrote (`rest = plan_embeddings(predictor, key)`) without having priced them. Estimate those tokens before the question.
4. **The work, each step with a labelled bar.** Then one line per model and back home:
   ```
     ✓ main updated · forecasts on for claude-opus-5-5 and claude-fable-5
     ✓ llm updated · forecasts on for claude-opus-5-5 and claude-fable-5
   ```
   Add one line for each AI model whose forecasts turned on or off, with the reason. Do not print the long test report here.

Cancel or Esc spends nothing and saves nothing. A second Update in the same session reuses what the first one read.

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
  Learns from: conversation, prompt text, LLM analysis

  › 1. Update        Learn from your newest work
    2. Make default  Use it when you don't pick one
    3. Rename        Give it another name
    4. Back
```

- **Update** is the same flow as choosing that model in the Update menu.
- **Make default** shows only on a model that is not the default already. It confirms: "llm is now the default. Estimate and Claude Code's status line use it."
- **Rename** asks for a new name (with the rules in `name_problem`). It renames the model's folder under `~/.yardstick/predictors` with the locks the code already uses. A default model stays the default, including when the old name was `default` and `default_name` only found it by that name.
- Add a `rename OLD NEW` command, so scripts can do the same.

## 6. Other changes

- Setup suggests the name `main` for the first model instead of `default`, so a name never clashes with the `(default)` mark.
- `ui.progress` keeps the current step's label everywhere, including in setup's scan.
- Scripts keep working. A bare run outside a terminal prints one line per model as today. Every command and `--json` output stays the same. `refresh NAME` in a terminal uses the new summary and question, and `--yes` still asks nothing.

## How to build it

- Move the new screens out of cli.py (about 2,700 lines) into small modules, for example `screens/home.py`, `screens/update.py`, `screens/models.py` and a key reading module. Keep each new file under about 400 lines with one clear purpose.
- Write the tests first:
  - the menu with an injected key reader (arrows, numbers, Enter, Esc and the typed fallback)
  - the text of each screen
  - the update summary appearing before any Voyage or LLM call
  - Cancel spending nothing
  - Rename keeping the default
  - unchanged output outside a terminal
  - `refresh --yes` asking nothing
- CI runs on Linux and Windows, so guard the `termios` import.
- Text in screens, docs, the changelog and commits uses plain words, with no dashes, no semicolons and no comma right before and, but, or, so or yet.
- Update the start of README.md and add an Unreleased entry to CHANGELOG.md.
