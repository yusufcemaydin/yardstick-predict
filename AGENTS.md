# Rules for AI coding agents

This file is for AI coding agents working on yardstick-predict, such as Claude Code and Codex. [CONTRIBUTING.md](CONTRIBUTING.md) says the same for people, at more length.

## Before you change anything

- Set up with `uv sync --locked`. It needs uv 0.11 or newer.
- Read the code you'll change and its tests. The commands are in `src/yardstick_predict/cli.py`. The home screen and its menus are in `src/yardstick_predict/screens/`. [How it works](docs/how-it-works.md) explains the data, the features and the model.
- Programs read the JSON answers. Within a version, their fields are never renamed or removed. [The format](docs/format.md) lists them.

## Before you say a change is done

Run these three and say what they printed:

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

- A change people can see gets a test that shows it.
- A screen must fit the terminal at any width. Test it at 80 and at 45 columns, as `tests/test_screens.py` does.
- Tests never call a real API. They use the stand-in for Voyage in `tests/conftest.py` and stand-ins for the AI models.
- Add a line to `CHANGELOG.md`, under Unreleased, for anything a user would notice.

## Writing

These rules cover everything a person reads: screens, messages, docs, the change log and commit messages.

- Use plain words. Say what a thing does before you name it. Explain a term the first time you use it.
- Put one idea in each sentence. Keep sentences short, with concrete nouns and the active voice.
- Use no dashes and no semicolons. Put no comma right before and, but, or, so or yet. Start a new sentence instead.
- On a screen, never say test, tested, passed or failed. Say not enough data. For an AI model never used, say no data.
- Never offer the default effort level on a screen. No one picks it.
- A long list numbers its choices on from one page to the next.
- Every screen, message, doc and command's help calls a predictor a model. The models a forecast is about, such as claude-opus-5-5, are AI models. The `--predictor` option, the JSON names and the error codes keep the word predictor, for the programs and scripts that use them.

## Keep private data out

- Never put an email address, a person's name or anything from someone's own records in the repository. That includes numbers worked out from their records. `tests/test_privacy.py` checks for some of this.
- Never write a real key or token into a file. Voyage's key lives in `VOYAGE_API_KEY` or the computer's password store.
- Text from someone's records goes to Voyage or an AI model only through `privacy.hide_secrets`. Hide it before you cut it to length.

## Commits and pull requests

- Commit only when you're asked to. Never push without being asked.
- Commit with a GitHub no-reply address, never a personal or work email.
- Add no Co-Authored-By line for an AI tool to a commit or a pull request. Add no "Generated with" line either.
