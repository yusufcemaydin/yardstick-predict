# Contributing to yardstick-predict

Thanks for helping. This guide says how to report problems and how to send changes.

## Set up and run the tests

You need [uv](https://docs.astral.sh/uv/). It installs the right Python and every dependency.

```bash
git clone https://github.com/yusufcemaydin/yardstick-predict
cd yardstick-predict
uv sync
uv run pytest
```

The tests never touch your own records, your Voyage key or your password store. They write their own small Claude Code and Codex records and use a stand-in for Voyage that runs on your computer.

On a Mac without Homebrew's OpenMP, the tests need to find the copy that scikit-learn carries. Run them the way the automatic tests do:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(uv run python -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
uv run pytest
```

New features and model types can live in a plugin, outside this repo. The front page's "Extend it" section shows how.

Before you send a change, check the style too:

```bash
uv run ruff check src tests
uv run ruff format src tests
```

[How it works](docs/how-it-works.md) explains the data, the model and the test. A change to the model must keep passing the test on real runs, not just the unit tests. Say in your pull request how it did.

## Report a problem or suggest an idea

Open an issue and pick a form:

- **Bug report:** something broke or gave an error.
- **Forecast was way off:** the predicted time or tokens were far from what really happened.
- **Idea or request:** a new feature or a change.

Search the open issues first. Someone may have reported it already.

Security problems are different. Report them privately, as the [security policy](SECURITY.md) explains.

## Send a change

1. For anything bigger than a small fix, open an issue first. Agreeing on the idea first saves you wasted work.
2. Fork the repo and make your change on a new branch.
3. Keep each pull request to one change. Small pull requests get reviewed faster.
4. Make sure the tests pass.
5. Open a pull request. Say what it changes and why.

A maintainer reads every pull request before it's merged.

## Changes that need extra care

These changes get a much closer review, and may be turned down:

- install steps or build scripts,
- the automatic checks in the `.github` folder,
- new dependencies,
- model files.

Model files must use the safetensors format, which holds only numbers. Pull requests that add pickle files will be closed. These usually end in `.pkl`, `.pt` or `.pth`, and they can run code when someone opens them.

Code that's minified, obfuscated or generated without its source will be closed too. Every line has to be readable.

## AI-written changes

You can use AI tools to write your change. You're still responsible for it. Read it, test it, and be ready to explain it.

## License

By sending a change, you agree that it's released under the [Apache License 2.0](LICENSE), like the rest of the project.
