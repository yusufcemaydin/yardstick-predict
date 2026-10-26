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

## Keep private data out

The repository is public, so the tests include a privacy check, `tests/test_privacy.py`. It reads every file git tracks, except binary files and `uv.lock`, and fails when one holds:

- an email address. GitHub's no-reply addresses, which reach no one, and the example domains `example.com`, `example.org` and `example.net` are fine.
- a home folder with a real-looking user name, such as `/Users/<name>`, `/home/<name>` or `C:\Users\<name>`. The few made-up names the tests use are fine.
- text shaped like a secret key: an Anthropic, OpenAI, Voyage, GitHub or AWS key, or a private key file.

It names the file and line of each find, never the text itself, so its output is safe to share. It reads files only, not commit messages.

Some words must stay private but can't be listed in a public file, such as the names of people, clients or other projects. Keep them in a file on your own computer, outside the repository, one a line. Case doesn't matter, and blank lines and lines that start with `#` are skipped. When the environment variable `YARDSTICK_PRIVATE_TERMS` names that file, the check fails on those words too:

```bash
YARDSTICK_PRIVATE_TERMS=~/private-terms.txt uv run pytest tests/test_privacy.py
```

To run it before every push, give git a pre-push hook: a script git runs before each push, which stops the push when the script fails. Save these two lines as `.git/hooks/pre-push`, then make the file runnable with `chmod +x .git/hooks/pre-push`:

```sh
#!/bin/sh
YARDSTICK_PRIVATE_TERMS="$HOME/private-terms.txt" uv run pytest -q tests/test_privacy.py
```

## Predictors and models saved by past releases

Each version keeps forecasting with the models the previous release saved, until a refresh trains them again. Anything older is asked to train again first. A version that meets a feature it doesn't know skips it, with a note.

The tests check this against what past releases really saved. `tests/fixtures` holds a predictor that 0.3.0's own code made, and what it forecast. When a release changes how predictors or models are saved, add that release's fixture the same way, with tests that this version still reads it. [The fixtures' README](tests/fixtures/README.md) shows how the 0.3.0 one was made. Its model files are plain JSON and plain numbers, as the tool saves them, never pickle files.

## Report a problem or suggest an idea

Open an issue and pick a form:

- **Bug report:** something broke or gave an error.
- **Forecast was way off:** the predicted time or tokens were far from what really happened.
- **Idea or request:** a new feature or a change.

Search the open issues first. Someone may have reported it already.

When yardstick-predict meets an error it didn't expect, it says so in one line. It keeps the details, the Python traceback, in `last-crash.txt` in the yardstick folder, usually `~/.yardstick`. They name the command, never what it was given. Still, read them before you paste them into a report: they may hold text from your work. To see the details where it happens, run the command again with `YARDSTICK_DEBUG=1`.

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

## Releasing

Between releases, the version on the main branch ends in `.dev0`, such as `0.4.0.dev0`, which means work toward 0.4.0. To release it, a maintainer:

1. Sets the version, such as `0.4.0`, in `pyproject.toml` and `src/yardstick_predict/__init__.py`, and runs `uv lock`.
2. In `CHANGELOG.md`, renames `## [Unreleased]` to the version and the day, such as `## [0.4.0] - 2026-11-02`, and starts a new, empty `## [Unreleased]` above it.
3. Merges that change into main once the tests pass.
4. Tags the merged commit with the version and pushes the tag:

   ```bash
   git tag v0.4.0
   git push origin v0.4.0
   ```

The tag starts the release workflow, `.github/workflows/release.yml`. It checks that the tag matches the version in both files, `v0.4.0` for `0.4.0`, and stops if not. Then it builds the package and publishes it on PyPI, the Python package index. Nothing else starts it: not pull requests, and not pushes to a branch.

After the release, set the next version on main, such as `0.5.0.dev0`.

Before the first release, set up PyPI once:

1. Make an account on [PyPI](https://pypi.org) and turn on two-factor sign-in.
2. Tell PyPI to trust this repository's release workflow, so it takes the upload with no password or token kept anywhere. PyPI calls this a trusted publisher. Before the project exists on PyPI, add it as a pending publisher, under Your account > Publishing, with the project name `yardstick-predict`, owner `yusufcemaydin`, repository `yardstick-predict`, workflow `release.yml` and environment `pypi`. The first release then creates the project.
3. On GitHub, under the repository's Settings > Environments, create an environment named `pypi`. To approve each publish yourself, add yourself as a required reviewer. Under deployment branches and tags, allow only tags that match `v*`.

## License

By sending a change, you agree that it's released under the [Apache License 2.0](LICENSE), like the rest of the project.
