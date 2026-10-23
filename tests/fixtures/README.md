# Fixtures from releases

Each version of yardstick-predict keeps forecasting with the models the previous release saved, until a refresh trains them again. The files here were saved by past releases' own code, so the tests can check that this version still reads them and forecasts with them as that release did. `tests/test_upgrade.py` holds those tests.

## predictor-0.3.0

`predictor-0.3.0` is a small predictor that yardstick-predict 0.3.0 created, analyzed, embedded and trained: its settings, its dataset and its model. 0.3.0 trained three versions of the model, and each AI model uses one of them.

`forecasts-0.3.0.json` holds some forecast requests and what 0.3.0 answered to them. It also holds the moment they were asked, and the vectors and LLM analysis answers the stand-ins gave, so a test can ask this version the same and compare.

Nothing in them comes from anyone's records:

- The 192 tasks are made up. Their prompts are random words, in three made-up projects.
- A stand-in for Voyage gives each text a vector of 16 numbers, from its words. Voyage's have 256. Fewer keep the files small.
- A stand-in plays LLM analysis.
- The records folders the settings name, under `/records`, don't exist. A refresh in a test finds no new tasks.

The model files are the ones 0.3.0 saves: plain JSON and plain numbers in `.npy` files. Loading them runs no code. All of it takes about 0.7 MB.

### How it was made

From the repository's top folder:

```bash
git worktree add --detach ../yardstick-0.3.0 v0.3.0
uv run --frozen --project ../yardstick-0.3.0 python -I tests/fixtures/make_predictor_0_3_0.py
git worktree remove ../yardstick-0.3.0
```

1. The first line checks out 0.3.0 in a folder of its own, next to the repository's.
2. The second runs `make_predictor_0_3_0.py` with 0.3.0's code. `uv` installs 0.3.0 and the exact libraries it locked, in that folder, first. `-I` keeps Python from loading code from the folder you're in. The script refuses to run with any other version.
3. The last line removes that folder again, with the libraries `uv` put in it.

The script works in a new temporary folder, which it also uses as the home folder. So it never reads your records, your settings or your keys. When it's done, it copies the predictor here, without its lock files, writes `forecasts-0.3.0.json` and deletes the temporary folder.

It also checks that each forecast's similar past tasks are clearly closer than the next ones in line. Another kind of processor rounds the last digits of a sum differently, and a near tie could then pick other similar tasks, so the test's comparison would depend on the computer.

Running it again makes the same tasks, the same model and the same answers, apart from the times the predictor was created and trained.
