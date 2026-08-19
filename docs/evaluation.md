# Engine Evaluation

The evaluation workflow has three layers:

- Build exact engine artifacts by commit.
- Run gates that compare a new artifact against a baseline.
- Save match records for later Elo-style progress plots.

## Build Artifacts

Build a clean commit into a commit-specific release directory:

```bash
make git-test PRESET=release
```

The output path is ordered by commit count and abbreviated SHA:

```text
build/by-commit/BUILD_ID/release/
```

To build an older commit, switch to it from a clean tree, run `make git-test PRESET=release`, then switch back. The old artifact stays under `build/by-commit/`.

## Strength Gate

Use a strength gate when a change may affect move choice or search behavior:

```bash
make eval-gate BASE=BASE_BUILD_ID NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy GAMES=2000
```

This target:

- requires a clean tree,
- builds and tests the current commit,
- runs the current engine as `engine_one` against the baseline,
- deterministically shuffles the holdout opening suite without replacement,
- preserves adjacent side-swapped pairs and checks evidence only after complete pairs,
- enables normalized-Elo GSPRT early stopping,
- appends one row to `eval/results.csv`,
- fails unless the sequential test reports `accept_alt`.

Default gate settings:

```text
BOOK=eval/openings/holdout.txt
GAMES=2000
WORKERS=1
SEQUENTIAL_NULL=0
SEQUENTIAL_ALT=20
SEQUENTIAL_ALPHA=0.05
SEQUENTIAL_BETA=0.05
GO_MOVETIME_MS=100
TIMEOUT_MS=1000
```

The gate uses a generalized sequential probability ratio test with normalized-Elo hypotheses. Each opening pair contributes one of five candidate score rates: `0`, `0.25`, `0.5`, `0.75`, or `1`; the quarter-score bins occur when one game has no winner. `accept_alt` crosses the upper log-likelihood boundary, `accept_null` crosses the lower boundary, and `continue` means the game budget ended without either crossing. The existing anytime-valid betting confidence sequence and evidence against an even score are still reported, but they do not control the gate decision.

Adjust them at the command line:

```bash
make eval-gate BASE=BASE_BUILD_ID NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy \
  GAMES=5000 WORKERS=8 GO_MOVETIME_MS=250 TIMEOUT_MS=1000 SEQUENTIAL_ALT=10
```

Every eval run names both engine binaries explicitly:

```bash
make eval-smoke BASE=build/by-commit/BASE_BUILD_ID/release \
  NEW_ENGINE=poe2_greedy \
  BASE_ENGINE=poe2_random_legal \
  BASE_ENGINE_ARGS='--seed 1' \
  SMOKE_GAMES=200
```

Override the game budget and search time directly when an experiment needs different settings:

```bash
make eval-gate \
  BASE=<baseline-build-id> \
  NEW_ENGINE=minimax/poe2_minimax \
  BASE_ENGINE=minimax/poe2_minimax \
  PRESET=release \
  GAMES=4000 \
  WORKERS=8 \
  GO_MOVETIME_MS=250 \
  SEQUENTIAL_ALT=10
```

The new build is always the current `HEAD` build, not the newest directory under `build/by-commit/`. Choose `BASE` explicitly; both engine names are required. `GAMES` must be even and cannot exceed twice the number of opening records. `OPENING_SEED=<n>` overrides the derived sampling seed when exact manual control is needed.

## Opening Suites

The committed corpus is divided into two books:

```text
eval/openings/development.txt
eval/openings/holdout.txt
```

`eval-smoke` uses the development book and `eval-gate` uses the holdout book.

Regenerate both books atomically with:

```bash
build/debug/runner/poe2_runner openings generate-corpus \
  --development-out eval/openings/development.txt \
  --holdout-out eval/openings/holdout.txt \
  --count 20000 \
  --plies 2,4,6,8,10,12,14 \
  --seed 20260816 \
  --max-score-gap 4
```

The legacy systematic book remains committed for reproducing historical evaluations:

```bash
build/debug/runner/poe2_runner openings generate-systematic \
  --out eval/openings/systematic-2ply-v1.txt \
  --plies 2
```

For evaluation runs, one opening is selected for each adjacent side-swapped pair. Selection is a deterministic shuffle without replacement. The default seed is derived from the candidate build ID, baseline build ID, book digest, and eval kind, so the same matchup is replayable while a different candidate receives a different ordering.

`BASE` can be a build id, a build directory, or an engine binary:

```bash
make eval-gate BASE=BASE_BUILD_ID NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy
make eval-gate BASE=build/by-commit/BASE_BUILD_ID/release NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy
make eval-gate BASE=build/by-commit/BASE_BUILD_ID/release/engines/poe2_greedy NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy
```

## Saved Runs

Every evaluation run is saved under:

```text
build/eval/runs/<timestamp>__<new-id>__<new-engine>__vs__<base-id>__<base-engine>[__args-hashes]/
```

Each run contains:

```text
manifest.json
command.txt
runner.log
summary.json
games.csv
ledger-row.csv
```

`ledger-row.csv` is a self-contained header and row for this run. It is retained even with
`--no-ledger`, so a separately authenticated workflow can deliberately promote the result later.

The committed master ledger is:

```text
eval/results.csv
```

It stores one summary row per evaluation run. Keep the raw logs in `build/eval/runs/`; they are intentionally not committed. Each row records both artifact identities as `new_id + new_engine + new_engine_args` and `base_id + base_engine + base_engine_args`.

Native evaluation and workflow promotion serialize master-ledger updates through the same persistent `<ledger>.lock` sidecar and append-mode write protocol, so concurrent writers cannot overwrite or interleave rows.

Every ledger row includes validity, sampling identity, the analysis version, statistical unit, pair-score counts, normalized-Elo estimate, GSPRT LLR and boundaries, betting diagnostics, and the final decision. Historical rows remain in their original order and are classified with their legacy model and score-rate units; unavailable fields remain blank. `summary.json` contains the complete analysis report, `manifest.json` records the resolved opening seed and book digest, and `games.csv` remains the raw source of truth.

The run command, ledger, manifest, summary, and runner log also record the requested and actual worker counts. The summary and ledger report games completed speculatively and discarded after a cutoff or invalid result. Parallel results are committed in opening order, so speculative results never enter the sequential test; deterministic, isolated engines retain the same cutoff prefix as a one-worker run. Each worker owns two engine processes.

Any timeout, disconnect, malformed or illegal move, protocol error, or startup failure immediately makes an eval run invalid. The partial artifacts and ledger row are still written, the abnormal game is excluded from statistical inference, and eval exits with status `3`. A valid gate that accepts the null or reaches its cap undecided exits with status `2`.

## Direct Runner Usage

The Make targets call `poe2_runner eval`, but the runner can also be used directly:

```bash
NEW_BUILD=build/by-commit/NEW_BUILD_ID/release
"$NEW_BUILD/runner/poe2_runner" eval \
  --new-build "$NEW_BUILD" \
  --base BASE_BUILD_ID \
  --new-engine poe2_greedy \
  --base-engine poe2_random_legal \
  --base-engine-args '--seed 1' \
  --opening-book eval/openings/holdout.txt \
  --shuffle-openings \
  --games 2000 \
  --workers 8 \
  --go-movetime-ms 100 \
  --timeout-ms 1000 \
  --sequential-stop \
  --sequential-null 0 \
  --sequential-alt 20 \
  --require-accept-alt
```

Use `--no-ledger` for throwaway local tests that should not append to `eval/results.csv`.
