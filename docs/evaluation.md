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

The output path is ordered by commit count and short SHA:

```text
build/by-commit/000015-d74d255e5cfd/release/
```

To build an older commit, switch to it from a clean tree, run `make git-test PRESET=release`, then switch back. The old artifact stays under `build/by-commit/`.

## Strength Gate

Use a strength gate when a change may affect move choice or search behavior:

```bash
make eval-gate BASE=000014-abcd1234 ENGINE=poe2_greedy GAMES=2000
```

This target:

- requires a clean tree,
- builds and tests the current commit,
- runs the current engine as `engine_one` against the baseline,
- enables SPRT early stopping,
- appends one row to `eval/results.csv`,
- fails unless SPRT reports `accept_alt`.

Default gate settings are intentionally simple:

```text
BOOK=eval/openings/systematic-2ply-v1.txt
SPRT_NULL=0.50
SPRT_ALT=0.55
SPRT_ALPHA=0.05
SPRT_BETA=0.05
GO_MOVETIME_MS=900
TIMEOUT_MS=1000
```

Adjust them at the command line:

```bash
make eval-gate BASE=000014-abcd1234 GAMES=5000 GO_MOVETIME_MS=100 TIMEOUT_MS=200
```

## Opening Suites

Evaluation games can start from a committed opening suite. The default gate suite is:

```text
eval/openings/systematic-2ply-v1.txt
```

Generate it from the runner:

```bash
build/debug/runner/poe2_runner openings generate-systematic \
  --out eval/openings/systematic-2ply-v1.txt \
  --plies 2
```

This enumerates all ordered two-ply prefixes on the 7x7 board and keeps one representative per
colored final position under board symmetry. Raw two-ply histories are `49 * 48 = 2352`; the
canonical suite has 315 openings.

Random deeper suites are useful for fresh or holdout checks:

```bash
build/debug/runner/poe2_runner openings generate-random \
  --out eval/openings/fresh/random-6ply-2026-07.txt \
  --count 200 \
  --plies 6 \
  --seed 20260707 \
  --max-score-gap 4
```

`BASE` can be a build id, a build directory, or an engine binary:

```bash
make eval-gate BASE=000014-abcd1234
make eval-gate BASE=build/by-commit/000014-abcd1234/release
make eval-gate BASE=build/by-commit/000014-abcd1234/release/engines/poe2_greedy
```

## Saved Runs

Every evaluation run is saved under:

```text
build/eval/runs/<timestamp>__<new-id>__vs__<base-id>/
```

Each run contains:

```text
manifest.json
command.txt
runner.log
summary.json
games.csv
```

The committed master ledger is:

```text
eval/results.csv
```

It stores one summary row per evaluation run. Keep the raw logs in `build/eval/runs/`; they are intentionally not committed.

## Direct Runner Usage

The Make targets call `poe2_runner eval`, but the runner can also be used directly:

```bash
build/by-commit/000015-d74d255e5cfd/release/runner/poe2_runner eval \
  --new-build build/by-commit/000015-d74d255e5cfd/release \
  --base 000014-abcd1234 \
  --engine poe2_greedy \
  --opening-book eval/openings/systematic-2ply-v1.txt \
  --games 2000 \
  --go-movetime-ms 100 \
  --timeout-ms 200 \
  --sprt-stop \
  --require-accept-alt
```

Use `--no-ledger` for throwaway local tests that should not append to `eval/results.csv`.
