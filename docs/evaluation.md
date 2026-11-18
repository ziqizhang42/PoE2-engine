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
BOOK=eval/openings/holdout-v2.txt
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

Each corpus is divided into a development book and a holdout book. `eval-smoke` uses the development book and `eval-gate` the holdout book of the balanced corpus ([Balanced Books](#balanced-books)):

```text
eval/openings/development-v2.txt
eval/openings/holdout-v2.txt
```

The rule-generated corpus stays committed. Results recorded up to 2026-11-18 used it, and `run1`'s recipe names it:

```text
eval/openings/development.txt
eval/openings/holdout.txt
```

A score on one corpus cannot be compared with a score on the other: the same difference between two engines shows as about twice the normalized Elo on the balanced books. Every row of `eval/results.csv` names its book and the book's digest.

Regenerate both rule-generated books atomically with:

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

For evaluation runs, one opening is selected for each adjacent side-swapped pair. Selection is a deterministic shuffle without replacement. The default seed is derived from the candidate build ID, baseline build ID, book digest, and eval kind, and from the SHA-256 of any net file a side plays, so the same matchup is replayable while a different candidate receives a different ordering.

`BASE` can be a build id, a build directory, or an engine binary:

```bash
make eval-gate BASE=BASE_BUILD_ID NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy
make eval-gate BASE=build/by-commit/BASE_BUILD_ID/release NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy
make eval-gate BASE=build/by-commit/BASE_BUILD_ID/release/engines/poe2_greedy NEW_ENGINE=poe2_greedy BASE_ENGINE=poe2_greedy
```

### Balanced Books

In the rule-generated books the opening often decides the game: the first player's mean final lead varies by about 8.6 points from one opening to another, so in matches between two close engines only about 7% of side-swapped pairs are won twice by the same engine. The other pairs say nothing about which engine is stronger. The balanced books keep only openings that a fixed-node search predicts to be close to even:

```text
eval/openings/development-v2.txt
eval/openings/holdout-v2.txt
eval/openings/balance-correction-v2.txt
```

They are the default books of `eval-gate`, `eval-smoke`, and training recipes that name no book. The gate's alternative stays at 20 normalized Elo, which on these books is a difference about half as large as on the rule-generated ones: a gate accepts a real improvement about 2.3 times sooner, takes as long as before to reject a change with no effect, and passes smaller gains. `SEQUENTIAL_ALT=40` is the same real difference as 20 was on the rule-generated books.

`poe2_runner openings generate-balanced` builds both books:

1. Candidates are every 2-ply position and, for each deeper length, random legal lines under the score-gap rule of the rule corpus, drawn from a stream of their own seeded by `--seed` and the length. Positions equal under the 8 board symmetries are one position, within each book and across the two.
2. Each candidate is searched by the engine for a fixed node budget, after `newgame` so the table starts empty. The score, in half-points from the side to move, already includes the handicap.
3. The raw search over-rates the first player, by about 3 points at 2 to 6 plies and about 1 point at 8 to 14. A per-length line, lead = intercept + slope × points, maps the score to the first player's expected final lead after the handicap. Without it the kept openings would be won by the first player only about 28% of the time.
4. A candidate passes when the predicted lead is below `--threshold` points in size (default 2). Each book keeps every passing 2-ply opening, of which there are only about 60, and the first `--per-length` passing openings of each deeper length in stream order.
5. An opening that is in a previous book stays on that book's side, so the new holdout book has no position the development book exposed to tuning, and the new development book has none from the holdout book. Every 2-ply position is in one of the previous books, so 2-ply openings keep their old side. Other openings are split by a hash of the seed and the position.

The correction comes from match results. `poe2_runner openings fit-balance` reads the `games.csv` of each finished run, takes each opening's mean final lead over its normal games, searches the opening the same way, and fits the line for each length by least squares. It writes the constants together with the engine version, the net, the node budget, and the table size, since the constants only hold for that search; `generate-balanced` takes the node budget and table size from the file and refuses an engine that plays another net. Both commands give the same output for any `--workers`.

`fit-balance` also checks the rule on openings it did not fit: it fits on half of the openings, split by position hash, and reports on the other half the bias of the corrected prediction by length, the share of openings kept, the share of outright pairs on kept and on all openings, and how often the first player won on kept openings. The file records this check. For `balance-correction-v2.txt`, fitted on 18 matches on `development.txt` (40,998 games, 8,451 openings) between v0.3.2 builds playing `run1/g002`: the rule kept 20.4% of held-out openings, outright pairs were 27.2% on kept openings against 7.8% on all, and the first player won 48.9% of games on kept openings.

Each book's header records the seed, the engine version and net, the node budget, the table size, the correction file with its digest and constants, the threshold, the quota, the previous books with their digests, and for each length the number of openings kept, of candidates searched, and of openings inherited from a previous book. Each opening line ends with its search score and predicted lead as a comment.

Regenerate both books byte for byte from the repository root (about 3 minutes with 8 workers; the paths are recorded as given, so use these):

```bash
cmake --preset release && cmake --build --preset release
build/release/runner/poe2_runner openings generate-balanced \
  --development-out eval/openings/development-v2.txt \
  --holdout-out eval/openings/holdout-v2.txt \
  --correction eval/openings/balance-correction-v2.txt \
  --engine build/release/engines/minimax/poe2_minimax \
  --plies 2,4,6,8,10,12,14 \
  --per-length 657 \
  --threshold 2 \
  --max-score-gap 4 \
  --seed 20261118 \
  --previous-development eval/openings/development.txt \
  --previous-holdout eval/openings/holdout.txt \
  --workers 8
```

The engine must be one whose fixed-node search matches v0.3.2 with `run1/g002`. Refit the correction from finished matches on the development book, for example when the champion's net changes, and give the new correction and books new names so earlier results stay comparable:

```bash
build/release/runner/poe2_runner openings fit-balance \
  --run build/eval/runs/<run> --run build/eval/runs/<another run> \
  --engine build/release/engines/minimax/poe2_minimax \
  --nodes 200000 \
  --hash-mb 16 \
  --workers 8 \
  --out eval/openings/balance-correction-v3.txt
```

`--run` accepts only directories with a `summary.json`, so unfinished runs are not read. Only lengths with at least `--min-openings` fitted openings (default 30) get constants.

## Saved Runs

Every evaluation run is saved under:

```text
build/eval/runs/<timestamp>__<new-id>__<new-engine>__vs__<base-id>__<base-engine>[__args-hashes][__newnet_<net ID>][__basenet_<net ID>]/
```

A name longer than 200 bytes keeps its first 182 bytes and ends in `__` and a 16-hex-digit digest of the full name, so it stays unique; the manifest records every part in full.

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

It stores one summary row per evaluation run. Keep the raw logs in `build/eval/runs/`; they are intentionally not committed. Each row records both artifact identities as `new_id + new_engine + new_engine_args` and `base_id + base_engine + base_engine_args`, and each side's engine and net as reported by the engine: `new_engine_version`, `new_net_name`, `new_net_id`, `new_net_sha256`, and the same four for `base`. A side that played a net file (`--new-net` or `--base-net`, see [runner.md](runner.md#eval-mode)) has its file named in the manifest and the run ID. Rows from before engines reported their nets leave these columns empty; the runner and the training workflow add the empty columns to an older ledger the first time they append to it.

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
  --opening-book eval/openings/holdout-v2.txt \
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
