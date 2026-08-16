# Match Runner

`poe2_runner` is an authoritative match runner. It can run in manual mode for debugging, supervise one game between two external engine processes, or run a repeated series to compare two engines.

## Build

```bash
cmake --build --preset debug --target poe2_runner poe2_first_legal poe2_greedy poe2_random_legal
```

## Manual Mode

```bash
build/debug/runner/poe2_runner
```

Commands:

- `a1` through `g7`: submit a move.
- `state`: print ply, side to move, scores, and empty count.
- `help`: print command summary.
- `quit` or `exit`: stop the runner.

Example:

```text
commands: a1-g7, state, quit, help
state ply=0 side=p1 p1=0 p2=0 empty=49
a1
accepted a1
state ply=1 side=p2 p1=1 p2=0 empty=48
a1
rejected a1 occupied
```

## Authority Model

The runner's `Position` is authoritative. Engines should keep their own local positions for search and return proposed text moves. The runner parses and applies those moves to its own state, then accepts or rejects them.

## One-Game Match Mode

```bash
build/debug/runner/poe2_runner match \
  --p1 ./build/debug/engines/poe2_first_legal \
  --p2 ./build/debug/engines/poe2_first_legal \
  --timeout-ms 1000 \
  --go-depth 1
```

The runner starts each engine as a separate process, sends the current move history before every turn, asks the side to move for `go`, validates the returned move, and prints the final result.

Engines can use the `poe2_engine_stdio` helper library to handle this local stdin/stdout protocol. With that helper, an engine implements `poe2::engine::Engine`, stores any engine-specific configuration or state on that object, and returns a `poe2::engine::EngineResult` from `choose_move`.

The runner executable uses the `poe2_match_runner` library for process supervision and one-game match execution. The executable itself is intentionally kept as CLI glue.

Engine failures are treated as game outcomes:

- timeout before `bestmove`: opponent wins with reason `timeout`
- process closes before `bestmove`: opponent wins with reason `disconnected`
- malformed coordinate: opponent wins with reason `malformed_move`
- illegal coordinate: opponent wins with reason `illegal_move`

## Series Mode

Series mode uses one reusable process pair per worker, sends `newgame` before each game, alternates sides by default, and prints aggregate comparison stats.

```bash
build/debug/runner/poe2_runner series \
  --engine-one ./build/debug/engines/poe2_first_legal \
  --engine-two "./build/debug/engines/poe2_random_legal --seed 1" \
  --games 100 \
  --workers 4 \
  --timeout-ms 1000 \
  --go-movetime-ms 100 \
  --opening-book eval/openings/development.txt \
  --shuffle-openings \
  --opening-seed 0 \
  --summary-only \
  --sequential-stop \
  --sequential-null 0 \
  --sequential-alt 20
```

Useful options:

- `--games <n>`: number of games to run.
- `--workers <n>`: number of games to process concurrently. Defaults to `1`; each worker starts one process for each engine.
- `--opening-book <path>`: start games after move prefixes from a text opening suite.
- `--shuffle-openings`: shuffle the opening suite once before the series starts.
- `--opening-seed <n>`: make the shuffled selection and ordering reproducible.
- `--go-depth <n>`: include `depth <n>` in every `go` command sent to engines.
- `--go-movetime-ms <ms>`: include `movetime <ms>` in every `go` command sent to engines.
- `--go-nodes <n>`: include `nodes <n>` in every `go` command sent to engines.
- `--fixed-sides`: keep engine one as player one for every game.
- `--summary-only`: suppress per-game result lines for larger local runs.
- `--verbose-games`: print the full one-game move/state log for every game.
- `--sequential-stop`: treat `--games` as the maximum game budget and stop early on `accept_alt` or `accept_null`.
- `--sequential-null <nelo>`: null normalized-Elo value for engine one. Defaults to `0`.
- `--sequential-alt <nelo>`: alternative normalized-Elo value for engine one. Defaults to `20`.
- `--sequential-alpha <p>`: false-positive risk for accepting the alternative. Defaults to `0.05`.
- `--sequential-beta <p>`: false-negative risk for accepting the null. Defaults to `0.05`.

The summary includes engine-one wins, engine-two wins, average plies, average scores, reason counts, normalized Elo, a paired normalized-Elo GSPRT report, and the anytime-valid betting confidence analysis. The betting analysis is diagnostic and does not control the GSPRT decision. The decision is `accept_alt`, `accept_null`, or `continue`; `continue` means the current sample is not decisive under the selected hypotheses and risk settings. Without `--sequential-stop`, the runner runs the requested budget and reports the decision at the end. With it, a boundary crossing stops the series early.

Decisions are directional for engine one. `accept_alt` crosses the upper GSPRT boundary in favor of `--sequential-alt`; `accept_null` crosses the lower boundary in favor of `--sequential-null`. Swap engine order to test the other engine.

When an opening book is provided, series mode never wraps around. It fails before starting if the requested games need more unique openings than the book contains. With alternating sides, each opening is used for the adjacent swapped-side game before advancing, and sequential stopping is checked only after the pair is complete.

With alternating sides, the statistical unit is the completed opening pair, not an individual game. Normal pairs contribute an engine-one score rate of `0`, `0.5`, or `1`. Diagnostic no-winner outcomes occupy the remaining pentanomial bins. Abnormal games and incomplete pairs are excluded from inference. With `--fixed-sides`, individual normal games remain the statistical units.

Parallel workers always execute a whole side-swapped pair as one work unit. Results are committed in opening order, regardless of which worker finishes first, and sequential stopping is evaluated after each committed pair. Once a boundary or invalid game stops the series, already-running later pairs are allowed to finish and are discarded; `games_discarded` reports that speculative work. No post-boundary result enters the analysis, and wasted work is limited to at most two games per other worker. For deterministic, isolated engines this produces the same cutoff prefix as a one-worker run. The runner does not interrupt an engine in the middle of a search.

Choose the worker count with resource headroom. A run starts `2 * workers` engine processes, wall-clock timeouts still include scheduling delays, and engines that use internal threads or a GPU can oversubscribe the machine.

## Eval Mode

Eval mode wraps series mode for saved engine comparisons:

```bash
build/release/runner/poe2_runner eval \
  --new-build build/by-commit/000015-d74d255e5cfd/release \
  --base 000014-abcd1234 \
  --new-engine poe2_greedy \
  --base-engine poe2_random_legal \
  --base-engine-args '--seed 1' \
  --opening-book eval/openings/holdout.txt \
  --shuffle-openings \
  --games 2000 \
  --workers 4 \
  --sequential-stop \
  --require-accept-alt
```

It writes `manifest.json`, `summary.json`, `games.csv`, `command.txt`, and `runner.log` under `build/eval/runs/`, and appends one summary row to `eval/results.csv` unless `--no-ledger` is passed.
Every eval run supplies both `--new-engine <name>` and `--base-engine <name>`. Use `--new-engine-args` or `--base-engine-args` for side-specific engine arguments.

Eval mode requires an even game budget, alternating sides, a shuffled opening book, and enough unique openings. If `--opening-seed` is omitted, eval derives it from both build IDs, the opening-book digest, and the eval kind. Any abnormal game invalidates the run immediately, writes partial artifacts, excludes that game from inference, and returns status `3`.

## Opening Generation

Generate the development and holdout corpus:

```bash
build/debug/runner/poe2_runner openings generate-corpus \
  --development-out eval/openings/development.txt \
  --holdout-out eval/openings/holdout.txt \
  --count 20000 \
  --plies 2,4,6,8,10,12,14 \
  --seed 20260816 \
  --max-score-gap 4
```

This includes the complete canonical two-ply stratum, generates deeper legal histories under the score-gap rule, deduplicates final colored positions under board symmetry, and deterministically balances every depth across the two outputs.

The systematic generator remains available for reproducing the historical two-ply suite:

```bash
build/debug/runner/poe2_runner openings generate-systematic \
  --out eval/openings/systematic-2ply-v1.txt \
  --plies 2
```

Generate a random symmetry-deduplicated suite:

```bash
build/debug/runner/poe2_runner openings generate-random \
  --out eval/openings/fresh/random-6ply-2026-07.txt \
  --count 200 \
  --plies 6 \
  --seed 20260707 \
  --max-score-gap 4
```

## Engine Stdio Protocol

The initial protocol is intentionally small and text based.

Startup:

```text
poe2
isready
```

The engine should eventually reply:

```text
readyok
```

Before a move, the runner sends the full move history:

```text
position startpos moves a1 b1 c1
go
```

The engine replies:

```text
bestmove d1
```

Engines may print `info ...` lines. The runner ignores all lines until it sees a `bestmove` line or the timeout expires.

Engines using the stdio helper can also accept optional `go` limits:

```text
go depth 5 movetime 1000 nodes 500000
```

All limits are optional. Engine implementations may ignore limits that do not apply to their algorithm.

The runner's `--timeout-ms` remains a hard supervision timeout. `--go-movetime-ms` is only a requested engine search limit and should usually be set lower than `--timeout-ms`.
