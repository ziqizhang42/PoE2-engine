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

Series mode reuses the same two engine processes across games, sends `newgame` before each game, alternates sides by default, and prints aggregate comparison stats.

```bash
build/debug/runner/poe2_runner series \
  --engine-one ./build/debug/engines/poe2_first_legal \
  --engine-two "./build/debug/engines/poe2_random_legal --seed 1" \
  --games 100 \
  --timeout-ms 1000 \
  --go-movetime-ms 900 \
  --summary-only \
  --sprt-stop \
  --sprt-null 0.50 \
  --sprt-alt 0.55
```

Useful options:

- `--games <n>`: number of games to run.
- `--opening-book <path>`: cycle through a text opening suite and start each game after the
  selected move prefix.
- `--go-depth <n>`: include `depth <n>` in every `go` command sent to engines.
- `--go-movetime-ms <ms>`: include `movetime <ms>` in every `go` command sent to engines.
- `--go-nodes <n>`: include `nodes <n>` in every `go` command sent to engines.
- `--fixed-sides`: keep engine one as player one for every game.
- `--summary-only`: suppress per-game result lines for larger local runs.
- `--verbose-games`: print the full one-game move/state log for every game.
- `--sprt-stop`: treat `--games` as the maximum game budget and stop early on `accept_alt` or `accept_null`.
- `--sprt-null <p>`: null score rate for engine one. Defaults to `0.50`.
- `--sprt-alt <p>`: alternative score rate for engine one. Defaults to `0.55`.
- `--sprt-alpha <p>`: false-positive risk for accepting the alternative. Defaults to `0.05`.
- `--sprt-beta <p>`: false-negative risk for accepting the null. Defaults to `0.05`.

The summary includes engine-one wins, engine-two wins, average plies, average scores, reason counts, a 95% Wilson confidence interval for engine one's score rate, and an SPRT-style likelihood report. The SPRT decision is `accept_alt`, `accept_null`, or `continue`; `continue` means the current sample is not decisive under the selected null/alternative rates and risk settings. Without `--sprt-stop`, the runner always runs exactly `--games` games and only reports the SPRT decision at the end. With `--sprt-stop`, `--games` becomes a maximum budget and a decisive SPRT result stops the series early.

SPRT decisions are directional for engine one: `accept_alt` supports engine one's `--sprt-alt` score-rate claim, while `accept_null` rejects it; swap engine order to test the other engine.

When an opening book is provided, series mode cycles through opening lines. With alternating sides,
each opening is reused for the adjacent swapped-side game before advancing to the next line.

## Eval Mode

Eval mode wraps series mode for saved engine comparisons:

```bash
build/release/runner/poe2_runner eval \
  --new-build build/by-commit/000015-d74d255e5cfd/release \
  --base 000014-abcd1234 \
  --new-engine poe2_greedy \
  --base-engine poe2_random_legal \
  --base-engine-args '--seed 1' \
  --games 2000 \
  --sprt-stop \
  --require-accept-alt
```

It writes `manifest.json`, `summary.json`, `games.csv`, `command.txt`, and `runner.log` under `build/eval/runs/`, and appends one summary row to `eval/results.csv` unless `--no-ledger` is passed.
Every eval run supplies both `--new-engine <name>` and `--base-engine <name>`. Use
`--new-engine-args` or `--base-engine-args` for side-specific engine arguments.

## Opening Generation

Generate the default systematic 2-ply gate suite:

```bash
build/debug/runner/poe2_runner openings generate-systematic \
  --out eval/openings/systematic-2ply-v1.txt \
  --plies 2
```

This enumerates all ordered two-ply prefixes and keeps one representative per final colored
position under board symmetry.

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
