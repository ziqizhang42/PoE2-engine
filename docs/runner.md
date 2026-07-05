# Match Runner

`poe2_runner` is an authoritative match runner. It can run in manual mode for debugging, or supervise one game between two external engine processes.

## Build

```bash
cmake --build --preset debug --target poe2_runner poe2_first_legal
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
  --timeout-ms 1000
```

The runner starts each engine as a separate process, sends the current move history before every turn, asks the side to move for `go`, validates the returned move, and prints the final result.

Engines can use the `poe2_engine_stdio` helper library to handle this local stdin/stdout protocol. With that helper, an engine only needs to provide a function that chooses a move from a `poe2::Position`.

The runner executable uses the `poe2_match_runner` library for process supervision and one-game match execution. The executable itself is intentionally kept as CLI glue.

Engine failures are treated as game outcomes:

- timeout before `bestmove`: opponent wins with reason `timeout`
- process closes before `bestmove`: opponent wins with reason `disconnected`
- malformed coordinate: opponent wins with reason `malformed_move`
- illegal coordinate: opponent wins with reason `illegal_move`

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
