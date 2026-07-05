# Match Runner

`poe2_runner` is a minimal authoritative match runner. It owns one `poe2::Position`,
reads text moves from standard input, validates them with the public move API, and
prints accepted or rejected results.

## Build

```bash
cmake --build --preset debug --target poe2_runner
```

## Usage

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

The runner's `Position` is authoritative. Engines should keep their own local
positions for search and return proposed text moves. The runner parses and applies
those moves to its own state, then accepts or rejects them.
