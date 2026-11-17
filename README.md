# PoE2 Engine

An engine for **Powers of Exponent 2 (PoE2)**, a two-player game on a 7x7 board. Players take
turns placing a piece on an empty cell until the board is full. Every maximal line of a
player's pieces, in any of the four directions, scores `2^(length-1)`. The second player
starts with 5.5 points, and the higher total wins ([rules](docs/rules.md)).

The engine is a minimax search with a small learned evaluator, called a net. Each new net is
trained on positions labeled by searches that use the net before it, and is installed only
after it beats that net over games.

## What is here

| Folder | Contents |
|---|---|
| `game/` | Rules and position state |
| `engines/` | The engines; `minimax/` is the real one |
| `runner/` | Match runner, and the statistical gate that compares two engines |
| `wasm/` | Browser (WebAssembly) package |
| `training/minimax/` | Python and PyTorch training pipeline |
| `models/`, `eval/` | Every gated net, opening books, and the ledger of match results |

## Build and run

The build needs a C++23 compiler, CMake, Ninja, and LLVM 22's clang-format and clang-tidy
([setup](docs/dev-setup.md)).

```bash
make PRESET=release
build/release/engines/minimax/poe2_minimax --version
```

`make` builds, tests, and lints. `--version` prints the engine version and its built-in net.

To play one game between two engines:

```bash
build/release/runner/poe2_runner match \
  --p1 build/release/engines/minimax/poe2_minimax \
  --p2 build/release/engines/poe2_greedy \
  --timeout-ms 1000 --go-movetime-ms 100
```

## Documentation

- Playing: [rules](docs/rules.md), [game API](docs/api.md), [match runner](docs/runner.md),
  [browser package](docs/wasm.md)
- Strength: [gates](docs/evaluation.md), [nets](docs/nets.md),
  [training workflow](docs/training-workflow.md), [research log](docs/research-log.md)
- Releases: [changelog](CHANGELOG.md), [releasing](docs/releasing.md)
