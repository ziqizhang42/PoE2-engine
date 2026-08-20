# Consolidated Training Workflow

`poe2-train` orchestrates the repository's existing source, labeling, feature, training, evaluation, export, parity, benchmark, and match tools. Those tools remain independently runnable and authoritative.

```text
run
├── datasets
│   ├── managed: source -> label shards -> features
│   └── imported: authenticate and pin completed features
└── iterations (TOML order)
    └── training -> sealed evaluation? -> candidate validation? -> engine gate?
```

Each iteration prepares its dataset automatically. Dependencies may name earlier iterations and must be complete unless `--with-dependencies` is used.

## Quick start

Select a checked-in recipe from the repository root:

```bash
ls training/minimax/recipes
export TRAIN_CONFIG=training/minimax/recipes/RECIPE_NAME.toml

make train-plan
make train-validate
make train-run
make train-status
```

Run one iteration with Make or include incomplete dependencies through the CLI:

```bash
make train-iteration TRAIN_ITERATION=ITERATION_NAME

cd training/minimax
CONFIG=recipes/RECIPE_NAME.toml
uv run --frozen poe2-train iteration --config "$CONFIG" \
  ITERATION_NAME --with-dependencies
uv run --frozen poe2-train status --config "$CONFIG" --json
```

`plan`, `validate`, and `status` are read-only; `validate` authenticates existing outputs. Mutating commands require a clean committed worktree and a tracked recipe. Repository-local run roots must be ignored; checked-in recipes use `build/`.

## Console progress

On an interactive terminal, mutating commands show one colored tqdm stage bar with elapsed time, estimated remaining time, and a compact postfix for the active source, label shard, build, or model. Completed stages and iterations each leave one short summary line; the full child-process output continues to be captured in the existing per-command log files instead of scrolling through the terminal.

Stages that reuse an existing complete artifact are labeled `authenticated`, so repeated dependency or exclusion preparation is distinguishable from artifact creation.

`status` prints the same ordered iteration graph, an overall stage bar, and cumulative recorded stage time. `status --json` remains undecorated and machine-readable. Redirected output is compact one-line stage reporting without ANSI control sequences.

The candidate promotion sequence is:

1. Run training and candidate validation.
2. Preview and apply candidate handoff.
3. Review and commit only the generated model header.
4. Add and commit the `engine_gate` table in a compatible recipe revision.
5. Run the iteration again to execute the gate.
6. Preview and explicitly apply ledger promotion.

## Configuration

The TOML parser rejects unknown fields, invalid references, unsafe paths, and unsupported stage combinations. Paths are repository-relative unless absolute. Imported datasets and opening books must not overlap the run root.

A minimal managed pattern-training recipe looks like this:

```toml
schema_version = 1

[run]
name = "example-v1"
output_dir = "build/training-runs/example-v1"
build_preset = "release"
# label_concurrency = 2  # omit to auto-size

[datasets.development]
kind = "managed"

[datasets.development.source]
corpus_id = "poe2-example-development-v1"
seed = 20260817
trajectories = 1000
shards = 4
workers = 6

[datasets.development.labels]
mode = "teacher"
nodes = 100000
workers = 6
require_all = true

[[iterations]]
name = "development-ladder"
dataset = "development"

[iterations.training]
type = "pattern"
device = "cuda"
seed = 20260818
suite = "default"
```

See the checked-in recipes for all source, labeling, and feature tuning fields.

An imported dataset points to one completed feature artifact. Its binary and manifest digests are pinned on first use; source and partial label imports are unsupported.

```toml
[datasets.existing]
kind = "imported"
path = "build/data/features/example/b-primitives"
```

Baseline training uses `type = "baseline"` and accepts `device`, `seed`, `close_margin`, and `ridge_lambdas`. Pattern suites are `default`, `frozen-pattern-gain`, and `line-pattern-audit`.

### Optional pattern stages

Sealed evaluation can exclude positions found in other named datasets:

```toml
[iterations.sealed_evaluation]
exclude_datasets = ["development"]
```

The `frozen-pattern-gain` suite can export and validate a candidate:

```toml
[iterations.candidate_validation]
promotable = true
samples = 4096
symmetry_samples = 512
benchmark_iterations = 100
opening_book = "eval/openings/development.txt"
search_positions = 32
search_movetime_ms = 100
```

Candidate validation builds isolated inference, engine, and runner binaries; checks Python/C++ and D4 parity; and records benchmarks, digests, logs, resources, and separate float and exported-quantized metrics. The deployable frozen suite fixes its quantization at the engine's scale-32 contract before sealed evaluation.

Add the gate only after the candidate header has been reviewed, installed, and committed:

```toml
[iterations.engine_gate]
base = "BASE_BUILD_ID"
base_engine = "minimax/poe2_minimax"
base_engine_args = "--evaluator pattern-gain"
opening_book = "eval/openings/holdout.txt"
games = 2000
workers = 6
timeout_ms = 1000
go_movetime_ms = 100
sequential_stop = true
sequential_null = 0
sequential_alt = 20
sequential_alpha = 0.05
sequential_beta = 0.05
require_accept_alt = true
```

The candidate side is fixed to `minimax/poe2_minimax --evaluator pattern-gain`. The gate requires a promotable candidate and an exact tracked-header digest match. Its build identity is derived from the model-header digest; the base engine and arguments remain configurable.

When `label_concurrency` is omitted, label processes are sized as:

```text
max(1, floor((logical CPUs - 1) / workers per label shard))
```

## State and recovery

Every mutating command holds `<run>/.orchestrator.lock`. Accepted TOML revisions are snapshotted, events are appended and fsynced, and `summary.json` is rebuilt atomically.

```text
<run>/
├── state/{configs,events.jsonl}
├── summary.json
├── logs/<stage>/attempt-...
├── build/<preset>/
├── datasets/<name>/{source,labels/shards,features}
└── iterations/<name>/{training/{report.json,training-metrics.svg},sealed-evaluation,candidate,engine-gate}
```

Build trees are private to each run or candidate. Once a stage starts, its configuration and input digests cannot change or be removed; unstarted stages and new iterations may be added in a later compatible recipe revision.

Artifacts are create-only and authenticated before reuse. On interruption, subprocess groups are terminated and completed label shards remain reusable. If an artifact is incomplete or malformed, the workflow reports its exact path and stops. Inspect it, then manually move aside or remove only that path before retrying.

## Candidate handoff

Handoff previews the difference by default:

```bash
cd training/minimax
CONFIG=recipes/RECIPE_NAME.toml
ITERATION=ITERATION_NAME

uv run --frozen poe2-train handoff --config "$CONFIG" "$ITERATION"
uv run --frozen poe2-train handoff --config "$CONFIG" "$ITERATION" --apply
```

Apply requires a promotable candidate, a clean worktree, and the run lock. It atomically replaces only `engines/minimax/src/frozen_pattern_gain_model.hpp`. Review and commit that file before adding the gate.

## Gate promotion

Gates run with `--no-ledger` and retain a self-contained `ledger-row.csv`. Preview and apply its publication explicitly:

```bash
uv run --frozen poe2-train promote-gate --config "$CONFIG" "$ITERATION"
uv run --frozen poe2-train promote-gate --config "$CONFIG" "$ITERATION" --apply
```

Promotion requires a clean tree and the run lock. It authenticates the saved row against the gate manifest and summary, validates the master schema, coordinates with native ledger writers, and rejects duplicate run IDs before appending to `eval/results.csv`.

## Lower-level references

- [Position sources](position-source.md)
- [Labeling](labeling.md)
- [Feature artifacts](feature-data.md)
- [Training experiments and parity](../training/minimax/README.md)
- [Engine evaluation](evaluation.md)
