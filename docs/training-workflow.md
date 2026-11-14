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

`status` prints the same ordered iteration graph, an overall stage bar, and cumulative recorded stage time. A completed gate also shows its recorded `gate_decision`. `status --json` remains undecorated and machine-readable. Redirected output is compact one-line stage reporting without ANSI control sequences.

The candidate promotion sequence is:

1. Commit a recipe whose candidate iteration includes an `engine_gate` table. A gate can also be added later in a compatible recipe revision, as long as that stage has not started.
2. Run the iteration: training, candidate validation, and the gate. The gate builds the candidate's model header privately, so nothing in the source tree changes.
3. Preview and apply gate promotion. This appends the ledger row, archives the model under `models/`, and adds a row to `models/index.csv`, whatever the decision.
4. Review and commit the ledger row, archive, and index.
5. Only if the gate decision is `accept_alt`: preview and apply candidate handoff, then review and commit the installed model header.

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

### Teachers and generations

A managed dataset can name a teacher: an archived model whose learned evaluator drives the noisy-search positions and the labels. Without one, both use the two-ply-closure evaluator.

```toml
[models]
g000 = "candidate-sha256-<header digest>"

[datasets.g001]
kind = "managed"
teacher = "g000"
```

Each `[models]` entry gives a recipe-local name to a model archived under `models/`. Loading the recipe requires the archived `.hpp` and `.json` to exist and the header to match its digest. The workflow builds a private data tool for each teacher under `<run>/build/teacher-<digest prefix>/`, with `-DPOE2_MINIMAX_MODEL_HEADER` pointing at the archived header, and passes `--evaluator pattern-gain` to position generation and labeling. Source, label, and feature authentication require the recorded evaluator and model digest to match the recipe. A teacher is part of the source and label stage fingerprints, so it cannot change once those stages start. Datasets without a teacher keep the fingerprints they had before teachers existed.

A run trains one architecture over successive generations named `g000`, `g001`, …. `g000` is the run's starting model, either imported or trained inside the run. Each later generation `gNNN` has a dataset taught by the generation before it, and its gate plays that generation with `base_model`. To start the next generation, promote the current one's gate so its model is archived, add it to `[models]` under the generation's own name, and append the new dataset and iteration in a compatible recipe revision. A `[models]` entry that shares an iteration's name must be the model that iteration produced. The workflow checks this once the iteration's candidate exists: when a run or `validate` starts, and before the entry is used as a teacher or gate base.

```toml
[models]
g000 = "candidate-sha256-<v0.2 digest>"
g001 = "candidate-sha256-<g001 digest>"

[datasets.g002]
kind = "managed"
teacher = "g001"

[[iterations]]
name = "g002"
dataset = "g002"

[iterations.engine_gate]
base_model = "g001"
``` A new run starts only when the architecture or a major setting changes. Offline metrics compare models only within one generation, because each generation's labels come from a different teacher; games compare generations.

Baseline training uses `type = "baseline"` and accepts `device`, `seed`, `close_margin`, and `ridge_lambdas`. Pattern suites are `default`, `frozen-pattern-gain`, and `line-pattern-audit`.

Iteration dependencies enforce ordering and authentication only; they never copy a selected model or settings into a later iteration. The exploratory suites report their own validation winner, while `frozen-pattern-gain` always trains the fixed engine-compatible deployment contract. A recipe containing both roles is therefore a reproducibility and audit pipeline, not an automatic nested model-selection pipeline.

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

A promotable candidate can be gated against an existing engine build:

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
```

The candidate side is fixed to `minimax/poe2_minimax --evaluator pattern-gain`. The gate requires a promotable candidate. It configures a private build with `-DPOE2_MINIMAX_MODEL_HEADER` pointing at the candidate's authenticated header, runs the full test suite, and plays the match. The tracked engine header is never read or changed. Its build identity, `candidate-sha256-<header digest>`, is derived from the model header; the base engine and arguments remain configurable.

A gate sets exactly one base. `base` names an existing build ID, build directory, or engine binary. `base_model` names a `[models]` entry; the gate then builds that archived header privately under `engine-gate/builds/<model ID>/` from the same commit as the candidate, so both sides differ only in their weights. The ledger records the base's `base_id` as that model ID. A gate cannot use its own candidate as its base.

Every valid gate completes and records its sequential decision, including `accept_null` and `continue`, so a gate that does not show a win can still be promoted to the ledger and archived. Only handoff requires a win.

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

## Gate promotion

Gates run with `--no-ledger` and retain a self-contained `ledger-row.csv`. Preview and apply its publication explicitly:

```bash
cd training/minimax
CONFIG=recipes/RECIPE_NAME.toml
ITERATION=ITERATION_NAME

uv run --frozen poe2-train promote-gate --config "$CONFIG" "$ITERATION"
uv run --frozen poe2-train promote-gate --config "$CONFIG" "$ITERATION" --apply
```

Promotion requires a clean tree and the run lock. It authenticates the saved row against the gate manifest and summary, validates the master schema, coordinates with native ledger writers, and rejects duplicate run IDs before appending to `eval/results.csv`.

Promotion also archives the gated model, whether it won or not, in two files named by its build identity:

```text
models/
├── candidate-sha256-<header digest>.hpp    the exact gated model header
└── candidate-sha256-<header digest>.json   offline metrics and provenance
```

The ledger row's `new_id` is the same `candidate-sha256-<header digest>`, so every gated row leads to its model. The JSON records the architecture, recipe and iteration, Git commit, corpus, report digests, float and quantized validation and sealed-test metrics, parity, and benchmarks. It contains only fields fixed by the authenticated candidate, so it is byte-identical on every promotion attempt.

`models/index.csv` gives one readable row per promoted gate, in promotion order, so progress across runs shows without opening the archive:

| Column | Meaning |
|---|---|
| `recorded_utc` | When the gate started, from its manifest |
| `run`, `iteration` | The recipe run and iteration that produced the model |
| `model` | The first 12 hex digits of the model header digest; the archive files and the ledger's `new_id` carry the full digest |
| `base` | The build the candidate played against, or a base model's run-scoped name such as `run1/g000` |
| `base_model` | The first 12 hex digits of a base model's digest, empty for a build base |
| `decision` | The gate's sequential decision, such as `accept_alt` or `accept_null` |
| `games`, `score_pct` | Games played and the candidate's score, counting a no-winner game as half |
| `sealed_mae` | Exported quantized sealed-test MAE, empty without a sealed evaluation |
| `gate_run_id` | The ledger row's `run_id` |

The archive and the index row are written before the ledger row, so an interrupted promotion can be retried. Existing archive files and index rows are reused only if they match exactly; otherwise promotion stops.

An archived model can be rebuilt for analysis or a later re-gate by configuring a build with `-DPOE2_MINIMAX_MODEL_HEADER=models/candidate-sha256-<header digest>.hpp`.

## Candidate handoff

Handoff installs a gate winner as the shipped model. It previews the difference by default:

```bash
uv run --frozen poe2-train handoff --config "$CONFIG" "$ITERATION"
uv run --frozen poe2-train handoff --config "$CONFIG" "$ITERATION" --apply
```

Apply requires a promotable candidate, a completed engine gate whose decision is `accept_alt`, a committed ledger row and archive from gate promotion, a clean worktree, and the run lock. It copies the archived header over `engines/minimax/src/frozen_pattern_gain_model.hpp`, atomically and touching nothing else. Review and commit that file.

## Lower-level references

- [Position sources](position-source.md)
- [Labeling](labeling.md)
- [Feature artifacts](feature-data.md)
- [Training experiments and parity](../training/minimax/README.md)
- [Engine evaluation](evaluation.md)
