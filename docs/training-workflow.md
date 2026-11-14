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

Each iteration prepares its dataset automatically. Dependencies may name earlier iterations and must be complete unless `--with-dependencies` is used. A run of model generations is usually written as a [compact recipe](#generations), which expands into these datasets and iterations.

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

`status` prints the same ordered iteration graph, an overall stage bar, and cumulative recorded stage time. A completed gate also shows its recorded `gate_decision`, and a generation that is [not started](#chains) because an ancestor lost its gate shows as blocked with that reason; once every other iteration is complete, the run is `stopped`. `status --json` remains undecorated and machine-readable. Redirected output is compact one-line stage reporting without ANSI control sequences.

The candidate promotion sequence is:

1. Commit a recipe whose candidate iteration includes an `engine_gate` table. A gate can also be added later in a compatible recipe revision, as long as that stage has not started.
2. Run the iteration: training, candidate validation, and the gate. The gate builds the candidate's model header privately, so nothing in the source tree changes.
3. Preview and apply gate promotion. This appends the ledger row, archives the model under `models/`, and adds a row to `models/index.csv`, whatever the decision. A generation taught by or gated against a model trained in the same recipe can be promoted only after that model is archived, so promote a chain's generations in order.
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

### Generations

A run trains one architecture over successive generations named `g000`, `g001`, …. `g000` is the run's starting model. Each later generation is trained on positions sampled and labeled by searches that use its parent's model, and is gated against its parent. A new run starts only when the architecture or a major setting changes. Offline metrics compare models only within one generation, because each generation's labels come from a different teacher; games compare generations.

A run is most simply written as a compact recipe: shared defaults plus one short entry per generation.

```toml
schema_version = 1

[run]
name = "run1"
output_dir = "build/training-runs/run1"
build_preset = "release"

[defaults.source]
trajectories = 40000
shards = 160
workers = 6
search_nodes = 100000

[defaults.labels]
mode = "teacher"
nodes = 5000000
workers = 6

[defaults.training]
type = "pattern"
device = "cuda"
suite = "frozen-pattern-gain"

[defaults.sealed_evaluation]

[defaults.candidate_validation]
promotable = true

[defaults.engine_gate]
games = 2000
workers = 6

[[generations]]
name = "g000"
model = "candidate-sha256-<v0.2 digest>"

[[generations]]
name = "g001"
parent = "g000"
seed = 20261114

[[generations]]
name = "g002"
parent = "g001"
seed = 20261115
[generations.source]
search_weight = 60
```

A compact recipe is shorthand only. It expands into the long form described below before parsing, so its stages, fingerprints, and artifacts are exactly those of the equivalent long-form recipe; `run1.toml` was rewritten this way without changing a fingerprint. A recipe uses one form, not both.

- A generation with `model` is pinned: an archived model that this recipe does not train. It takes no other field.
- Every other generation needs a `parent`, which names an earlier generation, and a `seed`. It becomes a dataset and an iteration, both named after the generation. The parent is the dataset's `teacher` and the gate's `base_model`.
- `seed` is both the position-source seed and the training seed, and the corpus ID is `poe2-<run>-<generation>-v1`. A generation's own `[generations.source]` or `[generations.training]` can override either.
- `[defaults.TABLE]` holds settings for any of `source`, `labels`, `features`, `training`, `sealed_evaluation`, `candidate_validation`, and `engine_gate`. A generation's `[generations.TABLE]` overrides them key by key. Defaults cannot set a corpus ID or seed, and no table sets a gate base. A table absent from both is omitted; every trained generation needs candidate validation and an engine gate.
- Every trained generation is also a model of the recipe named after itself, so later generations can name it as their parent.

#### Chains

A generation whose parent is trained by the same recipe learns from the parent's candidate straight from the run folder, so one `poe2-train run` can train several generations unattended. The child starts only once the parent's gate is complete and its decision is `accept_alt`, authenticated as promote-gate authenticates a gate. The parent does not need to be promoted or archived first. The child's data tool is built from the parent's candidate header under `<run>/build/teacher-iteration-<parent>/`, and its gate builds the same header privately as its base, from the same commit as the candidate. Every manifest records the parent's model digest, as it does for an archived teacher.

A child's stage fingerprints name its parent by iteration and by the fingerprint of the parent's gate stage, which the recipe alone determines, so `plan` works before the parent exists. The parent's actual model ID is locked as an input of the child's stages when they first start, like other upstream inputs.

`poe2-train run` executes generations in order. When a gate ends with anything other than `accept_alt`, including the inconclusive `continue`, no generation that descends from it is started. The command still finishes normally and names the generation that did not win and the ones it did not start:

```text
■ run stopped  run=run1  elapsed=…
  ■ g002 did not win its engine gate (decision accept_null); not started: g003, g004
```

`plan`, `status`, and `validate` show such generations as not started with the same reason, and `poe2-train iteration` refuses to run one. The number of generation entries in a recipe therefore sets how far a chain can go. To continue after a loss, append a generation whose parent is an earlier winner, with a new seed, in a compatible recipe revision:

```toml
[[generations]]
name = "g004"
parent = "g001"
seed = 20261117
```

A generation that was never started can also be removed from the recipe; only started stages are locked. `poe2-train iteration NAME --with-dependencies` also runs the generations that NAME descends from.

#### Long form

The long form spells out every dataset and iteration. A managed dataset can name a teacher whose learned evaluator drives the noisy-search positions and the labels; without one, both use the two-ply-closure evaluator. A teacher and a gate's `base_model` name entries of `[models]`, which gives recipe-local names to two kinds of model:

```toml
[models]
g000 = "candidate-sha256-<v0.2 digest>"
g001 = { iteration = "g001" }

[datasets.g002]
kind = "managed"
teacher = "g001"

[[iterations]]
name = "g002"
dataset = "g002"

[iterations.engine_gate]
base_model = "g001"
```

A model ID names a model archived under `models/`. Loading the recipe requires the archived `.hpp` and `.json` to exist and the header to match its digest. The workflow builds a private data tool for each archived teacher under `<run>/build/teacher-<digest prefix>/`, with `-DPOE2_MINIMAX_MODEL_HEADER` pointing at the archived header, and passes `--evaluator pattern-gain` to position generation and labeling. Source, label, and feature authentication require the recorded evaluator and model digest to match the recipe. A teacher is part of the source and label stage fingerprints, so it cannot change once those stages start. Datasets without a teacher keep the fingerprints they had before teachers existed.

`{ iteration = "NAME" }` names the model that an earlier iteration of this recipe trains, used as described under [chains](#chains). That iteration must have an engine gate, only later iterations may use its model, and an entry may share a name only with its own iteration. Keep this reference after the model is promoted: replacing it with the archived ID would change the fingerprints of the stages that already used it.

An archived `[models]` entry that shares an iteration's name must be the model that iteration produced. The workflow checks this once the iteration's candidate exists: when a run or `validate` starts, and before the entry is used as a teacher or gate base.

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

Every valid gate completes and records its sequential decision, including `accept_null` and `continue`, so a gate that does not show a win can still be promoted to the ledger and archived. Only handoff, and a [chain](#chains) continuing past the gate, require a win.

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
├── datasets/<name>/{source-chunks,source,labels/shards,features}
└── iterations/<name>/{training/{report.json,training-metrics.svg},sealed-evaluation,candidate,engine-gate}
```

Build trees are private to each run or candidate. Once a stage starts, its configuration and input digests cannot change or be removed; unstarted stages and new iterations may be added in a later compatible recipe revision.

Artifacts are create-only and authenticated before reuse. To pause a run, interrupt it with Ctrl-C, `SIGTERM`, or `SIGHUP` (closing its tmux session sends this). The workflow terminates its subprocess groups and waits for them to exit, then removes every artifact it had started in that invocation but not completed, such as an in-progress stage or label shard. It records each removal as an `interrupted_artifact_removed` event and prints it. Completed artifacts, including finished label shards, are kept, and paths it did not start are never touched, so rerunning the same command resumes. The position source is generated in chunks of 2,000 trajectories under `datasets/<name>/source-chunks/` and then assembled, so a pause during generation loses at most one chunk. Finished chunks are reused on resume, and must come from the same build commit as the rest, like label shards. The chunk size is not part of any fingerprint, because assembly is byte-identical for any chunk size. A failure that is not an interruption leaves its incomplete artifact for inspection: the workflow reports its exact path and stops. Inspect it, then manually move aside or remove only that path before retrying.

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

A generation whose teacher or gate base is a model trained by the same recipe is promoted only after that model is archived, so every archived model's teacher is archived too. Promote a chain in generation order, committing each result before the next. Recording stays manual: a chain that ran unattended is promoted afterwards, one generation at a time.

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
