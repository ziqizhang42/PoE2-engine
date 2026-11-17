# Consolidated Training Workflow

`poe2-train` orchestrates the repository's existing source, labeling, feature, training, evaluation, export, parity, benchmark, and match tools. Those tools remain independently runnable and authoritative.

```text
run
├── datasets
│   ├── managed: source -> label shards -> features
│   └── imported: authenticate and pin completed features
├── iterations (TOML order)
│   └── training -> sealed evaluation? -> candidate validation? -> engine gate?
└── comparisons: a match between two models, once both exist
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

Run one iteration or [comparison](#comparisons) with Make, or include incomplete dependencies through the CLI:

```bash
make train-iteration TRAIN_ITERATION=ITERATION_NAME
make train-comparison TRAIN_COMPARISON=COMPARISON_NAME

cd training/minimax
CONFIG=recipes/RECIPE_NAME.toml
uv run --frozen poe2-train iteration --config "$CONFIG" \
  ITERATION_NAME --with-dependencies
uv run --frozen poe2-train comparison --config "$CONFIG" \
  COMPARISON_NAME --with-dependencies
uv run --frozen poe2-train status --config "$CONFIG" --json
```

`plan`, `validate`, and `status` are read-only; `validate` authenticates existing outputs. Mutating commands require a clean committed worktree and a tracked recipe. Repository-local run roots must be ignored; checked-in recipes use `build/`.

## Console progress

On an interactive terminal, mutating commands show one colored tqdm stage bar with elapsed time, estimated remaining time, and a compact postfix for the active source, label shard, build, or model. Completed stages and iterations each leave one short summary line; the full child-process output continues to be captured in the existing per-command log files instead of scrolling through the terminal.

Stages that reuse an existing complete artifact are labeled `authenticated`, so repeated dependency or exclusion preparation is distinguishable from artifact creation.

`status` prints the same ordered iteration graph, an overall stage bar, and cumulative recorded stage time. A completed gate or comparison also shows its recorded decision. A generation that is [not started](#chains) because an ancestor lost its gate shows as blocked with that reason, as does a comparison of a model that is never trained; once all other work is complete, the run is `stopped`. `status --json` remains undecorated and machine-readable. Redirected output is compact one-line stage reporting without ANSI control sequences.

The candidate promotion sequence is:

1. Commit a recipe whose candidate iteration includes an `engine_gate` table. A gate can also be added later in a compatible recipe revision, as long as that stage has not started.
2. Run the iteration: training, candidate validation, and the gate. Candidate validation exports the model as a net file, and the gate plays that file on an engine built from the run's commit, so nothing in the source tree changes.
3. Preview and apply gate promotion. This appends the ledger row, archives the model under `models/`, and adds a row to `models/index.csv`, whatever the decision. A generation that is gated against a model trained in the same recipe, or reads data taught by one, can be promoted only after that model is archived, so promote a chain's generations in order.
4. Review and commit the ledger row, archive, and index.
5. Only if the gate decision is `accept_alt`: preview and apply candidate handoff, then review and commit the installed built-in net and the version it raised.

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

A generation whose parent is trained by the same recipe learns from the parent's candidate straight from the run folder, so one `poe2-train run` can train several generations unattended. The child starts only once the parent's gate is complete and its decision is `accept_alt`, authenticated as promote-gate authenticates a gate. The parent does not need to be promoted or archived first. The child's data comes from the run's shared data tool loading the parent's candidate net file, and its gate plays the same net as its base on the gate's engine build. Every manifest records the SHA-256 of the parent's net, as it does for an archived teacher. A parent validated before nets were files is loaded from its converted net in `models/`, so it must be promoted first.

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

A model ID names a model archived under `models/`. Loading the recipe requires the archived net file (`.poe2net`) and `.json` to exist and the net to carry that ID; a model archived before nets were files also keeps its `.hpp`, which must match its digest. Position generation and labeling run the run's one data tool, `<run>/build/<preset>/runner/poe2_minimax_data`, with `--evaluator pattern-gain --net models/<ID>.poe2net`, so a teacher needs no build of its own. Source, label, and feature authentication require the recorded evaluator and net to match the recipe: data made from a net file records its SHA-256 as `net_sha256`, and data made before nets were files records the teacher's header digest as `model_header_sha256`, which still authenticates. A teacher is part of the source and label stage fingerprints, so it cannot change once those stages start. Datasets without a teacher keep the fingerprints they had before teachers existed.

`{ iteration = "NAME" }` names the model that an earlier iteration of this recipe trains, used as described under [chains](#chains). That iteration must have an engine gate, only later iterations may use its model, and an entry may share a name only with its own iteration. Keep this reference after the model is promoted: replacing it with the archived ID would change the fingerprints of the stages that already used it.

An archived `[models]` entry that shares an iteration's name must be the model that iteration produced. The workflow checks this once the iteration's candidate exists: when a run or `validate` starts, and before the entry is used as a teacher or gate base.

Baseline training uses `type = "baseline"` and accepts `device`, `seed`, `close_margin`, and `ridge_lambdas`. Pattern suites are `default`, `frozen-pattern-gain`, and `line-pattern-audit`.

Iteration dependencies enforce ordering and authentication only; they never copy a selected model or settings into a later iteration. The exploratory suites report their own validation winner, while `frozen-pattern-gain` always trains the fixed engine-compatible deployment contract. A recipe containing both roles is therefore a reproducibility and audit pipeline, not an automatic nested model-selection pipeline.

### Training data

A pattern iteration trains on its own dataset, the primary, and can also learn from the train records of other datasets of the recipe. Model selection, the sealed test, candidate parity, and every reported validation metric use the primary only.

In a compact recipe, these settings go in `[defaults.training]` or a generation's own `[generations.training]`:

```toml
[datasets.v02-corpus]
kind = "imported"
path = "build/data/features/pattern-scale-320k-s20260818/b-primitives"

[defaults.training]
type = "pattern"
device = "cuda"
suite = "frozen-pattern-gain"
window = 3
age_weight = 0.5

[[generations]]
name = "g004"
parent = "g003"
seed = 20261117
[generations.training]
refresh = 1
extra = [{ dataset = "g003b", weight = 0.3 }, { dataset = "v02-corpus", weight = 0.2 }]
max_records = 2000000
piece_records = 250000
```

- `window = N` (default 1) adds the datasets of the previous N - 1 ancestors along the parent chain that this recipe trained. A pinned ancestor has no dataset and ends the window. A growing window is just larger values on later generations.
- `age_weight` (default 1) multiplies the weight per generation of age: the parent's data counts `age_weight`, the grandparent's `age_weight²`, and so on.
- `refresh = K` (less than `window`) replaces the datasets of the K newest ancestors in the window with copies of their positions relabeled by this generation's parent, the teacher of its own new data. A copy is a dataset named `GENERATION-refresh-ANCESTOR` with the same positions and splits and this generation's label settings, at the weight of the dataset it replaces.
- `extra` names more datasets, each with a weight (default 1): any trained generation, such as a sibling that lost its gate, or an imported dataset. A compact recipe may declare imported datasets under `[datasets]`; generations make every managed one.
- `max_records`, `piece_records`, and `method` are described below.

These expand into the long form's `extra_datasets`, newest first: the window by age, then `extra` in order. In the long form:

```toml
[datasets.g001-relabeled]
kind = "managed"
teacher = "g002"
positions = "g001"

[datasets.g001-relabeled.labels]
mode = "teacher"
nodes = 5000000

[iterations.training]
type = "pattern"
suite = "frozen-pattern-gain"
extra_datasets = [{ dataset = "g001-relabeled", weight = 0.5 },
                  { dataset = "v02-corpus", weight = 0.2 }]
max_records = 2000000
piece_records = 250000
```

A managed dataset with `positions` instead of `source` labels another managed dataset's position source with its own teacher. It has only labels and features stages, keeps that source's settings and train, validation, and test splits, and its labels stage depends on that source stage. The actual teacher digest is locked as an input of its labels stage. The baseline training type does not take these options.

The training set is built in one place, the training selection in `poe2_training/dataset.py` (see [feature data](feature-data.md#training-selection)):

1. It starts from the primary's train split and each extra dataset's train split, in order. No extra dataset's validation or test records are ever used.
2. An extra record whose canonical position is in the primary's validation or test split is dropped, so those splits stay unseen by the model.
3. A position present in several datasets is kept only in the first, newest one: the primary, then the list order.
4. `max_records` caps the total. Whole datasets are taken newest first; the first one that does not fit contributes a deterministic subset of whole games (families, in a fixed hashed order); later ones contribute nothing. The cap counts the primary too.

The training report's `training_data` lists every dataset that contributed: its corpus ID, feature digests, teacher, weight, train records, the records dropped by the holdout rule, by a newer copy, and by the cap, and the records used. The candidate report and the archived model's JSON carry the same list as `training_datasets` when more than one dataset contributed. None of these fields appear, and no stage fingerprint changes, unless an option is set.

Each record's loss is weighted by its dataset's weight, the primary's being 1, and the weights are normalized to sum to one, so the data term stays a weighted average. The L2 term is then exactly as strong relative to the data as with one dataset: adding older data at a low weight neither weakens nor strengthens regularization, and scaling every weight changes nothing. Feature standardization, the gain-control ridge fit that initializes the model, and the per-ply means of gain-free models use the same weighted moments. Standardization therefore matches the distribution the loss sees, and the L2 penalty on standardized weights treats each feature consistently with it. Export folds standardization away, so the deployed tables do not depend on it. When every used record has the same weight, as with `age_weight = 1`, training uses the plain mean and unweighted moments.

Without `piece_records`, training expands every record's 36 line patterns into 1,716 dense float32 counts, about 6.9 KB per record, in host memory and on the device. With `piece_records = N`, it keeps compact data on the device for each record (the pattern IDs, 19 standardized gain summaries, ply, target, and weight) and expands dense counts N records at a time. Standardization comes from streaming sums, the gain-control fit from accumulated normal equations, and each full-batch Adam step accumulates its gradient over the pieces, so it is the same optimisation, not mini-batch training. Floating-point terms are added in a different order, and Adam can magnify that where a gradient is nearly zero. On test data, standardization was identical and gain-control weights agreed to 1e-15, but after about 1,300 steps line weights differed by up to 0.01, early stopping picked a step up to 50 later, and validation MAE differed by at most about 0.02%. That is about as much as the same in-memory run differs between CPU and GPU. The exported fixed-point tables can differ in individual entries, so keep one mode for a run. On the 303,599-record v0.2 corpus (189,246 train records), pieces of 50,000 cut peak host memory from 5.4 to 2.1 GiB and peak GPU allocation from 1.8 to 1.1 GiB, and took 3.2 times as long. The dense counts that pieces avoid grow by about 6.9 KB per record, so the saving matters once a window holds millions of records.

With `method = "lookup"` (the default is `"dense"`), training never builds the dense counts. A record's standardized line term is the sum, over its 36 patterns, of each pattern's weight divided by its count scale, minus one constant per phase component for the centering. So training keeps only the 36 pattern IDs per record, plus a fixed plan that groups every pattern occurrence by pattern, and looks the weights up. The parameters, standardization, gain-control fit, loss, L2 term, Adam settings, evaluation schedule, model selection, and record weights are the same as for the dense method. The gradient is summed over the plan in a fixed order instead of scattered with atomics, so runs repeat bit for bit with deterministic algorithms on; two GPU runs on the v0.2 corpus wrote byte-identical reports. `method` is fingerprinted only when it is `"lookup"`, works with every suite and the other training data options, and cannot be combined with `piece_records`, which only bounds the dense counts. Measured on this machine (RX 6800 XT, 16 GB, 30 GB host) with the frozen pattern/gain model and seed 20260818:

| Train records | Method | Time per step | Peak host | Peak GPU |
| --- | --- | --- | --- | --- |
| 189,246 (v0.2) | dense | 13.1 ms | 5.4 GiB | 1.8 GiB |
| 189,246 | pieces of 50,000 | 39.2 ms | 2.1 GiB | 1.1 GiB |
| 189,246 | lookup | 6.2 ms | 2.4 GiB | 0.3 GiB |
| 189,246 | lookup on the CPU | 33.4 ms | 1.9 GiB | none |
| 946,230 | dense | 78 ms | 20.3 GiB | 6.8 GiB |
| 946,230 | pieces of 50,000 | 208 ms | 2.4 GiB | 1.2 GiB |
| 946,230 | lookup | 45 ms | 4.3 GiB | 1.1 GiB |
| 2,081,706 | dense | does not fit: over 20 GiB host | | |
| 2,081,706 | pieces of 50,000 | 455 ms | 3.0 GiB | 1.4 GiB |
| 2,081,706 | lookup | 96 ms | 7.7 GiB | 2.4 GiB |
| 4,163,412 | lookup | 189 ms | 13.9 GiB | 4.6 GiB |

The larger sets repeat the v0.2 train split, and each step includes the evaluation every 25 steps. Dense needs about 21 KB of host memory per record while it standardizes and so stops fitting this machine between 1 and 1.5 million records. Lookup needs about 3.3 KB of host memory and 1.2 KB of GPU memory per record. On v0.2 the ten-model default suite trained in 86 s with lookup and 281 s dense. Lookup and dense results differ only by rounding, but Adam magnifies rounding late in training, so the selected step can move a long way: on v0.2 dense selected step 1,050 and lookup step 1,925. Validation MAE was 4.4280 and 4.4271, and sealed-test MAE 4.9285 and 4.9268. At the same step, weights agreed within 3e-5 after 100 steps and within 0.005 after 400 to 2,000 steps. Dense on the GPU and dense on the CPU differ as much from each other (steps 1,050 and 1,375). Fixed-point tables differ in most entries between any two of these runs, so keep one method within a run.

The swap file is on the system disk, so a command that runs out of memory must fail rather than swap. Two guards ensure that. Neither is a recipe setting, and neither enters a fingerprint or a record, because both depend on the machine.

- **Estimate.** Before it allocates anything large, the trainer estimates its peak host and GPU memory from the records it is about to use and its method. The per-record figures come from the measurements above. Measured peaks were within 0 to 11% of the estimates on the GPU, and up to 28% below them on the CPU. The trainer compares the estimate with free memory, less a margin: for the host, the smaller of what the machine has available and what its own memory limit leaves, less 5% of physical memory (at least 1 GiB); for the GPU, the card's free memory less 5% of it. Sealed evaluation checks its dense test table the same way. A refusal stops the stage before anything is created, so it is an ordinary recorded failure:

  ```text
  pattern experiment failed: pattern training (dense, 1,513,968 train and 74,157 validation records) needs about 31.5 GiB of host memory, but 17.9 GiB is free within this command's memory limit after a 1.5 GiB margin; nothing was allocated. Ways out: method = "lookup" (about 6.5 GiB host, 1.8 GiB GPU), piece_records = 50000 (about 2.9 GiB host, 1.3 GiB GPU), max_records, a smaller window, or more free memory.
  ```

  To continue, change the generation that has not started yet: switch it to `method = "lookup"`, add `piece_records`, cap it with `max_records`, or shrink its window. The trainer run by hand performs the same check.
- **Hard limit.** The workflow runs every command in its own systemd user scope, with a host memory limit of physical memory less 4 GiB kept for the rest of the system (26 GiB here) and swap switched off: `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`. The scope adds about 4 ms per command and leaves the process, its signals, its inherited run lock, its log and its exit status unchanged. Every command is wrapped, not only training, because any of them can run away and none may swap. A command that exceeds the limit is killed by the kernel. It is reported as `command was stopped for using more than its 24.0 GiB memory limit` and handled like any recorded failure.
- **Overrides.** `POE2_TRAINING_MEMORY_LIMIT` sets another limit (for example `20G`) or `off`. `POE2_MEMORY_ESTIMATE=off` skips the estimates; it is meant only for testing the limit.
- **Where systemd user scopes are unavailable**, or do not apply the limits, a run prints a warning, records a `memory_limit_unavailable` event, and runs its commands without a limit. The estimates still apply.

An iteration that needs a dataset which can never be produced, because that dataset's teacher did not win its gate, is not started and shows the reason, as under [chains](#chains).

### Comparisons

A comparison plays two models of the recipe against each other, separately from any gate. Typical uses are the newest generation against the starting model, and a generation that lost to its parent against an older generation. Each side names a `[models]` entry, archived or trained by the recipe; in a compact recipe, each names a generation.

```toml
[[comparisons]]
new = "g003"
base = "g000"

[[comparisons]]
name = "g002-vs-g000-long"
new = "g002"
base = "g000"
games = 4000
```

`name` defaults to `NEW-vs-BASE`. An entry takes the engine gate's match settings: `opening_book`, `games`, `workers`, `timeout_ms`, `go_movetime_ms`, `opening_seed`, and the `sequential_*` bounds. In a compact recipe it starts from `[defaults.engine_gate]` and overrides it key by key, which expands into the same long-form entry; in a long-form recipe, unset settings take the gate's defaults. Both sides always run `minimax/poe2_minimax --evaluator pattern-gain`, so the four engine fields may be given only with those values. Comparing a model with itself, by name or by model ID, is rejected.

A side trained by the recipe needs only its iteration's authenticated, promotable candidate. Its own gate does not have to accept it, because comparing a generation that lost is a fair question. The result goes under `<run>/comparisons/<name>/` and is laid out like a gate: the run's commit is built once under `builds/<engine build ID>/`, both nets play on that one engine, the evaluator run is under `runs/`, and the result is create-only, authenticated before reuse, and discarded if a pause interrupts it. Comparison builds skip the test suite, which each candidate's gate already ran.

`poe2-train run` plays each comparison as soon as both of its models exist, checking before the first iteration and after each one. Comparisons play in recipe order, one at a time and never alongside other work, because matches are timed. If a side is never trained, because its generation is not started after a lost gate, the comparison is not started either. `plan`, `status`, and `validate` show it with the reason, and `run` skips it without an error:

```text
  ■ g002 did not win its engine gate (decision accept_null); not started: g003, comparison g003-vs-g000
```

`poe2-train comparison NAME` plays one comparison, and `--with-dependencies` first runs the iterations that train its models.

A comparison is its own stage. Its fingerprint covers its match settings and both models' identities. An archived model is identified by its ID. A trained model is identified by its iteration and the fingerprint of that iteration's candidate-validation stage, which the recipe alone determines. The actual model IDs are locked as inputs when the comparison first starts. Adding or removing a comparison that has not started changes no other stage's fingerprint; like any stage, a started one cannot change or be removed. A comparison never decides whether a chain continues and never makes a model installable.

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

Candidate validation exports the selected model as a net file, `candidate/net.poe2net` ([nets.md](nets.md)), named `<run>/<iteration>` and recording its teacher, parent, training date, corpus, and the digests of its training report and features. It builds the inference tool and engine once in the run's shared build, checks Python/C++ and D4 parity with the inference tool loading the net, benchmarks search with the engine loading it, and records digests, logs, resources, and separate float and exported-quantized metrics. The candidate's model ID is `candidate-sha256-<net file SHA-256>`. The deployable frozen suite fixes its quantization at the engine's scale-32 contract before sealed evaluation. A candidate validated before nets were files holds the C++ header and private build it was validated with, and still authenticates.

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

The candidate side is fixed to `minimax/poe2_minimax --evaluator pattern-gain`. The gate requires a promotable candidate. It builds the run's commit once under `engine-gate/builds/<engine build ID>/`, runs the full test suite, and plays the match with the candidate's net file (`--new-net`). The engine build ID is `<commit count>-<short SHA>`, as `make git-build` names builds; it is the ledger row's `new_id`, and the ledger also records each side's net. The base engine and arguments remain configurable.

A gate sets exactly one base. `base` names an existing build ID, build directory, or engine binary, which plays its own built-in net, so an engine from before nets were files can be a base. `base_model` names a `[models]` entry; the gate then plays that model's net (`--base-net`) on the same engine build as the candidate, so both sides differ only in their nets, and the ledger's `base_id` is the same engine build ID. A gate cannot use its own candidate as its base. Gates played before nets were files built each model's header privately under `builds/<model ID>/`; they still authenticate, but a candidate validated before nets were files cannot start a new gate.

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
├── datasets/<name>/{source-chunks,source,labels/shards,features}   relabeled: labels, features
├── iterations/<name>/{training/{report.json,training-metrics.svg},sealed-evaluation,candidate,engine-gate}
└── comparisons/<name>/{builds/<engine build ID>,runs}
```

The run's data tool, inference tool, and engine share `build/<preset>/`, and each gate and comparison builds its commit once in its own folder. A private build reuses a Catch2 checkout already on disk instead of fetching it from GitHub when Git confirms it is a clean checkout of the tag pinned in `tests/CMakeLists.txt`, cloned from the pinned URL. The first one found, from the run's own builds and then from the repository's `build/*/`, is copied to `build/catch2-<tag>/` in the run folder and verified again before each use; otherwise CMake fetches as usual. Engine, runner, and data binaries are byte-identical either way. Once a stage starts, its configuration and input digests cannot change or be removed; unstarted stages and new iterations may be added in a later compatible recipe revision.

Artifacts are create-only and authenticated before reuse. To pause a run, interrupt it with Ctrl-C, `SIGTERM`, or `SIGHUP` (closing its tmux session sends this). The workflow terminates its subprocess groups and waits for them to exit, then removes every artifact it had started in that invocation but not completed, such as an in-progress stage, source chunk, label shard, or a private build tree that was still building. It records each removal as an `interrupted_artifact_removed` event and prints it. Completed artifacts, including finished label shards, are kept, and paths it did not start are never touched, so rerunning the same command resumes. The position source is generated in chunks of 2,000 trajectories under `datasets/<name>/source-chunks/` and then assembled, so a pause during generation loses at most one chunk. Finished chunks are reused on resume, and must come from the same build commit as the rest, like label shards. The chunk size is not part of any fingerprint, because assembly is byte-identical for any chunk size. A failure that is not an interruption leaves its incomplete artifact for inspection: the workflow reports its exact path and stops. Inspect it, then manually move aside or remove only that path before retrying.

After a crash or power loss, rerun the same command. Before the workflow creates an artifact, it appends a synced `artifact_started` event naming it to `state/events.jsonl`, and it records the outcome when the artifact is finished, fails, or is removed. Every mutating command (`run`, `iteration`, `comparison`, and the `--apply` forms of `promote-gate`, `record-comparison`, and `handoff`) first takes the run lock and looks for artifacts that were started with no recorded outcome. Each such artifact that is not marked COMPLETE is a crash leftover: the command removes it, records a `crash_artifact_removed` event, and prints it the way a pause does, then continues as a resume. A stage attempt that was running is recorded as abandoned, so its inputs stay locked and the next attempt gets a new number. A private build tree that was still building is removed and rebuilt from scratch, because a build tool can take a truncated output with a fresh timestamp for an up-to-date one. Nothing is marked COMPLETE until its files and folders, and the command logs its report records, are synced to disk. A COMPLETE artifact that fails authentication is therefore treated as damage: it is reported and never removed automatically. Remaking it changes the inputs of any later stage that had already started, whose inputs stay locked from its first attempt, so such a run cannot continue in place unless the original artifact is restored. A recorded failure is kept for inspection as before. Paths without a start record are never touched, including everything in a run folder from before crash recovery existed, and neither is anything outside the run folder. If a crash cut the event log's last line short, the next mutating command drops that one record and records an `event_log_repaired` event with the number of bytes dropped. Read-only commands ignore it. Damage anywhere else in the log is still an error. Commands inherit the run lock, so if only the orchestrator was killed, the next command reports the lock as held until the commands it started have exited. The `--apply` commands also append to tracked files in the repository. If one stops midway, `git status` shows what it wrote. Restore those files, remove any new archive files, and rerun the apply.

A run folder inside the repository moves with it. Stage fingerprints, locked inputs, and every recorded path that is checked later (opening books, imported datasets, match runs and their builds, and the paths in saved ledger rows) name paths relative to the repository. The match runner is given those relative paths too. A finished or unfinished run therefore keeps its status, plan, validation, and resume after the repository is moved or copied, and its gates and comparisons can still be promoted and recorded. `validate` lists unfinished artifacts left by a stopped run instead of refusing them; the next mutating command removes them. Some things do not move:

- A path outside the repository is recorded as an absolute path: an imported dataset, a run folder, or a gate `base` outside the repository must stay where it is. So must a gate `base` written as an absolute path.
- Stages and reports recorded before paths were relative name them absolutely. They still validate where they were recorded, but not after a move. The pilot's `train-final-model` candidate validation is the only such stage. A copy of the pilot run at another path shows its status and plan, but `validate` and `run` stop at that stage and name the recipe location it was recorded at.
- CMake refuses a build tree configured at another path. Finished stages only authenticate the binaries they recorded and rebuild nothing. A moved run that configures a private build tree again, such as a teacher's data tool at a new commit, removes the moved tree, records `moved_build_removed`, prints it, and rebuilds the tree from scratch.
- Logs, events, summaries, and the command lines in reports keep the absolute paths of the place they ran. Nothing compares them.
- The training environment in `training/minimax/.venv` has absolute paths. `uv run`, which the make targets use, repairs it after a move.

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

A generation whose gate base, or the teacher of any dataset it reads, is a model trained by the same recipe is promoted only after that model is archived, so every archived model's teachers are archived too. Promote a chain in generation order, committing each result before the next. Recording stays manual: a chain that ran unattended is promoted afterwards, one generation at a time.

Promotion also archives the gated model, whether it won or not, in files named by its model ID:

```text
models/
├── candidate-sha256-<digest>.poe2net   the net file the engine loads
├── candidate-sha256-<digest>.json      training record, offline metrics, and provenance
└── candidate-sha256-<digest>.hpp       only for g000 and g001, trained before nets were files
```

The digest is the net file's SHA-256. For `run1/g000` and `run1/g001`, trained before nets were files, it is the SHA-256 of their C++ header: their net files were converted from those headers, carry the digest as `header_sha256`, and evaluate identically. A net's ID, shown in the index and reported by the engine, is the first 12 hex digits of the digest. Every gated row leads to its model: through `new_net_sha256` for gates that played net files, and through `new_id`, which was the model ID, for gates played before.

The JSON (schema version 2) records the net's name, ID, SHA-256, design, and parameter count; its lineage (parent and teacher net names, or `two-ply-closure`); how it was trained (method, training positions, distinct games behind them when every dataset was generated by a recipe, and the selected model's training curve); and the architecture, recipe and iteration, Git commit, corpus, report digests, float and quantized validation and sealed-test metrics, parity, and benchmarks. It contains only fields fixed by the authenticated candidate and its run, so it is byte-identical on every promotion attempt.

`models/index.csv` gives one readable row per promoted gate, in promotion order, so progress across runs shows without opening the archive:

| Column | Meaning |
|---|---|
| `recorded_utc` | When the gate started, from its manifest |
| `run`, `iteration` | The recipe run and iteration that produced the model |
| `model` | The net ID: the first 12 hex digits of the model ID's digest |
| `base` | The build the candidate played against, or a base model's run-scoped name such as `run1/g000` |
| `base_model` | The first 12 hex digits of a base model's digest, empty for a build base |
| `decision` | The gate's sequential decision, such as `accept_alt` or `accept_null` |
| `games`, `score_pct` | Games played and the candidate's score, counting a no-winner game as half |
| `sealed_mae` | Exported quantized sealed-test MAE, empty without a sealed evaluation |
| `gate_run_id` | The ledger row's `run_id` |
| `net_sha256` | The SHA-256 of the archived net file |
| `teacher` | The net that labeled the training data, or `two-ply-closure` |
| `method` | The pattern-training method, such as `dense` or `lookup` |
| `data_positions`, `data_games` | Training positions, and the distinct games they came from; empty when imported data is involved |

Rows recorded before these columns existed are completed from the archived records the next time the index is written.

The archive and the index row are written before the ledger row, so an interrupted promotion can be retried. Existing archive files and index rows are reused only if they match exactly; otherwise promotion stops.

Any engine, inference tool, or data tool loads an archived model for analysis or a later match with `--net models/candidate-sha256-<digest>.poe2net`.

## Comparison recording

`record-comparison` previews or records a finished comparison, like promote-gate:

```bash
uv run --frozen poe2-train record-comparison --config "$CONFIG" COMPARISON_NAME
uv run --frozen poe2-train record-comparison --config "$CONFIG" COMPARISON_NAME --apply
```

Both models must already be in the `models/` archive, so every hash in the record resolves. Promote the gate of each trained side first; promote-gate archives a model whatever its decision. Recording authenticates the comparison and its saved ledger row the way promote-gate checks a gate. It then appends a row to the tracked `models/comparisons.csv` and the ledger row to `eval/results.csv` with kind `training-comparison`. Like promote-gate, it requires a clean tree and the run lock, rejects a run ID already in the ledger, and reuses an identical index row on a retry. Review and commit both files.

| Column | Meaning |
|---|---|
| `recorded_utc` | When the match started, from its manifest |
| `run`, `comparison` | The recipe run and the comparison's name |
| `new`, `base` | Each side's run-scoped name, such as `run1/g003` |
| `new_model`, `base_model` | Each model's net ID |
| `decision` | The sequential decision for `new` against `base` |
| `games`, `score_pct` | Games played and `new`'s score, counting a no-winner game as half |
| `run_id` | The ledger row's `run_id` |
| `new_net_sha256`, `base_net_sha256` | The SHA-256 of each side's net file |

`models/index.csv` stays one row per promoted gate. The ledger convention that the newest accepted gate is the champion applies to gate rows (kind `training-gate`), not to comparison rows (kind `training-comparison`): a comparison that ends in `accept_alt` does not make its new model the champion.

## Candidate handoff

Handoff installs a gate winner as the engine's built-in net. It previews the difference by default:

```bash
uv run --frozen poe2-train handoff --config "$CONFIG" "$ITERATION"
uv run --frozen poe2-train handoff --config "$CONFIG" "$ITERATION" --apply
```

Apply requires a promotable candidate, a completed engine gate whose decision is `accept_alt`, a committed ledger row and archive from gate promotion, a clean worktree, and the run lock. It copies the archived net file over `engines/minimax/nets/default.poe2net` and raises the last number of the project version in `CMakeLists.txt` (for example 0.3.0 to 0.3.1), each atomically and touching nothing else; the preview shows both versions. A candidate validated before nets were files, such as `run1/g001`, installs the net archived for it, converted from its header. Review and commit both files, and add a changelog line naming the new version and its built-in net ([nets.md](nets.md#the-built-in-net)). If the command is cut short, restore both files with Git and rerun it.

## Lower-level references

- [Position sources](position-source.md)
- [Labeling](labeling.md)
- [Feature artifacts](feature-data.md)
- [Training experiments and parity](../training/minimax/README.md)
- [Engine evaluation](evaluation.md)
