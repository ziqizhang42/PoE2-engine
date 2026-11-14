# Deterministic Position Sources

`poe2_minimax_data source` creates the unlabeled positions that a later labeling run will search. The source and label stages are separate on purpose: position selection can be audited once, while different deterministic search budgets can label the same immutable inputs without silently changing the training population.

No source corpus is created by a build or test beyond small fixtures under the build directory. Creating a real corpus is an explicit command.

## Generate and audit

```bash
cmake --build --preset release --target poe2_minimax_data

build/release/runner/poe2_minimax_data source \
  --output-dir artifacts/sources/example-corpus \
  --corpus-id poe2-example-corpus \
  --seed 20260817 \
  --trajectories 5000 \
  --samples-per-trajectory 8 \
  --shards 20 \
  --workers 6 \
  --search-nodes 10000 \
  --search-hash-mb 8 \
  --noise-percent 15 \
  --random-weight 30 \
  --greedy-weight 20 \
  --opponent-weight 20 \
  --search-weight 30 \
  --progress-every 100

python3 tools/inspect_position_source.py artifacts/sources/example-corpus
```

The example values describe the planned policy mixture, not a requirement to launch that run. `--seed` is mandatory and may be zero. A trajectory owns an independent SplitMix64 stream with a specified rejection-based bounded sampler. Work scheduling therefore cannot affect its moves. Shards made with one worker and six workers contain identical bytes; the manifest records the requested worker count and therefore intentionally differs when that setting differs.

## Chunked generation

A long source can be generated in resumable pieces. `source-chunk` takes the same options plus a trajectory range and writes one create-only chunk directory: `chunk.jsonl` holds the trajectories' records before splits and source IDs exist, and `COMPLETE` records its SHA-256. `source-assemble` reads every chunk in `--chunks`, then performs the corpus-wide steps (D4 duplicate grouping, split assignment, and source IDs) and writes the usual source artifact.

```bash
build/release/runner/poe2_minimax_data source-chunk <source options> \
  --output-dir artifacts/chunks/chunk-0 --trajectory-begin 0 --trajectory-end 2000
build/release/runner/poe2_minimax_data source-assemble <source options> \
  --chunks artifacts/chunks --output-dir artifacts/sources/example-corpus
```

Because every trajectory depends only on the seed and its index, an assembled source is byte-identical to one `source` run with the same options, for any chunk boundaries and worker counts. Assembly refuses chunks that leave gaps or overlap, whose generation settings differ, whose digest fails, or that a different build commit, dirty state, or compiled model produced.

## What is sampled

One policy is selected for each trajectory:

- `random`: uniformly selects a legal square.
- `immediate-gain`: selects the move with the largest exact non-mutating score gain.
- `opponent-aware`: selects the exact immediate two-ply key: own gain minus the opponent's best remaining gain.
- `noisy-search`: normally uses deterministic node-limited search and selects a random legal move at the configured noise percentage. The search uses the two-ply-closure evaluator unless `--evaluator pattern-gain` selects the compiled learned model. The manifest's `generator.evaluator` records the choice, and a pattern-gain source also records `generator.model_header_sha256`. Sources written before this field existed used two-ply closure.

Ties are resolved by the trajectory's deterministic random stream. Each trajectory contributes at most one position from each of eight phase buckets spanning plies 4 through 46. With fewer than eight requested samples, the buckets are selected deterministically and then emitted in ply order. This avoids treating every adjacent state in a game as an independent example while retaining early-, middle-, and late-game coverage.

The generator stores both a family ID and trajectory ID, the global trajectory ordinal, policy, sample ordinal, ply, and raw bitboards. Those fields are part of the source-position ID. The IDs are not inferred later from file names or move-history text.

## Frozen splits and duplicates

Train, validation, and test assignment happens before labels exist. The default deterministic ratio is 70/15/15 at the trajectory-family level. D4-equivalent duplicate positions are detected across the complete corpus; trajectories connected by a duplicate are unioned before splitting. Consequently, the same canonical board cannot leak across partitions.

Exact duplicate records remain visible rather than being silently discarded. The manifest and auditor report their count, which leaves a later training pipeline free to deduplicate or weight them explicitly.

## Artifact transaction

The output directory is reserved before trajectory generation. Existing paths are refused. A failed or interrupted run remains visibly incomplete; readers reject it.

```text
example-corpus/
  COMPLETE
  manifest.json
  shards/
    shard-00000000-<sha256>.jsonl
    shard-00000001-<sha256>.jsonl
    ...
```

Each shard is canonical JSON Lines: one configuration header followed by position records. Its SHA-256 digest is embedded in its filename. The manifest lists the same digest, trajectory range, and record count. `COMPLETE` is installed last, after every file has been synced to disk, and authenticates the manifest, which in turn authenticates every shard. Shards are contiguous trajectory ranges, so a family is never split between files and an interrupted labeling campaign can restart shard by shard.

The independent Python auditor checks the transaction, SHA-256 chain, configuration agreement, bitboard legality, piece counts, source IDs, generated family/trajectory IDs, phase sampling, trajectory order, policy and split counts, D4 duplicates, and cross-split leakage.

## Label a verified shard

Pass the completed source directory and a zero-based source shard. The corpus identity, source digest, shard count, and every record's provenance are read from the verified artifact:

```bash
build/release/runner/poe2_minimax_data labels \
  --input artifacts/sources/example-corpus \
  --source-shard 0 \
  --output-dir artifacts/labels/example-shard-000 \
  --mode teacher \
  --nodes 1000000 \
  --hash-mb 64 \
  --workers 6 \
  --progress-every 100 \
  --require-all
```

`--corpus-id` is unnecessary for this path; if supplied, it must match the source. The older opening-book input remains supported for small fixtures and external corpora. It still requires an explicit corpus ID and treats each line as its own training family.
