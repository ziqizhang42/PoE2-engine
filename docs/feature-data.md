# Deterministic Minimax Feature Data

The feature-data boundary combines every completed label shard, verifies global provenance and split safety, removes repeated D4-canonical positions, and exports primitive evaluator inputs for offline learning.

## Corpus preflight

Run the reusable read-only loader before exporting features:

```bash
python3 -B tools/preflight_minimax_corpus.py \
  artifacts/labels/example-corpus \
  --source artifacts/sources/example-corpus \
  --json
```

The preflight independently audits the source and every label shard, requires homogeneous search and build provenance, matches every label to its source record, and rejects family, trajectory, or canonical-position leakage across train, validation, and test. Repeated canonical positions are represented once. The representative is the deepest completed terminal-parity label; equal-depth labels must have equal values, and remaining ties use trajectory/sample/source order. The record retains the duplicate multiplicity so corpus accounting remains exact.

## Feature export

Export a single immutable feature artifact with the production C++ evaluator:

```bash
build/release/runner/poe2_minimax_data features \
  --source artifacts/sources/example-corpus \
  --labels artifacts/labels/example-corpus \
  --output-dir artifacts/features/example-corpus

python3 -B tools/inspect_minimax_features.py \
  artifacts/features/example-corpus \
  --labels artifacts/labels/example-corpus
```

Each selected board stores the selected, deepest, and previous search results. Occupied cells use signed 16-bit minimum as a sentinel. No learned feature, weight, phase interpolation, or quantization decision is embedded at this stage.

The output directory is reserved before writing and is never overwritten. The `manifest.json` file authenticates the combined label inputs, records both label-generation and exporter build provenance, defines every primitive encoding, and reports deduplication and split counts.

## Line-pattern definition

The 36 scoring lines are ordered as seven rows, seven columns, eleven down-right diagonals, and eleven down-left diagonals. Within a line, the first cell is the least-significant base-three digit. Digits are `0=empty`, `1=side to move`, and `2=opponent`; the raw pattern identifier is the length-specific offset plus that base-three code. Reversal tying and phase interpolation are deliberately left to the later learning experiment.

## Binary schema

All integers are little-endian. The file contains one 128-byte header followed by fixed 432-byte records. Scores are signed 32-bit doubled margins. Gains are signed 16-bit raw score gains.

Header:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Magic bytes `POE2FTR\0` |
| 8 | 4 | Schema version |
| 12 | 4 | Header size |
| 16 | 4 | Record size |
| 20 | 4 | Endian marker (`0x01020304`) |
| 24 | 8 | Selected record count |
| 32 | 8 | Original source record count |
| 40 | 8 | Duplicate records removed |
| 48 | 32 | SHA-256 digest of the logical corpus ID |
| 80 | 32 | SHA-256 digest of ordered label binary/manifest digests |
| 112 | 4 | Scoring-line count |
| 116 | 4 | Cell count |
| 120 | 4 | Input shard count |
| 124 | 4 | Reserved zero |

Record:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 16 | Raw P1 and P2 bitboards |
| 16 | 16 | D4-canonical position key |
| 32 | 40 | Source, family, trajectory, parent, and trajectory-index IDs |
| 72 | 32 | Total, selected, deepest, and previous node counts |
| 104 | 8 | Source shard and shard-local ordinal |
| 112 | 24 | Selected, deepest, previous, static, two-ply-closure, and residual scores |
| 136 | 4 | Duplicate multiplicity |
| 140 | 4 | Policy and sample IDs |
| 144 | 16 | Ply, side, split, mode, depths, moves, flags, and reserved zeros |
| 160 | 72 | 36 unsigned 16-bit line-pattern IDs |
| 232 | 98 | 49 signed 16-bit side-to-move gains |
| 330 | 98 | 49 signed 16-bit opponent gains |
| 428 | 4 | Reserved zero |

The manifest's `inputs.label_search` carries the label evaluator. Every shard must share it, so a learned teacher's `model_header_sha256` reaches the features, and shards labeled by different evaluators cannot be exported together.

The independent auditor verifies framing and digests, canonical uniqueness and ordering, split and multiplicity accounting, line-pattern encoding, normalized static scores, two-ply-closure arithmetic, and a deterministic sample of marginal gains recomputed from the game rules.

## Training selection

`poe2_training.dataset.TrainingSelection` reads the records a pattern model trains on from one primary artifact and an ordered list of weighted extra artifacts, newest first. It applies the one leak rule for multi-dataset training:

1. Only train-split records are read; no extra artifact's validation or test records are used.
2. An extra record whose D4-canonical key is in the primary's validation or test split is dropped.
3. A key present in several artifacts is kept only in the first: the primary, then the list order.
4. An optional record cap takes whole artifacts in order, then a deterministic subset of whole families (games) of the first artifact that does not fit, in SplitMix64 order of family ID, and nothing after it.

It reports, per artifact, the train records, the records dropped by each rule, and the records used. It can materialize every selected record at once, or yield pieces of at most a given number of records with their weights, so a consumer never holds more than one piece of wide columns. Key membership uses the same sorted canonical-key search as the sealed-test overlap exclusion.

The isolated offline consumer and its manual uv/PyTorch setup are documented in [`training/minimax/README.md`](../training/minimax/README.md).
