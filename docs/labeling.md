# Deterministic Minimax Labels

`poe2_minimax_data` turns a completed deterministic position source, or a validated opening book, into reproducible minimax labels. It is the data boundary between production two-ply-closure search and offline evaluation experiments. Position generation and model training are deliberately separate. See [Deterministic Position Sources](position-source.md) for the preferred metadata-aware input and [Deterministic Minimax Feature Data](feature-data.md) for corpus-wide preflight, deduplication, and primitive feature export.

## Build and run

```bash
cmake --build --preset release --target poe2_minimax_data

build/release/runner/poe2_minimax_data labels \
  --input artifacts/sources/pattern-pilot \
  --source-shard 0 \
  --output-dir artifacts/labels/pilot-shard-000 \
  --mode teacher \
  --nodes 5000000 \
  --hash-mb 16 \
  --workers 6 \
  --progress-every 100 \
  --require-all
```

For a source-corpus directory, `--source-shard` selects one immutable shard. The reader verifies the completion marker, manifest digest, digest-named shard, fixed header, position metadata, ordering, and legality before search begins. Corpus and shard provenance come from that source artifact.

The existing opening-book format remains supported: one legal move prefix per non-comment line. Malformed, illegal, or terminal openings are rejected by the same parser used by the match runner. For that input only, `--corpus-id` is required and `--shard-index`/`--shard-count` describe the supplied file. The label command never generates or divides a corpus.

The required mode is one of:

- `exact`: requests the remaining terminal depth and emits a record only when that entire search finishes. Add `--require-all` when any unsolved input should fail the job.
- `teacher`: uses the requested node budget as a hard cap and selects the deepest completed result whose depth parity matches the known terminal depth. The deepest and immediately previous completed results are retained alongside that training target. A teacher record remains identified as a teacher record even if it reaches terminal depth.

Both modes require a positive `--nodes` budget. Every position starts with cleared transposition and history state. Search uses board-symmetry canonicalization and the production two-ply-closure evaluator. No clock, random seed, scheduling decision, progress message, path to the output directory, or wall-clock timestamp enters the artifacts.

`--workers` defaults to one. Each worker owns an isolated search, history, and transposition table; inputs are scheduled dynamically but restored to source order before serialization. Binary records are therefore independent of worker count. The manifest records requested and effective worker counts, so manifests made with different worker settings intentionally differ. `--hash-mb` is per worker.

Progress is written to stderr only. Each record stores total nodes consumed plus the cumulative node counts for the selected target, deepest completed iteration, and immediately previous iteration. This distinguishes label-producing work, parity selection, and an interrupted attempt at the next depth without rerunning the search.

## Output transaction

The command parses the source and then atomically reserves `--output-dir` before starting any search. An existing path is always refused. While work is underway, the directory contains an `INCOMPLETE` marker. A successful commit contains:

```text
pilot-shard-000/
  COMPLETE
  labels.bin
  manifest.json
```

`COMPLETE` is installed last, after both temporary files have been renamed into place, and contains the binary and manifest SHA-256 digests. A crash or failed exact run deliberately leaves its directory marked `INCOMPLETE`; readers must reject it. The command never deletes or reuses such a directory automatically. This makes failed jobs visible and prevents two processes from writing the same shard.

Validate a finished shard before consuming it:

```bash
python3 tools/inspect_minimax_labels.py artifacts/labels/pilot-shard-000 \
  --source artifacts/sources/pattern-pilot/shards/shard-00000000-<sha256>.jsonl
```

The auditor is independent of the C++ writer. It verifies SHA-256 digests, framing, fixed sizes, legal bitboards and piece counts, D4 canonical keys, move legality, node/depth invariants, shard coordinates, manifest counts, and completion state.

Compare two or more independently audited label artifacts from the same source shard with the read-only comparison report:

```bash
python3 tools/compare_minimax_labels.py \
  artifacts/labels/pilot-shard-000-1m \
  artifacts/labels/pilot-shard-000-5m \
  artifacts/labels/pilot-shard-000-10m \
  --source artifacts/sources/pattern-pilot/shards/shard-00000000-<sha256>.jsonl \
  --strata
```

The report rejects source or provenance misalignment before comparing results. It summarizes selected-target and deepest-result changes, completed depths, terminal counts, best moves, signs, same-parity stability, newly terminal positions, and phase/policy strata. Pass `--json` for the complete machine-readable report.

## Binary schema

All integers are little-endian. The file contains one 112-byte header followed by fixed 140-byte records. Bitboards use board indices `a1=0` through `g7=48`. Scores are signed two's-complement 32-bit doubled margins from the side-to-move perspective. The independent reader also accepts the earlier 112-byte record layout for existing calibration artifacts.

Header:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Magic bytes `POE2LBL\0` |
| 8 | 4 | Schema version (`2`) |
| 12 | 4 | Header size (`112`) |
| 16 | 4 | Record size (`140`) |
| 20 | 4 | Endian marker (`0x01020304`) |
| 24 | 8 | Emitted record count |
| 32 | 8 | Input position count |
| 40 | 32 | SHA-256 digest of the raw source bytes |
| 72 | 32 | SHA-256 digest of the logical corpus ID |
| 104 | 4 | Zero-based shard index |
| 108 | 4 | Shard count |

Record:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Player 1 bitboard |
| 8 | 8 | Player 2 bitboard |
| 16 | 8 | Canonical position-key low word |
| 24 | 8 | Canonical position-key high word |
| 32 | 8 | Compact source-position ID |
| 40 | 8 | Family grouping ID |
| 48 | 8 | Trajectory grouping ID |
| 56 | 8 | Parent/sibling grouping ID, or zero |
| 64 | 8 | Stable global trajectory ordinal |
| 72 | 8 | Total nodes, including an interrupted next iteration |
| 80 | 8 | Nodes consumed when the selected parity-aligned target completed |
| 88 | 4 | One-based source line number |
| 92 | 4 | Zero-based source ordinal within this shard |
| 96 | 4 | Selected parity-aligned signed minimax value |
| 100 | 1 | Position ply |
| 101 | 1 | Side to move (`0` Player 1, `1` Player 2) |
| 102 | 1 | Label mode (`1` exact, `2` teacher) |
| 103 | 1 | Selected parity-aligned explicit depth |
| 104 | 1 | Attempted explicit depth after the deepest completed iteration |
| 105 | 1 | Explicit depth required for terminal value with two-ply closure |
| 106 | 1 | Selected parity-aligned best-move board index |
| 107 | 1 | Dataset split (`1` train, `2` validation, `3` test) |
| 108 | 2 | Source policy ID |
| 110 | 2 | Sample ordinal within a trajectory |
| 112 | 8 | Nodes consumed when the deepest completed iteration finished |
| 120 | 4 | Deepest completed signed minimax value |
| 124 | 1 | Deepest completed explicit depth |
| 125 | 1 | Deepest completed best-move board index |
| 126 | 8 | Nodes consumed when the immediately previous iteration finished, or zero |
| 134 | 4 | Immediately previous signed minimax value, or zero when absent |
| 138 | 1 | Immediately previous completed explicit depth, or zero |
| 139 | 1 | Immediately previous best-move board index, or `255` when absent |

The canonical key supports exact-state deduplication. Family, trajectory, parent, stable trajectory ordinal, and frozen split support leakage-safe grouping. The position-source generator supplies these relationships directly. The opening-book adapter maps each independent line to its own family and trajectory and assigns it to the training split.

## Manifest and provenance

The manifest records strong source, corpus, and binary digests; schema sizes; shard coordinates; input/result counts; unsolved source lines; parity-backoff and previous-iteration counts; mode and node cap; requested and effective hash allocation; requested and effective workers; the `deepest_terminal_parity` target-selection rule; evaluator/search semantics; and build identity (Git commit and dirty state captured at CMake configure time, compiler, build type, target processor, and native-architecture setting).

Re-run CMake configuration before a production labeling build so the recorded source state is current. Operational measurements such as elapsed time, host name, CPU utilization, and peak memory belong in a separate run log and are intentionally absent from deterministic artifacts.
