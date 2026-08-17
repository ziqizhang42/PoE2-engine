# Deterministic Minimax Labels

`poe2_minimax_data` turns a validated opening book into reproducible minimax labels. It is the data boundary between the production B evaluator/search and future offline evaluation experiments. Corpus generation and model training are deliberately separate.

## Build and run

```bash
cmake --build --preset release --target poe2_minimax_data

build/release/runner/poe2_minimax_data labels \
  --input path/to/source-shard.txt \
  --output-dir artifacts/labels/pilot-shard-000 \
  --corpus-id poe2-pattern-pilot \
  --shard-index 0 \
  --shard-count 20 \
  --mode teacher \
  --nodes 1000000 \
  --hash-mb 64 \
  --workers 6 \
  --progress-every 100 \
  --require-all
```

The input uses the existing opening-book format: one legal move prefix per non-comment line. Malformed, illegal, or terminal openings are rejected by the same parser used by the match runner. `--shard-index` and `--shard-count` describe the supplied source shard; this command does not generate or divide a corpus.

The required mode is one of:

- `exact`: requests the remaining terminal depth and emits a record only when that entire search finishes. Add `--require-all` when any unsolved input should fail the job.
- `teacher`: uses the requested node budget as a hard cap and emits the deepest completed iterative-deepening result. A teacher record remains identified as a teacher record even if it reaches terminal depth.

Both modes require a positive `--nodes` budget. Every position starts with cleared transposition and history state. Search uses board-symmetry canonicalization and the production B two-ply closure. No clock, random seed, scheduling decision, progress message, path to the output directory, or wall-clock timestamp enters the artifacts.

`--workers` defaults to one. Each worker owns an isolated search, history, and transposition table; inputs are scheduled dynamically but restored to source order before serialization. Binary records are therefore independent of worker count. The manifest records requested and effective worker counts, so manifests made with different worker settings intentionally differ. `--hash-mb` is per worker.

Progress is written to stderr only. Each record stores both total nodes consumed and the node count at the last completed iteration. This distinguishes the label-producing work from an interrupted attempt at the next depth.

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
  --source path/to/source-shard.txt
```

The auditor is independent of the C++ writer. It verifies SHA-256 digests, framing, fixed sizes, legal bitboards and piece counts, D4 canonical keys, move legality, node/depth invariants, shard coordinates, manifest counts, and completion state.

## Binary schema

All integers are little-endian. The file contains one 112-byte header followed by fixed 104-byte records. Bitboards use board indices `a1=0` through `g7=48`. The score is a signed two's-complement 32-bit doubled margin from the side-to-move perspective.

Header:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Magic bytes `POE2LBL\0` |
| 8 | 4 | Schema version (`1`) |
| 12 | 4 | Header size (`112`) |
| 16 | 4 | Record size (`104`) |
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
| 64 | 8 | Total nodes, including an interrupted next iteration |
| 72 | 8 | Nodes consumed when the stored iteration completed |
| 80 | 4 | One-based source line number |
| 84 | 4 | Zero-based source ordinal |
| 88 | 4 | Signed minimax value |
| 92 | 1 | Position ply |
| 93 | 1 | Side to move (`0` Player 1, `1` Player 2) |
| 94 | 1 | Label mode (`1` exact, `2` teacher) |
| 95 | 1 | Deepest completed explicit depth |
| 96 | 1 | Attempted explicit depth |
| 97 | 1 | Explicit depth required for terminal value with B closure |
| 98 | 1 | Best-move board index |
| 99 | 1 | Reserved flags, zero |
| 100 | 2 | Source policy ID |
| 102 | 2 | Sample ordinal within a trajectory |

The canonical key supports exact-state deduplication. Family, trajectory, and parent IDs support leakage-safe grouping. The current opening-book adapter maps each independent opening line to its own family and trajectory; a later corpus source should assign shared IDs to related samples rather than trying to reconstruct those relationships after labeling.

## Manifest and provenance

The manifest records strong source, corpus, and binary digests; schema sizes; shard coordinates; input/result counts; unsolved source lines; mode and node cap; requested and effective hash allocation; requested and effective workers; evaluator/search semantics; and build identity (Git commit and dirty state captured at CMake configure time, compiler, build type, target processor, and native-architecture setting).

Re-run CMake configuration before a production labeling build so the recorded source state is current. Operational measurements such as elapsed time, host name, CPU utilization, and peak memory belong in a separate run log and are intentionally absent from deterministic artifacts.
