# Deterministic Minimax Labels

`poe2_minimax_data` turns a validated opening book into reproducible minimax labels. It is the data boundary between the production B evaluator/search and future offline evaluation experiments: the engine produces stable integer targets, while model training remains a separate concern.

## Build and Run

```bash
cmake --build --preset release --target poe2_minimax_data

build/release/runner/poe2_minimax_data labels \
  --input eval/openings/development.txt \
  --output build/labels/development.bin \
  --manifest build/labels/development.json \
  --mode teacher \
  --nodes 1000000 \
  --hash-mb 64 \
  --workers 6 \
  --require-all
```

The input uses the existing opening-book format: one legal move prefix per non-comment line. Malformed, illegal, or terminal openings are rejected by the same parser used by the match runner.

The required mode is one of:

- `exact`: requests the remaining terminal depth and emits a record only when that entire search finishes. Add `--require-all` when any unsolved input should fail the job.
- `teacher`: uses the requested node budget as a hard cap and emits the deepest completed iterative-deepening result. Teacher records are never represented as exact, even when a particular search happens to reach the terminal depth.

Both modes require a positive `--nodes` budget. Every position starts with a cleared transposition table and history state. Search uses board-symmetry canonicalization and the production B two-ply closure. No clock, random seed, scheduling decision, or wall-clock timestamp enters the output. Running the same executable with the same source bytes, path spelling, and arguments therefore produces byte-identical files.

`--workers` defaults to one. Each worker owns an isolated search, history, and transposition table; inputs are scheduled dynamically but results are restored to source order before serialization. The binary records are therefore identical across worker counts. The manifest records both the requested count and the count actually used, so manifests from different worker configurations are intentionally different. `--hash-mb` is per worker.

Output paths are create-only. The command refuses to overwrite either file, writes through sibling `.tmp` files, and removes partial output if committing the pair fails.

## Binary Schema

All integers are little-endian. The file contains one 48-byte header followed by fixed 64-byte records. Bitboards use board indices `a1=0` through `g7=48`. The stored score is the doubled-margin integer from the side-to-move perspective.

Header:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Magic bytes `POE2LBL\0` |
| 8 | 4 | Schema version (`1`) |
| 12 | 4 | Header size (`48`) |
| 16 | 4 | Record size (`64`) |
| 20 | 4 | Endian marker (`0x01020304`) |
| 24 | 8 | Emitted record count |
| 32 | 8 | Input position count |
| 40 | 8 | FNV-1a-64 digest of the raw source file |

Record:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 8 | Player 1 bitboard |
| 8 | 8 | Player 2 bitboard |
| 16 | 8 | Canonical position-key low word |
| 24 | 8 | Canonical position-key high word |
| 32 | 8 | FNV-1a-64 ID of the normalized source line |
| 40 | 8 | Nodes consumed by iterative deepening |
| 48 | 4 | One-based source line number |
| 52 | 4 | Signed 32-bit minimax value |
| 56 | 1 | Position ply |
| 57 | 1 | Side to move (`0` Player 1, `1` Player 2) |
| 58 | 1 | Label mode (`1` exact, `2` teacher) |
| 59 | 1 | Deepest completed explicit depth |
| 60 | 1 | Explicit depth required to reach the terminal value with B closure |
| 61 | 1 | Best-move board index |
| 62 | 2 | Reserved, zero |

The canonical key is included for grouping and leakage-safe deduplication; the raw colored bitboards preserve the actual training position. Consumers must check the schema, header size, record size, and endian marker before reading records.

## Manifest and Provenance

The JSON manifest records the source and binary digests, schema version, input/result counts, unsolved source lines, mode, node limit, hash size, requested and effective workers, evaluator name, symmetry setting, and closure setting. It deliberately contains no timestamp or machine-specific performance data.

Keep the `.bin`, `.json`, source opening book, engine commit, and exact invocation together for a training run. A future learner should split by canonical opening/solve-tree family rather than by individual record; the label schema supplies provenance for that grouping but does not choose the training framework or model.
