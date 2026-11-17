# Nets

The `pattern-gain` evaluator reads its parameters from a net: a small binary file the engine
loads at start-up. Every engine also has a net built in, used whenever no file is given.

## Using a net

```bash
build/release/engines/minimax/poe2_minimax --net models/<net>.poe2net
build/release/engines/minimax/poe2_minimax_infer --net models/<net>.poe2net < positions.txt
build/release/runner/poe2_minimax_data labels --evaluator pattern-gain --net <net>.poe2net ...
```

`--net` is accepted by the engine, the inference tool, and the data tool's `source`,
`source-chunk`, `source-assemble`, and `labels` commands (the data tool needs
`--evaluator pattern-gain` with it). A file that cannot be used stops the program before it
does anything else, with the reason: not a net file, damaged, a container format or design
this engine does not support, or tables that do not fit the design.

`poe2_minimax --version` prints what the engine is and which net it would use (add `--net`
to check a file):

```text
version 0.3.0
designs pattern-gain-v1
net run1/g000 633309adb322 a52d5a8118ba116bf07469dba37c08022e7416b19544e823a26265abd7e3e94a
```

The engine sends the same three facts during the protocol handshake, as `id version ...`,
`id designs ...`, and `id net ...` lines between `id name minimax` and `poe2ok`.

## Names and IDs

Every net has one readable name, `<run>/<generation>` (for example `run1/g001`), and one ID:
the first 12 hex digits of the SHA-256 of its file. The full hash is the net's identity
wherever it is recorded. The two nets trained before net files existed, `run1/g000` (v0.2)
and `run1/g001`, keep the IDs earlier records gave them, which are hashes of the C++ headers
they were converted from (`633309adb322` and `8e39040c1537`); their files record that header
hash as `header_sha256`, and their file hashes are recorded beside the IDs.

Training writes nets: candidate validation exports each trained model with
`python -m poe2_training.net_export` (run from `training/minimax`), and every gated model,
winner or not, is archived as `models/candidate-sha256-<digest>.poe2net` beside a record of
its lineage, training, and metrics ([training-workflow.md](training-workflow.md#gate-promotion)).
Teachers, gates, and comparisons load those files; nothing is rebuilt per model.

## Designs

A net states its design, the evaluator it was made for. An engine lists the designs it can
run (`designs` above) and refuses any other. Today there is one:

| Design | Evaluator |
|---|---|
| `pattern-gain-v1` | Two-ply closure plus learned line-pattern and gain-summary residuals: per-ply intercepts, three phase knots over 1,716 reversal-orbit line patterns, and five phase knots over 19 gain features, in scale-32 fixed point. |

Every engine keeps loading every design ever shipped. `tests/fixtures/nets/<design>/` holds a
reference net and 1,000 positions across all plies with the evaluations the design's first
engine computed; the test suite requires every supported design to have such a fixture and
to reproduce every value exactly.

## File format

All integers are little-endian. A net file is a fixed header, the design's tables, a metadata
block, and an integrity check:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | Magic `POE2NET\0` |
| 8 | 4 | Container format version, `1` |
| 12 | 4 | Header size in bytes, `60 + 4 × table count` |
| 16 | 32 | Design identifier, ASCII, padded with zero bytes |
| 48 | 4 | Fixed-point fractional bits (`5`: scale 32) |
| 52 | 4 | Table count |
| 56 | 4 each | Element count of each table, in order |
| … | 4 | Metadata size in bytes |
| header size | … | Tables, in order, then metadata |
| end − 32 | 32 | SHA-256 of every preceding byte |

`pattern-gain-v1` has five tables: line knots (3 × u8), gain knots (5 × u8), intercepts per
ply (50 × i32), line weights (5,148 × i16, knot-major: `knot × 1716 + orbit`), and gain
weights (95 × i16, knot-major: `knot × 19 + feature`). The knots are part of the design and
must be 0, 28, 49 and 0, 12, 24, 36, 49, so the engine interpolates between phases with
constant divisors; a net with other knots needs a new design. Intercepts must lie within
±2^24, which keeps every evaluation inside the search's 32-bit window.

The metadata block is UTF-8 text of `key=value` lines, each ending in a newline, with keys
(`[a-z][a-z0-9_]*`) in strictly ascending order, so that one set of values has exactly one
encoding and exporting the same net twice gives the same bytes. `name` is required. Nets the
training workflow writes also record `parent` and `teacher` (names; `two-ply-closure` for a
teacher that is not a net), `corpus` and `corpus_digest`, `feature_binary_sha256`, `created`
(the UTC date the net was trained), and `training_report_sha256`.

The engine fills its lookup tables from a net once, when it loads it; search then reads them
exactly as it read the tables compiled from a header before, so evaluation, speed, and
results do not depend on where a net came from.

`training/minimax/src/poe2_training/net_file.py` reads and writes the same format.
