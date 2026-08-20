# Minimax Training Environment

For named datasets, resumable iterations, candidate-specific builds, and deliberate promotion, start with the [consolidated training workflow](../../docs/training-workflow.md). This document keeps the independently runnable lower-level experiment commands as references.

This isolated uv project authenticates and memory-maps the deterministic minimax feature artifact. Python 3.14 is the primary interpreter, and PyTorch 2.12.1 is pinned to the ROCm 7.2 package index.

## Manual environment setup

The repository does not install or configure uv automatically. Install the pinned uv release yourself:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Then resolve and install the isolated project environment manually:

```bash
cd training/minimax
uv python find 3.14
uv lock
uv sync --frozen
```

## Tests and environment smoke check

Run the loader tests inside the resolved environment:

```bash
uv run --frozen python -m unittest discover -s tests -v
```

From `training/minimax`, verify the exact environment, run a deterministic forward/backward tensor operation, authenticate the complete feature binary, materialize its columns in bulk, and transfer the model inputs once to the ROCm device:

```bash
FEATURE_DIR=../../path/to/completed/features
uv run --frozen poe2-training-smoke \
  --dataset "$FEATURE_DIR" \
  --require-gpu
```

The output separates artifact authentication and mapping, bulk materialization, zero-copy CPU tensor creation, and device transfer. A second timing run may omit the SHA-256 pass, but it does not replace the authenticated run:

```bash
uv run --frozen poe2-training-smoke \
  --dataset "$FEATURE_DIR" \
  --require-gpu \
  --skip-digest
```

The loader never parses JSON records or performs per-position Python decoding. It memory-maps the 432-byte fixed records, validates important columns with vectorized NumPy operations, makes one contiguous copy of the selected model arrays, and transfers those arrays as a batch. Family, trajectory, and parent identifiers remain on the host for grouped validation; only model inputs, targets, and label-quality metadata are sent to the device.

## Baseline experiment

The baseline experiment fits the residual between the teacher and the exact two-ply-closure value. It compares closure unchanged, a 50-entry per-ply calibration, and a phase-plus-gain-summary model. Both learned models use deterministic, closed-form float64 ridge regression in PyTorch. Ridge strength is selected by validation MAE, then validation RMSE, then the smaller ridge value. Each D4-canonical position receives one vote; duplicate multiplicity is not used as a training weight.

The command materializes model batches only for the train and validation splits. The test split is schema-authenticated with the rest of the artifact and its count is recorded for accounting, but it is excluded from fitting, model selection, and metrics. The output directory is create-only and contains an authenticated `report.json` with feature definitions, standardization values, fitted weights, the complete ridge path, phase/parity/label-quality metrics, dataset digests, runtime versions, and a digest of the training source.

From `training/minimax`, run the authenticated GPU experiment with:

```bash
FEATURE_DIR=../../path/to/completed/features
OUTPUT_DIR=../../path/to/new/baseline-report
uv run --frozen poe2-train-baselines \
  --dataset "$FEATURE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda
```

## Pattern experiment

The pattern experiment fits reversal-tied ternary scoring-line tables to the same two-ply-closure residual, optionally combined with exact marginal-gain summaries. The exploratory `default` suite reports its validation winner but does not alter another iteration. The engine-compatible architecture was fixed by an earlier broad development ladder followed by a middle-knot refinement: `--suite frozen-pattern-gain` trains that fixed three-knot line model `(0, 28, 49)` with five gain knots, Huber delta 8, and L2 strength `1e-4`. Candidate export intentionally rejects other architectures.

The integrated `pattern-gain` evaluator is closure plus one frozen learned residual. Its inputs are the 36 board scoring lines encoded as own/opponent/empty ternary patterns, and 19 cheap summaries of legal-move gains: each side's top four gains, four threshold counts per side, contested-best and unique-opponent-best flags, and denied opponent gain. Line weights interpolate across plies 0/28/49; gain-summary weights interpolate across plies 0/12/24/36/49; a per-ply intercept completes the model.

Each trained model also folds the training standardization into raw lookup weights and quantizes int16 tables plus a per-ply int32 intercept. Exploratory suites select a fixed-point scale on validation data; the deployable `frozen-pattern-gain` suite fixes scale 32 to match the engine contract before sealed evaluation. The test split remains excluded from training, checkpoint selection, quantization decisions, and reported metrics.

From `training/minimax`, train the frozen model with:

```bash
FEATURE_DIR=../../path/to/completed/features
EXPERIMENT_DIR=../../path/to/new/pattern-experiment
uv run --frozen poe2-train-patterns \
  --dataset "$FEATURE_DIR" \
  --output-dir "$EXPERIMENT_DIR" \
  --device cuda \
  --seed 20260818 \
  --suite frozen-pattern-gain
```

After the training report is complete and no further model choice remains, open the sealed test split once with:

```bash
FEATURE_DIR=../../path/to/completed/features
EXPERIMENT_DIR=../../path/to/completed/pattern-experiment
DEVELOPMENT_FEATURE_DIR=../../path/to/completed/development-features
EVALUATION_DIR=../../path/to/new/sealed-evaluation
uv run --frozen poe2-evaluate-pattern \
  --dataset "$FEATURE_DIR" \
  --experiment "$EXPERIMENT_DIR" \
  --exclude-dataset "$DEVELOPMENT_FEATURE_DIR" \
  --output-dir "$EVALUATION_DIR"
```

The sealed-test report verifies that the experiment and feature digests match, removes canonical positions present in the development artifact, and evaluates two-ply closure, the gain-only control, and the float and validation-selected quantized pattern/gain models. It records phase, parity, exact-label, and teacher-label metrics. Reusing an output directory is rejected.

## C++ export and inference parity

Export the authenticated selected integer tables from the repository root, then build the release inference checker:

```bash
EXPERIMENT_DIR=path/to/completed/pattern-experiment
python3 tools/export_minimax_pattern_gain.py \
  --experiment "$EXPERIMENT_DIR" \
  --output engines/minimax/src/frozen_pattern_gain_model.hpp
cmake --build --preset release --target poe2_minimax_infer
```

The engine defaults to `--evaluator pattern-gain`. Use `--evaluator two-ply-closure` for the
unlearned closure evaluator or `--evaluator static` for the raw normalized board score.
Pattern/gain retains its scale-32 accumulator throughout alpha-beta and the transposition table,
rounds only scores returned through the public engine API, and uses exact two-ply closure for
terminal positions and positions with at most two empty squares.

The integration audit found that the original `denied_opponent_best_gain` implementation selected
the first row-major square when several moves tied under the closure objective. That single feature
was therefore not D4 invariant. Python and C++ integration inference instead take the minimum
denial over every closure-optimal square. The frozen table values and architecture remain
unchanged; this conservative tie rule is permutation invariant and must be used for parity with the
engine.

From `training/minimax`, compare the scaled C++ accumulator exactly with Python over deterministic corpus samples and their D4 transforms. The optional benchmark reports direct two-ply-closure and pattern/gain evaluator latency over the same position mix:

```bash
FEATURE_DIR=../../path/to/completed/features
EXPERIMENT_DIR=../../path/to/completed/pattern-experiment
uv run --frozen poe2-verify-pattern-gain \
  --dataset "$FEATURE_DIR" \
  --experiment "$EXPERIMENT_DIR" \
  --inference-binary ../../build/release/engines/minimax/poe2_minimax_infer \
  --samples 4096 \
  --symmetry-samples 512 \
  --benchmark-iterations 100
```

Benchmark complete fixed-time searches from the repository root on an identical deterministic
opening sample:

```bash
python3 tools/benchmark_minimax_search.py \
  --engine build/release/engines/minimax/poe2_minimax \
  --opening-book eval/openings/development.txt \
  --positions 32 \
  --movetime-ms 100
```
