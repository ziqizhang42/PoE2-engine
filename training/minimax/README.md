# Minimax Training Environment

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
uv run --frozen poe2-training-smoke \
  --dataset ../../build/data/features/pattern-dev-80k-s20260817/b-primitives \
  --require-gpu
```

The output separates artifact authentication and mapping, bulk materialization, zero-copy CPU tensor creation, and device transfer. A second timing run may omit the SHA-256 pass, but it does not replace the authenticated run:

```bash
uv run --frozen poe2-training-smoke \
  --dataset ../../build/data/features/pattern-dev-80k-s20260817/b-primitives \
  --require-gpu \
  --skip-digest
```

The loader never parses JSON records or performs per-position Python decoding. It memory-maps the 432-byte fixed records, validates important columns with vectorized NumPy operations, makes one contiguous copy of the selected model arrays, and transfers those arrays as a batch. Family, trajectory, and parent identifiers remain on the host for grouped validation; only model inputs, targets, and label-quality metadata are sent to the device.
