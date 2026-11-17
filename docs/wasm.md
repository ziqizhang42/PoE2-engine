# Browser WebAssembly Engine

## Toolchain

The `wasm-release` preset requires the Emscripten version pinned in [`.emscripten-version`](../.emscripten-version):

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
POE2_EMSDK_VERSION="$(sed -n '1p' /path/to/PoE2-engine/.emscripten-version)"
./emsdk install "${POE2_EMSDK_VERSION}"
./emsdk activate "${POE2_EMSDK_VERSION}"
source ./emsdk_env.sh
```

Source `emsdk_env.sh` in every shell that configures or builds the preset.

## Build and Install

For a local development build with an installed SDK:

```bash
cd /path/to/PoE2-engine
cmake --preset wasm-release
cmake --build --preset wasm-release
ctest --preset wasm-release
```

The generated npm package is `build/wasm-release/package`; install it from the frontend with:

```bash
npm install /path/to/PoE2-engine/build/wasm-release/package
```

For the reproducible release package, Docker supplies the pinned SDK and Node.js:

```bash
make release-package
```

This runs the WebAssembly smoke test and writes the npm tarball and its checksum under `build/dist/v<version>/`. It does not publish an image or package. See [Releasing](releasing.md) for the full release checklist.

## Use

`analyze()` is synchronous, so call it from a module Worker, reuse the engine between completed searches, and terminate the Worker to cancel an active search.

```ts
import createEngine from "@poe2/engine-wasm";
import wasmUrl from "@poe2/engine-wasm/poe2-engine.wasm?url";

const engine = await createEngine({ wasmUrl });
const result = engine.analyze(
  {
    moves: ["d4", "a1", "c4"],
    searchTimeMs: 1_000,
    multiPv: 3,
  },
  {
    onProgress(update) {
      self.postMessage({ type: "progress", update });
    },
  },
);
```

`moves` is a legal history in `a1`–`g7` notation, `searchTimeMs` is a positive integer, `multiPv` accepts 1–5 and defaults to 1, and `maxDepth` accepts 1–49 for bounded tests.

Successful responses contain `lines`, `completedDepth`, and cumulative `nodes`; failures contain a structured `error`. `evaluationHalfPoints` is signed for Player 1, and `lines` remains in engine preference order, so do not re-sort it numerically.

## Nets

The engine analyzes with its built-in net unless it is given the bytes of a net file:

```ts
const net = await (await fetch(netUrl)).arrayBuffer();
const engine = await createEngine({ wasmUrl, net });
console.log(engine.info.net); // { name: "run1/g000", id: "633309adb322", sha256: "…", design: "pattern-gain-v1" }
```

`net` accepts an `ArrayBuffer` or any view of one, such as a `Uint8Array`. The engine checks the file exactly as the native engine's `--net` does. When it cannot use the file, `createEngine` rejects with a `NetLoadError` whose `code` is `"invalid_net"` and whose `reason` says why, for example `net file is damaged: its integrity check does not match its contents` or `net design "pattern-gain-v9" is not supported by this engine (supported: pattern-gain-v1)`; it rejects with a `TypeError` when `net` is not binary data. `engine.info` reports `engineVersion`, `apiVersion`, the net `designs` the engine supports, and the `net` it analyzes with.

See [nets.md](nets.md) for the file format and the built-in net.
