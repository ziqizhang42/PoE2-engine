# Browser WebAssembly Engine

The WebAssembly build packages the existing game and minimax libraries behind a small browser API. It does not compile or wrap the stdin/stdout protocol, native engine executables, or match runner.

## Install the Toolchain

The required Emscripten version is recorded in [`.emscripten-version`](../.emscripten-version) and checked by CMake. Install that exact release with the officially supported `emsdk`:

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 6.0.6
./emsdk activate 6.0.6
source ./emsdk_env.sh
emcc --version
```

`emsdk_env.sh` sets `EMSDK`, which the CMake preset uses to locate the toolchain file. Source it in each new shell before configuring.

## Build and Test

From this repository:

```bash
cmake --preset wasm-release
cmake --build --preset wasm-release
ctest --preset wasm-release
```

The smoke test uses the Node runtime supplied by emsdk to import the public ES-module loader, instantiate the separate WebAssembly asset, analyze a fixed position, compare it with a native fixture, exercise structured errors, and prove that a repeated call reuses cached search entries.

## Browser API

The default export asynchronously creates an isolated engine instance:

```ts
import createEngine from "@poe2/engine-wasm";

const engine = await createEngine();
const result = engine.analyze({
  moves: ["d4", "a1", "c4"],
  searchTimeMs: 250,
  maxDepth: 8,
});
```

`moves` is a legal history from the empty board in `a1` through `g7` notation. Runtime parsing also accepts uppercase files, but returned moves are always lowercase. `searchTimeMs` must be an integer from 1 through 2,147,483,647. Optional `maxDepth` must be an integer from 1 through 49.

`engine.analyze()` is synchronous and CPU-bound, so call it inside a Worker. One engine owns one long-lived minimax `Search` and a transposition table. Reuse that engine for later positions instead of calling `createEngine()` for every request.

Successful results have this shape:

```ts
{
  ok: true,
  bestMove: "e4",
  evaluationHalfPoints: 13,
  completedDepth: 6,
  nodes: 18420,
  principalVariation: ["e4", "e5", "d5"],
  engineVersion: "0.1.0",
  apiVersion: 1,
}
```

`evaluationHalfPoints` is always normalized to Player 1: positive favors Player 1 and negative
favors Player 2. It includes Player 2's 5.5-point handicap and is an integer because one unit is a half-point. `completedDepth` counts the last fully completed explicit iterative-deepening depth. The principal variation is a legal prefix and begins with `bestMove`.

Expected failures return `{ ok: false, error, engineVersion, apiVersion }`. History errors include `field`, a zero-based `moveIndex`, and the offending `move` when it is astring.
