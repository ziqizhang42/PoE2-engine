# @poe2/engine-wasm

Browser WebAssembly package for the PoE2 minimax engine. Instantiate it once inside a module Web Worker and reuse the returned engine so its transposition table survives across analyses:

```ts
import createEngine from "@poe2/engine-wasm";
import wasmUrl from "@poe2/engine-wasm/poe2-engine.wasm?url";

const engine = await createEngine({ wasmUrl });
const result = engine.analyze({ moves: ["d4", "a1"], searchTimeMs: 250, maxDepth: 8 });
```

Results are discriminated unions. Successes contain a best move, Player 1-normalized evaluation in half-points, completed depth, nodes, principal variation, engine version, and API version. Expected input and search failures contain a structured error. `analyze` is synchronous and CPU-bound.

See the source repository's `docs/wasm.md` for the complete API, build instructions, and a Vite Worker example.
