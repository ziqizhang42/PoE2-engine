# @poe2/engine-wasm

Browser WebAssembly package for the PoE2 minimax engine.

## Use

Call the synchronous analyzer from a module Worker, reuse one engine between completed searches, and terminate the Worker to cancel an active search.

```ts
import createEngine from "@poe2/engine-wasm";
import wasmUrl from "@poe2/engine-wasm/poe2-engine.wasm?url";

const engine = await createEngine({ wasmUrl });
const result = engine.analyze(
  {
    moves: ["d4", "a1"],
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

Successful responses contain ranked `lines`, `completedDepth`, and cumulative `nodes`; failures contain a structured `error`. `evaluationHalfPoints` is signed for Player 1, and `lines` remains in engine preference order, so do not re-sort it numerically.

Build and installation instructions are in `docs/wasm.md` in the source repository.
