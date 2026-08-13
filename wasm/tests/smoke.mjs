import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const [loaderPath, packagePath, wasmPath] = process.argv.slice(2);
assert(loaderPath, "loader path is required");
assert(packagePath, "package path is required");
assert(wasmPath, "WASM path is required");

const packageMetadata = JSON.parse(await readFile(packagePath, "utf8"));
assert.equal(packageMetadata.name, "@poe2/engine-wasm");
assert.equal(packageMetadata.version, "0.1.0");
assert((await stat(wasmPath)).size > 0, "WASM asset must not be empty");

const { default: createEngine } = await import(pathToFileURL(loaderPath).href);
const packagedEngine = await createEngine();
assert.equal(packagedEngine.analyze({ moves: "a1", searchTimeMs: 10 }).error.code, "invalid_history");

const engine = await createEngine({ wasmUrl: wasmPath });
const engineVersion = packageMetadata.version;

assert.deepEqual(engine.analyze(null), {
  ok: false,
  error: {
    code: "invalid_request",
    message: "request must be an analysis request object",
    field: "request",
  },
  engineVersion,
  apiVersion: 1,
});

const moves = [];
for (let index = 0; index < 45; index += 1) {
  const row = Math.floor(index / 7);
  const col = index % 7;
  moves.push(`${String.fromCodePoint("a".codePointAt(0) + col)}${String(row + 1)}`);
}

const request = { moves, searchTimeMs: 5000, maxDepth: 2 };
const first = engine.analyze(request);
assert.deepEqual(first, {
  ok: true,
  bestMove: "g7",
  evaluationHalfPoints: 41,
  completedDepth: 2,
  nodes: 16,
  principalVariation: ["g7", "f7"],
  engineVersion,
  apiVersion: 1,
});

const second = engine.analyze(request);
assert.deepEqual(second, { ...first, nodes: 2 });

const malformed = engine.analyze({ moves: ["h1"], searchTimeMs: 10 });
assert.equal(malformed.ok, false);
assert.equal(malformed.error.code, "malformed_history_move");
assert.equal(malformed.error.moveIndex, 0);
assert.equal(malformed.error.move, "h1");

const nonStringMove = engine.analyze({ moves: [1], searchTimeMs: 10 });
assert.equal(nonStringMove.ok, false);
assert.equal(nonStringMove.error.code, "malformed_history_move");
assert.equal(nonStringMove.error.moveIndex, 0);

const illegal = engine.analyze({ moves: ["a1", "a1"], searchTimeMs: 10 });
assert.equal(illegal.ok, false);
assert.equal(illegal.error.code, "illegal_history_move");
assert.equal(illegal.error.reason, "occupied");

const invalidTime = engine.analyze({ moves: [], searchTimeMs: 1.5 });
assert.equal(invalidTime.ok, false);
assert.equal(invalidTime.error.code, "invalid_search_time");

const invalidDepth = engine.analyze({ moves: [], searchTimeMs: 10, maxDepth: 50 });
assert.equal(invalidDepth.ok, false);
assert.equal(invalidDepth.error.code, "invalid_max_depth");
