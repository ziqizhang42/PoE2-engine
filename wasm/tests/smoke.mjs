import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const [loaderPath, packagePath, wasmPath, expectedVersion] = process.argv.slice(2);
assert(loaderPath, "loader path is required");
assert(packagePath, "package path is required");
assert(wasmPath, "WASM path is required");
assert(expectedVersion, "expected version is required");

const packageMetadata = JSON.parse(await readFile(packagePath, "utf8"));
assert.equal(packageMetadata.name, "@poe2/engine-wasm");
assert.equal(packageMetadata.version, expectedVersion);
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
assert.deepEqual(
  {
    ok: first.ok,
    bestMove: first.bestMove,
    evaluationHalfPoints: first.evaluationHalfPoints,
    completedDepth: first.completedDepth,
    nodes: first.nodes,
    principalVariation: first.principalVariation,
    engineVersion: first.engineVersion,
    apiVersion: first.apiVersion,
  },
  {
  ok: true,
  bestMove: "g7",
  evaluationHalfPoints: 41,
  completedDepth: 2,
  nodes: 16,
  principalVariation: ["g7", "f7"],
  engineVersion,
  apiVersion: 1,
  },
);
assert.equal(first.lines.length, 1);
assert.equal(first.lines[0].rank, 1);
assert.equal(first.lines[0].move, first.bestMove);
assert.equal(first.lines[0].evaluationHalfPoints, first.evaluationHalfPoints);
assert.deepEqual(first.lines[0].principalVariation, first.principalVariation);
assert(first.lines[0].equivalentMoves.includes(first.bestMove));

const second = engine.analyze(request);
assert.deepEqual(second, { ...first, nodes: 2 });

const updates = [];
const multiResult = engine.analyze(
  { ...request, multiPv: 3 },
  {
    onProgress(update) {
      updates.push(update);
    },
  },
);
assert.deepEqual(
  updates.map((update) => update.completedDepth),
  [1, 2],
);
assert.deepEqual(multiResult, updates.at(-1));
assert.equal(multiResult.lines.length, 3);
assert.equal(multiResult.bestMove, multiResult.lines[0].move);
assert.equal(multiResult.evaluationHalfPoints, multiResult.lines[0].evaluationHalfPoints);
assert.deepEqual(multiResult.principalVariation, multiResult.lines[0].principalVariation);
for (let index = 1; index < multiResult.lines.length; index += 1) {
  assert(
    multiResult.lines[index - 1].evaluationHalfPoints <=
      multiResult.lines[index].evaluationHalfPoints,
    "Player 2 lines must remain in engine preference order",
  );
}

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

for (const multiPv of [0, 6, 1.5, "2"]) {
  const invalidMultiPv = engine.analyze({ moves: [], searchTimeMs: 10, multiPv });
  assert.equal(invalidMultiPv.ok, false);
  assert.equal(invalidMultiPv.error.code, "invalid_multi_pv");
  assert.equal(invalidMultiPv.error.field, "multiPv");
}

const fiveLines = engine.analyze({ moves: [], searchTimeMs: 5000, maxDepth: 1, multiPv: 5 });
assert.equal(fiveLines.ok, true);
assert.equal(fiveLines.lines.length, 5);
assert(
  fiveLines.lines.reduce((count, line) => count + line.equivalentMoves.length, 0) >
    fiveLines.lines.length,
);

const oneMoveLeft = engine.analyze({
  moves: [...moves, "d7", "e7", "f7"],
  searchTimeMs: 5000,
  multiPv: 5,
});
assert.equal(oneMoveLeft.ok, true);
assert.equal(oneMoveLeft.lines.length, 1);
assert.deepEqual(oneMoveLeft.lines[0].equivalentMoves, ["g7"]);
