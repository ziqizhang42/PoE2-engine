import createModule from "./poe2-engine.mjs";

const packagedWasmUrl = new URL("./poe2-engine.wasm", import.meta.url);

export default async function createEngine(options = {}) {
  const wasmUrl = options.wasmUrl ?? packagedWasmUrl;
  const module = await createModule({
    locateFile(path, prefix) {
      return path.endsWith(".wasm") ? String(wasmUrl) : `${prefix}${path}`;
    },
  });
  const analyzer = new module.BrowserAnalyzer();

  return Object.freeze({
    analyze(request, options) {
      const onProgress = options?.onProgress;
      return analyzer.analyze(request, typeof onProgress === "function" ? onProgress : undefined);
    },
  });
}
