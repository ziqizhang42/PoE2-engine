import createModule from "./poe2-engine.mjs";

const packagedWasmUrl = new URL("./poe2-engine.wasm", import.meta.url);

export class NetLoadError extends Error {
  constructor(reason) {
    super(`could not load net: ${reason}`);
    this.name = "NetLoadError";
    this.code = "invalid_net";
    this.reason = reason;
  }
}

function netBytes(net) {
  if (net instanceof ArrayBuffer) {
    return new Uint8Array(net);
  }
  if (ArrayBuffer.isView(net)) {
    return new Uint8Array(net.buffer, net.byteOffset, net.byteLength);
  }
  throw new TypeError("net must be an ArrayBuffer or a view of one holding a net file");
}

function frozenInfo(info) {
  return Object.freeze({
    ...info,
    designs: Object.freeze([...info.designs]),
    net: Object.freeze({ ...info.net }),
  });
}

export default async function createEngine(options = {}) {
  const wasmUrl = options.wasmUrl ?? packagedWasmUrl;
  const net = options.net === undefined ? undefined : netBytes(options.net);
  const module = await createModule({
    locateFile(path, prefix) {
      return path.endsWith(".wasm") ? String(wasmUrl) : `${prefix}${path}`;
    },
  });
  const analyzer =
    net === undefined ? new module.BrowserAnalyzer() : new module.BrowserAnalyzer(net);
  const loadError = analyzer.loadError();
  if (loadError !== "") {
    analyzer.delete();
    throw new NetLoadError(loadError);
  }
  const info = frozenInfo(analyzer.info());

  return Object.freeze({
    info,
    analyze(request, options) {
      const onProgress = options?.onProgress;
      return analyzer.analyze(request, typeof onProgress === "function" ? onProgress : undefined);
    },
  });
}
