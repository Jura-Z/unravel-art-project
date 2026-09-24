// WASM SIMD simulation (the bench kernel, bit-exact to the reference spec).
// All particle state lives in WASM linear memory. JS writes the uniform block straight
// into that memory, and the WebGL renderer uploads P and T from views of it: no copies
// in JS, no allocation per frame.
let WASM_MODULE = null;

async function compileWasmKernel() {
  const bytes = b64bytes(WASM_SIMD_B64);
  if (typeof WebAssembly !== 'object' || !WebAssembly.validate(bytes)) throw new Error('WebAssembly SIMD is not supported here');
  WASM_MODULE = await WebAssembly.compile(bytes);   // async compile once; instances are created synchronously
}

function createWasmSim(n) {
  const module = WASM_MODULE;
  const probe = new WebAssembly.Instance(module, { env: { memory: new WebAssembly.Memory({ initial: 2 }) } });
  const memory = new WebAssembly.Memory({ initial: Math.ceil(probe.exports.bytes_needed(n) / 65536) + 1 });
  const x = new WebAssembly.Instance(module, { env: { memory } }).exports;
  const p = new Int32Array(memory.buffer, x.setup(n), 10);
  const f32 = (off, len) => new Float32Array(memory.buffer, off, len);
  x.links_setup_order();                                 // particle indices sorted by link priority (once per count)
  const LOUT = new Int32Array(memory.buffer, x.links_out(), LINK_K * 2);
  return {
    kind: 'wasm', n,
    S: f32(p[0], n * 3), P: f32(p[1], n * 4), V: f32(p[2], n * 3), F: f32(p[3], n), T: f32(p[4], n),
    heal: f32(p[5], n), SS: f32(p[6], Math.ceil(n / 64) * 5), U: f32(p[8], UNI.SIZE),
    UI: new Int32Array(memory.buffer, p[8], UNI.SIZE),
    torn: 0,
    load(init) { this.S.set(init.S); this.P.fill(0); this.V.fill(0); this.F.fill(0); this.T.fill(0); this.heal.set(init.heal); x.load_soa(); },
    reseed(S) { this.S.set(S); x.load_s_soa(); },      // shape morph: new attractor state, keep the rest
    step() { x.grid(); this.torn = x.run_simd(0, n); },
    samples() { return this.SS; },
    // Cursor links: priority-ordered SIMD walk, stops at the K-th hit (bench/links.mjs winner).
    // The caller has written VP/SW/SH and the cursor (UNI.LCX/LCY/LR2); returns the count, indices in linkOut.
    links() { return x.links_order_simd(); },
    linkOut: LOUT,
  };
}
