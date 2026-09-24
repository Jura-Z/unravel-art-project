// WASM SIMD simulation (the bench kernel, bit-exact to the reference spec).
// All particle state lives in WASM linear memory. JS writes the uniform block straight
// into that memory, and the WebGL renderer uploads P and T from views of it: no copies
// in JS, no allocation per frame.
let WASM_MODULE = null, WASM_MT_MODULE = null;
// Threads need SharedArrayBuffer, which browsers only grant to cross-origin isolated pages
// (COOP + COEP headers). Without it, the single-threaded kernel runs as before.
// ?threads=N overrides the count (1 = the single-threaded kernel), for benchmarking.
const WASM_THREADS = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated
  ? Math.max(1, Math.min(+new URLSearchParams(location.search).get('threads') || navigator.hardwareConcurrency || 1, 32)) : 1;

async function compileWasmKernel() {
  const bytes = b64bytes(WASM_SIMD_B64);
  if (typeof WebAssembly !== 'object' || !WebAssembly.validate(bytes)) throw new Error('WebAssembly SIMD is not supported here');
  WASM_MODULE = await WebAssembly.compile(bytes);   // async compile once; instances are created synchronously
  if (WASM_THREADS > 1) {
    try { WASM_MT_MODULE = await WebAssembly.compile(b64bytes(WASM_SIMD_MT_B64)); }
    catch (err) { console.warn('[unravel] threaded WASM unavailable:', err); }
  }
}

function createWasmSim(n) {
  if (WASM_MT_MODULE) return createWasmMTSim(n, WASM_THREADS);
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

// ---- threads -----------------------------------------------------------------------------
// The same kernel built with shared memory (simd-mt.wasm). Every worker instantiates it on the
// sim's memory with its own stack and steps a fixed slice of particles; the main thread takes
// slice 0. Per frame: one Atomics.notify out, a countdown back. No messages, no copies.
// The pool outlives a sim: a new particle count re-points the same workers at new memory.
const MT_STACK = 256 * 1024;
const MT_WORKER = `onmessage = (e) => {
  const { module, memory, ctl, stackTop, i0, i1, slot } = e.data;
  const x = new WebAssembly.Instance(module, { env: { memory } }).exports;
  x.__stack_pointer.value = stackTop;
  const c = new Int32Array(ctl);
  let gen = Atomics.load(c, 0);
  Atomics.add(c, 2, 1);                                   // ready
  for (;;) {
    Atomics.wait(c, 0, gen);
    gen = Atomics.load(c, 0);
    if (gen < 0) return;                                  // retired: the next message brings a new sim
    c[4 + slot] = x.run_simd(i0, i1);
    if (Atomics.sub(c, 1, 1) === 1) Atomics.notify(c, 1);
  }
};`;
let mtPool = null, mtRetire = null;

function createWasmMTSim(n, threads) {
  const module = WASM_MT_MODULE;
  if (mtRetire) mtRetire();
  const probe = new WebAssembly.Instance(module, { env: { memory: new WebAssembly.Memory({ initial: 2, maximum: 64, shared: true }) } });
  const pages = Math.ceil((probe.exports.bytes_needed(n) + threads * MT_STACK) / 65536) + 2;
  const memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared: true });
  const x = new WebAssembly.Instance(module, { env: { memory } }).exports;
  const p = new Int32Array(memory.buffer, x.setup(n), 10);
  const heapEnd = (p[9] + 15) & ~15;
  const f32 = (off, len) => new Float32Array(memory.buffer, off, len);
  x.links_setup_order();
  const LOUT = new Int32Array(memory.buffer, x.links_out(), LINK_K * 2);
  // slices in groups of 4 particles (SIMD width)
  const groups = n >> 2, per = Math.ceil(groups / threads);
  const range = (t) => [Math.min(n, t * per * 4), t === threads - 1 ? n : Math.min(n, (t + 1) * per * 4)];
  const ctl = new SharedArrayBuffer(4 * (4 + threads));
  const c = new Int32Array(ctl);
  if (!mtPool) {
    const url = URL.createObjectURL(new Blob([MT_WORKER], { type: 'text/javascript' }));
    mtPool = Array.from({ length: threads - 1 }, () => new Worker(url));
  }
  mtPool.forEach((w, k) => {
    const t = k + 1, [i0, i1] = range(t);
    w.postMessage({ module, memory, ctl, stackTop: heapEnd + (t + 1) * MT_STACK, i0, i1, slot: t });
  });
  let alive = true;
  mtRetire = () => { alive = false; Atomics.store(c, 0, -1); Atomics.notify(c, 0); };
  const [m0, m1] = range(0);
  return {
    kind: 'wasm', n, threads,
    S: f32(p[0], n * 3), P: f32(p[1], n * 4), V: f32(p[2], n * 3), F: f32(p[3], n), T: f32(p[4], n),
    heal: f32(p[5], n), SS: f32(p[6], Math.ceil(n / 64) * 5), U: f32(p[8], UNI.SIZE),
    UI: new Int32Array(memory.buffer, p[8], UNI.SIZE),
    torn: 0,
    load(init) { this.S.set(init.S); this.P.fill(0); this.V.fill(0); this.F.fill(0); this.T.fill(0); this.heal.set(init.heal); x.load_soa(); },
    reseed(S) { this.S.set(S); x.load_s_soa(); },
    step() {
      x.grid();                                          // warp grid on the main thread; the notify below publishes it
      // Until every worker has instantiated (a few ms after a count change), run the whole range here.
      if (!alive || Atomics.load(c, 2) < threads - 1) { this.torn = x.run_simd(0, n); return; }
      Atomics.store(c, 1, threads - 1);
      Atomics.add(c, 0, 1); Atomics.notify(c, 0);
      let torn = x.run_simd(m0, m1);
      while (Atomics.load(c, 1) !== 0) {}                // the main thread may not Atomics.wait; slices finish together
      for (let t = 1; t < threads; t++) torn += c[4 + t];
      this.torn = torn;
    },
    samples() { return this.SS; },
    links() { return x.links_order_simd(); },
    linkOut: LOUT,
  };
}
