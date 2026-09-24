// SIMD kernel on shared memory across Web Workers (the browser twin of wasm-mt.mjs).
// Needs a cross-origin-isolated page (COOP + COEP) for SharedArrayBuffer. Every worker
// instantiates the same module on the same memory with its own stack and steps a fixed
// slice; the main thread takes slice 0. Per frame: one Atomics.notify out, a countdown back.
// The main thread may not Atomics.wait, so it spins on the countdown (slices finish together).
import { UNI } from '../core.js';

const STACK = 256 * 1024;
const WORKER = `onmessage = (e) => {
  const { module, memory, ctl, stackTop, i0, i1, slot } = e.data;
  const x = new WebAssembly.Instance(module, { env: { memory } }).exports;
  x.__stack_pointer.value = stackTop;
  const c = new Int32Array(ctl);
  let gen = Atomics.load(c, 0);
  Atomics.add(c, 2, 1);
  for (;;) {
    Atomics.wait(c, 0, gen);
    gen = Atomics.load(c, 0);
    if (gen < 0) return;
    c[4 + slot] = x.run_simd(i0, i1);
    if (Atomics.sub(c, 1, 1) === 1) Atomics.notify(c, 1);
  }
};`;

export async function makeWasmMTWeb(n, bytes, threads) {
  if (!crossOriginIsolated) throw new Error('needs a cross-origin-isolated page (COOP/COEP)');
  const module = await WebAssembly.compile(bytes);
  const probe = new WebAssembly.Instance(module, { env: { memory: new WebAssembly.Memory({ initial: 2, maximum: 64, shared: true }) } });
  const pages = Math.ceil((probe.exports.bytes_needed(n) + threads * STACK) / 65536) + 2;
  const memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared: true });
  const x = new WebAssembly.Instance(module, { env: { memory } }).exports;
  const p = new Int32Array(memory.buffer, x.setup(n), 10);
  const heapEnd = (p[9] + 15) & ~15;
  const f32 = (off, len) => new Float32Array(memory.buffer, off, len);
  const v = {
    S: f32(p[0], n * 3), P: f32(p[1], n * 4), V: f32(p[2], n * 3), F: f32(p[3], n), T: f32(p[4], n),
    heal: f32(p[5], n), SS: f32(p[6], Math.ceil(n / 64) * 5), U: f32(p[8], UNI.SIZE),
  };
  const groups = n >> 2, per = Math.ceil(groups / threads);
  const range = (t) => [Math.min(n, t * per * 4), t === threads - 1 ? n : Math.min(n, (t + 1) * per * 4)];
  const ctl = new SharedArrayBuffer(4 * (4 + threads));
  const c = new Int32Array(ctl);
  const url = URL.createObjectURL(new Blob([WORKER], { type: 'text/javascript' }));
  const workers = [];
  for (let t = 1; t < threads; t++) {
    const [i0, i1] = range(t), w = new Worker(url);
    w.postMessage({ module, memory, ctl, stackTop: heapEnd + (t + 1) * STACK, i0, i1, slot: t });
    workers.push(w);
  }
  URL.revokeObjectURL(url);
  const t0 = performance.now();
  while (Atomics.load(c, 2) < threads - 1) {                // wait for every worker to instantiate
    if (performance.now() - t0 > 10000) throw new Error('workers did not start');
    await new Promise((r) => setTimeout(r, 5));
  }
  const [m0, m1] = range(0);
  return {
    name: `wasm-simd-mt${threads}`,
    load(init) { for (const k of ['S', 'P', 'V', 'F', 'T', 'heal']) v[k].set(init[k]); x.load_soa(); },
    step(U) {
      v.U.set(U);
      x.grid();
      Atomics.store(c, 1, threads - 1);
      Atomics.add(c, 0, 1); Atomics.notify(c, 0);
      let torn = x.run_simd(m0, m1);
      while (Atomics.load(c, 1) !== 0) {}
      for (let t = 1; t < threads; t++) torn += c[4 + t];
      this.torn = torn;
    },
    samples() { return v.SS; },
    state() { x.store_soa(); return { S: v.S, P: v.P, V: v.V, F: v.F, T: v.T }; },
    close() { Atomics.store(c, 0, -1); Atomics.notify(c, 0); for (const w of workers) w.terminate(); },
  };
}
