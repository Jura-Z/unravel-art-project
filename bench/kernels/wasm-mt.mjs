// SIMD kernel on shared memory across worker threads (Node worker_threads).
// Every thread instantiates the same module on the same SharedArrayBuffer-backed
// memory, gets its own stack, and processes a contiguous slice of particles. The
// per-frame handshake is one Atomics.notify out and a countdown back, so there are
// no messages, no copies, and no allocation per frame.
import { Worker } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { UNI } from '../core.js';

const here = dirname(fileURLToPath(import.meta.url));
const STACK = 256 * 1024;

const WORKER_SRC = `
const { workerData } = require('node:worker_threads');
const { module, memory, ctl, stackTop, i0, i1, slot } = workerData;
const inst = new WebAssembly.Instance(module, { env: { memory } });
inst.exports.__stack_pointer.value = stackTop;
const c = new Int32Array(ctl);
let gen = 0;
for (;;) {
  Atomics.wait(c, 0, gen);
  gen = Atomics.load(c, 0);
  if (gen < 0) break;
  c[4 + slot] = inst.exports.run_simd(i0, i1);
  if (Atomics.sub(c, 1, 1) === 1) Atomics.notify(c, 1);
}`;

export async function makeWasmMT(n, threads = availableParallelism()) {
  const module = await WebAssembly.compile(readFileSync(join(here, '..', 'build', 'simd-mt.wasm')));
  const probe = new WebAssembly.Instance(module, { env: { memory: new WebAssembly.Memory({ initial: 2, maximum: 64, shared: true }) } });
  const need = probe.exports.bytes_needed(n) + threads * STACK;
  const pages = Math.ceil(need / 65536) + 2;
  // Only reserve what we use: a 2 GB shared reservation per instance can fail late (inside a worker).
  const memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared: true });
  const main = new WebAssembly.Instance(module, { env: { memory } });
  const x = main.exports;
  const p = new Int32Array(memory.buffer, x.setup(n), 10);
  const heapEnd = (p[9] + 15) & ~15;
  const f32 = (off, len) => new Float32Array(memory.buffer, off, len);
  const v = {
    S: f32(p[0], n * 3), P: f32(p[1], n * 4), V: f32(p[2], n * 3), F: f32(p[3], n), T: f32(p[4], n),
    heal: f32(p[5], n), SS: f32(p[6], Math.ceil(n / 64) * 5), U: f32(p[8], UNI.SIZE),
  };
  // slices in groups of 4 particles (SIMD width); main thread takes slice 0
  const groups = n >> 2, per = Math.ceil(groups / threads);
  const range = (t) => [Math.min(n, t * per * 4), Math.min(n, (t + 1) * per * 4)];
  const ctl = new SharedArrayBuffer(4 * (4 + threads));
  const c = new Int32Array(ctl);
  const workers = [];
  let failed = null;
  for (let t = 1; t < threads; t++) {
    const [i0, i1] = range(t);
    const w = new Worker(WORKER_SRC, { eval: true, workerData: { module, memory, ctl, stackTop: heapEnd + (t + 1) * STACK, i0, i1, slot: t } });
    w.on('error', (e) => { failed = e; Atomics.store(c, 1, 0); Atomics.notify(c, 1); });
    workers.push(w);
  }
  await new Promise((r) => setTimeout(r, 50));   // let workers instantiate
  const [m0, m1] = range(0);
  return {
    name: `wasm-simd-mt${threads}`,
    load(init) { for (const k of ['S', 'P', 'V', 'F', 'T', 'heal']) v[k].set(init[k]); x.load_soa(); },
    step(U) {
      v.U.set(U);
      x.grid();                                   // warp grid: 1728 curls, main thread
      Atomics.store(c, 1, threads - 1);
      Atomics.add(c, 0, 1); Atomics.notify(c, 0);
      c[4] = x.run_simd(m0, m1);
      while (Atomics.load(c, 1) !== 0) Atomics.wait(c, 1, Atomics.load(c, 1), 2000);
      if (failed) throw new Error('worker failed: ' + failed.message);
      let torn = 0; for (let t = 0; t < threads; t++) torn += c[4 + t];
      this.torn = torn;
    },
    samples() { return v.SS; },
    state() { x.store_soa(); return { S: v.S, P: v.P, V: v.V, F: v.F, T: v.T }; },
    async close() { Atomics.store(c, 0, -1); Atomics.notify(c, 0); await Promise.all(workers.map((w) => w.terminate())); },
  };
}
