#!/usr/bin/env node
// Cursor-link search experiments. Question: "which K=40 particles (among every 4th) with the
// lowest hash priority project inside the 150 px cursor circle?" The answer is a unique set
// (priority is a bijection), so every variant must return exactly the same indices.
//
// Dataset: P snapshots from the deterministic session (replayed with the WASM SIMD kernel,
// which is bit-exact with the reference), every `--every` frames, x 12 cursor positions
// (dense centre, sparse edges, empty corners, half off-screen).
//
//   node bench/links.mjs --n 600000 --frames 300 --every 10
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { UNI } from './core.js';
import { makeHost } from './host.js';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, arr) => (v.startsWith('--') ? [...a, [v.slice(2), arr[i + 1]]] : a), []));
const n = +(args.n || 600000), frames = +(args.frames || 300), every = +(args.every || 10), shape = args.shape || 'Halvorsen';
const REPS = +(args.reps || 5);
const K = 40, STRIDE = 4, RADIUS = 150, R2 = RADIUS * RADIUS;
const U_LCX = 86, U_LCY = 87, U_LR2 = 88;
const fr = Math.fround, W_MIN = Math.fround(0.05);
const CURSORS = [[0.5, 0.5], [0.45, 0.55], [0.6, 0.4], [0.3, 0.5], [0.7, 0.6], [0.5, 0.2], [0.5, 0.85], [0.05, 0.05], [0.95, 0.95], [0, 0.5], [1, 0.3], [0.2, 0.8]];

// ---------- shared-memory WASM instance (SIMD + threads build; single-thread variants use it too) ----------
const bytes = readFileSync(join(here, 'build', 'simd-mt.wasm'));
const module = await WebAssembly.compile(bytes);
const THREADS = +(args.threads || Math.min(8, availableParallelism()));
const STACK = 256 * 1024;
const probe = new WebAssembly.Instance(module, { env: { memory: new WebAssembly.Memory({ initial: 2, maximum: 64, shared: true }) } });
const pages = Math.ceil((probe.exports.bytes_needed(n) + THREADS * STACK) / 65536) + 2;
const memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared: true });
const x = new WebAssembly.Instance(module, { env: { memory } }).exports;
const ptrs = new Int32Array(memory.buffer, x.setup(n), 10);
const f32 = (off, len) => new Float32Array(memory.buffer, off, len);
const v = { S: f32(ptrs[0], n * 3), P: f32(ptrs[1], n * 4), V: f32(ptrs[2], n * 3), F: f32(ptrs[3], n), T: f32(ptrs[4], n), heal: f32(ptrs[5], n), SS: f32(ptrs[6], Math.ceil(n / 64) * 5), U: f32(ptrs[8], UNI.SIZE) };
const LOUT = new Int32Array(memory.buffer, x.links_out(), 64 * K * 2);

// ---------- dataset ----------
const t0 = performance.now();
const host = makeHost({ n, shape });
const init = host.initState();
for (const k of ['S', 'P', 'V', 'F', 'T', 'heal']) v[k].set(init[k]);
x.load_soa();
const snaps = [];
for (let f = 0; f < frames; f++) {
  const { U } = host.prepare(f);
  v.U.set(U); x.grid(); x.run_simd(0, n); host.after(v.SS);
  if ((f + 1) % every === 0) snaps.push({ P: v.P.slice(), U: v.U.slice() });
}
console.log(`dataset: ${shape}, ${n.toLocaleString()} particles, ${snaps.length} snapshots x ${CURSORS.length} cursors (${((performance.now() - t0) / 1000).toFixed(1)} s)`);

// ---------- JS variants (exact f32: fround after every op, same order as links.c) ----------
const idxJ = new Int32Array(K), priJ = new Uint32Array(K);
function push(m, i) {
  const pr = Math.imul(i, 2654435761) >>> 0;
  if (m < K) { idxJ[m] = i; priJ[m] = pr; return m + 1; }
  let worst = 0; for (let q = 1; q < K; q++) if (priJ[q] > priJ[worst]) worst = q;
  if (pr < priJ[worst]) { idxJ[worst] = i; priJ[worst] = pr; }
  return m;
}
function jsBrute(P, U, cx, cy) {
  const vp = U.subarray(UNI.VP, UNI.VP + 16), sw = U[UNI.SW], sh = U[UNI.SH], r2 = fr(R2);
  const a0 = vp[0], a1 = vp[1], a3 = vp[3], a4 = vp[4], a5 = vp[5], a7 = vp[7], a8 = vp[8], a9 = vp[9], a11 = vp[11], a12 = vp[12], a13 = vp[13], a15 = vp[15];
  let m = 0;
  for (let i = 0; i < n; i += STRIDE) {
    const j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
    const w = fr(fr(fr(fr(a3 * x) + fr(a7 * y)) + fr(a11 * z)) + a15);
    if (!(w > W_MIN)) continue;
    const cxp = fr(fr(fr(fr(a0 * x) + fr(a4 * y)) + fr(a8 * z)) + a12);
    const cyp = fr(fr(fr(fr(a1 * x) + fr(a5 * y)) + fr(a9 * z)) + a13);
    const ex = fr(fr(fr(fr(fr(cxp / w) * 0.5) + 0.5) * sw) - cx);
    const ey = fr(fr(fr(0.5 - fr(fr(cyp / w) * 0.5)) * sh) - cy);
    if (fr(fr(ex * ex) + fr(ey * ey)) < r2) m = push(m, i);
  }
  return m;
}
// The app's current code: plain doubles (not bit-matched to C; reported separately).
function jsBruteDouble(P, U, cx, cy) {
  const vp = U.subarray(UNI.VP, UNI.VP + 16), W = U[UNI.SW], H = U[UNI.SH];
  let m = 0;
  for (let i = 0; i < n; i += STRIDE) {
    const j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
    const w = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    if (w <= 0.05) continue;
    const ex = ((vp[0] * x + vp[4] * y + vp[8] * z + vp[12]) / w * 0.5 + 0.5) * W - cx;
    const ey = (0.5 - (vp[1] * x + vp[5] * y + vp[9] * z + vp[13]) / w * 0.5) * H - cy;
    if (ex * ex + ey * ey < R2) m = push(m, i);
  }
  return m;
}
// Priority-ordered walk: indices pre-sorted by priority once per particle count, stop at the K-th hit.
const INV = (() => { let y = 1; for (let k = 0; k < 5; k++) y = Math.imul(y, 2 - Math.imul(2654435761, y)); return y; })();  // Newton: inverse mod 2^32
const ORDER = (() => {
  const m = Math.ceil(n / STRIDE), keys = new Uint32Array(m);
  for (let q = 0; q < m; q++) keys[q] = Math.imul(q * STRIDE, 2654435761) >>> 0;
  keys.sort();
  for (let q = 0; q < m; q++) keys[q] = Math.imul(keys[q], INV) >>> 0;   // priority -> index
  return new Int32Array(keys.buffer);
})();
function jsOrder(P, U, cx, cy) {
  const vp = U.subarray(UNI.VP, UNI.VP + 16), sw = U[UNI.SW], sh = U[UNI.SH], r2 = fr(R2);
  const a0 = vp[0], a1 = vp[1], a3 = vp[3], a4 = vp[4], a5 = vp[5], a7 = vp[7], a8 = vp[8], a9 = vp[9], a11 = vp[11], a12 = vp[12], a13 = vp[13], a15 = vp[15];
  let m = 0;
  for (let q = 0, e = ORDER.length; q < e && m < K; q++) {
    const i = ORDER[q], j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
    const w = fr(fr(fr(fr(a3 * x) + fr(a7 * y)) + fr(a11 * z)) + a15);
    if (!(w > W_MIN)) continue;
    const cxp = fr(fr(fr(fr(a0 * x) + fr(a4 * y)) + fr(a8 * z)) + a12);
    const cyp = fr(fr(fr(fr(a1 * x) + fr(a5 * y)) + fr(a9 * z)) + a13);
    const ex = fr(fr(fr(fr(fr(cxp / w) * 0.5) + 0.5) * sw) - cx);
    const ey = fr(fr(fr(0.5 - fr(fr(cyp / w) * 0.5)) * sh) - cy);
    if (fr(fr(ex * ex) + fr(ey * ey)) < r2) { idxJ[m] = i; priJ[m] = Math.imul(i, 2654435761) >>> 0; m++; }
  }
  return m;
}
// Same walk in plain doubles, and a hybrid: walk at most BUDGET probes in priority order;
// if that didn't find K hits (sparse region), fall back to the full scan. Both return the exact top-K.
function jsOrderF64(P, U, cx, cy, budget = ORDER.length) {
  const vp = U.subarray(UNI.VP, UNI.VP + 16), W = U[UNI.SW], H = U[UNI.SH];
  const a0 = vp[0], a1 = vp[1], a3 = vp[3], a4 = vp[4], a5 = vp[5], a7 = vp[7], a8 = vp[8], a9 = vp[9], a11 = vp[11], a12 = vp[12], a13 = vp[13], a15 = vp[15];
  let m = 0;
  for (let q = 0, e = Math.min(budget, ORDER.length); q < e; q++) {
    const i = ORDER[q], j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
    const w = a3 * x + a7 * y + a11 * z + a15;
    if (w <= 0.05) continue;
    const ex = ((a0 * x + a4 * y + a8 * z + a12) / w * 0.5 + 0.5) * W - cx;
    const ey = (0.5 - (a1 * x + a5 * y + a9 * z + a13) / w * 0.5) * H - cy;
    if (ex * ex + ey * ey < R2) { idxJ[m] = i; priJ[m] = Math.imul(i, 2654435761) >>> 0; if (++m === K) return m; }
  }
  return -m - 1;   // not finished
}
const BUDGET = +(args.budget || 4096);
function jsHybrid(P, U, cx, cy) {
  const m = jsOrderF64(P, U, cx, cy, BUDGET);
  return m >= 0 ? m : jsBruteDouble(P, U, cx, cy);
}
// Screen grid: project all, counting-sort into 150 px cells (off-screen clamped to border cells), query the bbox.
const CELL = 150, M = Math.ceil(n / STRIDE);
const GSX = new Float32Array(M), GSY = new Float32Array(M), GCELL = new Int32Array(M), GITEMS = new Int32Array(M);
const GSTART = new Int32Array(4097), FILL = new Int32Array(4096);
function jsGrid(P, U, cx, cy) {
  const vp = U.subarray(UNI.VP, UNI.VP + 16), sw = U[UNI.SW], sh = U[UNI.SH], r2 = fr(R2);
  const a0 = vp[0], a1 = vp[1], a3 = vp[3], a4 = vp[4], a5 = vp[5], a7 = vp[7], a8 = vp[8], a9 = vp[9], a11 = vp[11], a12 = vp[12], a13 = vp[13], a15 = vp[15];
  const gw = ((sw / CELL) | 0) + 1, gh = ((sh / CELL) | 0) + 1, nc = gw * gh;
  GSTART.fill(0, 0, nc + 1);
  let cnt = 0;
  for (let i = 0; i < n; i += STRIDE, cnt++) {
    const j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
    const w = fr(fr(fr(fr(a3 * x) + fr(a7 * y)) + fr(a11 * z)) + a15);
    let cell = -1;
    if (w > W_MIN) {
      const cxp = fr(fr(fr(fr(a0 * x) + fr(a4 * y)) + fr(a8 * z)) + a12);
      const cyp = fr(fr(fr(fr(a1 * x) + fr(a5 * y)) + fr(a9 * z)) + a13);
      const sx = fr(fr(fr(fr(cxp / w) * 0.5) + 0.5) * sw), sy = fr(fr(0.5 - fr(fr(cyp / w) * 0.5)) * sh);
      GSX[cnt] = sx; GSY[cnt] = sy;
      let fx = sx / CELL, fy = sy / CELL;
      if (fx === fx && fy === fy) {
        fx = fx < 0 ? 0 : fx > gw - 1 ? gw - 1 : fx; fy = fy < 0 ? 0 : fy > gh - 1 ? gh - 1 : fy;
        cell = (fy | 0) * gw + (fx | 0); GSTART[cell + 1]++;
      }
    }
    GCELL[cnt] = cell;
  }
  for (let c = 0; c < nc; c++) GSTART[c + 1] += GSTART[c];
  for (let c = 0; c < nc; c++) FILL[c] = GSTART[c];
  for (let q = 0; q < cnt; q++) { const c = GCELL[q]; if (c >= 0) GITEMS[FILL[c]++] = q; }
  const r = Math.sqrt(r2), cl = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v | 0);
  const x0 = cl((cx - r) / CELL, gw - 1), x1 = cl((cx + r) / CELL, gw - 1), y0 = cl((cy - r) / CELL, gh - 1), y1 = cl((cy + r) / CELL, gh - 1);
  let m = 0;
  for (let gy = y0; gy <= y1; gy++) for (let gx = x0; gx <= x1; gx++) {
    const c = gy * gw + gx;
    for (let s = GSTART[c], e = GSTART[c + 1]; s < e; s++) {
      const q = GITEMS[s], ex = fr(GSX[q] - cx), ey = fr(GSY[q] - cy);
      if (fr(fr(ex * ex) + fr(ey * ey)) < r2) m = push(m, q * STRIDE);
    }
  }
  return m;
}

// ---------- MT: workers each scan a slice into their own top-K slot, main merges ----------
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
  c[4 + slot] = inst.exports.links_brute_simd(i0, i1, slot);
  if (Atomics.sub(c, 1, 1) === 1) Atomics.notify(c, 1);
}`;
const heapEnd = (ptrs[9] + 15) & ~15;
const per = Math.ceil(n / 16 / THREADS) * 16, range = (t) => [Math.min(n, t * per), Math.min(n, (t + 1) * per)];
const ctl = new SharedArrayBuffer(4 * (4 + THREADS)), c = new Int32Array(ctl);
const workers = [];
for (let t = 1; t < THREADS; t++) {
  const [i0, i1] = range(t);
  workers.push(new Worker(WORKER_SRC, { eval: true, workerData: { module, memory, ctl, stackTop: heapEnd + (t + 1) * STACK, i0, i1, slot: t } }));
}
await new Promise((r) => setTimeout(r, 200));
function wasmMT() {
  Atomics.store(c, 1, THREADS - 1);
  Atomics.add(c, 0, 1); Atomics.notify(c, 0);
  c[4] = x.links_brute_simd(...range(0), 0);
  while (Atomics.load(c, 1) !== 0) Atomics.wait(c, 1, Atomics.load(c, 1), 2000);
  // merge per-thread top-Ks into slot 0 (slot t's candidates are lower-priority-first only by chance: full top-K merge)
  let m = 0;
  for (let t = 0; t < THREADS; t++) for (let q = 0; q < c[4 + t]; q++) m = push(m, LOUT[t * K * 2 + q]);
  return m;
}

// ---------- runner ----------
const canon = (ids, m) => Array.from(ids.subarray(0, m)).sort((a, b) => (Math.imul(a, 2654435761) >>> 0) - (Math.imul(b, 2654435761) >>> 0)).join(',');
const fromJS = (m) => canon(idxJ, m), fromWasm = (m) => canon(LOUT, m);
const VARIANTS = {
  'js-brute-f64 (app today)': { run: jsBruteDouble, js: true, exact: false },
  'js-brute': { run: jsBrute, js: true },
  'js-grid': { run: jsGrid, js: true },
  'js-order': { run: jsOrder, js: true },
  'js-order-f64': { run: (P, U, cx, cy) => { const m = jsOrderF64(P, U, cx, cy); return m >= 0 ? m : -m - 1; }, js: true },
  'js-hybrid-f64': { run: jsHybrid, js: true },
  'wasm-brute': { run: () => x.links_brute(0, n, 0) },
  'wasm-brute-simd': { run: () => x.links_brute_simd(0, n, 0) },
  [`wasm-simd-mt${THREADS}`]: { run: wasmMT, js: true },
  'wasm-grid': { run: () => x.links_grid() },
  'wasm-order': { run: () => x.links_order() },
  'wasm-order-simd': { run: () => x.links_order_simd() },
};
x.links_setup_order();

const results = {};
const truth = [];
let mismatches = {};
for (const name in VARIANTS) { results[name] = []; mismatches[name] = 0; }
for (let rep = 0; rep < REPS; rep++) {
  for (let s = 0; s < snaps.length; s++) {
    const { P, U } = snaps[s];
    v.P.set(P); v.U.set(U);
    for (let ci = 0; ci < CURSORS.length; ci++) {
      const cx = fr(CURSORS[ci][0] * U[UNI.SW]), cy = fr(CURSORS[ci][1] * U[UNI.SH]);
      v.U[U_LCX] = cx; v.U[U_LCY] = cy; v.U[U_LR2] = R2;
      const key = s * CURSORS.length + ci;
      for (const name in VARIANTS) {
        const V = VARIANTS[name];
        const t = performance.now();
        const m = V.run(v.P, v.U, cx, cy);
        const dt = performance.now() - t;
        const set = V.js ? fromJS(m) : fromWasm(m);
        if (rep === 0) {
          if (name === 'js-brute') truth[key] = set;
          else if (truth[key] !== undefined && set !== truth[key]) mismatches[name]++;
        }
        if (rep > 0) results[name].push({ dt, key, m });   // rep 0 is warm-up
      }
    }
  }
}
// js-brute-f64 ran before js-brute in rep 0: compare now
{
  for (let s = 0; s < snaps.length; s++) for (let ci = 0; ci < CURSORS.length; ci++) {
    const { P, U } = snaps[s]; const cx = fr(CURSORS[ci][0] * U[UNI.SW]), cy = fr(CURSORS[ci][1] * U[UNI.SH]);
    if (fromJS(jsBruteDouble(P, U, cx, cy)) !== truth[s * CURSORS.length + ci]) mismatches['js-brute-f64 (app today)']++;
  }
}
Atomics.store(c, 0, -1); Atomics.notify(c, 0); await Promise.all(workers.map((w) => w.terminate()));

const q = snaps.length * CURSORS.length;
const hits = results['js-brute'].slice(0, q).map((r) => r.m);
console.log(`queries: ${q}; hits per query: mean ${(hits.reduce((a, b) => a + b, 0) / q).toFixed(1)}, ${hits.filter((h) => h < K).length} queries with < ${K} hits (worst case for early exit)`);
console.log('\nvariant                      mean ms   p50 ms   p95 ms   max ms   set mismatches');
const rows = [];
for (const name in VARIANTS) {
  const d = results[name].map((r) => r.dt).sort((a, b) => a - b);
  const mean = d.reduce((a, b) => a + b, 0) / d.length, pct = (p) => d[Math.min(d.length - 1, Math.floor(p * d.length))];
  rows.push({ name, mean, p50: pct(0.5), p95: pct(0.95), max: d[d.length - 1], mm: mismatches[name] });
  console.log(`${name.padEnd(28)} ${mean.toFixed(3).padStart(7)}  ${pct(0.5).toFixed(3).padStart(7)}  ${pct(0.95).toFixed(3).padStart(7)}  ${d[d.length - 1].toFixed(3).padStart(7)}   ${mismatches[name]}/${q}`);
}
if (args.json) console.log(JSON.stringify({ n, shape, q, threads: THREADS, rows }));
