#!/usr/bin/env node
// Deterministic benchmark: record a trace with the reference kernel (cached on disk),
// replay it through every experiment, check bit-exactness at checkpoints, report speed.
//
//   node bench/run.mjs                         # all CPU experiments, defaults
//   node bench/run.mjs --impl js-ref,wasm-simd --n 300000 --frames 300 --runs 3
//   node bench/run.mjs --shape Thomas
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { UNI, fnv } from './core.js';
import { makeHost } from './host.js';
import { JsRef } from './kernels/js-ref.js';
import { REGISTRY } from './kernels/registry.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, arr) => (v.startsWith('--') ? [...a, [v.slice(2), arr[i + 1]]] : a), []));
const n = +(args.n || 150000), frames = +(args.frames || 300), shape = args.shape || 'Halvorsen', runs = +(args.runs || 2);
const impls = (args.impl || Object.keys(REGISTRY).join(',')).split(',');
const CHECK = [0, 9, 99, frames - 1];

// ---------- record (or load cached) trace ----------
const cacheDir = join(here, '.cache'); mkdirSync(cacheDir, { recursive: true });
const tag = `${shape}-${n}-${frames}`;
const tracePath = join(cacheDir, `trace-${tag}.bin`), initPath = join(cacheDir, `init-${tag}.bin`), refPath = join(cacheDir, `ref-${tag}.json`);
let trace, init, ref;
const fields = ['S', 'P', 'V', 'F', 'T', 'heal'], sizes = { S: 3, P: 4, V: 3, F: 1, T: 1, heal: 1 };
if (existsSync(tracePath) && existsSync(initPath) && existsSync(refPath) && !args.rerecord) {
  trace = new Float32Array(readFileSync(tracePath).buffer.slice(0));
  const ib = new Float32Array(readFileSync(initPath).buffer.slice(0));
  init = {}; let off = 0;
  for (const f of fields) { init[f] = ib.subarray(off, off + n * sizes[f]); off += n * sizes[f]; }
  ref = JSON.parse(readFileSync(refPath, 'utf8'));
} else {
  const t0 = performance.now();
  const host = makeHost({ n, shape });
  init = host.initState();
  const k = new JsRef(n); k.load(init);
  trace = new Float32Array(frames * UNI.SIZE);
  ref = { checks: {} };
  for (let f = 0; f < frames; f++) {
    const { U, UI } = host.prepare(f);
    trace.set(U, f * UNI.SIZE);
    k.step(U, UI);
    host.after(k.samples());
    if (CHECK.includes(f)) ref.checks[f] = fnv(Object.values(k.state()));
    if (f === 0 || f === 9) writeFileSync(join(cacheDir, `ref-${tag}-f${f}.bin`), Buffer.concat(Object.values(k.state()).map((v) => Buffer.from(v.buffer, v.byteOffset, v.byteLength))));
  }
  writeFileSync(tracePath, Buffer.from(trace.buffer));
  const all = new Float32Array(fields.reduce((s, f) => s + n * sizes[f], 0)); let off = 0;
  for (const f of fields) { all.set(init[f], off); off += n * sizes[f]; }
  writeFileSync(initPath, Buffer.from(all.buffer));
  writeFileSync(refPath, JSON.stringify(ref));
  console.log(`recorded trace ${tag} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

// ---------- replay ----------
async function replay(k) {
  const buf = new ArrayBuffer(UNI.SIZE * 4), U = new Float32Array(buf), UI = new Int32Array(buf);
  let best = null;
  for (let r = 0; r < runs; r++) {
    k.load(init);
    const times = new Float64Array(frames), checks = {};
    for (let f = 0; f < frames; f++) {
      for (let q = 0, b = f * UNI.SIZE; q < UNI.SIZE; q++) U[q] = trace[b + q];   // no per-frame view allocation
      const t0 = performance.now();
      await k.step(U, UI);
      times[f] = performance.now() - t0;
      if (CHECK.includes(f)) checks[f] = fnv(Object.values(await k.state()));
    }
    const total = times.reduce((a, b) => a + b, 0);
    const sorted = Array.from(times).sort((a, b) => a - b);
    const res = { mean: total / frames, median: sorted[frames >> 1], p95: sorted[Math.floor(frames * 0.95)], checks };
    if (!best || res.mean < best.mean) best = res;
  }
  return best;
}

console.log(`\nUnravel kernel benchmark · ${shape} · ${n.toLocaleString()} particles · ${frames} frames · best of ${runs} · ${process.version}\n`);
const rows = [];
let baseline = null;
for (const name of impls) {
  const make = REGISTRY[name];
  if (!make) { console.log(`unknown impl ${name}`); continue; }
  let r;
  try {
    const k = await make(n, { frames, trace, init, cacheDir, tag, CHECK, runs });
    r = k.native ? await k.run() : await replay(k);
    if (k.close) await k.close();
  } catch (e) { console.log(`${name}: FAILED ${e.message}`); continue; }
  const exact = CHECK.every((f) => r.checks[f] === ref.checks[f]);
  const firstBad = CHECK.find((f) => r.checks[f] !== ref.checks[f]);
  if (name === 'js-ref') { baseline = r.mean; ref.jsRefMs = r.mean; writeFileSync(refPath, JSON.stringify(ref)); }
  if (baseline === null) baseline = ref.jsRefMs || r.mean;   // cached js-ref time when it isn't re-run
  rows.push({ name, ...r, exact, firstBad });
  console.log(`${name.padEnd(16)} ${r.mean.toFixed(2).padStart(8)} ms/frame   median ${r.median.toFixed(2).padStart(7)}   p95 ${r.p95.toFixed(2).padStart(7)}   ${(baseline / r.mean).toFixed(2).padStart(6)}×   ${exact ? 'bit-exact ✓' : `MISMATCH ✗ (first at frame ${firstBad + 1})`}`);
}
writeFileSync(join(cacheDir, `results-${tag}.json`), JSON.stringify({ n, frames, shape, rows, ref }, null, 1));
