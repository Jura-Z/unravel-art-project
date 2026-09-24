#!/usr/bin/env node
// Runs every check and exits non-zero on the first class of failure. `npm test`.
// Needs: bench/build/*.wasm (sh bench/kernels/c/build.sh) and Playwright's Chromium.
import { spawnSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
mkdirSync('shots', { recursive: true });

const T = [
  ['build', 'node build.mjs', (o) => /built [\d.]+ KB/.test(o)],
  ['noise: gradient and divergence', 'node test/noise_test.cjs', (o) => {
    const g = +o.match(/grad max abs err ([\d.e-]+)/)[1], d = +o.match(/max divergence ([\d.e-]+)/)[1];
    return g < 1e-6 && d < 1e-4;
  }],
  ['attractor pads: no collapse', 'node test/attractor_sweep.cjs', (o) => !/\s(\d|1\d)[! ]\s/.test(o.split('==').slice(1).join(''))],
  ['kernels bit-exact (JS ↔ C/WASM)', 'node bench/run.mjs --impl js-ref,wasm-scalar,wasm-simd --frames 300 --runs 1', (o) => (o.match(/bit-exact ✓/g) || []).length === 3 && !/MISMATCH/.test(o)],
  ['cursor search: all variants identical', 'node bench/links.mjs --n 150000 --frames 100 --every 10 --reps 2 --threads 2', (o) => {
    const rows = o.split('\n').filter((l) => /\d+\/\d+$/.test(l.trim()));
    return rows.length >= 10 && rows.every((l) => /\s0\/\d+$/.test(l.trim()));
  }],
  ['cursor search in the app = full scan', 'node test/links-wasm.mjs', (o) => (o.match(/"same":true/g) || []).length === 8 && !/"same":false|pageerror/.test(o)],
  ['UI: wheel, pad, demo links, modal keys', 'node test/ui.mjs', () => true],
  ['WebGL backends run', 'node test/backends.mjs wasm-webgl,js-webgl', (o) => /wasm-webgl .*WASM SIMD kernel/.test(o) && /js-webgl .*attractor/.test(o) && !/pageerror/.test(o)],
  ['WebGPU offscreen frame', 'node test/gpu-offscreen.mjs', (o) => { const m = o.match(/"mean":([\d.]+)/); return m && +m[1] > 5 && /"hasSamples":true/.test(o); }],
  ['WebGPU shape change: fast + flash', 'node test/shape-flash.mjs', (o) => {
    const ms = o.match(/morphTo ms \(first, cached orbit\): ([\d.]+), ([\d.]+)/), f = o.match(/before ([\d.]+)\nflash ([\d.]+) flash value ([\d.]+)/);
    return ms && +ms[2] < 60 && f && +f[2] > +f[1] && +f[3] > 0.5 && !/pageerror/.test(o);
  }],
  ['morph (WebGL)', 'node test/morph.mjs', (o) => /errors \[\]/.test(o)],
  ['"How it got fast" renders', 'node test/journey.mjs', (o) => /journey rows \d+ height \d+ \[\]/.test(o) && /journey-phone rows \d+ height \d+ \[\]/.test(o)],
];

// --node: only the checks that don't need a browser (the browser ones also run on a real
// GPU via dist/selftest.html).
const nodeOnly = process.argv.includes('--node');
const BROWSER = /test\/(links-wasm|ui|backends|gpu-offscreen|shape-flash|morph|journey)\.mjs/;
let failed = 0;
for (const [name, cmd, check] of T) {
  if (nodeOnly && BROWSER.test(cmd)) continue;
  const t = Date.now();
  const r = spawnSync('sh', ['-c', cmd], { encoding: 'utf8', timeout: 600000 });
  const out = (r.stdout || '') + (r.stderr || '');
  let ok = r.status === 0;
  try { ok = ok && check(out); } catch { ok = false; }
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  (${((Date.now() - t) / 1000).toFixed(0)} s)`);
  if (!ok) { failed++; console.log(out.split('\n').filter((l) => l && !/TUNNEL|agent-proxy|connect_rejected/.test(l)).slice(-15).map((l) => '      ' + l).join('\n')); }
}
console.log(failed ? `\n${failed} failed` : '\nall passed');
process.exit(failed ? 1 : 0);
