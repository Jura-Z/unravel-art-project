// Where does the time go? Replays the trace with passes masked out (results not checked).
import { readFileSync } from 'node:fs';
import { UNI, FLAG_WARP, FLAG_TEAR, FLAG_BURST } from './core.js';
import { REGISTRY } from './kernels/registry.mjs';
const n = 150000, frames = 300, tag = `Halvorsen-${n}-${frames}`;
const trace = new Float32Array(readFileSync(`.cache/trace-${tag}.bin`).buffer.slice(0));
const ib = new Float32Array(readFileSync(`.cache/init-${tag}.bin`).buffer.slice(0));
const init = {}; let off = 0; for (const [f, s] of [['S', 3], ['P', 4], ['V', 3], ['F', 1], ['T', 1], ['heal', 1]]) { init[f] = ib.subarray(off, off + n * s); off += n * s; }
const impl = process.argv[2] || 'wasm-scalar';
const k = await REGISTRY[impl](n, {});
const buf = new ArrayBuffer(UNI.SIZE * 4), U = new Float32Array(buf), UI = new Int32Array(buf);
for (const [label, mask] of [['all', 0], ['no warp', FLAG_WARP], ['no tear/burst', FLAG_TEAR | FLAG_BURST], ['no warp, no tear', FLAG_WARP | FLAG_TEAR | FLAG_BURST]]) {
  let best = 1e9;
  for (let r = 0; r < 2; r++) {
    k.load(init); let t = 0, torn = 0;
    for (let f = 0; f < frames; f++) {
      U.set(trace.subarray(f * UNI.SIZE, (f + 1) * UNI.SIZE)); UI[UNI.FLAGS] &= ~mask;
      const t0 = performance.now(); k.step(U, UI); t += performance.now() - t0; torn += k.torn || 0;
    }
    best = Math.min(best, t / frames); if (r) console.log(`${impl} ${label.padEnd(18)} ${best.toFixed(2)} ms/frame   avg torn ${(torn / frames / n * 100).toFixed(1)}%`);
  }
}
