// Loads a C/WASM kernel build. State lives in wasm linear memory; JS only writes the
// 96-float uniform block per frame (no per-frame copies of particle data). The views
// returned by state() alias wasm memory directly, so a renderer could upload from them.
import { UNI } from '../core.js';

// bytes: the .wasm file contents (Node reads them from build/, the browser page fetches/embeds them)
export async function makeWasm(n, variant, bytes) {
  const module = await WebAssembly.compile(bytes);
  // Probe instance to learn how much memory we need, then the real one.
  const probeMem = new WebAssembly.Memory({ initial: 2 });
  const probe = await WebAssembly.instantiate(module, { env: { memory: probeMem } });
  const need = probe.exports.bytes_needed(n);
  const memory = new WebAssembly.Memory({ initial: Math.ceil(need / 65536) + 1 });
  const inst = await WebAssembly.instantiate(module, { env: { memory } });
  const x = inst.exports;
  const p = new Int32Array(memory.buffer, x.setup(n), 10);
  const f32 = (off, len) => new Float32Array(memory.buffer, off, len);
  const v = {
    S: f32(p[0], n * 3), P: f32(p[1], n * 4), V: f32(p[2], n * 3), F: f32(p[3], n), T: f32(p[4], n),
    heal: f32(p[5], n), SS: f32(p[6], Math.ceil(n / 64) * 5), U: f32(p[8], UNI.SIZE),
  };
  const simd = variant.startsWith('simd');
  return {
    name: `wasm-${variant}`,
    load(init) { for (const k of ['S', 'P', 'V', 'F', 'T', 'heal']) v[k].set(init[k]); if (simd) x.load_soa(); },
    step(U) { v.U.set(U); x.grid(); this.torn = simd ? (variant === 'simd-v1' ? x.run_simd1(0, n) : x.run_simd(0, n)) : x.run_scalar(0, n); },
    samples() { return v.SS; },
    state() { if (simd) x.store_soa(); return { S: v.S, P: v.P, V: v.V, F: v.F, T: v.T }; },
  };
}
