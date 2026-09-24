// Every experiment: name -> async factory(n, ctx) returning { load(init), step(U, UI), state() }
// (or { native: true, run() } for out-of-process runners).
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { JsRef } from './js-ref.js';
import { JsFused } from './js-fused.js';
import { readFileSync } from 'node:fs';
import { makeWasm as makeWasmBytes } from './wasm-core.js';
import { makeWasmMT } from './wasm-mt.mjs';

const build = join(dirname(fileURLToPath(import.meta.url)), '..', 'build');
const makeWasm = (n, variant) => makeWasmBytes(n, variant, readFileSync(join(build, `${variant === 'simd-v1' ? 'simd' : variant}.wasm`)));

export const REGISTRY = {
  'js-ref': async (n) => new JsRef(n),
  'js-fused': async (n) => new JsFused(n),
  'wasm-scalar': async (n) => makeWasm(n, 'scalar'),
  'wasm-simd-v1': async (n) => makeWasm(n, 'simd-v1'),
  'wasm-simd': async (n) => makeWasm(n, 'simd'),
  'wasm-simd-mt': async (n) => makeWasmMT(n),
  'native-c': async (n, ctx) => ({
    native: true,
    async run() {
      const out = execFileSync(join(build, 'native'), [
        join(ctx.cacheDir, `init-${ctx.tag}.bin`), join(ctx.cacheDir, `trace-${ctx.tag}.bin`),
        String(n), String(ctx.frames), String(ctx.runs || 2), ...ctx.CHECK.map(String)], { encoding: 'utf8' });
      return JSON.parse(out);
    },
  }),
};
