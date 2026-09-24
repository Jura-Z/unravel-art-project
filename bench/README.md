# Unravel kernel bench

A deterministic benchmark for the Unravel particle step. Every experiment replays
the same recorded session, must reproduce the reference state **bit for bit**, and
is then judged only on speed. See `SPEC.md` for the determinism rules.

```sh
kernels/c/build.sh                       # clang 18: native + 3 WASM variants
node run.mjs                             # all experiments, 150k particles, 300 frames
node run.mjs --impl wasm-simd,wasm-simd-mt --runs 3
node run.mjs --shape Thomas --n 50000    # records a new trace on first use (cached in .cache/)
node profile.mjs wasm-simd               # cost of each pass (by masking flags)
node links.mjs --n 600000                # cursor-link search variants (JS, WASM, SIMD, threads, grid)
node web/build.mjs                       # self-contained browser page -> bench/dist/bench.html
```

The fast loop is to edit a kernel, run `build.sh`, then run
`node run.mjs --impl <yours>,wasm-simd`. The reference trace and checksums are
cached, so iterations don't re-run the slow JS reference.

## Experiments
| name | what it is |
|---|---|
| `js-ref` | App structure (4 passes), exact f32 via `Math.fround`. **The spec.** |
| `js-fused` | Same maths, one sweep per particle, no temp arrays |
| `native-c` | C, `-O3 -march=native`, scalar fused (x86-64 reference point) |
| `wasm-scalar` | Same C → WASM `-O3` |
| `wasm-simd-v1` | + SIMD128 for ODE/home only |
| `wasm-simd` | + SoA state, vec4 warp grid, rows = vertex format, torn-particle compaction, vectorised simplex/curl |
| `wasm-simd-mt` | `wasm-simd` on shared memory across worker threads (Atomics handshake, no copies); Node workers here, Web Workers on the page (`kernels/wasm-mt-web.js`, needs COOP/COEP) |
| `webgpu` | WGSL compute, one invocation per particle (browser page only; tolerance-checked) |

## Results
Desktop Chrome 153, 32-thread CPU, NVIDIA Ada Lovelace GPU, bench page served with COOP/COEP.
CPU rows are wall time per frame (best of 3); WebGPU is GPU timestamp time over a batch
(best of several runs; GPU clocks vary about 2x between runs).

150k particles, the full 300-frame session (every CPU row bit-exact at every checkpoint):

| impl | ms/frame |
|---|---|
| `wasm-scalar` | 8.0 |
| `wasm-simd` | 3.3 |
| `wasm-simd-mt` (32 threads) | 0.51 |
| `webgpu` | 0.033 |

Scale (120 frames: idle, then the first tear):

| particles | wasm-simd | wasm-simd-mt (32) | webgpu |
|---|---|---|---|
| 150k | 2.16 | 0.42 (5.1x) | 0.013 |
| 1M | 14.4 | 1.61 (8.9x) | 0.044 |
| 4M | 58.0 | 6.72 (8.6x) | 0.42 |

Threads stop scaling near 9x: from 1M up the step is bound by memory bandwidth. Measure in a
plain browser window: with DevTools attached, Chrome keeps WebAssembly on its baseline
compiler and every WASM row roughly doubles.

## Memory discipline
- **Nothing allocates per frame in any timed kernel.** The JS kernels copy the few
  uniform vectors they need into preallocated scratch.
- **The WASM backends own their state in linear memory.** Per frame, JS writes only
  the 384-byte uniform block. `state()` returns views that alias wasm memory, so a
  renderer can upload straight from them.
- **WebGPU keeps everything in storage buffers.** `P` is `vec4` xyzw, so it can be
  bound as the vertex buffer directly. Readback happens only at checkpoints.
