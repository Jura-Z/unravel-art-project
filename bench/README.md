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
node web/build.mjs                       # self-contained browser page -> dist/bench.html
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
| `wasm-simd-mt` | `wasm-simd` on shared memory across worker threads (Atomics handshake, no copies) |
| `webgpu` | WGSL compute, one invocation per particle (browser page only; tolerance-checked) |

## Memory discipline
- **Nothing allocates per frame in any timed kernel.** The JS kernels copy the few
  uniform vectors they need into preallocated scratch.
- **The WASM backends own their state in linear memory.** Per frame, JS writes only
  the 384-byte uniform block. `state()` returns views that alias wasm memory, so a
  renderer can upload straight from them.
- **WebGPU keeps everything in storage buffers.** `P` is `vec4` xyzw, so it can be
  bound as the vertex buffer directly. Readback happens only at checkpoints.
