# Unravel kernel — benchmark spec

The benchmark runs the Unravel particle step (everything except rendering and the
cursor-links pass) for a fixed, scripted session and checks that every experiment
produces **bit-identical** state. Experiments are judged only on speed.

## Determinism rules (every CPU implementation must follow them)

1. **Arithmetic is IEEE-754 binary32.** Every `+ - * /` and `sqrt` result is
   rounded to f32. JS emulates this with `Math.fround` after every operation
   (exact, because f64 has more than 2·24+2 bits). C uses `float` with
   `-ffp-contract=off` (no FMA) and no fast-math.
2. **Operation order is fixed** and written out left to right in the kernels
   exactly as in `js-ref.js`. Reassociating (e.g. summing in a different order)
   changes bits and is not allowed.
3. **No platform transcendentals** in the step. `sin` is `fsin()`: Cody–Waite
   range reduction + odd polynomial, all in f32 ops. `exp` only appears in
   per-frame constants computed by the host.
4. **Randomness is a counter hash** `hash(i, frame, salt)` (lowbias32), turned
   into `[0,1)` as `(h >>> 8) * 2^-24`, exact in f32. It needs no state, so it
   works the same across threads and on the GPU.
5. **Per-particle independence.** A particle reads only its own state plus the
   read-only frame uniforms and the warp grid. An escaped particle respawns near
   the attractor's seed point (it used to copy a random other particle, which
   was a data race once threads were involved).
6. **Reductions happen on the host** from the sample array (every 64th particle,
   in index order), so thread count and chunking can't change them.

## Record once, replay everywhere

`host.js` owns the scenario (camera, tear strokes, burst, turbulence, pad bend) and
the auto-fit state (centre / scale / mean speed). The reference run (`js-ref`)
records the per-frame uniform block into a **trace**. Every experiment replays
that same trace, so it doesn't need host logic or samples, and a native binary
can replay it from a file.

## Checks

- FNV-1a over the raw bits of `S, P, V, F, T` at checkpoint frames (1, 10, 100, last).
- CPU experiments must match the reference checksum at **every** checkpoint.
- WebGPU cannot be bit-exact by specification (WGSL allows ≤2.5 ULP division and
  sqrt, and implementations may fuse multiply-adds), so it is checked with a
  tolerance at frames 1 and 10. It is not compared at 300: the attractor is chaotic,
  so 1-ULP differences grow until the particle positions decorrelate.

## Uniform block layout (Float32Array(96), ints via an Int32Array view)
See `UNI` in `core.js`.
