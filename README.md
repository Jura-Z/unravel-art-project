# Unravel

An interactive particle instrument for the browser. Up to 2.4 million particles ride a
strange attractor (Halvorsen, Thomas, Lorenz, Aizawa). Drag across the shape to tear
threads off: they turn to curl-noise smoke and drift back home. Bend the attractor with
the pad, stir it with turbulence, and let it morph from shape to shape without a restart.

![Unravel](docs/screenshot.jpg)

It is also a performance study. The same particle step exists as plain JavaScript,
C compiled to WebAssembly SIMD, and a WebGPU compute shader. Every CPU version is
checked **bit for bit** against a reference before its speed counts. The page's
"How it got fast" article (`#journey`) tells the story. The numbers are in
[`bench/`](bench/README.md).

| path | simulation | rendering | used when |
|---|---|---|---|
| `#webgpu` | WGSL compute | WebGPU, straight from the particle buffer | the browser has WebGPU |
| `#wasm-webgl` | C → WASM SIMD128 (bit-exact) | WebGL2, uploads from WASM memory | WebGPU is missing |
| `#js-webgl` | plain JavaScript | WebGL2 | last resort |

With no hash, the page picks the first path that works. On start-up it also picks a
particle count from its own measured frame time (up to 120 fps). The − and + buttons
override it.

## Controls
- **hover**: lines to nearby particles
- **drag**: tear
- **click**: burst
- **scroll**: zoom
- **shift + scroll**: turbulence
- **right-drag**: orbit
- **space**: freeze
- **[ ]**: particle count
- **C**: next shape
- **H**: hide UI

## Build
Requires Node 22+. There are no runtime dependencies: every page is a single
self-contained HTML file. The one exception is the fonts, which load from Google Fonts.

```sh
sh bench/kernels/c/build.sh   # C → WASM (clang 18+ with the wasm32 target); outputs bench/build/*.wasm
node build.mjs                # → dist/standalone.html, dist/pages/{webgpu,wasm-webgl,js-webgl}.html, dist/selftest.html
```

`bench/build/simd.wasm` is committed, so `node build.mjs` works without clang.
`dist/index.html` is the same page without the `<html>` wrapper, used for hosting as a
claude.ai artifact.

## Tests
```sh
npm install                   # playwright, esbuild (dev only)
npm test -- --node            # no browser: kernels bit-exact, cursor search, noise, attractor pads
npm test                      # + headless browser checks (set CHROME_PATH to use a specific Chrome)
```

GPU behaviour and performance should be checked on a real GPU. Open
`dist/selftest.html` in a desktop browser. It runs every backend in turn on your GPU and
shows a PASS/FAIL table covering: backend selection, auto particle count, cursor links
(compared with a full scan), shape-change cost and the worst frame at the largest count,
input mapping, the pad, the demo cursor, and the article. Headless software GPUs are fine
for "does it render" checks, but their timings mean nothing.

Test hooks:
- `window.__engine`: the running engine
- `window.__MAX_FRAMES`: stop after N frames, no auto particle count
- `window.__GPU_OFFSCREEN`: render WebGPU offscreen, then read it back with `renderer.readPixels()`

## Layout
```
src/            the app (concatenated by build.mjs in a fixed order; shared globals)
  modes/unravel.js   host logic: input, morphs, auto-fit, demo, JS simulation, cursor links
  engine.js          WebGL2 renderer, frame loop, input, HUD, auto particle count
  gpu/backend.js     WebGPU simulation + renderer
  sim/wasm-sim.js    WASM SIMD simulation (the bench kernel)
  journey.js         the "How it got fast" article
bench/          deterministic benchmark, kernels (JS, C, WGSL), spec, browser bench page
test/           Node and Playwright checks, run.mjs, selftest-probe.js
```

## Credits
Built by Iurii Zakipnyi with Claude (Anthropic) as a pair programmer. See the article in
the page for how.

## License
MIT. See [LICENSE](LICENSE).
