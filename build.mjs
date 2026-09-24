// Inline all sources into self-contained pages (no third-party scripts; fonts come from Google Fonts).
//   dist/index.html            artifact body (auto backend: WebGPU -> WASM+WebGL -> JS+WebGL)
//   dist/pages/<backend>.html  standalone pages, one per backend (debug entry points)
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
const files = ['noise.js', 'camera.js', 'uni.js', 'attractors.js', 'engine.js', 'sim/wasm-sim.js', 'gpu/backend.js',
  'modes/unravel.js', 'journey.js', 'main.js'];
const wasm = readFileSync('bench/build/simd.wasm').toString('base64');
const wgsl = readFileSync('bench/kernels/webgpu/kernel.wgsl', 'utf8');
const src = files.map((f) => `// ---- ${f}\n` + readFileSync(`src/${f}`, 'utf8')).join('\n');
const shell = readFileSync('src/shell.html', 'utf8');
const page = (forced) => `${shell}\n<script>\n(() => {\n'use strict';\nconst FORCED_BACKEND = ${JSON.stringify(forced)};\nconst WASM_SIMD_B64 = ${JSON.stringify(wasm)};\nconst KERNEL_WGSL = ${JSON.stringify(wgsl)};\n${src}\n})();\n</script>\n`;
const wrap = (body) => `<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n${body}</body>\n</html>\n`;
mkdirSync('dist/pages', { recursive: true });
const auto = page('auto');
writeFileSync('dist/index.html', auto);
writeFileSync('dist/standalone.html', wrap(auto));
for (const b of ['webgpu', 'wasm-webgl', 'js-webgl']) writeFileSync(`dist/pages/${b}.html`, wrap(page(b)));
// dist/selftest.html: the app with test/selftest-probe.js injected. Open it in a real browser
// to test every backend on a real GPU; it cycles the backends itself (not part of the app).
const probe = readFileSync('test/selftest-probe.js', 'utf8');
writeFileSync('dist/selftest.html', wrap(auto).replace('<head>', `<head>\n<script>\n${probe}\n</script>`));
console.log('built', (auto.length / 1024).toFixed(1), 'KB');
