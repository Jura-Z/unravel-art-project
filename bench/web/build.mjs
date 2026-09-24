// Builds the self-contained bench page: trace + reference checksums + wasm inlined.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fnv } from '../core.js';
const here = dirname(fileURLToPath(import.meta.url)), root = join(here, '..');
const n = 150000, frames = 300, shape = 'Halvorsen', tag = `${shape}-${n}-${frames}`;
const cache = join(root, '.cache');
const ref = JSON.parse(readFileSync(join(cache, `ref-${tag}.json`), 'utf8'));
const results = JSON.parse(readFileSync(join(cache, `results-${tag}.json`), 'utf8'));
const ib = new Float32Array(readFileSync(join(cache, `init-${tag}.bin`)).buffer.slice(0));
const initSum = fnv([ib.subarray(0, n * 3), ib.subarray(n * 12, n * 13)]);   // S and heal
const b64 = (p) => readFileSync(p).toString('base64');
const data = {
  meta: { n, frames, shape, checks: ref.checks, jsRefMs: ref.jsRefMs, initSum },
  trace: b64(join(cache, `trace-${tag}.bin`)),
  wasm: { scalar: b64(join(root, 'build', 'scalar.wasm')), simd: b64(join(root, 'build', 'simd.wasm')) },
  node: results.rows.map((r) => ({ name: r.name, mean: r.mean, exact: r.exact })),
};
writeFileSync(join(here, 'bundle-data.js'), `export const BUNDLE = ${JSON.stringify(data)};\n`);
const js = execFileSync('esbuild', [join(here, 'page.js'), '--bundle', '--format=iife', '--minify-syntax', '--loader:.wgsl=text', '--target=es2022'], { encoding: 'utf8', maxBuffer: 64 << 20 });
const shell = readFileSync(join(here, 'shell.html'), 'utf8');
const nodeRows = `<script>
(function(){const rows=${JSON.stringify(data.node)};const ref=${JSON.stringify(ref.jsRefMs || null)};
const tb=document.getElementById('node-rows');for(const r of rows){const tr=document.createElement('tr');
tr.innerHTML='<th scope="row">'+r.name+'</th><td class="num">'+r.mean.toFixed(2)+'</td><td class="num">'+(ref?(ref/r.mean).toFixed(1)+'×':'')+'</td><td>'+(r.exact?'<span class="good">bit-exact</span>':'<span class="bad">mismatch</span>')+'</td>';tb.appendChild(tr);}})();
</script>`;
const body = `${shell}\n<script>\n${js}\n</script>\n${nodeRows}\n`;
mkdirSync(join(root, 'dist'), { recursive: true });
writeFileSync(join(root, 'dist', 'bench.html'), body);
writeFileSync(join(root, 'dist', 'bench-standalone.html'), `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>${body}</body></html>`);
console.log('bench page', (body.length / 1024).toFixed(0), 'KB');
