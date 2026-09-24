// Browser bench page: replays the recorded trace through each backend in this browser,
// checks bit-exactness against the reference checksums recorded in Node, times it, and
// saves every run to the artifact's database so results can be read back and compared.
import { UNI, fnv } from '../core.js';
import { makeHost } from '../host.js';
import { JsRef } from '../kernels/js-ref.js';
import { makeWasm } from '../kernels/wasm-core.js';
import { makeWasmMTWeb } from '../kernels/wasm-mt-web.js';
import { makeWebGPU } from '../kernels/webgpu/webgpu.js';
import WGSL from '../kernels/webgpu/kernel.wgsl';
import { BUNDLE } from './bundle-data.js';   // generated: trace, reference checksums, wasm bytes

const $ = (id) => document.getElementById(id);
const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const { n, frames, shape, checks: REF, initSum: INIT_SUM } = BUNDLE.meta;
const CHECK = [0, 9, 99, frames - 1];
const trace = new Float32Array(b64(BUNDLE.trace).buffer);
const wasmBytes = { scalar: b64(BUNDLE.wasm.scalar), simd: b64(BUNDLE.wasm.simd), simdMt: b64(BUNDLE.wasm.simdMt) };
const THREADS = Math.min(navigator.hardwareConcurrency || 1, 32);
const SCALE_N = [150000, 1000000, 4000000];
const SCALE_FRAMES = 120;
const SCALE_IDS = ['wasm-simd', 'wasm-simd-mt', 'webgpu'];   // frames 0-119: idle, then the first tear stroke

const BACKENDS = [
  { id: 'js-ref', label: 'JS reference (exact f32 via Math.fround)', make: async (m) => new JsRef(m), slow: true },
  { id: 'wasm-scalar', label: 'C → WASM -O3, scalar', make: (m) => makeWasm(m, 'scalar', wasmBytes.scalar) },
  { id: 'wasm-simd', label: 'C → WASM -O3, SIMD128 + compaction', make: (m) => makeWasm(m, 'simd', wasmBytes.simd) },
  { id: 'wasm-simd-mt', label: `WASM SIMD128, ${THREADS} threads on shared memory`, make: (m) => makeWasmMTWeb(m, wasmBytes.simdMt, THREADS), mt: true },
  { id: 'webgpu', label: 'WebGPU compute (WGSL)', make: (m) => makeWebGPU(m, WGSL), gpu: true },
];

const inits = new Map();
function log(msg) { $('status').textContent = msg; }
const tick = () => new Promise((r) => setTimeout(r, 0));
const frameU = (U, f) => { for (let q = 0, b = f * UNI.SIZE; q < UNI.SIZE; q++) U[q] = trace[b + q]; };

async function initFor(m) {
  if (inits.has(m)) return inits.get(m);
  log(`Building the seeded initial state for ${m.toLocaleString()} particles…`); await tick();
  const init = makeHost({ n: m, shape }).initState();
  if (m === n) {
    const sum = fnv([init.S, init.heal]);
    if (sum !== INIT_SUM) throw new Error(`initial state differs from the recorded one (${sum} vs ${INIT_SUM})`);
  }
  inits.set(m, init);
  return init;
}

async function replayCPU(k, init, runs, nFrames, withChecks) {
  const buf = new ArrayBuffer(UNI.SIZE * 4), U = new Float32Array(buf), UI = new Int32Array(buf);
  let best = null;
  for (let r = 0; r < runs; r++) {
    k.load(init);
    const times = new Float64Array(nFrames), checks = {};
    for (let f = 0; f < nFrames; f++) {
      frameU(U, f);
      const t0 = performance.now(); k.step(U, UI); times[f] = performance.now() - t0;
      if (withChecks && CHECK.includes(f)) checks[f] = fnv(Object.values(k.state()));
      if (f % 20 === 19) { log(`${k.name} · ${init.heal.length.toLocaleString()} particles · run ${r + 1}/${runs} · frame ${f + 1}/${nFrames}`); await tick(); }
    }
    const sorted = Array.from(times).sort((a, b) => a - b);
    const res = { mean: times.reduce((a, b) => a + b, 0) / nFrames, median: sorted[nFrames >> 1], checks };
    if (!best || res.mean < best.mean) best = res;
  }
  return best;
}

// Exact reference states at frames 1 and 10 (from the bit-exact SIMD kernel).
async function refStates(m, init) {
  const k = await makeWasm(m, 'simd', wasmBytes.simd);
  k.load(init);
  const buf = new ArrayBuffer(UNI.SIZE * 4), U = new Float32Array(buf), UI = new Int32Array(buf);
  const out = {};
  for (let f = 0; f < 10; f++) {
    frameU(U, f); k.step(U, UI);
    if (f === 0 || f === 9) out[f] = Object.fromEntries(Object.entries(k.state()).map(([key, v]) => [key, v.slice()]));
  }
  return out;
}

function compare(a, b) {
  let exact = 0, total = 0, maxAbs = 0;
  for (const key of ['S', 'P', 'V', 'F', 'T']) {
    const x = a[key], y = b[key];
    for (let i = 0; i < x.length; i++) { total++; if (Object.is(x[i], y[i])) exact++; else maxAbs = Math.max(maxAbs, Math.abs(x[i] - y[i])); }
  }
  return { exactPct: +(exact / total * 100).toFixed(3), maxAbs };
}

async function runGPU(k, init, runs, nFrames, withAccuracy) {
  const buf = new ArrayBuffer(UNI.SIZE * 4), U = new Float32Array(buf), UI = new Int32Array(buf);
  const m = init.heal.length;
  const res = { gpuName: k.gpuName };
  if (withAccuracy) {
    log('WebGPU: comparing frames 1 and 10 with the exact reference…'); await tick();
    const ref = await refStates(m, init);
    k.load(init); res.acc = {};
    for (let f = 0; f < 10; f++) { frameU(U, f); k.step(U, UI); if (f === 0 || f === 9) res.acc[f] = compare(await k.state(), ref[f]); }
  }
  k.uploadTrace(trace, nFrames);
  // (a) realistic: one submission per frame, wall time until the queue drains
  let wall = Infinity;
  for (let r = 0; r < runs; r++) {
    k.load(init); await k.finish();
    log(`WebGPU · ${m.toLocaleString()} particles · per-frame submits, run ${r + 1}/${runs}`); await tick();
    const t0 = performance.now();
    for (let f = 0; f < nFrames; f++) k.step(null, null, false, f);
    await k.finish();
    wall = Math.min(wall, (performance.now() - t0) / nFrames);
  }
  // (b) GPU throughput: all frames in one pass, bracketed by timestamps
  let gpu = null;
  if (k.gpuTimeMs) {
    const g = [];
    for (let r = 0; r < Math.max(2, runs); r++) {
      k.load(init); await k.finish();
      k.runBatch(0, nFrames);
      const t = await k.gpuTimeMs();
      if (t !== null && t > 0) g.push(t / nFrames);
    }
    if (g.length) gpu = Math.min(...g);
  }
  return { ...res, mean: wall, gpuMs: gpu };
}

function makeRow(tb, id, label) {
  const tr = document.createElement('tr');
  tr.id = id;
  tr.innerHTML = `<th scope="row">${label}</th><td class="num" data-k="ms">–</td><td class="num" data-k="x">–</td><td data-k="ok">–</td><td class="bar"><span></span></td>`;
  tb.appendChild(tr);
  return tr;
}

async function saveRun(record) {
  try {
    const db = await window.claude?.use?.('db');
    if (!db) return;                     // only claude.ai artifacts have a results log; elsewhere, say nothing
    await db.collection('runs').add(record);
    $('saved').textContent = `Saved to this page's results log (${new Date(record.at).toLocaleTimeString()}).`;
  } catch (e) {
    $('saved').textContent = 'Results were not saved: ' + (e.code || e.message);
  }
}

async function runAll() {
  $('run').disabled = true;
  const runs = +$('runs').value;
  const record = { at: new Date().toISOString(), ua: navigator.userAgent, cores: navigator.hardwareConcurrency || null,
    n, frames, shape, results: {}, scale: [] };
  try {
    const init = await initFor(n);
    // ---- main table: correctness + time at the recorded size ----
    for (const b of BACKENDS) {
      if (!$('use-' + b.id).checked) continue;
      const tr = $('row-' + b.id);
      tr.querySelector('[data-k=ok]').textContent = 'running…';
      let r;
      try {
        const k = await b.make(n);
        r = b.gpu ? await runGPU(k, init, runs, frames, true) : await replayCPU(k, init, b.slow ? 1 : runs, frames, true);
        if (k.close) k.close();
      } catch (e) {
        tr.querySelector('[data-k=ok]').innerHTML = `<span class="bad">${e.message}</span>`;
        record.results[b.id] = { error: e.message };
        continue;
      }
      const out = { mean: +r.mean.toFixed(4) };
      if (b.gpu) {
        Object.assign(out, { gpuMs: r.gpuMs, gpuName: r.gpuName, acc: r.acc });
        record.gpuName = r.gpuName;
        const a1 = r.acc[0], a10 = r.acc[9];
        tr.querySelector('[data-k=ms]').textContent = r.mean.toFixed(3) + (r.gpuMs != null ? ` · GPU ${r.gpuMs.toFixed(3)}` : '');
        tr.querySelector('[data-k=ok]').innerHTML =
          `<span class="${a1.exactPct === 100 ? 'good' : 'warn'}">f1 ${a1.exactPct}% bit-equal (max |Δ| ${a1.maxAbs.toExponential(1)}) · f10 ${a10.exactPct}% (max |Δ| ${a10.maxAbs.toExponential(1)})</span>`;
      } else {
        out.exact = CHECK.every((f) => r.checks[f] === REF[f]);
        out.checks = r.checks;
        tr.querySelector('[data-k=ms]').textContent = r.mean.toFixed(2);
        const bad = CHECK.find((f) => r.checks[f] !== REF[f]);
        tr.querySelector('[data-k=ok]').innerHTML = out.exact ? '<span class="good">bit-exact at every checkpoint</span>' : `<span class="bad">differs from frame ${bad + 1}</span>`;
      }
      record.results[b.id] = out;
      bars('rows', record.results, (id) => 'row-' + id);
    }
    // ---- scale: WASM SIMD vs WebGPU at growing particle counts ----
    if ($('use-scale').checked) {
      for (const m of SCALE_N) {
        const initM = await initFor(m);
        const entry = { n: m, frames: SCALE_FRAMES };
        for (const id of SCALE_IDS) {
          const b = BACKENDS.find((x) => x.id === id);
          if (!$('use-' + id).checked) continue;
          const tr = $(`scale-${id}-${m}`);
          tr.querySelector('[data-k=ok]').textContent = 'running…';
          try {
            const k = await b.make(m);
            const r = b.gpu ? await runGPU(k, initM, 1, SCALE_FRAMES, m <= 1000000) : await replayCPU(k, initM, 1, SCALE_FRAMES, false);
            if (k.close) k.close();
            entry[id] = { mean: +r.mean.toFixed(4), gpuMs: r.gpuMs ?? null, acc: r.acc ?? null };
            tr.querySelector('[data-k=ms]').textContent = r.mean.toFixed(3) + (r.gpuMs != null ? ` · GPU ${r.gpuMs.toFixed(3)}` : '');
            tr.querySelector('[data-k=ok]').innerHTML = r.acc ? `<span class="warn">f1 ${r.acc[0].exactPct}% bit-equal</span>` : '<span class="dim">timing only</span>';
          } catch (e) {
            entry[id] = { error: e.message };
            tr.querySelector('[data-k=ok]').innerHTML = `<span class="bad">${e.message}</span>`;
          }
        }
        // Compare the GPU with the best CPU that ran: every thread when they're available, else one.
        const w1 = entry['wasm-simd']?.mean, wN = entry['wasm-simd-mt']?.mean, g = entry.webgpu?.gpuMs ?? entry.webgpu?.mean;
        if (w1 && wN) $(`scale-wasm-simd-mt-${m}`).querySelector('[data-k=x]').textContent = `${(w1 / wN).toFixed(1)}× vs 1 thread`;
        const w = wN || w1;
        if (w && g) $(`scale-webgpu-${m}`).querySelector('[data-k=x]').textContent = `${(w / g).toFixed(0)}× vs WASM ${wN ? `${THREADS} threads` : '1 thread'}`;
        record.scale.push(entry);
      }
    }
    log('Done. CPU rows: wall time per frame. WebGPU: wall time with one submit per frame · GPU = timestamp time per frame over a batched run.');
    await saveRun(record);
  } catch (e) {
    log('Error: ' + e.message);
    record.error = e.message;
    await saveRun(record);
  }
  $('run').disabled = false;
}

function bars(tbodyId, results, rowId) {
  const entries = Object.entries(results).filter(([, r]) => r && r.mean);
  const ref = results['js-ref']?.mean || results['wasm-scalar']?.mean;
  const label = results['js-ref']?.mean ? 'vs JS ref' : 'vs WASM scalar';
  const max = Math.max(...entries.map(([, r]) => r.mean));
  for (const [id, r] of entries) {
    const tr = $(rowId(id)); if (!tr) continue;
    tr.querySelector('.bar span').style.width = (r.mean / max * 100).toFixed(1) + '%';
    const t = r.gpuMs || r.mean;
    tr.querySelector('[data-k=x]').textContent = ref ? `${(ref / t).toFixed(1)}× ${label}` : '';
  }
}

function boot() {
  $('meta').textContent = `${shape} · ${n.toLocaleString()} particles · ${frames} frames · scripted session (tear, burst, turbulence swell, second tear)`;
  const tb = $('rows');
  for (const b of BACKENDS) makeRow(tb, 'row-' + b.id, b.label);
  const st = $('scale-rows');
  for (const m of SCALE_N) for (const id of SCALE_IDS) makeRow(st, `scale-${id}-${m}`, `${id} · ${m >= 1e6 ? m / 1e6 + 'M' : m / 1e3 + 'k'}`);
  const opts = $('opts');
  for (const b of BACKENDS) {
    const l = document.createElement('label');
    const avail = b.gpu ? !!navigator.gpu : b.mt ? crossOriginIsolated : true;
    l.innerHTML = `<input type="checkbox" id="use-${b.id}" ${!b.slow && avail ? 'checked' : ''} ${avail ? '' : 'disabled'}> ${b.id}${b.slow ? ' <small>(slow, ~25 s)</small>' : ''}${avail ? '' : b.mt ? ' <small>(page not cross-origin isolated)</small>' : ' <small>(no WebGPU here)</small>'}`;
    opts.appendChild(l);
  }
  const l = document.createElement('label');
  // The 4M run holds several copies of the particle state; phones and small-memory devices can lose the tab.
  const small = matchMedia('(pointer: coarse)').matches || (navigator.deviceMemory && navigator.deviceMemory < 8);
  l.innerHTML = `<input type="checkbox" id="use-scale" ${small ? '' : 'checked'}> scale test <small>(150k / 1M / 4M, ~1 min${small ? '; off on phones: the 4M run needs a lot of memory' : ''})</small>`;
  opts.appendChild(l);
  $('mt').textContent = crossOriginIsolated
    ? `This page is cross-origin isolated: WASM runs on ${THREADS} threads here.`
    : 'WASM threads need a cross-origin-isolated page (COOP/COEP headers), which this host does not send. Thread results come from the Node runner (table below).';
  if (!window.claude?.use) $('saved').textContent = '';   // no results log outside claude.ai artifacts
  $('run').addEventListener('click', runAll);
}
boot();
