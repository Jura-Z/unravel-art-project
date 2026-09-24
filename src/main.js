// Boot: pick a backend, falling back in order WebGPU -> WASM + WebGL2 -> JS + WebGL2.
// Debug entry points force one path: #webgpu, #wasm-webgl, #js-webgl (or a build that
// sets FORCED_BACKEND). Each attempt gets a fresh canvas, because a canvas that has
// handed out a WebGPU context can never give a WebGL one.
const BACKENDS = {
  'webgpu': { label: 'WebGPU compute · WebGPU render', sim: 'gpu', defaultCount: 5,
    async make(canvas) { return GpuBackend.create(canvas); } },
  'wasm-webgl': { label: () => WASM_MT_MODULE ? `WASM SIMD sim ×${WASM_THREADS} threads · WebGL2 render` : 'WASM SIMD sim · WebGL2 render',
    sim: 'wasm', defaultCount: () => (WASM_MT_MODULE ? 3 : 1),
    async make(canvas) { await compileWasmKernel(); return new Renderer(canvas); } },
  'js-webgl': { label: 'JS sim · WebGL2 render', sim: 'js', defaultCount: 1,
    async make(canvas) { return new Renderer(canvas); } },
};

const GPU_BAD_KEY = 'unravel-webgpu-bad';

function freshCanvas() {
  const old = document.getElementById('stage');
  const c = old.cloneNode(false);
  old.replaceWith(c);
  return c;
}

(async function boot() {
  const hash = (location.hash || '').slice(1);
  const want = BACKENDS[hash] ? hash : (typeof FORCED_BACKEND === 'string' && BACKENDS[FORCED_BACKEND] ? FORCED_BACKEND : 'auto');
  let gpuBad = null;
  try { gpuBad = localStorage.getItem(GPU_BAD_KEY); } catch (e) {}
  const auto = ['webgpu', 'wasm-webgl', 'js-webgl'].filter((n) => n !== 'webgpu' || gpuBad !== navigator.userAgent);
  const order = want === 'auto' ? auto : [want];
  const failures = [];
  for (const name of order) {
    const b = BACKENDS[name];
    try {
      const canvas = freshCanvas();
      const renderer = await b.make(canvas);
      const opt = (v) => (typeof v === 'function' ? v() : v);
      // iOS (iPadOS reports as a Mac with touch) starts at 600k on any backend; auto-count moves from there.
      const ios = /iP(hone|ad|od)/.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
      UnravelMode.countIndex = ios ? UnravelMode.counts.indexOf(600000) : opt(b.defaultCount);
      const engine = new Engine(UnravelMode, { canvas, renderer, simBackend: b.sim, label: opt(b.label) });
      engine.start();
      window.__engine = engine;
      markBackend(name, want, failures);
      // If the GPU goes away mid-session (driver reset, sleep), fall back instead of freezing.
      if (name === 'webgpu' && want === 'auto') {
        window.addEventListener('unravel-gpu-lost', () => { location.hash = '#wasm-webgl'; });
        // WebGPU that starts but renders nothing (seen on iOS): remember it for this browser
        // version and come back on WASM + WebGL2.
        const check = async (tries) => {
          const ok = await renderer.selfCheck();
          if (ok === false && tries > 0) return setTimeout(() => check(tries - 1), 700);
          if (ok !== false) return;
          console.warn('[unravel] WebGPU renders nothing here; falling back to WASM + WebGL2');
          try { localStorage.setItem(GPU_BAD_KEY, navigator.userAgent); } catch (e) {}
          location.reload();
        };
        setTimeout(() => check(2), 2500);
      }
      return;
    } catch (err) {
      console.warn(`[unravel] ${name} unavailable:`, err);
      failures.push(`${name}: ${err.message}`);
    }
  }
  const fb = document.getElementById('fallback');
  fb.textContent = 'This needs WebGPU or WebGL2. ' + failures.join(' · ');
  fb.hidden = false;
  document.querySelectorAll('.chrome').forEach((el) => { el.hidden = true; });
})();

// Debug switcher: separate entry points per backend (plain #hash links reload the page).
function markBackend(active, want, failures) {
  const box = document.getElementById('debug-backends');
  if (!box) return;
  box.querySelectorAll('a').forEach((a) => {
    const id = a.getAttribute('href').slice(1) || 'auto';
    if (id === active || (id === 'auto' && want === 'auto')) a.setAttribute('aria-current', 'page');
  });
  if (failures.length) box.title = 'Skipped: ' + failures.join(' · ');
  window.addEventListener('hashchange', () => { const h = location.hash.slice(1); if (!h || BACKENDS[h]) location.reload(); });
}
