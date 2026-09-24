// Injected into dist/selftest.html (build.mjs): the app page plus this probe. It runs the
// checks inside the real page on the real GPU, once per backend (switching by hash + reload,
// results kept in sessionStorage), then shows a PASS/FAIL table. Not part of the app.
(() => {
  const ORDER = ['webgpu', 'wasm-webgl', 'js-webgl'], KEY = 'unravel-selftest';
  let state = null;
  try { state = JSON.parse(sessionStorage.getItem(KEY) || 'null'); } catch { state = null; }
  if (!state || state.done) state = { i: 0, rows: [], done: false };
  const EXPECT = ORDER[state.i];
  history.replaceState(null, '', '#' + EXPECT);          // before main.js reads it; no hashchange
  // The app's constants live inside its closure; the few needed here are repeated.
  const SHAPES = 4, LINK_RADIUS_PX = 150;
  const fmtCount = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1e3)}k`);
  const results = [], errors = [];
  window.addEventListener('error', (e) => errors.push(String(e.message)));
  window.addEventListener('unhandledrejection', (e) => errors.push(String(e.reason)));
  const origError = console.error.bind(console);
  console.error = (...a) => { errors.push(a.map(String).join(' ')); origError(...a); };

  // Count real frames and keep their intervals.
  const raf = window.requestAnimationFrame.bind(window);
  let ticks = 0, last = 0; const iv = [];
  window.requestAnimationFrame = (cb) => raf((t) => { if (last) iv.push(t - last); last = t; ticks++; cb(t); });
  const frames = (n) => new Promise((res) => { const end = ticks + n; const w = () => (ticks >= end ? res() : setTimeout(w, 5)); w(); });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const check = (name, ok, info = '') => results.push({ name, ok: !!ok, info: String(info) });
  const step = (name) => { const b = document.getElementById('selftest-status') || banner(); b.textContent = `self-test ${state.i + 1}/${ORDER.length} · ${EXPECT} · ${name} · ${rate()} fps`; };
  function banner() {
    const b = document.createElement('div'); b.id = 'selftest-status';
    b.style.cssText = 'position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:99;background:#000c;color:#fff;font:13px ui-monospace,monospace;padding:6px 12px;border-radius:0 0 8px 8px';
    document.body.appendChild(b); return b;
  }
  const rate = () => { const r = iv.slice(-30); return r.length ? (1000 / (r.reduce((a, b) => a + b, 0) / r.length)).toFixed(0) : '?'; };
  function finish() {
    for (const r of results) state.rows.push({ b: EXPECT, ...r });
    state.i++;
    state.done = state.i >= ORDER.length;
    try { sessionStorage.setItem(KEY, JSON.stringify(state)); } catch { state.done = true; }   // no storage: one backend only
    if (!state.done) { history.replaceState(null, '', '#' + ORDER[state.i]); location.reload(); return; }
    const fails = state.rows.filter((r) => !r.ok).length;
    const el = document.createElement('div');
    el.style.cssText = 'position:fixed;inset:0;z-index:100;overflow:auto;background:#0d0c14f2;color:#e8e6f0;font:13px/1.6 ui-monospace,monospace;padding:20px';
    el.innerHTML = `<h2 style="margin:0 0 10px;color:${fails ? '#ff7a8a' : '#7dffa8'}">Unravel self-test: ${fails ? fails + ' failed' : 'all passed'}</h2><table style="border-collapse:collapse"></table><p><button>Run again</button></p>`;
    const t = el.querySelector('table');
    for (const r of state.rows) {
      const tr = t.insertRow();
      for (const [v, c] of [[r.b, '#9a96b0'], [r.name, ''], [r.ok ? 'PASS' : 'FAIL', r.ok ? '#7dffa8' : '#ff7a8a'], [r.info, '#9a96b0']]) {
        const td = tr.insertCell(); td.textContent = v; td.style.cssText = `padding:2px 12px 2px 0;color:${c}`;
      }
    }
    el.querySelector('button').onclick = () => { try { sessionStorage.removeItem(KEY); } catch {} history.replaceState(null, '', '#webgpu'); location.reload(); };
    document.body.appendChild(el);
    const st = document.getElementById('selftest-status'); if (st) st.remove();
  }

  async function run() {
    step('waiting for engine');
    const tw = performance.now();
    while (!window.__engine) {
      if (performance.now() - tw > 20000) { check('backend starts', false, document.getElementById('fallback')?.textContent || 'no engine after 20 s'); return finish(); }
      await sleep(50);
    }
    const E = window.__engine, m = E.mode;
    await frames(20);
    const label = document.getElementById('backend').textContent;
    check('backend is the one requested', { webgpu: 'gpu', 'wasm-webgl': 'wasm', 'js-webgl': 'js' }[EXPECT] === m.backend, label);

    step('auto count');
    // Auto particle count: give it a few seconds, report what it picked.
    const t0 = performance.now();
    while (E.auto && E.auto.on && performance.now() - t0 < 12000) await sleep(100);
    check('auto particle count settles', !(E.auto && E.auto.on), `${fmtCount(m.count)} · ${E.stats.frame.toFixed(1)} ms/frame`);

    // Keep the demo out of the way from here on.
    E.stopDemo(); E.lastUserInput = 1e12;
    await frames(10);

    step('links');
    // Cursor links.
    m.linksOn = true;
    E.dispatch('hover', E.makePointer(E.cssW * 0.5, E.cssH * 0.5));
    await frames(8);
    if (m.backend === 'gpu') {
      check('cursor links (GPU pick)', m.sim.linkCount > 0, `${m.sim.linkCount} in radius`);
    } else {
      const got = Array.from(m.linkIdx.slice(0, m.lines.count)).sort((a, b) => a - b).join();
      const n = m.linksScan(E.camera.vp, E.cssW, E.cssH, m.hover.x, m.hover.y, LINK_RADIUS_PX * LINK_RADIUS_PX);
      const ref = Array.from(m.linkIdx.slice(0, n)).sort((a, b) => a - b).join();
      check('cursor links = full scan', m.lines.count > 0 && got === ref, `${m.lines.count} links · ${m.ms.links.toFixed(2)} ms`);
    }
    E.dispatch('leave', null);

    step('shape change');
    // Shape change: JS cost and the worst frame during the morph, at the largest count.
    const want = m.backend === 'gpu' ? m.counts.length - 1 : m.countIndex;
    if (want !== m.countIndex) { m.countIndex = want; m.init(E, m.counts[want]); E.syncCountButtons(); await frames(30); }
    const next = (m.sys + 1) % SHAPES;
    iv.length = 0;
    const tm = performance.now(); m.morphTo(next); const morphMs = performance.now() - tm;
    let flash = 0;
    for (let k = 0; k < 20; k++) { await frames(1); flash = Math.max(flash, m.flash); }   // peak (the attack is 60 ms)
    await frames(40);
    const worst = Math.max(...iv);
    check(`shape change at ${fmtCount(m.count)}`, morphMs < 60 && worst < 100, `morphTo ${morphMs.toFixed(1)} ms · worst frame ${worst.toFixed(1)} ms`);
    check('shape change flashes white', flash > 0.5, `flash ${flash.toFixed(2)}`);

    step('input');
    // Input mapping.
    const c = E.canvas, d0 = E.camera.dist, tb = m.turb;
    c.dispatchEvent(new WheelEvent('wheel', { deltaY: 200, cancelable: true }));
    const zoom = E.camera.dist > d0 && m.turb === tb, d1 = E.camera.dist;
    c.dispatchEvent(new WheelEvent('wheel', { deltaY: -200, shiftKey: true, cancelable: true }));
    check('wheel = zoom, shift+wheel = turbulence', zoom && m.turb > tb && E.camera.dist === d1);

    // Pad knob on the pad for every shape's default.
    const inb = [];
    for (let i = 0; i < SHAPES; i++) {
      m.morphTo(i); E.updatePad();
      const l = parseFloat(E.padKnob.style.left), t = parseFloat(E.padKnob.style.top);
      inb.push(l >= 0 && l <= 100 && t >= 0 && t <= 100);
    }
    check('pad knob stays on the pad', inb.every(Boolean), inb.map((b) => (b ? '✓' : '✗')).join(''));

    // Demo ghost: links appear with it and vanish with it (CPU backends expose the lines).
    if (m.backend !== 'gpu') {
      const real = m.demo;
      E.demoActive = true; E.lastUserInput = -1e12;
      m.demo = () => ({ u: 0.5, v: 0.5 }); await frames(4);
      const on = m.lines.count;
      m.demo = () => null; await frames(4);
      check('demo links vanish with the ghost', on > 0 && m.lines.count === 0 && !m.hover, `${on} → ${m.lines.count}`);
      m.demo = real; E.stopDemo(); E.lastUserInput = 1e12;
    }

    step('article');
    // The article: renders, and the instrument ignores keys under it.
    document.getElementById('journey-link').click();
    await frames(2);
    const j = document.getElementById('journey');
    const fz = E.frozen;
    window.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));
    check('"How it got fast" opens, keys blocked under it', j && !j.hidden && j.querySelectorAll('.jrow').length > 5 && j.querySelector('.jtable') && E.frozen === fz);
    j.querySelector('.jclose').click();

    check('no errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    finish();
  }
  const start = () => run().catch((e) => { check('selftest crashed', false, e && e.stack || e); finish(); });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
