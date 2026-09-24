import pw from 'playwright';
const b = await pw.chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
for (const page of ['wasm-webgl', 'js-webgl']) {
  const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
  p.on('pageerror', (e) => console.log('pageerror', e.message));
  await p.addInitScript(() => { window.requestAnimationFrame = (cb) => { window.__tick = cb; return 1; }; });
  await p.goto('file://' + process.cwd() + `/dist/pages/${page}.html`);
  await p.waitForFunction(() => window.__engine && window.__tick, null, { timeout: 60000, polling: 200 });
  const r = await p.evaluate(() => {
    const E = window.__engine, m = E.mode; E.auto = null; E.demoActive = false; E.lastUserInput = 1e12;
    let now = E.lastFrame; const adv = (k) => { for (let i = 0; i < k; i++) { now += 1000 / 60; window.__tick(now); } };
    adv(30);
    const out = [];
    for (const [u, v] of [[0.5, 0.5], [0.45, 0.4], [0.6, 0.55], [0.05, 0.05]]) {
      E.dispatch('hover', E.makePointer(u * E.cssW, v * E.cssH)); adv(1);
      const got = Array.from(m.linkIdx.slice(0, m.lines.count)).sort((a, b) => a - b).join();
      const cnt = m.lines.count;
      // reference: full scan on the same P (glow writes only touch P.w, not xyz)
      const ref = m.linksScan(E.camera.vp, E.cssW, E.cssH, m.hover.x, m.hover.y, 150 * 150);
      out.push({ cnt, ref, same: got === Array.from(m.linkIdx.slice(0, ref)).sort((a, b) => a - b).join(), ms: m.ms.links.toFixed(3) });
    }
    return { out, readout: document.getElementById('readout').textContent };
  });
  console.log(page, JSON.stringify(r));
  await p.close();
}
await b.close();
