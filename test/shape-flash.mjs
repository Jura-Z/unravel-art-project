// Shape change: JS cost of morphTo at 2.4M (WebGPU, offscreen), the flash frame, and the settled new shape.
import pw from 'playwright';
import { launch, root } from './browser.mjs';
import { writeFileSync } from 'node:fs';
const b = await launch();
const p = await b.newPage({ viewport: { width: 800, height: 500 } });
p.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text().slice(0, 300)); });
p.on('pageerror', (e) => console.log('pageerror', e.message));
await p.addInitScript(() => { window.__GPU_OFFSCREEN = true; window.requestAnimationFrame = (cb) => { window.__tick = cb; return 1; }; });
await p.goto(root + '/dist/pages/webgpu.html');
await p.waitForFunction(() => window.__engine, null, { timeout: 60000, polling: 200 });
const run = (n) => p.evaluate(async (n) => {
  const E = window.__engine; let now = E.lastFrame || performance.now();
  for (let i = 0; i < n; i++) { now += 1000 / 60; window.__tick(now); await E.renderer.device.queue.onSubmittedWorkDone(); await new Promise((r) => setTimeout(r, 0)); }
}, n);
const shot = async (name) => {
  const r = await p.evaluate(async () => {
    const px = await window.__engine.renderer.readPixels();
    const c = document.createElement('canvas'); c.width = px.w; c.height = px.h;
    c.getContext('2d').putImageData(new ImageData(px.data, px.w, px.h), 0, 0);
    let sum = 0; for (let i = 0; i < px.data.length; i += 4) sum += px.data[i] + px.data[i + 1] + px.data[i + 2];
    return { url: c.toDataURL('image/png'), mean: sum / (px.data.length / 4) / 3 };
  });
  writeFileSync(`shots/${name}.png`, Buffer.from(r.url.split(',')[1], 'base64'));
  return r.mean.toFixed(1);
};
// 2.4M: JS cost of the shape change
await p.evaluate(() => { const E = window.__engine; E.auto = null; E.demoActive = false; E.lastUserInput = 1e12; E.mode.countIndex = 6; E.mode.init(E, E.mode.counts[6]); });
const cost = await p.evaluate(() => { const m = window.__engine.mode, t = performance.now(); m.morphTo(1); const a = performance.now() - t; const t2 = performance.now(); m.morphTo(2); return [a, performance.now() - t2]; });
console.log('2.4M morphTo ms (first, cached orbit):', cost.map((x) => x.toFixed(1)).join(', '));
// visual at 100k
await p.evaluate(() => { const E = window.__engine; E.mode.countIndex = 0; E.mode.init(E, E.mode.counts[0]); E.renderer.clear(); });
await run(60);
console.log('before', await shot('flash-0'));
await p.evaluate(() => window.__engine.mode.morphTo(0));
await run(4);
console.log('flash', await shot('flash-1'), 'flash value', await p.evaluate(() => window.__engine.mode.flash.toFixed(2)));
await run(150);
console.log('settled', await shot('flash-2'), await p.evaluate(() => window.__engine.mode.flash));
await b.close();
