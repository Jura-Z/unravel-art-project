// Renders the WebGPU path offscreen (headless can't present a WebGPU canvas) and saves a PNG.
import pw from 'playwright';
import { launch, root } from './browser.mjs';
const b = await launch();
const p = await b.newPage({ viewport: { width: 800, height: 500 } });
p.on('console', (m) => { if (m.type() === 'error') console.log('console:', m.text().slice(0, 300)); });
await p.addInitScript(() => { window.__GPU_OFFSCREEN = true; window.requestAnimationFrame = (cb) => { window.__tick = cb; return 1; }; });
await p.goto(root + '/dist/pages/webgpu.html');
await p.waitForFunction(() => window.__engine, null, { timeout: 60000, polling: 200 });
await p.evaluate(async () => { const E = window.__engine; E.mode.countIndex = 0; E.reset(); });
const steps = +(process.env.FRAMES || 150);
await p.evaluate(async (steps) => {
  const E = window.__engine; let now = performance.now();
  for (let i = 0; i < steps; i++) {
    now += 1000 / 60; window.__tick(now);
    await E.renderer.device.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 0));
  }
}, steps);
const r = await p.evaluate(async () => {
  const E = window.__engine, px = await E.renderer.readPixels();
  const c = document.createElement('canvas'); c.width = px.w; c.height = px.h;
  const g = c.getContext('2d'); g.putImageData(new ImageData(px.data, px.w, px.h), 0, 0);
  let sum = 0; for (let i = 0; i < px.data.length; i += 4) sum += px.data[i] + px.data[i + 1] + px.data[i + 2];
  const SS = E.mode.sim.samples(); const st = { meanSpeed: E.mode.meanSpeed, scale: E.mode.scale, center: E.mode.center, floorY: E.mode.floor.y, ss0: Array.from(SS.slice(0, 10)), camDist: E.camera.dist };
  return { st, url: c.toDataURL('image/png'), mean: sum / (px.data.length / 4) / 3, readout: document.getElementById('readout').textContent, hasSamples: E.mode.sim.hasSamples, frame: document.getElementById('framems').textContent };
});
console.log(JSON.stringify({ ...r, url: undefined }));
(await import('node:fs')).writeFileSync('shots/gpu-offscreen.png', Buffer.from(r.url.split(',')[1], 'base64'));
await b.close();
