import pw from 'playwright';
import { launch, root } from './browser.mjs';
const { chromium } = pw;
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errs = []; page.on('pageerror', (e) => errs.push(e.message));
await page.addInitScript(() => { window.requestAnimationFrame = (cb) => { window.__tick = cb; return 1; }; });
await page.goto(root + '/dist/pages/wasm-webgl.html');   // the WebGL path; auto would pick WebGPU on a real GPU
await page.waitForFunction(() => window.__engine);
const adv = (secs) => page.evaluate((secs) => {
  const E = window.__engine, dt = 1 / 60; let now = E.lastFrame;
  for (let i = 0; i < Math.round(secs * 60); i++) { now += 1000 / 60; window.__tick(now); }
  return document.getElementById('readout').textContent;
}, secs);
await page.evaluate(() => { const E = window.__engine; E.demoActive = false; E.lastUserInput = 1e9; });
await adv(1.5);
const t0 = Date.now();
await page.evaluate(() => { const t = performance.now(); window.__engine.mode.morphTo(1); window.__morphMs = performance.now() - t; });
console.log('morphTo cost ms', await page.evaluate(() => window.__morphMs.toFixed(1)));
for (const [s, name] of [[0.9, 'm1'], [1.3, 'm2'], [2.5, 'm3']]) { console.log(name, await adv(s)); await page.screenshot({ path: `shots/${name}.png` }); }
console.log('errors', errs);
await browser.close();
