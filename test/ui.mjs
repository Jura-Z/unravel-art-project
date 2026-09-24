// UI checks: wheel = zoom / shift+wheel = turbulence, pad knob stays on the pad for every
// shape, the demo's links vanish with its ghost cursor, and keys are ignored under the article.
import pw from 'playwright';
import { launch, root } from './browser.mjs';
const b = await launch();
const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
const errs = []; p.on('pageerror', (e) => errs.push(e.message));
await p.addInitScript(() => { window.__MAX_FRAMES = 1e9; window.requestAnimationFrame = (cb) => { window.__tick = cb; return 1; }; });
await p.goto(root + '/dist/pages/js-webgl.html');
await p.waitForFunction(() => window.__engine && window.__tick, null, { timeout: 60000, polling: 200 });
const r = await p.evaluate(() => {
  const E = window.__engine, m = E.mode, c = E.canvas; let now = E.lastFrame;
  const adv = (k) => { for (let i = 0; i < k; i++) { now += 1000 / 60; window.__tick(now); } };
  const out = {};
  adv(5);
  const d0 = E.camera.dist, t0 = m.turb;
  c.dispatchEvent(new WheelEvent('wheel', { deltaY: 200, cancelable: true }));
  out.wheelZooms = E.camera.dist > d0 && m.turb === t0;
  const d1 = E.camera.dist;
  c.dispatchEvent(new WheelEvent('wheel', { deltaY: -200, shiftKey: true, cancelable: true }));
  out.shiftWheelTurb = m.turb > t0 && E.camera.dist === d1;
  c.dispatchEvent(new WheelEvent('wheel', { deltaX: -200, shiftKey: true, cancelable: true }));   // shift+wheel as horizontal scroll
  out.shiftWheelX = m.turb > t0;
  // pad knob inside the pad for every shape's default
  out.padInBounds = [0, 1, 2, 3].map((i) => {
    m.sys = i; m.reset(true); E.updatePad();
    const k = E.padKnob.style, l = parseFloat(k.left), t = parseFloat(k.top);
    return l >= 0 && l <= 100 && t >= 0 && t <= 100;
  }).every(Boolean);
  // demo: ghost cursor hover creates links, hiding it removes them
  m.linksOn = true;
  E.demoActive = true; E.lastUserInput = -1e9;
  const realDemo = m.demo;
  m.demo = () => ({ u: 0.5, v: 0.5 });
  adv(2);
  out.demoLinks = m.lines.count;
  m.demo = () => null;
  adv(2);
  out.linksAfterHide = m.lines.count;
  m.demo = realDemo;
  // keys ignored while the article is open
  E.stopDemo();
  document.getElementById('journey-link').click();
  const fz = E.frozen;
  window.dispatchEvent(new KeyboardEvent('keydown', { key: ' ' }));
  out.keysBlockedUnderArticle = E.frozen === fz;
  return out;
});
console.log(JSON.stringify({ ...r, errs }));
const ok = r.wheelZooms && r.shiftWheelTurb && r.shiftWheelX && r.padInBounds && r.demoLinks > 0 && r.linksAfterHide === 0 && r.keysBlockedUnderArticle && !errs.length;
await b.close();
process.exit(ok ? 0 : 1);
