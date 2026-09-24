import pw from 'playwright';
const { chromium } = pw;
const which = process.argv[2] ? process.argv[2].split(',') : ['webgpu', 'wasm-webgl', 'js-webgl'];
const b = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}), 
  args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
for (const name of which) {
  const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
  const errs = [];
  p.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  p.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.type() + ': ' + m.text().slice(0, 300)); });
  await p.goto('file://' + process.cwd() + `/dist/pages/${name}.html`);
  await p.waitForTimeout(+process.env.WAIT || 9000);
  const info = await p.evaluate(() => ({ backend: document.getElementById('backend')?.textContent, readout: document.getElementById('readout')?.textContent, frame: document.getElementById('framems')?.textContent, fallback: document.getElementById('fallback')?.hidden === false ? document.getElementById('fallback').textContent : null }));
  await p.screenshot({ path: `shots/be-${name}.png` });
  console.log(name, JSON.stringify(info)); if (errs.length) console.log('  ' + errs.filter((e) => !e.includes('ERR_TUNNEL')).join('\n  '));
  await p.close();
}
await b.close();
