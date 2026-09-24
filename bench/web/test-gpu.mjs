import pw from 'playwright';
const { chromium } = pw;
const b = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}), 
  args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
const p = await b.newPage();
const errs = []; p.on('pageerror', (e) => errs.push(e.message)); p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
await p.goto('file://' + process.cwd() + '/dist/bench-standalone.html');
await p.evaluate(() => { for (const id of ['js-ref', 'wasm-scalar']) document.getElementById('use-' + id).checked = false; document.getElementById('runs').value = '1'; document.getElementById('use-scale').checked = false; });
await p.click('#run');
await p.waitForFunction(() => !document.getElementById('run').disabled && /Done|Error/.test(document.getElementById('status').textContent), null, { timeout: 600000 });
const rows = await p.evaluate(() => [...document.querySelectorAll('#rows tr, #saved')].map((tr) => tr.innerText.replace(/\s+/g, ' ')));
console.log(rows.join('\n')); console.log('status:', await p.textContent('#status')); console.log('errors:', errs);
await p.screenshot({ path: 'dist/bench-shot.png', fullPage: true });
await b.close();
