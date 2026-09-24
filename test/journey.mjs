import pw from 'playwright';
const b = await pw.chromium.launch({ headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
for (const [w, h, name] of [[1280, 900, 'journey'], [400, 860, 'journey-phone']]) {
  const p = await b.newPage({ viewport: { width: w, height: h } });
  const errs = []; p.on('pageerror', (e) => errs.push(e.message));
  await p.goto('file://' + process.cwd() + '/dist/pages/js-webgl.html#journey');
  await p.waitForTimeout(2500);
  await p.screenshot({ path: `shots/${name}.png`, fullPage: false });
  const bars = await p.evaluate(() => document.querySelectorAll('.jrow').length);
  const sh = await p.evaluate(() => document.querySelector('.jcard').scrollHeight);
  console.log(name, 'rows', bars, 'height', sh, errs);
  await p.evaluate(() => { document.querySelector('.journey').scrollTop = 1000; });
  await p.waitForTimeout(300);
  await p.screenshot({ path: `shots/${name}-2.png` });
  await p.close();
}
await b.close();
