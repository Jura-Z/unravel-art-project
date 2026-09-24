// Pad sweep for every attractor (node test/attractor_sweep.cjs) with collapse detection: final ensemble spread relative to the default's.
const vm = require('vm'); const fs = require('fs');
vm.runInThisContext(fs.readFileSync('src/attractors.js', 'utf8').replace('const ATTRACTORS', 'globalThis.ATTRACTORS'));
function run(sys, u, v) {
  const A = ATTRACTORS[sys]; const p = { ...A.def };
  const kind = { Aizawa: 0, Thomas: 1, Halvorsen: 2, Lorenz: 3 }[A.name];
  const [k0, lo0, hi0] = A.pad[0], [k1, lo1, hi1] = A.pad[1];
  p[k0] = lo0 + u * (hi0 - lo0); p[k1] = lo1 + v * (hi1 - lo1);
  const M = 300, S = [];
  for (let m = 0; m < M; m++) S.push([A.init[0] + (Math.random() - .5) * A.spread * 2, A.init[1] + (Math.random() - .5) * A.spread * 2, A.init[2] + (Math.random() - .5) * A.spread * 2]);
  let esc = 0;
  for (let s = 0; s < 3000; s++) for (const q of S) {
    let [x, y, z] = q, dx, dy, dz;
    if (kind === 0) { const zb = z - p.b; dx = zb * x - p.d * y; dy = p.d * x + zb * y; dz = p.c + p.a * z - z * z * z / 3 - (x * x + y * y) * (1 + p.e * z) + p.f * z * x * x * x; }
    else if (kind === 1) { dx = Math.sin(p.k * y) - p.b * x; dy = Math.sin(p.k * z) - p.b * y; dz = Math.sin(p.k * x) - p.b * z; }
    else if (kind === 2) { dx = -p.a * x - p.k * y - p.k * z - y * y; dy = -p.a * y - p.k * z - p.k * x - z * z; dz = -p.a * z - p.k * x - p.k * y - x * x; }
    else { dx = p.s * (y - x); dy = x * (p.r - z) - y; dz = x * y - p.b * z; }
    x += dx * A.h; y += dy * A.h; z += dz * A.h;
    if (!(Math.abs(x) < A.spread * 12 && Math.abs(y) < A.spread * 12 && Math.abs(z) < A.spread * 12)) { esc++; x = A.init[0]; y = A.init[1]; z = A.init[2]; }
    q[0] = x; q[1] = y; q[2] = z;
  }
  const mean = [0, 1, 2].map((i) => S.reduce((a, q) => a + q[i], 0) / M);
  const rms = Math.sqrt(S.reduce((a, q) => a + (q[0] - mean[0]) ** 2 + (q[1] - mean[1]) ** 2 + (q[2] - mean[2]) ** 2, 0) / M);
  // "line-ness": fraction of particles within 2% of rms from the ensemble's principal axis is hard; use distinct-cell count instead
  const cells = new Set(S.map((q) => q.map((c, i) => Math.floor((c - mean[i]) / (rms * 0.08))).join(',')));
  return { rms, cells: cells.size, esc };
}
for (let sys = 0; sys < 4; sys++) {
  const A = ATTRACTORS[sys]; console.log(`\n== ${A.name}  pad x=${A.pad[0][0]} [${A.pad[0][1]}, ${A.pad[0][2]}]  y=${A.pad[1][0]} [${A.pad[1][1]}, ${A.pad[1][2]}]   (cells of 300; low = collapsed)`);
  const rows = [];
  for (let v = 1; v >= -0.001; v -= 0.25) {
    let line = `y=${v.toFixed(2)} `;
    for (let u = 0; u <= 1.001; u += 0.25) { const r = run(sys, u, v); line += `${String(r.cells).padStart(4)}${r.esc ? '!' : ' '}`; }
    rows.push(line);
  }
  console.log(rows.join('\n'));
}
