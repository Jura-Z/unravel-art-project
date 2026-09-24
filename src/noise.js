// 3D simplex noise with analytic gradient (after Gustavson; kernel radius^2 = 0.5 so the field is continuous).
// Writes the gradient into `out` at offset `o` and returns the value.
// Allocation-free so it can run in a hot loop over typed arrays.

const NOISE_PERM = new Uint8Array(512);
const NOISE_GRAD = new Float32Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
  1, 1, 0, 0, -1, 1, -1, 1, 0, 0, -1, -1,
]);

(function seedNoise(seed) {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  let s = seed >>> 0;
  for (let i = 255; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  for (let i = 0; i < 512; i++) NOISE_PERM[i] = p[i & 255];
})(1337);

const F3 = 1 / 3, G3 = 1 / 6;

function snoise3d(x, y, z, out, o) {
  const s = (x + y + z) * F3;
  const i = Math.floor(x + s), j = Math.floor(y + s), k = Math.floor(z + s);
  const t = (i + j + k) * G3;
  const x0 = x - (i - t), y0 = y - (j - t), z0 = z - (k - t);

  let i1, j1, k1, i2, j2, k2;
  if (x0 >= y0) {
    if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
    else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
  } else {
    if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
    else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
    else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
  }

  const x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
  const x2 = x0 - i2 + 2 * G3, y2 = y0 - j2 + 2 * G3, z2 = z0 - k2 + 2 * G3;
  const x3 = x0 - 1 + 3 * G3, y3 = y0 - 1 + 3 * G3, z3 = z0 - 1 + 3 * G3;

  const ii = i & 255, jj = j & 255, kk = k & 255;
  const P = NOISE_PERM, G = NOISE_GRAD;

  let n = 0, dx = 0, dy = 0, dz = 0;
  let tt, t2, t4, gi, gx, gy, gz, gd;

  tt = 0.5 - x0 * x0 - y0 * y0 - z0 * z0;
  if (tt > 0) {
    gi = (P[ii + P[jj + P[kk]]] & 15) * 3; gx = G[gi]; gy = G[gi + 1]; gz = G[gi + 2];
    t2 = tt * tt; t4 = t2 * t2; gd = gx * x0 + gy * y0 + gz * z0;
    n += t4 * gd;
    const m = -8 * t2 * tt * gd;
    dx += m * x0 + t4 * gx; dy += m * y0 + t4 * gy; dz += m * z0 + t4 * gz;
  }
  tt = 0.5 - x1 * x1 - y1 * y1 - z1 * z1;
  if (tt > 0) {
    gi = (P[ii + i1 + P[jj + j1 + P[kk + k1]]] & 15) * 3; gx = G[gi]; gy = G[gi + 1]; gz = G[gi + 2];
    t2 = tt * tt; t4 = t2 * t2; gd = gx * x1 + gy * y1 + gz * z1;
    n += t4 * gd;
    const m = -8 * t2 * tt * gd;
    dx += m * x1 + t4 * gx; dy += m * y1 + t4 * gy; dz += m * z1 + t4 * gz;
  }
  tt = 0.5 - x2 * x2 - y2 * y2 - z2 * z2;
  if (tt > 0) {
    gi = (P[ii + i2 + P[jj + j2 + P[kk + k2]]] & 15) * 3; gx = G[gi]; gy = G[gi + 1]; gz = G[gi + 2];
    t2 = tt * tt; t4 = t2 * t2; gd = gx * x2 + gy * y2 + gz * z2;
    n += t4 * gd;
    const m = -8 * t2 * tt * gd;
    dx += m * x2 + t4 * gx; dy += m * y2 + t4 * gy; dz += m * z2 + t4 * gz;
  }
  tt = 0.5 - x3 * x3 - y3 * y3 - z3 * z3;
  if (tt > 0) {
    gi = (P[ii + 1 + P[jj + 1 + P[kk + 1]]] & 15) * 3; gx = G[gi]; gy = G[gi + 1]; gz = G[gi + 2];
    t2 = tt * tt; t4 = t2 * t2; gd = gx * x3 + gy * y3 + gz * z3;
    n += t4 * gd;
    const m = -8 * t2 * tt * gd;
    dx += m * x3 + t4 * gx; dy += m * y3 + t4 * gy; dz += m * z3 + t4 * gz;
  }

  out[o] = 32 * dx; out[o + 1] = 32 * dy; out[o + 2] = 32 * dz;
  return 32 * n;
}

// Divergence-free velocity: curl of a vector potential built from three
// decorrelated noise fields. Writes (vx, vy, vz) into out[0..2].
const _cg = new Float64Array(9);
function curlNoise(x, y, z, out) {
  snoise3d(x, y, z, _cg, 0);
  snoise3d(x + 31.416, y - 17.23, z + 5.71, _cg, 3);
  snoise3d(x - 12.87, y + 47.31, z - 23.9, _cg, 6);
  // psi = (n1, n2, n3); curl = (d n3/dy - d n2/dz, d n1/dz - d n3/dx, d n2/dx - d n1/dy)
  out[0] = _cg[7] - _cg[5];
  out[1] = _cg[2] - _cg[6];
  out[2] = _cg[3] - _cg[1];
}
