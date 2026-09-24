// Shared, bit-exact building blocks for every JS kernel + the uniform layout.
// Every arithmetic result is rounded with fr() = Math.fround, i.e. exact binary32.

export const fr = Math.fround;

// ---- uniform block layout (indices into Float32Array U / Int32Array UI) -------
export const UNI = {
  KIND: 0, FRAME: 1, FLAGS: 2,               // ints
  A: 4, B: 5, C: 6, D: 7, E: 8, FP: 9, K: 10, R: 11, SG: 12,
  H: 13, BOUND: 14, JIT: 15, INIT: 16,       // INIT..INIT+2
  SC: 19, CX: 20, CY: 21, CZ: 22, INV_SPEED: 23, INV_HOT: 24,
  ORIENT: 25,                                // 9 floats
  WARP_AMP: 34, WOX: 35, WOY: 36, WOZ: 37,
  TX: 39, TY: 40, TR2: 41, WVX: 42, WVY: 43, WVZ: 44,
  BX: 45, BY: 46, BR2: 47, RIGHT: 48, UP: 51, FWD: 54,
  SW: 57, SH: 58, VP: 59,                    // 16 floats
  FREQ: 75, AMP: 76, SOX: 77, SOY: 78, SOZ: 79, RELAX: 80, BACK: 81, BACK_MORPH: 82, DT: 83,
  SIZE: 96,
};
export const FLAG_WARP = 1, FLAG_TEAR = 2, FLAG_BURST = 4;
export const KIND = { Aizawa: 0, Thomas: 1, Halvorsen: 2, Lorenz: 3 };

// Warp grid constants (compile-time constants in C / WGSL too)
export const WG = 12;
export const W_LO = fr(-2.2);
export const W_CELL = fr(fr(4.4) / fr(11));
export const W_INV = fr(fr(1) / W_CELL);
export const W_NF = fr(0.9);
export const W_GMAX = fr(fr(12) - fr(1.001));
export const MORPH_EASE = fr(1.4);

// ---- counter-based RNG ----------------------------------------------------------
export function hash(i, frame, salt) {
  let x = (Math.imul(i, 0x9E3779B1) ^ Math.imul(frame, 0x85EBCA77) ^ Math.imul(salt, 0xC2B2AE3D)) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15; x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}
const INV24 = fr(1 / 16777216);
export function rnd(i, frame, salt) { return fr((hash(i, frame, salt) >>> 8) * INV24); }

// ---- sin in f32 ops (Cody–Waite + odd polynomial) --------------------------------
const INV_PI = fr(1 / Math.PI), PI_A = fr(3.140625), PI_B = fr(Math.PI - 3.140625);
const S1 = fr(-1 / 6), S2 = fr(1 / 120), S3 = fr(-1 / 5040), S4 = fr(1 / 362880), S5 = fr(-1 / 39916800);
export function fsin(x) {
  const q = Math.floor(fr(fr(x * INV_PI) + fr(0.5)));
  const r = fr(fr(x - fr(q * PI_A)) - fr(q * PI_B));
  const r2 = fr(r * r);
  let p = fr(S4 + fr(r2 * S5));
  p = fr(S3 + fr(r2 * p));
  p = fr(S2 + fr(r2 * p));
  p = fr(S1 + fr(r2 * p));
  p = fr(r + fr(fr(r * r2) * p));
  return (q & 1) ? -p : p;
}

// ---- simplex noise + curl, exact f32 ------------------------------------------
export const PERM = new Uint8Array(512);
export const GRAD = new Float32Array([
  1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1, 0,
  1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, -1,
  0, 1, 1, 0, -1, 1, 0, 1, -1, 0, -1, -1,
  1, 1, 0, 0, -1, 1, -1, 1, 0, 0, -1, -1,
]);
(function seed(seed) {
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  let s = seed >>> 0;
  for (let i = 255; i > 0; i--) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const j = s % (i + 1);
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
})(1337);

const F3 = fr(1 / 3), G3 = fr(1 / 6), G3x2 = fr(2 * G3), G3x3 = fr(3 * G3);

// Adds one simplex corner's contribution; returns [n, dx, dy, dz] via acc (Float64Array(4) holding f32 values)
function corner(x, y, z, gi, acc) {
  const tt = fr(fr(fr(fr(0.5) - fr(x * x)) - fr(y * y)) - fr(z * z));
  if (tt > 0) {
    const gx = GRAD[gi], gy = GRAD[gi + 1], gz = GRAD[gi + 2];
    const t2 = fr(tt * tt), t4 = fr(t2 * t2);
    const gd = fr(fr(fr(gx * x) + fr(gy * y)) + fr(gz * z));
    acc[0] = fr(acc[0] + fr(t4 * gd));
    const m = fr(fr(fr(-8 * t2) * tt) * gd);
    acc[1] = fr(acc[1] + fr(fr(m * x) + fr(t4 * gx)));
    acc[2] = fr(acc[2] + fr(fr(m * y) + fr(t4 * gy)));
    acc[3] = fr(acc[3] + fr(fr(m * z) + fr(t4 * gz)));
  }
}

const _acc = new Float64Array(4);
// Writes gradient (f32) into out[o..o+2]
export function snoiseGrad(x, y, z, out, o) {
  const s = fr(fr(fr(x + y) + z) * F3);
  const i = Math.floor(fr(x + s)), j = Math.floor(fr(y + s)), k = Math.floor(fr(z + s));
  const t = fr(fr(fr(i + j) + k) * G3);
  const x0 = fr(x - fr(i - t)), y0 = fr(y - fr(j - t)), z0 = fr(z - fr(k - t));
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
  const x1 = fr(fr(x0 - i1) + G3), y1 = fr(fr(y0 - j1) + G3), z1 = fr(fr(z0 - k1) + G3);
  const x2 = fr(fr(x0 - i2) + G3x2), y2 = fr(fr(y0 - j2) + G3x2), z2 = fr(fr(z0 - k2) + G3x2);
  const x3 = fr(fr(x0 - 1) + G3x3), y3 = fr(fr(y0 - 1) + G3x3), z3 = fr(fr(z0 - 1) + G3x3);
  const ii = i & 255, jj = j & 255, kk = k & 255, P = PERM;
  _acc[0] = 0; _acc[1] = 0; _acc[2] = 0; _acc[3] = 0;
  corner(x0, y0, z0, (P[ii + P[jj + P[kk]]] & 15) * 3, _acc);
  corner(x1, y1, z1, (P[ii + i1 + P[jj + j1 + P[kk + k1]]] & 15) * 3, _acc);
  corner(x2, y2, z2, (P[ii + i2 + P[jj + j2 + P[kk + k2]]] & 15) * 3, _acc);
  corner(x3, y3, z3, (P[ii + 1 + P[jj + 1 + P[kk + 1]]] & 15) * 3, _acc);
  out[o] = fr(32 * _acc[1]); out[o + 1] = fr(32 * _acc[2]); out[o + 2] = fr(32 * _acc[3]);
}

const OFF1 = [fr(31.416), fr(17.23), fr(5.71)], OFF2 = [fr(12.87), fr(47.31), fr(23.9)];
const _g = new Float64Array(9);
export function curl(x, y, z, out) {
  snoiseGrad(x, y, z, _g, 0);
  snoiseGrad(fr(x + OFF1[0]), fr(y - OFF1[1]), fr(z + OFF1[2]), _g, 3);
  snoiseGrad(fr(x - OFF2[0]), fr(y + OFF2[1]), fr(z - OFF2[2]), _g, 6);
  out[0] = fr(_g[7] - _g[5]);
  out[1] = fr(_g[2] - _g[6]);
  out[2] = fr(_g[3] - _g[1]);
}

// Build the 12³ warp grid for this frame (1728 curl evaluations).
const _c = new Float64Array(3);
export function buildWarpGrid(W, ox, oy, oz) {
  for (let gz = 0, q = 0; gz < WG; gz++) {
    const pz = fr(fr(fr(W_LO + fr(gz * W_CELL)) * W_NF) + oz);
    for (let gy = 0; gy < WG; gy++) {
      const py = fr(fr(fr(W_LO + fr(gy * W_CELL)) * W_NF) + oy);
      for (let gx = 0; gx < WG; gx++, q += 3) {
        const px = fr(fr(fr(W_LO + fr(gx * W_CELL)) * W_NF) + ox);
        curl(px, py, pz, _c);
        W[q] = _c[0]; W[q + 1] = _c[1]; W[q + 2] = _c[2];
      }
    }
  }
}

// ---- checksum ---------------------------------------------------------------------
export function fnv(views) {
  let h = 0x811c9dc5 >>> 0;
  for (const v of views) {
    const u = new Uint32Array(v.buffer, v.byteOffset, v.length);
    for (let i = 0; i < u.length; i++) { h ^= u[i]; h = Math.imul(h, 0x01000193); }
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}
