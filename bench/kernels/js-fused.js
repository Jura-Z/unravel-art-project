// JS experiment: same exact-f32 maths as js-ref, but one sweep per particle
// (ODE -> warp -> tear -> smoke) instead of four passes, and no H/B temp arrays.
// Tests how much of js-ref's cost is memory traffic vs. arithmetic.
import { fr, UNI as u, FLAG_WARP, FLAG_TEAR, FLAG_BURST, WG, W_LO, W_INV, W_GMAX, MORPH_EASE,
  rnd, fsin, curl, buildWarpGrid } from '../core.js';

export class JsFused {
  constructor(n) {
    this.name = 'js-fused';
    this.n = n;
    this.S = new Float32Array(n * 3); this.P = new Float32Array(n * 4); this.V = new Float32Array(n * 3);
    this.F = new Float32Array(n); this.T = new Float32Array(n); this.heal = new Float32Array(n);
    this.SS = new Float32Array(Math.ceil(n / 64) * 5);
    this.W = new Float32Array(WG * WG * WG * 3);
    this.h = new Float64Array(3); this.c = new Float64Array(3);
    this.o9 = new Float64Array(9); this.vp16 = new Float64Array(16); this.r3 = new Float64Array(3); this.u3 = new Float64Array(3); this.f3 = new Float64Array(3);
  }
  load(init) { for (const k of ['S', 'P', 'V', 'F', 'T', 'heal']) this[k].set(init[k]); }
  state() { return { S: this.S, P: this.P, V: this.V, F: this.F, T: this.T }; }
  samples() { return this.SS; }

  step(U, UI) {
    const flags = UI[u.FLAGS], warp = flags & FLAG_WARP, tearing = flags & (FLAG_TEAR | FLAG_BURST);
    if (warp) buildWarpGrid(this.W, U[u.WOX], U[u.WOY], U[u.WOZ]);
    const n = this.n, S = this.S, P = this.P, V = this.V, F = this.F, T = this.T, SS = this.SS, W = this.W, h = this.h, c = this.c, heal = this.heal;
    const kind = UI[u.KIND], frame = UI[u.FRAME];
    const a = U[u.A], b = U[u.B], cc = U[u.C], d = U[u.D], e = U[u.E], fp = U[u.FP], k = U[u.K], r = U[u.R], sg = U[u.SG];
    const hh = U[u.H], bound = U[u.BOUND], nb = -bound, jit = U[u.JIT];
    const i0 = U[u.INIT], i1 = U[u.INIT + 1], i2 = U[u.INIT + 2];
    const sc = U[u.SC], cx = U[u.CX], cy = U[u.CY], cz = U[u.CZ], invSpeed = U[u.INV_SPEED], invHot = U[u.INV_HOT];
    const o0 = U[u.ORIENT], o1 = U[u.ORIENT + 1], o2 = U[u.ORIENT + 2], o3 = U[u.ORIENT + 3], o4 = U[u.ORIENT + 4], o5 = U[u.ORIENT + 5], o6 = U[u.ORIENT + 6], o7 = U[u.ORIENT + 7], o8 = U[u.ORIENT + 8];
    const amp = U[u.WARP_AMP], G = WG, G2 = WG * WG;
    const vp = this.vp16, Wd = U[u.SW], Hh = U[u.SH];
    for (let q = 0; q < 16; q++) vp[q] = U[u.VP + q];
    const tearOn = flags & FLAG_TEAR, burstOn = flags & FLAG_BURST;
    const tx = U[u.TX], ty = U[u.TY], tr2 = U[u.TR2], wvx = U[u.WVX], wvy = U[u.WVY], wvz = U[u.WVZ];
    const bx0 = U[u.BX], by0 = U[u.BY], br2 = U[u.BR2];
    const R = this.r3, Up = this.u3, Fw = this.f3;
    for (let q = 0; q < 3; q++) { R[q] = U[u.RIGHT + q]; Up[q] = U[u.UP + q]; Fw[q] = U[u.FWD + q]; }
    const freq = U[u.FREQ], samp = U[u.AMP], sox = U[u.SOX], soy = U[u.SOY], soz = U[u.SOZ];
    const relax = U[u.RELAX], back = U[u.BACK], backMorph = U[u.BACK_MORPH], dt = U[u.DT];
    const half = fr(0.5), one = fr(1), third = fr(3), negEase = -MORPH_EASE, m036 = fr(-0.36), m035 = fr(-0.35);
    let torn = 0;
    for (let i = 0; i < n; i++) {
      const q = i * 3, j = i * 4;
      // ---- ODE + home ----
      let x = S[q], y = S[q + 1], z = S[q + 2], dx, dy, dz;
      if (kind === 2) {
        dx = fr(fr(fr(fr(-a * x) - fr(k * y)) - fr(k * z)) - fr(y * y));
        dy = fr(fr(fr(fr(-a * y) - fr(k * z)) - fr(k * x)) - fr(z * z));
        dz = fr(fr(fr(fr(-a * z) - fr(k * x)) - fr(k * y)) - fr(x * x));
      } else if (kind === 1) {
        dx = fr(fsin(fr(k * y)) - fr(b * x)); dy = fr(fsin(fr(k * z)) - fr(b * y)); dz = fr(fsin(fr(k * x)) - fr(b * z));
      } else if (kind === 0) {
        const zb = fr(z - b);
        dx = fr(fr(zb * x) - fr(d * y)); dy = fr(fr(d * x) + fr(zb * y));
        dz = fr(fr(fr(fr(cc + fr(a * z)) - fr(fr(fr(z * z) * z) / third)) - fr(fr(fr(x * x) + fr(y * y)) * fr(1 + fr(e * z)))) + fr(fr(fr(fr(fp * z) * x) * x) * x));
      } else {
        dx = fr(sg * fr(y - x)); dy = fr(fr(x * fr(r - z)) - y); dz = fr(fr(x * y) - fr(b * z));
      }
      x = fr(x + fr(dx * hh)); y = fr(y + fr(dy * hh)); z = fr(z + fr(dz * hh));
      if (!(x > nb && x < bound && y > nb && y < bound && z > nb && z < bound)) {
        x = fr(i0 + fr(fr(rnd(i, frame, 1) - half) * jit)); y = fr(i1 + fr(fr(rnd(i, frame, 2) - half) * jit)); z = fr(i2 + fr(fr(rnd(i, frame, 3) - half) * jit));
      }
      S[q] = x; S[q + 1] = y; S[q + 2] = z;
      const speed = fr(Math.sqrt(fr(fr(fr(dx * dx) + fr(dy * dy)) + fr(dz * dz))));
      const rx = fr(fr(x - cx) * sc), ry = fr(fr(y - cy) * sc), rz = fr(fr(z - cz) * sc);
      let hx = fr(fr(fr(o0 * rx) + fr(o1 * ry)) + fr(o2 * rz));
      let hy = fr(fr(fr(o3 * rx) + fr(o4 * ry)) + fr(o5 * rz));
      let hz = fr(fr(fr(o6 * rx) + fr(o7 * ry)) + fr(o8 * rz));
      let f = F[i];
      T[i] = fr(fr(speed * invSpeed) + (f > 0 ? fr(f * fr(0.3)) : 0));
      const hs = fr(fr(speed * invHot) - fr(1.05));
      const bb = hs > 0 ? fr(fr(0.55) + Math.min(fr(5.5), fr(fr(fr(fr(3.2) * hs) * hs) + fr(fr(1.2) * hs)))) : fr(0.55);
      if ((i & 63) === 0) { const s = (i >> 6) * 5; SS[s] = x; SS[s + 1] = y; SS[s + 2] = z; SS[s + 3] = speed; SS[s + 4] = hy; }
      // ---- warp ----
      if (warp) {
        let fx = fr(fr(hx - W_LO) * W_INV), fy = fr(fr(hy - W_LO) * W_INV), fz = fr(fr(hz - W_LO) * W_INV);
        fx = fx < 0 ? 0 : fx > W_GMAX ? W_GMAX : fx; fy = fy < 0 ? 0 : fy > W_GMAX ? W_GMAX : fy; fz = fz < 0 ? 0 : fz > W_GMAX ? W_GMAX : fz;
        const ix = fx | 0, iy = fy | 0, iz = fz | 0;
        const ax = fr(fx - ix), ay = fr(fy - iy), az = fr(fz - iz), bx = fr(one - ax), by = fr(one - ay), bz = fr(one - az);
        const w0 = fr(fr(bx * by) * bz), w1 = fr(fr(ax * by) * bz), w2 = fr(fr(bx * ay) * bz), w3 = fr(fr(ax * ay) * bz);
        const w4 = fr(fr(bx * by) * az), w5 = fr(fr(ax * by) * az), w6 = fr(fr(bx * ay) * az), w7 = fr(fr(ax * ay) * az);
        const b0 = (iz * G2 + iy * G + ix) * 3, b1 = b0 + 3, b2 = b0 + G * 3, b3 = b2 + 3, b4 = b0 + G2 * 3, b5 = b4 + 3, b6 = b4 + G * 3, b7 = b6 + 3;
        h[0] = hx; h[1] = hy; h[2] = hz;
        for (let dd = 0; dd < 3; dd++) {
          const s = fr(fr(fr(fr(fr(fr(fr(fr(W[b0 + dd] * w0) + fr(W[b1 + dd] * w1)) + fr(W[b2 + dd] * w2)) + fr(W[b3 + dd] * w3))
            + fr(W[b4 + dd] * w4)) + fr(W[b5 + dd] * w5)) + fr(W[b6 + dd] * w6)) + fr(W[b7 + dd] * w7));
          h[dd] = fr(h[dd] + fr(amp * s));
        }
        hx = h[0]; hy = h[1]; hz = h[2];
      }
      // ---- tear / burst (old position) ----
      if (tearing) {
        const px = P[j], py = P[j + 1], pz = P[j + 2];
        const w = fr(fr(fr(fr(vp[3] * px) + fr(vp[7] * py)) + fr(vp[11] * pz)) + vp[15]);
        if (w > fr(0.05)) {
          const cxp = fr(fr(fr(fr(vp[0] * px) + fr(vp[4] * py)) + fr(vp[8] * pz)) + vp[12]);
          const cyp = fr(fr(fr(fr(vp[1] * px) + fr(vp[5] * py)) + fr(vp[9] * pz)) + vp[13]);
          const qx = fr(fr(fr(fr(cxp / w) * half) + half) * Wd), qy = fr(fr(half - fr(fr(cyp / w) * half)) * Hh);
          if (tearOn) {
            const ex = fr(qx - tx), ey = fr(qy - ty), d2 = fr(fr(ex * ex) + fr(ey * ey));
            if (d2 < tr2) {
              let ff = fr(one - fr(d2 / tr2)); ff = fr(ff * ff);
              if (f <= 0) { V[q] = 0; V[q + 1] = 0; V[q + 2] = 0; }
              const pull = fr(ff * fr(0.35)), jt = fr(fr(0.06) * ff);
              V[q] = fr(fr(V[q] + fr(fr(fr(wvx * fr(1.2)) - V[q]) * pull)) + fr(fr(rnd(i, frame, 11) - half) * jt));
              V[q + 1] = fr(fr(V[q + 1] + fr(fr(fr(wvy * fr(1.2)) - V[q + 1]) * pull)) + fr(fr(rnd(i, frame, 12) - half) * jt));
              V[q + 2] = fr(fr(V[q + 2] + fr(fr(fr(wvz * fr(1.2)) - V[q + 2]) * pull)) + fr(fr(rnd(i, frame, 13) - half) * jt));
              if (ff > f) f = ff;
            }
          }
          if (burstOn) {
            const ex = fr(qx - bx0), ey = fr(qy - by0), d2 = fr(fr(ex * ex) + fr(ey * ey));
            if (d2 < br2) {
              const t = fr(one - fr(d2 / br2)), ff = fr(t * fr(Math.sqrt(t)));
              const l = fr(fr(Math.sqrt(d2)) + fr(1e-3)), ux = fr(ex / l), uy = fr(-ey / l), s = fr(fr(1.6) * ff);
              const zr = fr(fr(rnd(i, frame, 14) - half) * fr(0.8));
              if (f <= 0) { V[q] = 0; V[q + 1] = 0; V[q + 2] = 0; }
              V[q] = fr(V[q] + fr(fr(fr(fr(R[0] * ux) + fr(Up[0] * uy)) + fr(Fw[0] * zr)) * s));
              V[q + 1] = fr(V[q + 1] + fr(fr(fr(fr(R[1] * ux) + fr(Up[1] * uy)) + fr(Fw[1] * zr)) * s));
              V[q + 2] = fr(V[q + 2] + fr(fr(fr(fr(R[2] * ux) + fr(Up[2] * uy)) + fr(Fw[2] * zr)) * s));
              if (ff > f) f = ff;
            }
          }
        }
      }
      // ---- smoke / return / attach ----
      if (f === 0) { P[j] = hx; P[j + 1] = hy; P[j + 2] = hz; P[j + 3] = bb; F[i] = 0; continue; }
      let px = P[j], py = P[j + 1], pz = P[j + 2];
      if (f < 0) {
        if (f < negEase) { F[i] = fr(f + dt); P[j + 3] = bb; continue; }
        const rr = f < m036 ? backMorph : back;
        px = fr(px + fr(fr(hx - px) * rr)); py = fr(py + fr(fr(hy - py) * rr)); pz = fr(pz + fr(fr(hz - pz) * rr));
        f = fr(f + dt); if (f > 0) f = 0;
        P[j] = px; P[j + 1] = py; P[j + 2] = pz; P[j + 3] = bb; F[i] = f;
        continue;
      }
      torn++;
      curl(fr(fr(px * freq) + sox), fr(fr(py * freq) + soy), fr(fr(pz * freq) + soz), c);
      const g = fr(one - f), pull = fr(fr(fr(fr(7) * g) * g) * g);
      const fa = fr(samp * fr(fr(0.35) + fr(fr(0.65) * f)));
      let vx = V[q], vy = V[q + 1], vz = V[q + 2];
      vx = fr(vx + fr(fr(fr(fr(c[0] * fa) + fr(fr(hx - px) * pull)) - vx) * relax));
      vy = fr(vy + fr(fr(fr(fr(c[1] * fa) + fr(fr(hy - py) * pull)) - vy) * relax));
      vz = fr(vz + fr(fr(fr(fr(c[2] * fa) + fr(fr(hz - pz) * pull)) - vz) * relax));
      V[q] = vx; V[q + 1] = vy; V[q + 2] = vz;
      P[j] = fr(px + fr(vx * dt)); P[j + 1] = fr(py + fr(vy * dt)); P[j + 2] = fr(pz + fr(vz * dt));
      P[j + 3] = fr(fr(0.7) + fr(fr(fr(4.5) * f) * f));
      f = fr(f - fr(heal[i] * dt));
      F[i] = f > 0 ? f : m035;
    }
    this.torn = torn;
  }
}
