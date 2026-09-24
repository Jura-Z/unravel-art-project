// Scenario host: the scripted session, camera, initial state and per-frame uniforms.
// Host math runs in f64 (like the app) and is packed into the f32 uniform block, so
// every kernel sees identical bits. The reference run records those blocks as a trace.
import { fr, UNI as u, FLAG_WARP, FLAG_TEAR, FLAG_BURST, KIND, rnd, fsin } from './core.js';

export const ATTRACTORS = {
  Halvorsen: { h: 0.0035, spread: 2.5, init: [-1.48, -1.51, 2.04], def: { a: 1.89, k: 4 }, pad: [['a', 1.7, 2.2], ['k', 3.6, 4.3]], orient: 'diag' },
  Thomas: { h: 0.05, spread: 3, init: [1, 0.5, 0], def: { b: 0.19, k: 1.0 }, pad: [['b', 0.12, 0.2], ['k', 0.95, 1.3]], orient: 'diag' },
  Lorenz: { h: 0.003, spread: 10, init: [1, 1, 20], def: { s: 10, r: 28, b: 8 / 3 }, pad: [['r', 24, 45], ['s', 7, 14]], orient: [1, 0, 0, 0, 0, 1, 0, 1, 0] },
  Aizawa: { h: 0.009, spread: 1.5, init: [0.1, 0, 0.3], def: { a: 0.95, b: 0.7, c: 0.6, d: 3.5, e: 0.25, f: 0.1 }, pad: [['d', 2.0, 4.2], ['a', 0.65, 0.93]], orient: [1, 0, 0, 0, 0, 1, 0, 1, 0] },
};
const DIAG = (() => {
  const s3 = 1 / Math.sqrt(3), s2 = 1 / Math.sqrt(2);
  const up = [s3, s3, s3], r = [s2, -s2, 0];
  const f = [up[1] * r[2] - up[2] * r[1], up[2] * r[0] - up[0] * r[2], up[0] * r[1] - up[1] * r[0]];
  return [r[0], r[1], r[2], up[0], up[1], up[2], f[0], f[1], f[2]];
})();

// Fixed camera (same maths as the app's OrbitCamera)
function camera(W, H) {
  const yaw = 0.5, pitch = 0.26, dist = 4.3, t = [0, -0.45, 0], fov = 45 * Math.PI / 180;
  const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
  const e = [t[0] + dist * cp * sy, t[1] + dist * sp, t[2] + dist * cp * cy];
  let f = [t[0] - e[0], t[1] - e[1], t[2] - e[2]]; let l = Math.hypot(...f); f = f.map((v) => v / l);
  let r = [-f[2], 0, f[0]]; l = Math.hypot(...r); r = r.map((v) => v / l);
  const up = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const view = [r[0], up[0], -f[0], 0, r[1], up[1], -f[1], 0, r[2], up[2], -f[2], 0,
    -(r[0] * e[0] + r[1] * e[1] + r[2] * e[2]), -(up[0] * e[0] + up[1] * e[1] + up[2] * e[2]), f[0] * e[0] + f[1] * e[1] + f[2] * e[2], 1];
  const ff = 1 / Math.tan(fov / 2), near = 0.05, far = 100, nf = 1 / (near - far), aspect = W / H;
  const proj = [ff / aspect, 0, 0, 0, 0, ff, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
  const vp = new Array(16);
  for (let c = 0; c < 4; c++) for (let rr = 0; rr < 4; rr++) {
    vp[c * 4 + rr] = proj[rr] * view[c * 4] + proj[4 + rr] * view[c * 4 + 1] + proj[8 + rr] * view[c * 4 + 2] + proj[12 + rr] * view[c * 4 + 3];
  }
  return { vp, right: r, up, fwd: f, wpp: 2 * dist * Math.tan(fov / 2) / H };
}

// Scripted input: what a person does during the benchmark session.
function script(fr_, dt) {
  const lerp = (a, b, t) => a + (b - a) * t;
  let cursor = null, burst = null;
  if (fr_ >= 60 && fr_ < 110) { const t = (fr_ - 60) / 50; cursor = [lerp(420, 880, t), lerp(380, 430, t) + 30 * Math.sin(t * 6)]; }
  if (fr_ >= 215 && fr_ < 260) { const t = (fr_ - 215) / 45; cursor = [lerp(760, 500, t), lerp(250, 560, t)]; }
  if (fr_ === 130) burst = [700, 470];
  let turb = 0.35;
  if (fr_ >= 140 && fr_ < 200) turb = 0.35 + 0.55 * Math.sin(Math.PI * (fr_ - 140) / 60);
  let pad = null;
  if (fr_ >= 150 && fr_ < 230) { const s = (fr_ - 150) / 80 * Math.PI * 2; pad = [0.32 * Math.sin(s), 0.3 * Math.sin(2 * s)]; }
  return { cursor, burst, turb, pad };
}

export function makeHost({ n, shape = 'Halvorsen', W = 1280, H = 800 }) {
  const A = ATTRACTORS[shape];
  const kind = KIND[shape];
  const orient = A.orient === 'diag' ? DIAG : A.orient;
  const cam = camera(W, H);
  const dt = 1 / 60;
  const params = { ...A.def }, target = { ...A.def };
  const defPad = A.pad.map(([key, lo, hi]) => (A.def[key] - lo) / (hi - lo));
  const st = { center: [0, 0, 0], scale: 0.95 / A.spread, meanSpeed: 1, first: true, svx: 0, svy: 0, pcx: null, pcy: null };
  const buf = new ArrayBuffer(u.SIZE * 4);
  const U = new Float32Array(buf), UI = new Int32Array(buf);

  // ---- initial state (seeded, deterministic) ----
  function initState() {
    const S = new Float32Array(n * 3);
    const p = params, h = A.h;
    let x = A.init[0], y = A.init[1], z = A.init[2];
    const adv = () => {
      let dx, dy, dz;
      if (kind === 0) { const zb = z - p.b; dx = zb * x - p.d * y; dy = p.d * x + zb * y; dz = p.c + p.a * z - z * z * z / 3 - (x * x + y * y) * (1 + p.e * z) + p.f * z * x * x * x; }
      else if (kind === 1) { dx = fsin(p.k * y) - p.b * x; dy = fsin(p.k * z) - p.b * y; dz = fsin(p.k * x) - p.b * z; }
      else if (kind === 2) { dx = -p.a * x - p.k * y - p.k * z - y * y; dy = -p.a * y - p.k * z - p.k * x - z * z; dz = -p.a * z - p.k * x - p.k * y - x * x; }
      else { dx = p.s * (y - x); dy = x * (p.r - z) - y; dz = x * y - p.b * z; }
      x += dx * h; y += dy * h; z += dz * h;
    };
    for (let s = 0; s < 3000; s++) adv();
    const jit = A.spread * 0.18;
    let mx = 0, my = 0, mz = 0;
    for (let i = 0; i < n; i++) {
      adv(); adv(); adv();
      S[i * 3] = fr(x + (rnd(i, 0x7fffffff, 21) - 0.5) * jit);
      S[i * 3 + 1] = fr(y + (rnd(i, 0x7fffffff, 22) - 0.5) * jit);
      S[i * 3 + 2] = fr(z + (rnd(i, 0x7fffffff, 23) - 0.5) * jit);
      mx += S[i * 3]; my += S[i * 3 + 1]; mz += S[i * 3 + 2];
    }
    st.center = [mx / n, my / n, mz / n];
    const heal = new Float32Array(n);
    for (let i = 0; i < n; i++) heal[i] = fr(0.16 + rnd(i, 0x7ffffffe, 24) * 0.14);
    return { S, P: new Float32Array(n * 4), V: new Float32Array(n * 3), F: new Float32Array(n), T: new Float32Array(n), heal };
  }

  // ---- per-frame uniforms ----
  function prepare(frame) {
    const time = (frame + 1) * dt;
    const sc = script(frame, dt);
    if (sc.pad) {
      const [[k0, lo0, hi0], [k1, lo1, hi1]] = A.pad;
      target[k0] = lo0 + Math.min(1, Math.max(0, defPad[0] + sc.pad[0])) * (hi0 - lo0);
      target[k1] = lo1 + Math.min(1, Math.max(0, defPad[1] + sc.pad[1])) * (hi1 - lo1);
    }
    const follow = 1 - Math.exp(-dt * 2.5);
    for (const k in target) params[k] += (target[k] - params[k]) * follow;

    U.fill(0);
    UI[u.KIND] = kind; UI[u.FRAME] = frame;
    const p = params;
    U[u.A] = p.a || 0; U[u.B] = p.b || 0; U[u.C] = p.c || 0; U[u.D] = p.d || 0; U[u.E] = p.e || 0;
    U[u.FP] = p.f || 0; U[u.K] = p.k || 0; U[u.R] = p.r || 0; U[u.SG] = p.s || 0;
    U[u.H] = A.h * Math.min(2, dt * 60); U[u.BOUND] = A.spread * 12; U[u.JIT] = A.spread * 0.18;
    U.set(A.init, u.INIT);
    U[u.SC] = st.scale; U[u.CX] = st.center[0]; U[u.CY] = st.center[1]; U[u.CZ] = st.center[2];
    U[u.INV_SPEED] = 0.35 / st.meanSpeed; U[u.INV_HOT] = 1 / st.meanSpeed;
    U.set(orient, u.ORIENT);

    let flags = 0;
    const turb = sc.turb;
    const warpAmp = 0.42 * turb * turb + 0.08 * turb;
    if (warpAmp > 1e-3) { flags |= FLAG_WARP; U[u.WARP_AMP] = warpAmp; U[u.WOX] = time * 0.12; U[u.WOY] = time * 0.07; U[u.WOZ] = -time * 0.09; }

    if (sc.cursor) {
      const [x, y] = sc.cursor;
      const pvx = st.pcx === null ? 0 : (x - st.pcx) / dt, pvy = st.pcy === null ? 0 : (y - st.pcy) / dt;
      st.svx = st.svx * 0.5 + pvx * 0.5; st.svy = st.svy * 0.5 + pvy * 0.5;
      st.pcx = x; st.pcy = y;
      const vx = st.svx * cam.wpp, vy = -st.svy * cam.wpp;
      flags |= FLAG_TEAR;
      U[u.TX] = x; U[u.TY] = y; U[u.TR2] = 80 * 80;
      U[u.WVX] = cam.right[0] * vx + cam.up[0] * vy; U[u.WVY] = cam.right[1] * vx + cam.up[1] * vy; U[u.WVZ] = cam.right[2] * vx + cam.up[2] * vy;
    } else { st.pcx = st.pcy = null; st.svx = st.svy = 0; }
    if (sc.burst) { flags |= FLAG_BURST; U[u.BX] = sc.burst[0]; U[u.BY] = sc.burst[1]; U[u.BR2] = 150 * 150; }
    U.set(cam.right, u.RIGHT); U.set(cam.up, u.UP); U.set(cam.fwd, u.FWD);
    U[u.SW] = W; U[u.SH] = H; U.set(cam.vp, u.VP);

    U[u.FREQ] = 0.6 + 1.8 * turb; U[u.AMP] = (0.2 + 0.85 * turb) / 1.6;
    U[u.SOX] = time * 0.05; U[u.SOY] = -time * 0.035; U[u.SOZ] = time * 0.065;
    U[u.RELAX] = 1 - Math.exp(-dt * 2.2); U[u.BACK] = 1 - Math.exp(-dt * 12); U[u.BACK_MORPH] = 1 - Math.exp(-dt * 3.5);
    U[u.DT] = dt;
    UI[u.FLAGS] = flags;
    return { U, UI };
  }

  // ---- auto-fit from the kernel's samples (sequential f64 sums, like the app) ----
  function after(SS) {
    const m = SS.length / 5;
    let sx = 0, sy = 0, sz = 0, sv = 0;
    for (let s = 0; s < m; s++) { sx += SS[s * 5]; sy += SS[s * 5 + 1]; sz += SS[s * 5 + 2]; sv += SS[s * 5 + 3]; }
    const mx = sx / m, my = sy / m, mz = sz / m;
    let sr = 0;
    for (let s = 0; s < m; s++) { const ex = SS[s * 5] - mx, ey = SS[s * 5 + 1] - my, ez = SS[s * 5 + 2] - mz; sr += ex * ex + ey * ey + ez * ez; }
    const rms = Math.sqrt(sr / m) || 1;
    const a = st.first ? 1 : 1 - Math.exp(-dt * 1.5);
    st.center[0] += (mx - st.center[0]) * a; st.center[1] += (my - st.center[1]) * a; st.center[2] += (mz - st.center[2]) * a;
    st.scale += (0.95 / rms - st.scale) * a;
    st.meanSpeed += (sv / m - st.meanSpeed) * a;
    st.first = false;
  }

  return { initState, prepare, after, U, UI };
}
