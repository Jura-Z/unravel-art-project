// Unravel — attractor + tear + smoke + turbulence + links in one instrument.
//
// Every particle rides a strange attractor (its "home" keeps moving along the flow).
// Turbulence warps the whole shape through a small baked curl-noise grid. Dragging
// tears particles off: they become smoke, advected by per-particle curl noise with the
// stroke's momentum, and after a few seconds they drift back to wherever their home
// has moved. Particles near the cursor are linked to it with faint lines.
//
// Three simulation backends share this host logic (input, morphs, auto-fit, demo):
//   js   – the passes below, in plain JS over typed arrays
//   wasm – the C kernel (bench/kernels/c) compiled to WASM SIMD, bit-exact with the spec
//   gpu  – the same kernel in WGSL, run by the WebGPU renderer (gpu/backend.js)
// The JS step is split into passes, each timed on its own:
//   ode   – integrate the attractor, write home positions         O(N)
//   warp  – bake a 12³ curl grid, displace homes trilinearly      O(N + G³)
//   tear  – project to screen and apply the brush (while touching) O(N)
//   smoke – curl-noise advection for torn particles                O(torn)
//   links – K particles near the cursor, walked in priority order  O(K..N/4)

const UNRAVEL_PALETTES = [
  { name: 'Ember', stops: ['#4744d6', '#8a5cff', '#ff5cb8', '#ffa066', '#fff0d6'] },
  { name: 'Aurora', stops: ['#2a62d8', '#2cc6e8', '#8cffe2', '#c2b6ff', '#ff86d0'] },
  { name: 'Ice', stops: ['#2a4a9a', '#4a86ff', '#9ad8ff', '#f2fbff', '#c2caff'] },
];
const LINK_STRIDE = 4, LINK_BUDGET = 4096;
const FLASH_ATTACK = 0.06;          // shape-change flash: seconds to full white
// Seeding: one cached orbit per attractor (default params), sampled by particle index.
const ORBIT_LEN = 1 << 17, ORBITS = [];
function orbitFor(sys) {
  if (ORBITS[sys]) return ORBITS[sys];
  const A = ATTRACTORS[sys], p = A.def, h = A.h, kind = KIND_OF[A.name];
  const out = new Float32Array(ORBIT_LEN * 3);
  let x = A.init[0], y = A.init[1], z = A.init[2];
  for (let s = 0, e = 3000 + ORBIT_LEN * 3; s < e; s++) {
    let dx, dy, dz;
    if (kind === 0) { const zb = z - p.b; dx = zb * x - p.d * y; dy = p.d * x + zb * y; dz = p.c + p.a * z - z * z * z / 3 - (x * x + y * y) * (1 + p.e * z) + p.f * z * x * x * x; }
    else if (kind === 1) { dx = Math.sin(p.k * y) - p.b * x; dy = Math.sin(p.k * z) - p.b * y; dz = Math.sin(p.k * x) - p.b * z; }
    else if (kind === 2) { dx = -p.a * x - p.k * y - p.k * z - y * y; dy = -p.a * y - p.k * z - p.k * x - z * z; dz = -p.a * z - p.k * x - p.k * y - x * x; }
    else { dx = p.s * (y - x); dy = x * (p.r - z) - y; dz = x * y - p.b * z; }
    x += dx * h; y += dy * h; z += dz * h;
    if (s >= 3000 && (s - 3000) % 3 === 2) { const o = ((s - 3000 - 2) / 3) * 3; out[o] = x; out[o + 1] = y; out[o + 2] = z; }
  }
  return (ORBITS[sys] = out);
}
// Same lowbias32 counter hash as the kernels (hash3(i, 0x7ffffff0 + axis, salt)); the WGSL seed pass matches it.
function seedRnd(i, salt, axis) {
  let x = (Math.imul(i, 0x9E3779B1) ^ Math.imul(0x7ffffff0 + axis, 0x85EBCA77) ^ Math.imul(salt, 0xC2B2AE3D)) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x7feb352d); x ^= x >>> 15; x = Math.imul(x, 0x846ca68b); x ^= x >>> 16;
  return (x >>> 8) * (1 / 16777216);
}
const LINK_INV = (() => { let y = 1; for (let k = 0; k < 5; k++) y = Math.imul(y, 2 - Math.imul(2654435761, y)); return y; })();   // 2654435761^-1 mod 2^32
const SHAPE_PERIOD = 22, MORPH_EASE = 1.4, MORPH_SPREAD = 1.6;
const WARP_G = 12, WARP_LO = -2.2, WARP_HI = 2.2;

const UnravelMode = {
  hint: '<b>Drag</b> across the shape to tear threads off; they turn to smoke and drift home. <b>Click</b> for a burst. Use the panel to bend the shape and add turbulence.',
  accent: '#ff8fb1',
  palette: UNRAVEL_PALETTES[0].stops,
  sky: { low: '#100f22', high: '#040509', floor: '#23203f' },
  floor: { y: -1.3, reflGain: 0.7, reflFall: 0.9 },   // y follows the shape's lowest point
  decay: 0.66, exposure: 1.3, pointSize: 1.0, gain: 0.03, refCount: 300000, grain: 0.028,
  lineGain: 0.35,
  counts: [100000, 150000, 200000, 300000, 600000, 1200000, 2400000], countIndex: 1,
  camera: { yaw: 0.5, pitch: 0.26, dist: 4.3, target: [0, -0.45, 0] },
  minPitch: 0.03,
  autoRotate: 0.06,

  sys: 0,
  paletteIndex: 0,
  shapeCycle: true,      // morph to the next attractor every SHAPE_PERIOD seconds
  turb: 0.35,
  linksOn: true,
  trails: true,
  // HDR, bloom (where float render targets exist) and colour cycling are always on.
  bloom: { on: true, threshold: 0.8, strength: 0.8 },
  paletteRate: 0.35,

  get optionGroups() {
    return [
      { label: 'shape', items: [...ATTRACTORS.map((a) => a.name), 'Cycle'],
        index: this.shapeCycle ? ATTRACTORS.length : this.sys,
        current: this.sys,                            // shown even while Cycle is selected
        select: (i) => {
          if (i === ATTRACTORS.length) { this.shapeCycle = true; this.shapeT = 0; return; }
          this.shapeCycle = false;
          if (i !== this.sys) this.morphTo(i);
        } },
      { label: 'blur', items: ['motion blur', 'off'], index: this.trails ? 0 : 1,
        select: (i) => { this.trails = i === 0; if (this.E) this.E.renderer.clear(); } },
    ];
  },

  get pad() {
    const A = ATTRACTORS[this.sys];
    return {
      get: () => this.padUV,
      set: (u, v) => this.setPad(u, v),
      reset: () => this.setPad(this.defPad(0), this.defPad(1)),
      labels: () => [`${A.axes[0]} →`, `${A.axes[1]} →`],
    };
  },

  setTurb(v) { this.turb = Math.max(0, Math.min(1, v)); },

  setPalette(i) {
    this.paletteIndex = i;
    this.palette = UNRAVEL_PALETTES[i].stops;
    if (this.E) this.E.renderer.setPalette(this.palette);
  },

  // Colour always drifts through the palettes (the renderer crossfades); shapes
  // morph from one attractor to the next when Cycle is on.
  tickColor(dt, E) {
    // Shape-change flash: fast attack, ~0.8 s exponential release (the particles "react").
    if (this.flashT !== undefined) {
      this.flashT += dt;
      const t = this.flashT;
      this.flash = t < FLASH_ATTACK ? t / FLASH_ATTACK : Math.exp(-(t - FLASH_ATTACK) * 4.5);
      if (this.flash < 0.004) { this.flash = 0; this.flashT = undefined; }
    }
    this.cycT = (this.cycT || 0) + dt;
    if (this.cycT > 9) { this.cycT = 0; this.setPalette((this.paletteIndex + 1) % UNRAVEL_PALETTES.length); }
    if (this.shapeCycle && !E.frozen) {
      this.shapeT = (this.shapeT || 0) + dt;
      if (this.shapeT > SHAPE_PERIOD) { this.shapeT = 0; this.morphTo((this.sys + 1) % ATTRACTORS.length); }
    }
  },

  // Switch attractor without restarting: particles keep their current positions and
  // peel off top-to-bottom, easing onto their homes on the new shape. The new system is
  // seeded on its own trajectory, so nothing has to converge from a random cloud.
  morphTo(i) {
    this.flashT = 0;
    if (this.backend === 'gpu') {
      const SS = this.sim.samples();
      let lo = Infinity, hi = -Infinity;
      if (SS) for (let q = 4; q < SS.length; q += 5) { if (SS[q] < lo) lo = SS[q]; if (SS[q] > hi) hi = SS[q]; }
      if (!(hi > lo)) { lo = -1.5; hi = 1.5; }
      this.sys = i;
      this.reset(true);
      this.sim.morph(hi + 0.2, Math.max(1e-3, hi - lo + 0.4));
      this.morphLeft = MORPH_EASE + MORPH_SPREAD + 0.2;
      if (this.E) { this.E.renderOptions(); this.E.toast(ATTRACTORS[i].name); }
      return;
    }
    const P = this.P, F = this.F, n = this.count;
    let lo = Infinity, hi = -Infinity;
    for (let q = 0; q < n; q += 64) { const y = P[q * 4 + 1]; if (y < lo) lo = y; if (y > hi) hi = y; }
    const span = Math.max(1e-3, hi - lo);
    this.sys = i;
    this.reset(true);
    for (let q = 0; q < n; q++) {
      const top = (hi - P[q * 4 + 1]) / span;                 // 0 at the top, 1 at the bottom
      F[q] = -(MORPH_EASE + MORPH_SPREAD * Math.min(1, Math.max(0, top)) + Math.random() * 0.15);
    }
    this.morphLeft = MORPH_EASE + MORPH_SPREAD + 0.2;
    if (this.E) { this.E.renderOptions(); this.E.toast(ATTRACTORS[i].name); }
  },
  init(E, n) {
    this.E = E;
    if (!E.renderer.bloomSupported) this.bloom.on = false;
    this.count = n;
    this.backend = E.simBackend || 'js';      // 'js' | 'wasm' | 'gpu'
    this.S = this.backend === 'gpu' ? null : new Float32Array(n * 3);   // raw attractor state (JS: live; WASM: seed buffer; GPU: seeded on the GPU)
    this.heal = new Float32Array(n);
    for (let i = 0; i < n; i++) this.heal[i] = 0.16 + Math.random() * 0.14;
    if (this.backend === 'wasm') {
      // Kernel owns the state in WASM memory; P, T, F are views (the WebGL renderer uploads from them).
      this.sim = createWasmSim(n);
      this.P = this.sim.P; this.T = this.sim.T; this.F = this.sim.F;
      this.H = this.V = this.B = this.warp = null;
    } else if (this.backend === 'gpu') {
      this.sim = E.renderer.createSim(n);      // state lives in GPU buffers
      this.P = this.T = this.F = this.H = this.V = this.B = this.warp = null;
    } else {
      this.sim = null;
      this.H = new Float32Array(n * 3);   // home position (world)
      this.P = new Float32Array(n * 4);   // rendered position + brightness
      this.V = new Float32Array(n * 3);   // velocity while torn
      this.F = new Float32Array(n);       // >0 torn (1 = fully smoke), <0 returning, 0 attached
      this.B = new Float32Array(n);       // emissive intensity; >1 is HDR (fast threads burn hot)
      this.T = new Float32Array(n);
      this.warp = new Float32Array(WARP_G * WARP_G * WARP_G * 3);
    }
    this.ms = { ode: 0, warp: 0, tear: 0, smoke: 0, links: 0, kernel: 0 };
    this.torn = 0;
    this.frameNo = 0;
    this.cursor = null;
    this.burst = null;
    this._c = new Float64Array(3);
    this.setupLinks();
    this.reset();
  },

  setupLinks() {
    this.linkIdx = new Int32Array(LINK_K);
    this.linkD2 = new Float32Array(LINK_K);
    this.linkPri = new Uint32Array(LINK_K);
    this.lines = { data: new Float32Array(LINK_K * 8), count: 0 };
  },

  defPad(axis) {
    const A = ATTRACTORS[this.sys];
    const [key, lo, hi] = A.pad[axis];
    return Math.max(0, Math.min(1, (A.def[key] - lo) / (hi - lo)));   // the knob always stays on the pad
  },

  setPad(u, v) {
    const A = ATTRACTORS[this.sys];
    const [k0, lo0, hi0] = A.pad[0], [k1, lo1, hi1] = A.pad[1];
    u = Math.max(0, Math.min(1, u)); v = Math.max(0, Math.min(1, v));
    this.padUV = [u, v];
    this.target[k0] = lo0 + u * (hi0 - lo0);
    this.target[k1] = lo1 + v * (hi1 - lo1);
  },

  reset(morph) {
    const A = ATTRACTORS[this.sys];
    this.params = Object.assign({}, A.def);
    this.target = Object.assign({}, A.def);
    this.padUV = [this.defPad(0), this.defPad(1)];
    // Seed particles on the attractor, jittered so the opening is a quick "focus pull".
    // Each particle takes orbit[i mod L] from the shape's cached orbit plus a counter-hash
    // jitter. WebGPU does the fill in a compute pass, so a shape change uploads nothing
    // (this keeps a morph at 2.4M particles to ~14 ms of JS).
    const S = this.S, n = this.count, orb = orbitFor(this.sys), L = ORBIT_LEN;
    const jit = A.spread * 0.18, salt = this.seedSalt = ((this.seedSalt | 0) + 1) & 0xffff;
    const gpu = this.backend === 'gpu';
    if (!gpu) {
      for (let i = 0; i < n; i++) {
        const k = i * 3, o = (i & (L - 1)) * 3;
        S[k] = orb[o] + (seedRnd(i, salt, 0) - 0.5) * jit;
        S[k + 1] = orb[o + 1] + (seedRnd(i, salt, 1) - 0.5) * jit;
        S[k + 2] = orb[o + 2] + (seedRnd(i, salt, 2) - 0.5) * jit;
      }
    }
    // Initial fit from every 64th particle (recomputed here, so it's backend-independent),
    // so the first frames are framed and coloured before any samples come back.
    {
      const p = this.params, kind = KIND_OF[A.name];
      let mx = 0, my = 0, mz = 0, m = 0;
      const smp = this.seedSamples || (this.seedSamples = new Float32Array(Math.ceil(this.counts[this.counts.length - 1] / 64) * 3));
      for (let i = 0; i < n; i += 64, m++) {
        const o = (i & (L - 1)) * 3, q = m * 3;
        smp[q] = orb[o] + (seedRnd(i, salt, 0) - 0.5) * jit;
        smp[q + 1] = orb[o + 1] + (seedRnd(i, salt, 1) - 0.5) * jit;
        smp[q + 2] = orb[o + 2] + (seedRnd(i, salt, 2) - 0.5) * jit;
        mx += smp[q]; my += smp[q + 1]; mz += smp[q + 2];
      }
      const cx = mx / m, cy = my / m, cz = mz / m;
      let sr = 0, sv = 0;
      for (let q = 0; q < m * 3; q += 3) {
        const x = smp[q], y = smp[q + 1], z = smp[q + 2];
        sr += (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
        let dx, dy, dz;
        if (kind === 0) { const zb = z - p.b; dx = zb * x - p.d * y; dy = p.d * x + zb * y; dz = p.c + p.a * z - z * z * z / 3 - (x * x + y * y) * (1 + p.e * z) + p.f * z * x * x * x; }
        else if (kind === 1) { dx = Math.sin(p.k * y) - p.b * x; dy = Math.sin(p.k * z) - p.b * y; dz = Math.sin(p.k * x) - p.b * z; }
        else if (kind === 2) { dx = -p.a * x - p.k * y - p.k * z - y * y; dy = -p.a * y - p.k * z - p.k * x - z * z; dz = -p.a * z - p.k * x - p.k * y - x * x; }
        else { dx = p.s * (y - x); dy = x * (p.r - z) - y; dz = x * y - p.b * z; }
        sv += Math.sqrt(dx * dx + dy * dy + dz * dz);
      }
      this.center = [cx, cy, cz];
      this.scale = 0.95 / (Math.sqrt(sr / m) || 1);
      this.meanSpeed = sv / m || 1;
    }
    if (gpu) {
      if (!morph) { this.sim.load({ S: null, heal: this.heal }); this.floorReady = false; }
      this.sim.seed(orb, jit, salt);
    } else if (this.sim) {
      if (morph) this.sim.reseed(S); else { this.sim.load({ S, heal: this.heal }); this.floorReady = false; }
    } else {
      this.V.fill(0);
      if (!morph) { this.F.fill(0); this.floorReady = false; }
    }
    // Warm the next shape's orbit while idle, so Cycle never pays for it on the frame.
    const next = (this.sys + 1) % ATTRACTORS.length;
    if (!ORBITS[next]) {
      if (window.requestIdleCallback) requestIdleCallback(() => orbitFor(next), { timeout: 1500 });   // busy pages get no idle time
      else setTimeout(() => orbitFor(next), 500);
    }
    this.first = true;
  },

  // ---- input ----------------------------------------------------------------
  pointer(type, p) {
    if (type === 'leave') { this.hover = null; return; }
    if (p) this.hover = { x: p.x, y: p.y, nx: p.nx, ny: p.ny };
    if (type === 'down') {
      this.cursor = { x: p.x, y: p.y, x0: p.x, y0: p.y, t0: p.t, moved: false };
    } else if (type === 'move') {
      const c = this.cursor;
      if (!c) return;
      c.x = p.x; c.y = p.y;
      if (Math.hypot(p.x - c.x0, p.y - c.y0) > 6) c.moved = true;
    } else if (type === 'up') {
      const c = this.cursor;
      if (c && !c.moved && p.t - c.t0 < 350) this.burst = { x: p.x, y: p.y };
      this.cursor = null;
      this._pcx = this._pcy = undefined;
    }
  },

  key(k, E) {
    if (k === 'c' || k === 'C') { this.shapeCycle = false; this.morphTo((this.sys + 1) % ATTRACTORS.length); }
  },

  colorAnim(t) {
    // The gradient runs along a slowly turning direction and scrolls over time.
    const a = t * 0.11;
    const x = Math.cos(a), y = 0.45 * Math.sin(t * 0.07), z = Math.sin(a);
    const l = Math.hypot(x, y, z);
    return { dir: [x / l, y / l, z / l], scale: 0.55, phase: t * 0.05 };
  },

  // While the demo plays, the camera slowly dollies out and back in (engine eases to it).
  demoDist(t) { return 4.3 + 1.6 * Math.sin(t * 0.21 - 0.4) + 0.5 * Math.sin(t * 0.083); },

  demo(t) {
    if (t < 1.6) return null;                                   // shape comes into focus
    const c = (t - 1.6) % 17;
    if (c < 1.8) {                                              // tear a strip across the middle
      const s = c / 1.8;
      return { u: 0.3 + 0.42 * s, v: 0.42 + 0.08 * Math.sin(s * 3.3), down: c > 0.1 };
    }
    if (c < 3.2) return { u: 0.72, v: 0.48 + (c - 1.8) * 0.04, down: false };
    if (c < 3.3) return { u: 0.58, v: 0.56, down: true };      // click → burst
    if (c < 5.5) return { u: 0.58, v: 0.56, down: false };
    if (c < 10.5) {                                             // play the pad
      const s = (c - 5.5) / 5 * Math.PI * 2;
      return { padU: this.defPad(0) + 0.32 * Math.sin(s), padV: this.defPad(1) + 0.3 * Math.sin(s * 2) };
    }
    if (c < 10.7) return { padU: this.defPad(0), padV: this.defPad(1) };
    if (c < 14) {                                               // turbulence swell
      const s = (c - 10.7) / 3.3;
      return { turb: 0.35 + 0.55 * Math.sin(s * Math.PI) };
    }
    if (c < 15.5) {                                             // a second, diagonal tear
      const s = (c - 14) / 1.5;
      return { u: 0.62 - 0.3 * s, v: 0.28 + 0.4 * s, down: c > 14.1, turb: 0.35 };
    }
    return null;
  },

  // ---- simulation (kernel backends: WASM / WebGPU) ------------------------------
  // The host fills the kernel's uniform block (same layout as the benchmark) and the
  // kernel steps every particle. Auto-fit and the floor come from the 1-in-64 samples.
  stepKernel(dt, time, E) {
    const A = ATTRACTORS[this.sys], sim = this.sim;
    const p = this.params, tg = this.target;
    const follow = 1 - Math.exp(-dt * 2.5);
    for (const k in tg) p[k] += (tg[k] - p[k]) * follow;
    const U = sim.U, UI = sim.UI, u = UNI;
    UI[u.KIND] = KIND_OF[A.name]; UI[u.FRAME] = ++this.frameNo;
    U[u.A] = p.a || 0; U[u.B] = p.b || 0; U[u.C] = p.c || 0; U[u.D] = p.d || 0; U[u.E] = p.e || 0;
    U[u.FP] = p.f || 0; U[u.K] = p.k || 0; U[u.R] = p.r || 0; U[u.SG] = p.s || 0;
    U[u.H] = A.h * Math.min(2, dt * 60); U[u.BOUND] = A.spread * 12; U[u.JIT] = A.spread * 0.18;
    U[u.INIT] = A.init[0]; U[u.INIT + 1] = A.init[1]; U[u.INIT + 2] = A.init[2];
    U[u.SC] = this.scale; U[u.CX] = this.center[0]; U[u.CY] = this.center[1]; U[u.CZ] = this.center[2];
    U[u.INV_SPEED] = 0.35 / this.meanSpeed; U[u.INV_HOT] = 1 / this.meanSpeed;
    for (let q = 0; q < 9; q++) U[u.ORIENT + q] = A.orient[q];
    let flags = 0;
    const warpAmp = 0.42 * this.turb * this.turb + 0.08 * this.turb;
    if (warpAmp > 1e-3) { flags |= UFLAG_WARP; U[u.WARP_AMP] = warpAmp; U[u.WOX] = time * 0.12; U[u.WOY] = time * 0.07; U[u.WOZ] = -time * 0.09; }
    const cam = E.camera, cur = this.cursor, burst = this.burst;
    if (cur) {
      const pvx = this._pcx === undefined ? 0 : (cur.x - this._pcx) / Math.max(dt, 1e-3);
      const pvy = this._pcy === undefined ? 0 : (cur.y - this._pcy) / Math.max(dt, 1e-3);
      this._svx = (this._svx || 0) * 0.5 + pvx * 0.5; this._svy = (this._svy || 0) * 0.5 + pvy * 0.5;
      this._pcx = cur.x; this._pcy = cur.y;
      const wpp = cam.worldPerPx(), vx = this._svx * wpp, vy = -this._svy * wpp;
      flags |= UFLAG_TEAR;
      U[u.TX] = cur.x; U[u.TY] = cur.y; U[u.TR2] = 80 * 80;
      U[u.WVX] = cam.right[0] * vx + cam.up[0] * vy; U[u.WVY] = cam.right[1] * vx + cam.up[1] * vy; U[u.WVZ] = cam.right[2] * vx + cam.up[2] * vy;
    }
    if (burst) { flags |= UFLAG_BURST; U[u.BX] = burst.x; U[u.BY] = burst.y; U[u.BR2] = 150 * 150; this.burst = null; }
    for (let q = 0; q < 3; q++) { U[u.RIGHT + q] = cam.right[q]; U[u.UP + q] = cam.up[q]; U[u.FWD + q] = cam.fwd[q]; }
    U[u.SW] = E.cssW; U[u.SH] = E.cssH;
    for (let q = 0; q < 16; q++) U[u.VP + q] = cam.vp[q];
    U[u.FREQ] = 0.6 + 1.8 * this.turb; U[u.AMP] = (0.2 + 0.85 * this.turb) / 1.6;
    U[u.SOX] = time * 0.05; U[u.SOY] = -time * 0.035; U[u.SOZ] = time * 0.065;
    U[u.RELAX] = 1 - Math.exp(-dt * 2.2); U[u.BACK] = 1 - Math.exp(-dt * 12); U[u.BACK_MORPH] = 1 - Math.exp(-dt * 3.5);
    U[u.DT] = dt;
    UI[u.FLAGS] = flags;

    const t0 = performance.now();
    sim.step();
    this.ms.kernel += (performance.now() - t0 - this.ms.kernel) * 0.1;
    if (sim.torn !== null) this.torn = sim.torn;

    // auto-fit + floor from the samples (WebGPU: the latest async readback, 1-2 frames old)
    const SS = sim.samples();
    if (SS) {
      const m = SS.length / 5;
      let sx = 0, sy = 0, sz = 0, sv = 0, minY = Infinity;
      for (let q = 0; q < m; q++) { const b = q * 5; sx += SS[b]; sy += SS[b + 1]; sz += SS[b + 2]; sv += SS[b + 3]; if (SS[b + 4] < minY) minY = SS[b + 4]; }
      const mx = sx / m, my = sy / m, mz = sz / m;
      let sr = 0;
      for (let q = 0; q < m; q++) { const b = q * 5, ex = SS[b] - mx, ey = SS[b + 1] - my, ez = SS[b + 2] - mz; sr += ex * ex + ey * ey + ez * ez; }
      const rms = Math.sqrt(sr / m) || 1;
      const a = this.first ? 1 : 1 - Math.exp(-dt * 1.5);
      this.center[0] += (mx - this.center[0]) * a; this.center[1] += (my - this.center[1]) * a; this.center[2] += (mz - this.center[2]) * a;
      this.scale += (0.95 / rms - this.scale) * a;
      this.meanSpeed += (sv / m - this.meanSpeed) * a;
      const fy = Math.min(0, minY) - 0.25 - 0.3 * this.turb;
      this.floor.y += (fy - this.floor.y) * (this.floorReady ? 1 - Math.exp(-dt * 0.5) : 1);
      this.floorReady = true;
      this.first = false;
    }
    if (this.backend === 'wasm') {
      const tl = performance.now();
      this.stepLinks(E);
      this.ms.links += (performance.now() - tl - this.ms.links) * 0.1;
    } else this.lines.count = 0;             // WebGPU draws cursor links itself
  },

  // ---- simulation (plain JS) ---------------------------------------------------------
  step(dt, time, E) {
    if (this.sim) return this.stepKernel(dt, time, E);
    const A = ATTRACTORS[this.sys];
    const p = this.params, tg = this.target;
    const follow = 1 - Math.exp(-dt * 2.5);
    for (const k in tg) p[k] += (tg[k] - p[k]) * follow;

    const n = this.count, S = this.S, H = this.H, P = this.P, V = this.V, F = this.F, T = this.T;
    const t0 = performance.now();

    // ---- pass 1: ODE + home -------------------------------------------------
    const h = A.h * Math.min(2, dt * 60);
    const bound = A.spread * 12, o = A.orient;
    const sc = this.scale, cx = this.center[0], cy = this.center[1], cz = this.center[2];
    const invSpeed = 0.35 / this.meanSpeed, invHot = 1 / this.meanSpeed, B = this.B;
    const kind = KIND_OF[A.name];
    let sx = 0, sy = 0, sz = 0, sv = 0, samples = 0, minY = 0;
    for (let i = 0; i < n; i++) {
      const k = i * 3;
      let x = S[k], y = S[k + 1], z = S[k + 2], dx, dy, dz;
      switch (kind) {
        case 0: {
          const zb = z - p.b;
          dx = zb * x - p.d * y; dy = p.d * x + zb * y;
          dz = p.c + p.a * z - z * z * z / 3 - (x * x + y * y) * (1 + p.e * z) + p.f * z * x * x * x;
          break;
        }
        case 1:
          dx = Math.sin(p.k * y) - p.b * x; dy = Math.sin(p.k * z) - p.b * y; dz = Math.sin(p.k * x) - p.b * z;
          break;
        case 2:
          dx = -p.a * x - p.k * y - p.k * z - y * y;
          dy = -p.a * y - p.k * z - p.k * x - z * z;
          dz = -p.a * z - p.k * x - p.k * y - x * x;
          break;
        default:
          dx = p.s * (y - x); dy = x * (p.r - z) - y; dz = x * y - p.b * z;
      }
      x += dx * h; y += dy * h; z += dz * h;
      if (!(x > -bound && x < bound && y > -bound && y < bound && z > -bound && z < bound)) {
        const r = ((Math.random() * n) | 0) * 3;
        x = S[r] + (Math.random() - 0.5) * 1e-3; y = S[r + 1] + (Math.random() - 0.5) * 1e-3; z = S[r + 2] + (Math.random() - 0.5) * 1e-3;
      }
      S[k] = x; S[k + 1] = y; S[k + 2] = z;
      const speed = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if ((i & 63) === 0) { sx += x; sy += y; sz += z; sv += speed; samples++; }
      const rx = (x - cx) * sc, ry = (y - cy) * sc, rz = (z - cz) * sc;
      H[k] = o[0] * rx + o[1] * ry + o[2] * rz;
      const hy = o[3] * rx + o[4] * ry + o[5] * rz;
      H[k + 1] = hy;
      H[k + 2] = o[6] * rx + o[7] * ry + o[8] * rz;
      if ((i & 63) === 0 && hy < minY) minY = hy;
      const f = F[i];
      T[i] = speed * invSpeed + (f > 0 ? f * 0.3 : 0);
      // HDR emission: most of the shape sits below 1.0, the fastest threads go to ~6×.
      const hs = speed * invHot - 1.05;
      B[i] = hs > 0 ? 0.55 + Math.min(5.5, 3.2 * hs * hs + 1.2 * hs) : 0.55;
    }
    if (samples) {
      const mx = sx / samples, my = sy / samples, mz = sz / samples;
      let sr = 0;
      for (let i = 0; i < n; i += 64) {
        const k = i * 3, ex = S[k] - mx, ey = S[k + 1] - my, ez = S[k + 2] - mz;
        sr += ex * ex + ey * ey + ez * ez;
      }
      const rms = Math.sqrt(sr / samples) || 1;
      const a = this.first ? 1 : 1 - Math.exp(-dt * 1.5);
      this.center[0] += (mx - this.center[0]) * a;
      this.center[1] += (my - this.center[1]) * a;
      this.center[2] += (mz - this.center[2]) * a;
      this.scale += (0.95 / rms - this.scale) * a;
      this.meanSpeed += (sv / samples - this.meanSpeed) * a;
      // keep the floor just under the shape (slow, so it never jitters)
      const fy = minY - 0.25 - 0.3 * this.turb;
      this.floor.y += (fy - this.floor.y) * (this.floorReady ? 1 - Math.exp(-dt * 0.5) : 1);
      this.floorReady = true;
    }
    const t1 = performance.now();

    // ---- pass 1b: turbulence warp of the homes (baked curl grid) -------------
    const warpAmp = 0.42 * this.turb * this.turb + 0.08 * this.turb;
    if (warpAmp > 1e-3) {
      const G = WARP_G, W = this.warp, c = this._c;
      const cell = (WARP_HI - WARP_LO) / (G - 1);
      const f = 0.9, ox = time * 0.12, oy = time * 0.07, oz = -time * 0.09;
      for (let gz = 0, q = 0; gz < G; gz++) for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++, q += 3) {
        curlNoise((WARP_LO + gx * cell) * f + ox, (WARP_LO + gy * cell) * f + oy, (WARP_LO + gz * cell) * f + oz, c);
        W[q] = c[0]; W[q + 1] = c[1]; W[q + 2] = c[2];
      }
      const inv = 1 / cell, gmax = G - 1.001, G2 = G * G;
      for (let i = 0; i < n; i++) {
        const k = i * 3;
        let fx = (H[k] - WARP_LO) * inv, fy = (H[k + 1] - WARP_LO) * inv, fz = (H[k + 2] - WARP_LO) * inv;
        fx = fx < 0 ? 0 : fx > gmax ? gmax : fx;
        fy = fy < 0 ? 0 : fy > gmax ? gmax : fy;
        fz = fz < 0 ? 0 : fz > gmax ? gmax : fz;
        const ix = fx | 0, iy = fy | 0, iz = fz | 0;
        const ax = fx - ix, ay = fy - iy, az = fz - iz;
        const b = (iz * G2 + iy * G + ix) * 3;
        const b1 = b + 3, b2 = b + G * 3, b3 = b2 + 3, b4 = b + G2 * 3, b5 = b4 + 3, b6 = b4 + G * 3, b7 = b6 + 3;
        const w0 = (1 - ax) * (1 - ay) * (1 - az), w1 = ax * (1 - ay) * (1 - az), w2 = (1 - ax) * ay * (1 - az), w3 = ax * ay * (1 - az);
        const w4 = (1 - ax) * (1 - ay) * az, w5 = ax * (1 - ay) * az, w6 = (1 - ax) * ay * az, w7 = ax * ay * az;
        H[k] += warpAmp * (W[b] * w0 + W[b1] * w1 + W[b2] * w2 + W[b3] * w3 + W[b4] * w4 + W[b5] * w5 + W[b6] * w6 + W[b7] * w7);
        H[k + 1] += warpAmp * (W[b + 1] * w0 + W[b1 + 1] * w1 + W[b2 + 1] * w2 + W[b3 + 1] * w3 + W[b4 + 1] * w4 + W[b5 + 1] * w5 + W[b6 + 1] * w6 + W[b7 + 1] * w7);
        H[k + 2] += warpAmp * (W[b + 2] * w0 + W[b1 + 2] * w1 + W[b2 + 2] * w2 + W[b3 + 2] * w3 + W[b4 + 2] * w4 + W[b5 + 2] * w5 + W[b6 + 2] * w6 + W[b7 + 2] * w7);
      }
    }
    const t1b = performance.now();

    // ---- pass 2: tear brush / burst ------------------------------------------
    const cur = this.cursor, burst = this.burst;
    if (cur || burst) {
      const cam = E.camera, vp = cam.vp, Wd = E.cssW, Hh = E.cssH, wpp = cam.worldPerPx();
      let tx = 0, ty = 0, tr2 = 0, wvx = 0, wvy = 0, wvz = 0;
      if (cur) {
        const pvx = this._pcx === undefined ? 0 : (cur.x - this._pcx) / Math.max(dt, 1e-3);
        const pvy = this._pcy === undefined ? 0 : (cur.y - this._pcy) / Math.max(dt, 1e-3);
        this._svx = (this._svx || 0) * 0.5 + pvx * 0.5;
        this._svy = (this._svy || 0) * 0.5 + pvy * 0.5;
        this._pcx = cur.x; this._pcy = cur.y;
        tx = cur.x; ty = cur.y; tr2 = 80 * 80;
        const vx = this._svx * wpp, vy = -this._svy * wpp;
        wvx = cam.right[0] * vx + cam.up[0] * vy;
        wvy = cam.right[1] * vx + cam.up[1] * vy;
        wvz = cam.right[2] * vx + cam.up[2] * vy;
      }
      const bx = burst ? burst.x : 0, by = burst ? burst.y : 0, br2 = burst ? 150 * 150 : 0;
      const R = cam.right, U = cam.up, Fw = cam.fwd;
      for (let i = 0; i < n; i++) {
        const j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
        const w = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        if (w <= 0.05) continue;
        const qx = ((vp[0] * x + vp[4] * y + vp[8] * z + vp[12]) / w * 0.5 + 0.5) * Wd;
        const qy = (0.5 - (vp[1] * x + vp[5] * y + vp[9] * z + vp[13]) / w * 0.5) * Hh;
        const k = i * 3;
        if (tr2) {
          const ex = qx - tx, ey = qy - ty, d2 = ex * ex + ey * ey;
          if (d2 < tr2) {
            const ff = (1 - d2 / tr2) ** 2;
            if (F[i] <= 0) { V[k] = V[k + 1] = V[k + 2] = 0; }
            // Threads are dragged along with the finger: velocity eases toward the
            // brush velocity instead of stacking an impulse every frame.
            const pull = ff * 0.35;
            V[k] += (wvx * 1.2 - V[k]) * pull + (Math.random() - 0.5) * 0.06 * ff;
            V[k + 1] += (wvy * 1.2 - V[k + 1]) * pull + (Math.random() - 0.5) * 0.06 * ff;
            V[k + 2] += (wvz * 1.2 - V[k + 2]) * pull + (Math.random() - 0.5) * 0.06 * ff;
            if (ff > F[i]) F[i] = ff;
          }
        }
        if (br2) {
          const ex = qx - bx, ey = qy - by, d2 = ex * ex + ey * ey;
          if (d2 < br2) {
            const ff = (1 - d2 / br2) ** 1.5;
            const l = Math.sqrt(d2) + 1e-3, ux = ex / l, uy = -ey / l, s = 1.6 * ff;
            const zr = (Math.random() - 0.5) * 0.8;
            if (F[i] <= 0) { V[k] = V[k + 1] = V[k + 2] = 0; }
            V[k] += (R[0] * ux + U[0] * uy + Fw[0] * zr) * s;
            V[k + 1] += (R[1] * ux + U[1] * uy + Fw[1] * zr) * s;
            V[k + 2] += (R[2] * ux + U[2] * uy + Fw[2] * zr) * s;
            if (ff > F[i]) F[i] = ff;
          }
        }
      }
      this.burst = null;
    }
    const t2 = performance.now();

    // ---- pass 3: smoke for torn particles, snap-to-home for the rest ---------
    const c = this._c;
    const freq = 0.6 + 1.8 * this.turb;
    const amp = (0.2 + 0.85 * this.turb) / 1.6;
    const ox = time * 0.05, oy = -time * 0.035, oz = time * 0.065;
    const relax = 1 - Math.exp(-dt * 2.2);
    const back = 1 - Math.exp(-dt * 12), backMorph = 1 - Math.exp(-dt * 3.5);
    const heal = this.heal;
    let torn = 0;
    for (let i = 0; i < n; i++) {
      const j = i * 4, k = i * 3;
      let f = F[i];
      if (f === 0) {                                   // attached: ride the attractor
        P[j] = H[k]; P[j + 1] = H[k + 1]; P[j + 2] = H[k + 2]; P[j + 3] = this.B[i];
        continue;
      }
      let x = P[j], y = P[j + 1], z = P[j + 2];
      if (f < 0) {                                     // returning: ease onto the moving home
        if (f < -MORPH_EASE) {                         // morph: still waiting for the wave
          F[i] = f + dt; P[j + 3] = this.B[i];
          continue;
        }
        const r = f < -0.36 ? backMorph : back;
        x += (H[k] - x) * r; y += (H[k + 1] - y) * r; z += (H[k + 2] - z) * r;
        f = Math.min(0, f + dt);
        P[j] = x; P[j + 1] = y; P[j + 2] = z; P[j + 3] = this.B[i];
        F[i] = f;
        continue;
      }
      torn++;
      curlNoise(x * freq + ox, y * freq + oy, z * freq + oz, c);
      const g = 1 - f, pull = 7 * g * g * g;
      const fa = amp * (0.35 + 0.65 * f);
      let vx = V[k], vy = V[k + 1], vz = V[k + 2];
      vx += (c[0] * fa + (H[k] - x) * pull - vx) * relax;
      vy += (c[1] * fa + (H[k + 1] - y) * pull - vy) * relax;
      vz += (c[2] * fa + (H[k + 2] - z) * pull - vz) * relax;
      V[k] = vx; V[k + 1] = vy; V[k + 2] = vz;
      P[j] = x + vx * dt; P[j + 1] = y + vy * dt; P[j + 2] = z + vz * dt;
      P[j + 3] = 0.7 + 4.5 * f * f;                    // freshly torn smoke burns hot, cools as it drifts
      f -= heal[i] * dt;
      F[i] = f > 0 ? f : -0.35;                        // start the short return phase
    }
    const t3 = performance.now();

    // ---- pass 5: cursor links ---------------------------------------------------
    this.stepLinks(E);
    const t4 = performance.now();

    this.torn = torn;
    this.first = false;
    const e = 0.1, m = this.ms;
    m.ode += (t1 - t0 - m.ode) * e;
    m.warp += (t1b - t1 - m.warp) * e;
    m.tear += (t2 - t1b - m.tear) * e;
    m.smoke += (t3 - t2 - m.smoke) * e;
    m.links += (t4 - t3 - m.links) * e;
  },

  // Link the cursor to K particles within a screen radius: among every LINK_STRIDE-th
  // particle, the K with the lowest hash priority. The priority is a bijection, so the
  // answer is a unique set, and bench/links.mjs checks that every search variant finds
  // exactly that set. Winners there: WASM walks the particles in priority order (SIMD)
  // and stops at the K-th hit. JS does the same walk for LINK_BUDGET probes, then falls
  // back to a full scan when the cursor is over a sparse area.
  stepLinks(E) {
    const L = this.lines;
    L.count = 0;
    const hv = this.hover;
    if (!this.linksOn || !hv) return;
    const cam = E.camera, vp = cam.vp, W = E.cssW, H = E.cssH;
    const P = this.P, idx = this.linkIdx, dd = this.linkD2;
    const R2 = LINK_RADIUS_PX * LINK_RADIUS_PX;
    let m;
    if (this.sim && this.sim.links) {
      const U = this.sim.U;
      U.set(vp, UNI.VP); U[UNI.SW] = W; U[UNI.SH] = H; U[UNI.LCX] = hv.x; U[UNI.LCY] = hv.y; U[UNI.LR2] = R2;
      m = this.sim.links();
      const out = this.sim.linkOut;
      for (let q = 0; q < m; q++) {
        const i = out[q], j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
        const w = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
        const ex = ((vp[0] * x + vp[4] * y + vp[8] * z + vp[12]) / w * 0.5 + 0.5) * W - hv.x;
        const ey = (0.5 - (vp[1] * x + vp[5] * y + vp[9] * z + vp[13]) / w * 0.5) * H - hv.y;
        idx[q] = i; dd[q] = Math.min(R2, ex * ex + ey * ey);
      }
    } else {
      m = this.linksOrdered(vp, W, H, hv.x, hv.y, R2);
      if (m < 0) m = this.linksScan(vp, W, H, hv.x, hv.y, R2);
    }
    if (!m) return;
    // Anchor the cursor in 3D at the average depth of the linked particles.
    const ray = cam.ray(hv.nx, hv.ny), e = ray.o, d = ray.d;
    let tSum = 0;
    for (let q = 0; q < m; q++) {
      const j = idx[q] * 4;
      tSum += (P[j] - e[0]) * d[0] + (P[j + 1] - e[1]) * d[1] + (P[j + 2] - e[2]) * d[2];
    }
    const t = tSum / m, ax = e[0] + d[0] * t, ay = e[1] + d[1] * t, az = e[2] + d[2] * t;
    const D = L.data;
    for (let q = 0; q < m; q++) {
      const j = idx[q] * 4, f = 1 - Math.sqrt(dd[q] / R2), a = f * f, o = q * 8;
      D[o] = ax; D[o + 1] = ay; D[o + 2] = az; D[o + 3] = a * 0.4;
      D[o + 4] = P[j]; D[o + 5] = P[j + 1]; D[o + 6] = P[j + 2]; D[o + 7] = a;
      P[j + 3] *= 1 + 1.5 * a;                            // linked particles glow a little
    }
    L.count = m;
  },

  // Priority-ordered walk over at most LINK_BUDGET candidates; -1 if it didn't find K.
  linksOrdered(vp, W, H, cx, cy, R2) {
    const n = this.count, P = this.P, idx = this.linkIdx, dd = this.linkD2;
    if (!this.linkOrder || this.linkOrderN !== n) {
      // Sort the priorities, then map back to indices with the multiplicative inverse (no comparator, no objects).
      const cnt = Math.ceil(n / LINK_STRIDE), keys = new Uint32Array(cnt);
      for (let q = 0; q < cnt; q++) keys[q] = Math.imul(q * LINK_STRIDE, 2654435761) >>> 0;
      keys.sort();
      for (let q = 0; q < cnt; q++) keys[q] = Math.imul(keys[q], LINK_INV) >>> 0;
      this.linkOrder = new Int32Array(keys.buffer); this.linkOrderN = n;
    }
    const O = this.linkOrder, e = Math.min(LINK_BUDGET, O.length);
    const a0 = vp[0], a1 = vp[1], a3 = vp[3], a4 = vp[4], a5 = vp[5], a7 = vp[7], a8 = vp[8], a9 = vp[9], a11 = vp[11], a12 = vp[12], a13 = vp[13], a15 = vp[15];
    let m = 0;
    for (let q = 0; q < e; q++) {
      const i = O[q], j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
      const w = a3 * x + a7 * y + a11 * z + a15;
      if (w <= 0.05) continue;
      const ex = ((a0 * x + a4 * y + a8 * z + a12) / w * 0.5 + 0.5) * W - cx;
      const ey = (0.5 - (a1 * x + a5 * y + a9 * z + a13) / w * 0.5) * H - cy;
      const d2 = ex * ex + ey * ey;
      if (d2 < R2) { idx[m] = i; dd[m] = d2; if (++m === LINK_K) return m; }
    }
    return O.length <= LINK_BUDGET ? m : -1;
  },

  // Full scan with a small top-K list (the sparse-cursor fallback).
  linksScan(vp, W, H, cx, cy, R2) {
    const P = this.P, n = this.count, idx = this.linkIdx, dd = this.linkD2, pri = this.linkPri;
    let m = 0, worst = 0, worstD = -1;
    for (let i = 0; i < n; i += LINK_STRIDE) {
      const j = i * 4, x = P[j], y = P[j + 1], z = P[j + 2];
      const w = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
      if (w <= 0.05) continue;
      const ex = ((vp[0] * x + vp[4] * y + vp[8] * z + vp[12]) / w * 0.5 + 0.5) * W - cx;
      const ey = (0.5 - (vp[1] * x + vp[5] * y + vp[9] * z + vp[13]) / w * 0.5) * H - cy;
      const d2 = ex * ex + ey * ey;
      if (d2 >= R2) continue;
      // Rank by a fixed per-particle hash, not distance: the set spreads over the whole
      // radius (nearest-K clumps into one fan) and stays stable, so lines don't flicker.
      const pr = Math.imul(i, 2654435761) >>> 0;
      if (m < LINK_K) {
        idx[m] = i; dd[m] = d2; pri[m] = pr;
        if (pr > worstD) { worstD = pr; worst = m; }
        m++;
      } else if (pr < worstD) {
        idx[worst] = i; dd[worst] = d2; pri[worst] = pr;
        worstD = -1;
        for (let q = 0; q < LINK_K; q++) if (pri[q] > worstD) { worstD = pri[q]; worst = q; }
      }
    }
    return m;
  },

  readout() {
    const m = this.ms;
    if (this.backend === 'gpu') return `sim + render on the GPU (${this.E.renderer.gpuName || 'WebGPU'}) · host ${m.kernel.toFixed(2)} ms`;
    if (this.backend === 'wasm') return `torn ${(this.torn / this.count * 100).toFixed(0)}% · WASM SIMD kernel ${m.kernel.toFixed(1)} ms · links ${m.links.toFixed(1)} ms`;
    const links = this.linksOn ? ` · links ${m.links.toFixed(1)} (${this.lines.count})` : '';
    return `torn ${(this.torn / this.count * 100).toFixed(0)}% · attractor ${m.ode.toFixed(1)} · warp ${m.warp.toFixed(1)} · tear ${m.tear.toFixed(1)} · smoke ${m.smoke.toFixed(1)}${links} ms`;
  },
};
