// WebGPU path: simulation (the bench's WGSL kernel) and rendering on the GPU, in one
// command buffer per frame. Particles never leave GPU memory. The CPU writes a 384-byte
// sim uniform block and a 448-byte render uniform block per frame, and reads back only
// the 1-in-64 sample array (async, for auto-fit and the floor) and the link counter.

const GPU_SHADE = /* wgsl */`
fn pal(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0) * 4.0;
  let i = i32(min(floor(t), 3.0));
  let f = t - f32(i);
  let p0 = R.pal[max(i - 1, 0)].xyz; let p1 = R.pal[i].xyz; let p2 = R.pal[i + 1].xyz; let p3 = R.pal[min(i + 2, 4)].xyz;
  let c = 0.5 * (2.0 * p1 + (p2 - p0) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f + (3.0 * p1 - p0 - 3.0 * p2 + p3) * f * f * f);
  return max(c, vec3f(0.0));
}
fn shade(t: f32, pos: vec3f) -> vec3f {
  var s = t;
  if (R.p1.x > 0.5) { s = t + dot(pos, R.colorDir.xyz) * R.colorDir.w + R.p1.y; s = 0.5 - 0.5 * cos(3.14159265 * s); }
  return mix(pal(s), vec3f(2.6), R.misc2.y * (0.8 + 0.2 * sin(t * 37.0)));   // shape-change flash
}
fn fogOf(w: f32) -> f32 { return 1.0 - 0.75 * smoothstep(R.p0.z, R.p0.w, w); }
`;

const GPU_RENDER_STRUCT = /* wgsl */`
struct RU {
  vp: mat4x4f, invVP: mat4x4f, eye: vec4f, pal: array<vec4f, 5>, colorDir: vec4f,
  p0: vec4f,   // size, gain, fogNear, fogFar
  p1: vec4f,   // colorAnim, colorPhase, reflGain, reflFall
  p2: vec4f,   // floorY, floorOn, exposure, grain
  p3: vec4f,   // viewport w, h, lineGain, decay
  skyLow: vec4f, skyHigh: vec4f, floorTint: vec4f,   // .w: bloomOn, bloomStrength, bloomThreshold
  cursor: vec4f,  // css x, css y, radius (css px), active
  ray: vec4f,     // cursor ray dir xyz, dpr
  fwd: vec4f,     // camera forward xyz, time
  misc: vec4f,    // reflSize (device px), particle count, link threshold (0..1), css w
  misc2: vec4f,   // css h, flash, _, _
};
@group(0) @binding(0) var<uniform> R: RU;
`;

const GPU_PARTICLE_WGSL = GPU_RENDER_STRUCT + /* wgsl */`
@group(0) @binding(1) var<storage, read> P: array<vec4f>;
@group(0) @binding(2) var<storage, read> T: array<f32>;
` + GPU_SHADE + /* wgsl */`
override REFLECT: bool = false;
override QUAD: bool = false;
override SOFT: bool = false;
struct VO { @builtin(position) pos: vec4f, @location(0) c: vec3f, @location(1) uv: vec2f };

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var o: VO;
  let idx = select(vi, ii, QUAD);
  let p = P[idx];
  var w = p.xyz;
  var k = 1.0;
  if (REFLECT) {
    let h = w.y - R.p2.x;
    k = select(0.0, exp(-h * R.p1.w), h > 0.0);
    w.y = R.p2.x - h;
  }
  var c = R.vp * vec4f(w, 1.0);
  var gain = R.p0.y;
  if (REFLECT) { gain = gain * R.p1.z / (R.misc.x * R.misc.x * 0.35); }
  o.c = shade(T[idx], p.xyz) * p.w * gain * fogOf(c.w) * k;
  if (QUAD) {
    let corner = vec2f(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
    let size = select(R.p0.x, R.misc.x, REFLECT);
    c = vec4f(c.xy + corner * size / vec2f(R.p3.x, R.p3.y) * c.w, c.zw);
    o.uv = corner;
  } else { o.uv = vec2f(0.0); }
  o.pos = select(vec4f(2.0, 2.0, 2.0, 1.0), c, p.w > 0.0 && k > 0.002);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  var a = 1.0;
  if (SOFT) { a = max(0.0, 1.0 - dot(i.uv, i.uv)); a = a * a; }
  return vec4f(i.c * a, 1.0);
}
`;

const GPU_LINKS_WGSL = GPU_RENDER_STRUCT + /* wgsl */`
@group(0) @binding(1) var<storage, read> P: array<vec4f>;
struct LinkStats { n: atomic<u32>, placed: atomic<u32>, wsum: atomic<u32>, _pad: u32 };
@group(0) @binding(2) var<storage, read_write> LS: LinkStats;
@group(0) @binding(3) var<storage, read_write> LIST: array<u32>;
const K = ${LINK_K}u;
fn hsh(i: u32) -> u32 { var x = i * 0x9E3779B1u; x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
fn screen(p: vec3f) -> vec3f {
  let c = R.vp * vec4f(p, 1.0);
  return vec3f((c.x / c.w * 0.5 + 0.5) * R.misc.w, (0.5 - c.y / c.w * 0.5) * R.misc2.x, c.w);
}
// Candidates are a fixed hashed subset (stable from frame to frame), sized by the CPU so
// ~K fall inside the cursor radius; the first K found are kept.
@compute @workgroup_size(64) fn pick(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= u32(R.misc.y) || R.cursor.w < 0.5) { return; }
  if (f32(hsh(i) >> 8u) * (1.0 / 16777216.0) >= R.misc.z) { return; }
  let s = screen(P[i].xyz);
  if (s.z <= 0.05) { return; }
  let d = s.xy - R.cursor.xy;
  if (dot(d, d) >= R.cursor.z * R.cursor.z) { return; }
  let k = atomicAdd(&LS.n, 1u);
  if (k < K) { LIST[k] = i; atomicAdd(&LS.placed, 1u); atomicAdd(&LS.wsum, u32(s.z * 1024.0)); }
}
`;

const GPU_LINE_DRAW_WGSL = GPU_RENDER_STRUCT + /* wgsl */`
@group(0) @binding(1) var<storage, read> P: array<vec4f>;
struct LinkStats { n: u32, placed: u32, wsum: u32, _pad: u32 };
@group(0) @binding(2) var<storage, read> LS: LinkStats;
@group(0) @binding(3) var<storage, read> LIST: array<u32>;
` + GPU_SHADE + /* wgsl */`
struct VO { @builtin(position) pos: vec4f, @location(0) c: vec3f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  var o: VO;
  let link = vi >> 1u;
  let placed = min(LS.placed, ${LINK_K}u);
  if (link >= placed) { o.pos = vec4f(2.0, 2.0, 2.0, 1.0); o.c = vec3f(0.0); return o; }
  let p = P[LIST[link]].xyz;
  let cp = R.vp * vec4f(p, 1.0);
  let sx = (cp.x / cp.w * 0.5 + 0.5) * R.misc.w; let sy = (0.5 - cp.y / cp.w * 0.5) * R.misc2.x;
  let f = 1.0 - clamp(length(vec2f(sx, sy) - R.cursor.xy) / R.cursor.z, 0.0, 1.0);
  let a = f * f;
  let avgW = f32(LS.wsum) / 1024.0 / f32(placed);
  let anchor = R.eye.xyz + R.ray.xyz * (avgW / dot(R.ray.xyz, R.fwd.xyz));
  var w = p; var alpha = a;
  if ((vi & 1u) == 0u) { w = anchor; alpha = a * 0.4; }
  let c = R.vp * vec4f(w, 1.0);
  o.pos = c;
  o.c = shade(0.55, w) * alpha * R.p3.z * fogOf(c.w);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f { return vec4f(i.c, 1.0); }
`;

const GPU_FULLSCREEN = /* wgsl */`
struct FO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) vi: u32) -> FO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: FO; o.pos = vec4f(p[vi], 0.0, 1.0); o.uv = vec2f(p[vi].x * 0.5 + 0.5, 0.5 - p[vi].y * 0.5); return o;
}
`;

const GPU_FADE_WGSL = GPU_RENDER_STRUCT + GPU_FULLSCREEN + /* wgsl */`
@fragment fn fs() -> @location(0) vec4f { return vec4f(0.0, 0.0, 0.0, R.p3.w); }
`;

const GPU_BLOOM_WGSL = GPU_RENDER_STRUCT + GPU_FULLSCREEN + /* wgsl */`
@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var smp: sampler;
@group(1) @binding(2) var<uniform> texel: vec4f;   // 1/w, 1/h of the source
fn tap(uv: vec2f, o: vec2f) -> vec3f { return textureSampleLevel(src, smp, uv + texel.xy * o, 0.0).rgb; }
@fragment fn pre(i: FO) -> @location(0) vec4f {
  var c = (tap(i.uv, vec2f(-0.5, -0.5)) + tap(i.uv, vec2f(0.5, -0.5)) + tap(i.uv, vec2f(-0.5, 0.5)) + tap(i.uv, vec2f(0.5, 0.5))) * 0.25 * R.p2.z;
  let thr = R.floorTint.w;
  let l = max(c.r, max(c.g, c.b));
  let knee = thr * 0.5;
  var s = clamp(l - thr + knee, 0.0, 2.0 * knee);
  s = s * s / (4.0 * knee + 1e-4);
  return vec4f(c * max(s, l - thr) / max(l, 1e-4), 1.0);
}
@fragment fn down(i: FO) -> @location(0) vec4f {
  let c = tap(i.uv, vec2f(0.0)) * 4.0 + tap(i.uv, vec2f(-1.0, -1.0)) + tap(i.uv, vec2f(1.0, -1.0)) + tap(i.uv, vec2f(-1.0, 1.0)) + tap(i.uv, vec2f(1.0, 1.0));
  return vec4f(c / 8.0, 1.0);
}
@fragment fn up(i: FO) -> @location(0) vec4f {
  var c = tap(i.uv, vec2f(-2.0, 0.0)) + tap(i.uv, vec2f(2.0, 0.0)) + tap(i.uv, vec2f(0.0, -2.0)) + tap(i.uv, vec2f(0.0, 2.0));
  c += (tap(i.uv, vec2f(-1.0, -1.0)) + tap(i.uv, vec2f(1.0, -1.0)) + tap(i.uv, vec2f(-1.0, 1.0)) + tap(i.uv, vec2f(1.0, 1.0))) * 2.0;
  return vec4f(c / 12.0, 1.0);
}
`;

const GPU_TONE_WGSL = GPU_RENDER_STRUCT + GPU_FULLSCREEN + /* wgsl */`
@group(1) @binding(0) var accum: texture_2d<f32>;
@group(1) @binding(1) var bloomTex: texture_2d<f32>;
@group(1) @binding(2) var smp: sampler;
fn aces(x: vec3f) -> vec3f { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0)); }
fn hash2(p0: vec2f) -> f32 { var p = fract(p0 * vec2f(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
@fragment fn fs(i: FO) -> @location(0) vec4f {
  let ndc = vec2f(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0);
  var a = R.invVP * vec4f(ndc, -1.0, 1.0); a /= a.w;
  var b = R.invVP * vec4f(ndc, 1.0, 1.0); b /= b.w;
  let dir = normalize(b.xyz - a.xyz);
  var bg = mix(R.skyLow.xyz, R.skyHigh.xyz, smoothstep(-0.05, 0.55, dir.y));
  bg += R.skyLow.xyz * 0.6 * exp(-abs(dir.y) * 22.0);
  // floor (evaluated unconditionally so fwidth stays in uniform control flow)
  let t = (R.p2.x - R.eye.y) / min(dir.y, -1e-4);
  let hit = R.eye.xyz + dir * t;
  let r = length(hit.xz);
  var fl = R.floorTint.xyz * (0.35 + 1.6 * exp(-r * r * 0.22));
  let gp = hit.xz * 1.25;
  let g = abs(fract(gp - 0.5) - 0.5) / max(fwidth(gp), vec2f(1e-4));
  let line = 1.0 - min(min(g.x, g.y), 1.0);
  fl += R.floorTint.xyz * line * 0.9 * exp(-r * 0.45);
  let fogF = exp(-max(t - 2.0, 0.0) * 0.12);
  if (R.p2.y > 0.5 && dir.y < -1e-4) { bg = mix(bg, fl, fogF); }

  let dims = vec2f(textureDimensions(accum));
  var hdr = textureLoad(accum, vec2i(i.uv * dims), 0).rgb * R.p2.z;
  if (R.skyLow.w > 0.5) { hdr += textureSampleLevel(bloomTex, smp, i.uv, 0.0).rgb * R.skyHigh.w; }
  var q = i.uv - 0.5; q.x *= R.p3.x / R.p3.y;
  let vig = clamp(1.0 - dot(q, q) * 1.1, 0.0, 1.0);
  var col = aces(hdr) + bg;
  col = pow(col, vec3f(1.0 / 2.2));
  col *= mix(0.72, 1.0, vig);
  col += (hash2(i.pos.xy + fract(R.fwd.w * 7.13) * 311.0) - 0.5) * R.p2.w;
  return vec4f(col, 1.0);
}
`;

const SEED_WGSL = `
override N: u32;
struct Par { jit: f32, salt: u32, len: u32, pad: u32 }
@group(0) @binding(0) var<uniform> par: Par;
@group(0) @binding(1) var<storage, read> ORB: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
fn rnd(i: u32, axis: u32) -> f32 {       // lowbias32, same as seedRnd() in unravel.js
  var x = (i * 0x9E3779B1u) ^ ((0x7ffffff0u + axis) * 0x85EBCA77u) ^ (par.salt * 0xC2B2AE3Du);
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return f32(x >> 8u) * (1.0 / 16777216.0);
}
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let o = (i & (par.len - 1u)) * 3u;
  for (var a = 0u; a < 3u; a++) { S[i * 3u + a] = ORB[o + a] + (rnd(i, a) - 0.5) * par.jit; }
}`;

class GpuBackend {
  static async create(canvas) {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10) },
    });
    // Test hook: headless Linux Chromium loses the device as soon as a canvas presents, so the
    // test harness renders into an offscreen texture instead and reads pixels back.
    const ctx = window.__GPU_OFFSCREEN ? { canvas, offscreen: true } : canvas.getContext('webgpu');
    if (!ctx) throw new Error('No WebGPU canvas context');
    const b = new GpuBackend(device, ctx, adapter);
    await b.checkShaders();
    return b;
  }

  constructor(device, ctx, adapter) {
    this.kind = 'gpu';
    this.device = device; this.ctx = ctx; this.adapter = adapter;   // keep the adapter alive (Dawn loses the device if its instance is collected)
    this.gpuName = [adapter.info?.vendor, adapter.info?.architecture].filter(Boolean).join(' ');
    this.format = ctx.offscreen ? 'rgba8unorm' : navigator.gpu.getPreferredCanvasFormat();
    if (!ctx.offscreen) ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.hdr = true; this.bloomSupported = true;
    this.palette = new Float32Array(15); this.paletteTarget = new Float32Array(15);
    this.w = 0; this.h = 0;
    this.ru = new Float32Array(112);          // 448 bytes of render uniforms
    this.ruBuf = device.createBuffer({ size: 448, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.lost = false;
    device.lost.then((info) => {
      this.lost = true;
      if (info.reason !== 'destroyed') { console.error('WebGPU device lost:', info.message); window.dispatchEvent(new CustomEvent('unravel-gpu-lost')); }
    });
    let reported = 0;
    device.addEventListener('uncapturederror', (e) => { if (reported++ < 5) console.error('WebGPU error:', e.error.message); });
    this.buildPipelines();
  }

  async checkShaders() {
    for (const m of this.modules) {
      const info = await m.getCompilationInfo();
      const errs = info.messages.filter((x) => x.type === 'error');
      if (errs.length) throw new Error('WGSL: ' + errs.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '));
    }
  }

  buildPipelines() {
    const d = this.device;
    const mod = (code) => d.createShaderModule({ code });
    const mParticle = mod(GPU_PARTICLE_WGSL), mLinks = mod(GPU_LINKS_WGSL), mLine = mod(GPU_LINE_DRAW_WGSL);
    const mFade = mod(GPU_FADE_WGSL), mBloom = mod(GPU_BLOOM_WGSL), mTone = mod(GPU_TONE_WGSL);
    this.modules = [mParticle, mLinks, mLine, mFade, mBloom, mTone];
    const HDR = 'rgba16float';
    const add = { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } };
    const fade = { color: { srcFactor: 'zero', dstFactor: 'src-alpha' }, alpha: { srcFactor: 'zero', dstFactor: 'src-alpha' } };
    const auto = 'auto';
    const particle = (constants, topology) => d.createRenderPipeline({
      layout: auto, vertex: { module: mParticle, entryPoint: 'vs', constants },
      fragment: { module: mParticle, entryPoint: 'fs', constants, targets: [{ format: HDR, blend: add }] },
      primitive: { topology },
    });
    this.pPoints = particle({ REFLECT: 0, QUAD: 0, SOFT: 0 }, 'point-list');
    this.pQuads = particle({ REFLECT: 0, QUAD: 1, SOFT: 1 }, 'triangle-strip');   // matches WebGL: soft discs above 1.6 px
    this.pReflect = particle({ REFLECT: 1, QUAD: 1, SOFT: 1 }, 'triangle-strip');
    this.pLine = d.createRenderPipeline({
      layout: auto, vertex: { module: mLine, entryPoint: 'vs' },
      fragment: { module: mLine, entryPoint: 'fs', targets: [{ format: HDR, blend: add }] }, primitive: { topology: 'line-list' },
    });
    this.pFade = d.createRenderPipeline({
      layout: auto, vertex: { module: mFade, entryPoint: 'vs' },
      fragment: { module: mFade, entryPoint: 'fs', targets: [{ format: HDR, blend: fade }] },
    });
    const bloomPipe = (entry, blend) => d.createRenderPipeline({
      layout: auto, vertex: { module: mBloom, entryPoint: 'vs' },
      fragment: { module: mBloom, entryPoint: entry, targets: [{ format: HDR, blend }] },
    });
    this.pPre = bloomPipe('pre'); this.pDown = bloomPipe('down'); this.pUp = bloomPipe('up', add);
    this.pTone = d.createRenderPipeline({
      layout: auto, vertex: { module: mTone, entryPoint: 'vs' },
      fragment: { module: mTone, entryPoint: 'fs', targets: [{ format: this.format }] },
    });
    this.pPick = d.createComputePipeline({ layout: auto, compute: { module: mLinks, entryPoint: 'pick' } });
    // links scratch
    this.linkStats = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.linkList = d.createBuffer({ size: LINK_K * 4, usage: GPUBufferUsage.STORAGE });
    this.linkThreshold = 1 / 16;
  }

  // ---------------------------------------------------------------- simulation
  createSim(n) {
    const d = this.device, self = this;
    if (this.sim) this.sim.destroy();
    const module = d.createShaderModule({ code: KERNEL_WGSL });
    const G = WARP_KERNEL_G, cellF = Math.fround(Math.fround(4.4) / (G - 1));
    const constants = { W_CELL: cellF, W_INV: Math.fround(1 / cellF), W_GMAX: Math.fround(Math.fround(G) - Math.fround(1.001)), N: n };
    const pipe = (entryPoint) => d.createComputePipeline({ layout: 'auto', compute: { module, entryPoint, constants } });
    const pGrid = pipe('grid'), pStep = pipe('step'), pMorph = pipe('morph');
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mk = (bytes) => d.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: S });
    const nSS = Math.ceil(n / 64) * 5;
    const buf = { U: mk(UNI.SIZE * 4), S: mk(n * 12), P: mk(n * 16), V: mk(n * 12), F: mk(n * 4), T: mk(n * 4), heal: mk(n * 4), SS: mk(nSS * 4), W: mk(G * G * G * 12), TB: mk(560 * 4) };
    const tables = new Uint32Array(560);
    tables.set(NOISE_PERM, 0);
    new Float32Array(tables.buffer, 512 * 4, 48).set(NOISE_GRAD);
    d.queue.writeBuffer(buf.TB, 0, tables);
    const order = ['U', 'S', 'P', 'V', 'F', 'T', 'heal', 'SS', 'W', 'TB'];
    // 'auto' layouts only contain the bindings an entry point uses, so each gets its own group
    const bindUsed = (p, used) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: used.map((k) => ({ binding: order.indexOf(k), resource: { buffer: buf[k] } })) });
    const bStep = bindUsed(pStep, order);
    const bGrid = bindUsed(pGrid, ['U', 'W', 'TB']);
    const bMorph = bindUsed(pMorph, ['U', 'P', 'V', 'F']);
    // Seed pass: S[i] = orbit[i mod L] + hash jitter (replaces a JS loop + a 12n-byte upload per shape change).
    const pSeed = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: SEED_WGSL }), entryPoint: 'seed', constants: { N: n } } });
    const seedPar = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const orbitBufs = new Map();                // orbit array -> GPU buffer (4 shapes x 1.5 MB, uploaded once each)
    const seedBinds = new Map();
    // sample readback ring (SS + link stats), mapped asynchronously
    const ring = [0, 1, 2].map(() => ({ buf: d.createBuffer({ size: nSS * 4 + 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false }));
    const latest = new Float32Array(nSS);
    const U = new Float32Array(UNI.SIZE), UI = new Int32Array(U.buffer);
    const sim = {
      kind: 'gpu', n, U, UI, buf, torn: null, hasSamples: false, linkCount: 0,
      P: null, T: null,                      // stays on the GPU
      pendingMorph: null, stepPending: false,
      load(init) {
        if (init.S) d.queue.writeBuffer(buf.S, 0, init.S);
        d.queue.writeBuffer(buf.heal, 0, init.heal);
        this.pendingClear = true;                 // P, V, F, T zeroed on the GPU (no 16n-byte zero upload)
        this.hasSamples = false;
      },
      reseed(S) { d.queue.writeBuffer(buf.S, 0, S); },
      seed(orbit, jit, salt) {
        let ob = orbitBufs.get(orbit);
        if (!ob) {
          ob = d.createBuffer({ size: orbit.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          d.queue.writeBuffer(ob, 0, orbit);
          orbitBufs.set(orbit, ob);
          seedBinds.set(orbit, d.createBindGroup({ layout: pSeed.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: seedPar } }, { binding: 1, resource: { buffer: ob } }, { binding: 2, resource: { buffer: buf.S } }] }));
        }
        const par = new ArrayBuffer(16);
        new Float32Array(par, 0, 1)[0] = jit; new Uint32Array(par, 4, 3).set([salt, orbit.length / 3, 0]);
        d.queue.writeBuffer(seedPar, 0, par);
        this.pendingSeed = seedBinds.get(orbit);
      },
      morph(top, span) { this.pendingMorph = [top, span]; },
      step() { this.stepPending = true; },
      samples() { return this.hasSamples ? latest : null; },
      encode(enc) {                           // called by the renderer inside the frame's command buffer
        if (this.pendingMorph) {
          U[UNI.MORPH_TOP] = this.pendingMorph[0]; U[UNI.MORPH_SPAN] = this.pendingMorph[1];
        }
        d.queue.writeBuffer(buf.U, 0, U);
        if (this.pendingClear) { for (const k of ['P', 'V', 'F', 'T']) enc.clearBuffer(buf[k]); this.pendingClear = false; }
        const pass = enc.beginComputePass();
        if (this.pendingSeed) { pass.setPipeline(pSeed); pass.setBindGroup(0, this.pendingSeed); pass.dispatchWorkgroups(Math.ceil(n / 64)); this.pendingSeed = null; }
        if (this.pendingMorph) { pass.setPipeline(pMorph); pass.setBindGroup(0, bMorph); pass.dispatchWorkgroups(Math.ceil(n / 64)); this.pendingMorph = null; }
        if (this.stepPending) {
          pass.setPipeline(pGrid); pass.setBindGroup(0, bGrid); pass.dispatchWorkgroups(Math.ceil(G * G * G / 64));
          pass.setPipeline(pStep); pass.setBindGroup(0, bStep); pass.dispatchWorkgroups(Math.ceil(n / 64));
          this.stepPending = false;
        }
        pass.end();
      },
      readback(enc) {                         // copy samples + link stats into a free staging buffer
        const slot = ring.find((r) => !r.busy);
        if (!slot) return null;
        enc.copyBufferToBuffer(buf.SS, 0, slot.buf, 0, nSS * 4);
        enc.copyBufferToBuffer(self.linkStats, 0, slot.buf, nSS * 4, 16);
        slot.busy = true;
        return slot;
      },
      afterSubmit(slot) {
        if (!slot) return;
        slot.buf.mapAsync(GPUMapMode.READ).then(() => {
          const r = slot.buf.getMappedRange();
          latest.set(new Float32Array(r, 0, nSS));
          sim.linkCount = new Uint32Array(r, nSS * 4, 4)[0];
          slot.buf.unmap(); slot.busy = false; sim.hasSamples = true;
        }).catch(() => { slot.busy = false; });
      },
      destroy() { for (const b of orbitBufs.values()) b.destroy(); seedPar.destroy(); for (const k in buf) buf[k].destroy(); for (const r of ring) r.buf.destroy(); },
    };
    this.sim = sim;
    this.particleBinds = null;
    return sim;
  }

  // ---------------------------------------------------------------- renderer API (matches the WebGL Renderer)
  resize(w, h) {
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.ctx.canvas.width = w; this.ctx.canvas.height = h;
    const d = this.device;
    for (const t of [this.accum, ...(this.levels || []).map((l) => l.tex)]) if (t) t.destroy();
    const tex = (tw, th) => d.createTexture({ size: [tw, th], format: 'rgba16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.accum = tex(w, h);
    this.levels = [];
    let bw = w, bh = h;
    for (let i = 0; i < 6; i++) {
      bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
      const t = tex(bw, bh);
      const texel = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.levels.push({ tex: t, view: t.createView(), w: bw, h: bh, texel });
    }
    this.accumTexel = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.accumTexel, 0, new Float32Array([1 / w, 1 / h, 0, 0]));
    for (const l of this.levels) d.queue.writeBuffer(l.texel, 0, new Float32Array([1 / l.w, 1 / l.h, 0, 0]));
    this.accumView = this.accum.createView();
    if (this.ctx.offscreen) {
      if (this.offTex) this.offTex.destroy();
      this.offTex = d.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    }
    this.sizeBinds = null;
    this.needClear = true;
  }

  clear() { this.needClear = true; }

  // Test hook: read the offscreen frame back as RGBA bytes.
  async readPixels() {
    const d = this.device, bpr = Math.ceil(this.w * 4 / 256) * 256;
    const buf = d.createBuffer({ size: bpr * this.h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.offTex }, { buffer: buf, bytesPerRow: bpr }, [this.w, this.h]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange()), out = new Uint8ClampedArray(this.w * this.h * 4);
    for (let y = 0; y < this.h; y++) out.set(src.subarray(y * bpr, y * bpr + this.w * 4), y * this.w * 4);
    buf.unmap(); buf.destroy();
    return { w: this.w, h: this.h, data: out };
  }
  upload() {}                                  // particles are already on the GPU
  uploadLines() {}

  setPalette(hexes, immediate) {
    for (let i = 0; i < 5; i++) {
      const c = hexLin(hexes[i]);
      this.paletteTarget[i * 3] = c[0]; this.paletteTarget[i * 3 + 1] = c[1]; this.paletteTarget[i * 3 + 2] = c[2];
    }
    if (immediate) this.palette.set(this.paletteTarget);
  }
  tickPalette(dt, rate) {
    const a = 1 - Math.exp(-dt * rate);
    for (let i = 0; i < 15; i++) this.palette[i] += (this.paletteTarget[i] - this.palette[i]) * a;
  }

  makeBinds() {
    const d = this.device, sim = this.sim, ub = { buffer: this.ruBuf };
    const pb = (p) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: ub }, { binding: 1, resource: { buffer: sim.buf.P } }, { binding: 2, resource: { buffer: sim.buf.T } }] });
    this.bPoints = pb(this.pPoints); this.bQuads = pb(this.pQuads); this.bReflect = pb(this.pReflect);
    this.bPick = d.createBindGroup({ layout: this.pPick.getBindGroupLayout(0), entries: [{ binding: 0, resource: ub }, { binding: 1, resource: { buffer: sim.buf.P } }, { binding: 2, resource: { buffer: this.linkStats } }, { binding: 3, resource: { buffer: this.linkList } }] });
    this.bLine = d.createBindGroup({ layout: this.pLine.getBindGroupLayout(0), entries: [{ binding: 0, resource: ub }, { binding: 1, resource: { buffer: sim.buf.P } }, { binding: 2, resource: { buffer: this.linkStats } }, { binding: 3, resource: { buffer: this.linkList } }] });
    this.bFade = d.createBindGroup({ layout: this.pFade.getBindGroupLayout(0), entries: [{ binding: 0, resource: ub }] });
    this.particleBinds = sim;
  }

  makeSizeBinds() {
    const d = this.device, ub = { buffer: this.ruBuf };
    const g0 = (p) => d.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [{ binding: 0, resource: ub }] });
    const g1 = (p, view, texel) => d.createBindGroup({ layout: p.getBindGroupLayout(1), entries: [{ binding: 0, resource: view }, { binding: 1, resource: this.sampler }, { binding: 2, resource: { buffer: texel } }] });
    const L = this.levels;
    this.bloomBinds = {
      pre0: g0(this.pPre),
      down0: d.createBindGroup({ layout: this.pDown.getBindGroupLayout(0), entries: [] }),
      up0: d.createBindGroup({ layout: this.pUp.getBindGroupLayout(0), entries: [] }),
      pre: g1(this.pPre, this.accumView, this.accumTexel),
      down: L.map((l, i) => (i ? g1(this.pDown, L[i - 1].view, L[i - 1].texel) : null)),
      up: L.map((l, i) => (i ? g1(this.pUp, l.view, l.texel) : null)),
    };
    this.bTone0 = g0(this.pTone);
    this.bTone1 = d.createBindGroup({ layout: this.pTone.getBindGroupLayout(1), entries: [{ binding: 0, resource: this.accumView }, { binding: 1, resource: L[0].view }, { binding: 2, resource: this.sampler }] });
    this.sizeBinds = true;
  }

  writeUniforms(o) {
    const u = this.ru;
    u.set(o.vp, 0);
    mat4Invert(this.invTmp || (this.invTmp = new Float32Array(16)), o.vp); u.set(this.invTmp, 16);
    u[32] = o.eye[0]; u[33] = o.eye[1]; u[34] = o.eye[2]; u[35] = o.time;
    for (let i = 0; i < 5; i++) { u[36 + i * 4] = this.palette[i * 3]; u[37 + i * 4] = this.palette[i * 3 + 1]; u[38 + i * 4] = this.palette[i * 3 + 2]; u[39 + i * 4] = 0; }
    const ca = o.colorAnim;
    u[56] = ca ? ca.dir[0] : 0; u[57] = ca ? ca.dir[1] : 0; u[58] = ca ? ca.dir[2] : 0; u[59] = ca ? ca.scale : 0;
    u[60] = o.size; u[61] = o.gain; u[62] = o.fogNear; u[63] = o.fogFar;
    u[64] = ca ? 1 : 0; u[65] = ca ? ca.phase : 0; u[66] = o.floor ? o.floor.reflGain : 0; u[67] = o.floor ? o.floor.reflFall : 1;
    u[68] = o.floor ? o.floor.y : 0; u[69] = o.floor ? 1 : 0; u[70] = o.exposure; u[71] = o.grain;
    u[72] = this.w; u[73] = this.h; u[74] = o.lineGain; u[75] = o.decay;
    const bloomOn = o.bloom && o.bloom.on;
    u[76] = o.sky.low[0]; u[77] = o.sky.low[1]; u[78] = o.sky.low[2]; u[79] = bloomOn ? 1 : 0;
    u[80] = o.sky.high[0]; u[81] = o.sky.high[1]; u[82] = o.sky.high[2]; u[83] = bloomOn ? o.bloom.strength : 0;
    u[84] = o.sky.floor[0]; u[85] = o.sky.floor[1]; u[86] = o.sky.floor[2]; u[87] = o.bloom ? o.bloom.threshold : 1;
    const cur = o.cursor;
    u[88] = cur ? cur.x : 0; u[89] = cur ? cur.y : 0; u[90] = LINK_RADIUS_PX; u[91] = cur ? 1 : 0;
    u[92] = cur ? cur.ray[0] : 0; u[93] = cur ? cur.ray[1] : 0; u[94] = cur ? cur.ray[2] : 1; u[95] = o.dpr;
    u[96] = o.fwd[0]; u[97] = o.fwd[1]; u[98] = o.fwd[2]; u[99] = o.time;
    u[100] = o.size * 2.2; u[101] = this.sim.n; u[102] = this.linkThreshold; u[103] = o.cssW;
    u[104] = o.cssH; u[105] = o.flash || 0;
    this.device.queue.writeBuffer(this.ruBuf, 0, u);
  }

  draw(o) {
    if (this.lost || !this.sim) return;
    const d = this.device, sim = this.sim;
    if (this.particleBinds !== sim) this.makeBinds();
    if (!this.sizeBinds) this.makeSizeBinds();
    // Link candidates: keep ~LINK_K hashed candidates inside the cursor radius (adaptive).
    if (o.cursor && sim.hasSamples) {
      const c = sim.linkCount;
      const ratio = c > 0 ? LINK_K / c : 2;
      this.linkThreshold = Math.min(1, Math.max(1e-4, this.linkThreshold * Math.pow(ratio, 0.3)));
    }
    this.writeUniforms(o);
    const enc = d.createCommandEncoder();
    sim.encode(enc);
    if (o.cursor) {
      enc.clearBuffer(this.linkStats);
      const cp = enc.beginComputePass();
      cp.setPipeline(this.pPick); cp.setBindGroup(0, this.bPick); cp.dispatchWorkgroups(Math.ceil(sim.n / 64));
      cp.end();
    } else enc.clearBuffer(this.linkStats);

    // accumulation: fade (trails) or clear, then particles, reflection, links
    const clear = this.needClear || o.decay <= 0;
    this.needClear = false;
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.accumView, loadOp: clear ? 'clear' : 'load', clearValue: [0, 0, 0, 0], storeOp: 'store' }] });
    if (!clear && o.decay < 1) { pass.setPipeline(this.pFade); pass.setBindGroup(0, this.bFade); pass.draw(3); }
    if (o.size > 1.6) { pass.setPipeline(this.pQuads); pass.setBindGroup(0, this.bQuads); pass.draw(4, sim.n); }
    else { pass.setPipeline(this.pPoints); pass.setBindGroup(0, this.bPoints); pass.draw(sim.n); }
    if (o.floor) { pass.setPipeline(this.pReflect); pass.setBindGroup(0, this.bReflect); pass.draw(4, sim.n); }
    if (o.cursor) { pass.setPipeline(this.pLine); pass.setBindGroup(0, this.bLine); pass.draw(LINK_K * 2); }
    pass.end();

    // bloom chain
    const B = this.bloomBinds, L = this.levels;
    if (o.bloom && o.bloom.on) {
      const rp = (view, load, pipe, g1) => {
        const p = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: load ? 'load' : 'clear', clearValue: [0, 0, 0, 0], storeOp: 'store' }] });
        p.setPipeline(pipe); p.setBindGroup(0, pipe === this.pPre ? B.pre0 : pipe === this.pDown ? B.down0 : B.up0); p.setBindGroup(1, g1); p.draw(3); p.end();
      };
      rp(L[0].view, false, this.pPre, B.pre);
      for (let i = 1; i < L.length; i++) rp(L[i].view, false, this.pDown, B.down[i]);
      for (let i = L.length - 1; i > 0; i--) rp(L[i - 1].view, true, this.pUp, B.up[i]);
    }

    // tone map + sky/floor to the canvas
    const target = this.ctx.offscreen ? this.offTex.createView() : this.ctx.getCurrentTexture().createView();
    const tp = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: 'clear', clearValue: [0, 0, 0, 1], storeOp: 'store' }] });
    tp.setPipeline(this.pTone); tp.setBindGroup(0, this.bTone0); tp.setBindGroup(1, this.bTone1); tp.draw(3); tp.end();

    const slot = sim.readback(enc);
    d.queue.submit([enc.finish()]);
    sim.afterSubmit(slot);
  }
}
