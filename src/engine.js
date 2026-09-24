// Engine: WebGL2 renderer (HDR accumulation, trails, reflection, links, sky + floor,
// tone map), frame loop with timing, input, ghost demo and HUD.
// The WebGPU path has its own renderer (gpu/backend.js) with the same draw() interface.
// Simulation lives in the mode (modes/unravel.js), on one of three backends.

const AUTO_WINDOW = 30, AUTO_MIN_FRAME_MS = 1000 / 120;   // auto particle count
const fmtCount = (n) => (n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : `${Math.round(n / 1e3)}k`);

// ---------------------------------------------------------------------------
// Shaders

// Shared colour code: Catmull-Rom palette + optional gradient animated over
// space and time (computed on the GPU; the CPU only supplies a per-point offset).
const GLSL_SHADE = `
uniform vec3 uPal[5];
uniform float uColorAnim;
uniform vec3 uColorDir;
uniform float uColorScale;
uniform float uColorPhase;
uniform float uFlash;              // shape-change flash: 0..1, pushes colour to HDR white
vec3 pal(float t) {
  t = clamp(t, 0.0, 1.0) * 4.0;
  int i = int(min(floor(t), 3.0));
  float f = t - float(i);
  vec3 p0 = uPal[max(i - 1, 0)], p1 = uPal[i], p2 = uPal[i + 1], p3 = uPal[min(i + 2, 4)];
  vec3 c = 0.5 * (2.0 * p1 + (p2 - p0) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f
                  + (3.0 * p1 - p0 - 3.0 * p2 + p3) * f * f * f);
  return max(c, vec3(0.0));
}
vec3 shade(float t, vec3 pos) {
  float s = t;
  if (uColorAnim > 0.5) {
    s = t + dot(pos, uColorDir) * uColorScale + uColorPhase;
    s = 0.5 - 0.5 * cos(3.14159265 * s);   // smooth ping-pong: seamless while scrolling
  }
  return mix(pal(s), vec3(2.6), uFlash * (0.8 + 0.2 * sin(t * 37.0)));
}`;

const GLSL_FULLSCREEN_VS = `#version 300 es
const vec2 P[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUv;
void main() { vec2 p = P[gl_VertexID]; vUv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;

const GLSL_FADE_FS = `#version 300 es
precision highp float;
uniform float uDecay;
out vec4 o;
void main() { o = vec4(0.0, 0.0, 0.0, uDecay); }`;

const GLSL_POINTS_VS = `#version 300 es
layout(location = 0) in vec4 aP;   // xyz world, w brightness
layout(location = 1) in float aT;  // per-point colour offset
uniform mat4 uVP;
uniform float uSize;
uniform float uGain;
uniform vec2 uFog;
uniform float uReflect;            // 1 = draw the mirror image below the floor
uniform float uFloorY;
uniform float uReflFall;
out vec3 vC;
${GLSL_SHADE}
void main() {
  vec3 p = aP.xyz;
  float k = 1.0;
  if (uReflect > 0.5) {
    float h = p.y - uFloorY;
    k = h > 0.0 ? exp(-h * uReflFall) : 0.0;   // reflection fades with height above the floor
    p.y = uFloorY - h;
  }
  vec4 c = uVP * vec4(p, 1.0);
  float fog = 1.0 - 0.75 * smoothstep(uFog.x, uFog.y, c.w);
  vC = shade(aT, aP.xyz) * aP.w * uGain * fog * k;
  gl_Position = (aP.w > 0.0 && k > 0.002) ? c : vec4(2.0, 2.0, 2.0, 1.0);
  gl_PointSize = uSize;
}`;

const GLSL_POINTS_FS = `#version 300 es
precision highp float;
in vec3 vC;
uniform float uSoft;
out vec4 o;
void main() {
  float a = 1.0;
  if (uSoft > 0.5) { vec2 d = gl_PointCoord * 2.0 - 1.0; a = max(0.0, 1.0 - dot(d, d)); a *= a; }
  o = vec4(vC * a, 1.0);
}`;

const GLSL_LINES_VS = `#version 300 es
layout(location = 0) in vec4 aL;   // xyz world, w alpha
uniform mat4 uVP;
uniform float uGain;
uniform vec2 uFog;
out vec3 vC;
${GLSL_SHADE}
void main() {
  vec4 c = uVP * vec4(aL.xyz, 1.0);
  float fog = 1.0 - 0.75 * smoothstep(uFog.x, uFog.y, c.w);
  vC = shade(0.55, aL.xyz) * aL.w * uGain * fog;
  gl_Position = c;
}`;

const GLSL_LINES_FS = `#version 300 es
precision highp float;
in vec3 vC;
out vec4 o;
void main() { o = vec4(vC, 1.0); }`;

// Bloom: bright-pass prefilter, then a dual-filter (Kawase-style) down/up chain on
// half-float targets. Only enabled when float render targets exist.
const GLSL_BLOOM_PRE_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexel;        // source texel size
uniform float uExposure;
uniform float uThreshold;
out vec4 o;
void main() {
  vec3 c = texture(uTex, vUv + uTexel * vec2(-0.5, -0.5)).rgb + texture(uTex, vUv + uTexel * vec2(0.5, -0.5)).rgb
         + texture(uTex, vUv + uTexel * vec2(-0.5, 0.5)).rgb + texture(uTex, vUv + uTexel * vec2(0.5, 0.5)).rgb;
  c *= 0.25 * uExposure;
  float l = max(c.r, max(c.g, c.b));
  float knee = uThreshold * 0.5;
  float s = clamp(l - uThreshold + knee, 0.0, 2.0 * knee);
  s = s * s / (4.0 * knee + 1e-4);
  o = vec4(c * max(s, l - uThreshold) / max(l, 1e-4), 1.0);
}`;

const GLSL_BLOOM_DOWN_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexel;
out vec4 o;
void main() {
  vec3 c = texture(uTex, vUv).rgb * 4.0;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, -1.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(1.0, -1.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, 1.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(1.0, 1.0)).rgb;
  o = vec4(c / 8.0, 1.0);
}`;

const GLSL_BLOOM_UP_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform float uWeight;
out vec4 o;
void main() {
  vec3 c = vec3(0.0);
  c += texture(uTex, vUv + uTexel * vec2(-2.0, 0.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(2.0, 0.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(0.0, -2.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(0.0, 2.0)).rgb;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, -1.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(1.0, -1.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(-1.0, 1.0)).rgb * 2.0;
  c += texture(uTex, vUv + uTexel * vec2(1.0, 1.0)).rgb * 2.0;
  o = vec4(c / 12.0 * uWeight, 1.0);
}`;

// Tone map + background. The background is a sky gradient and, below the horizon,
// a floor plane found by casting a ray per pixel (glow under the shape, faint grid).
const GLSL_TONEMAP_FS = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTex;
uniform sampler2D uBloom;
uniform float uBloomOn;
uniform float uBloomStrength;
uniform float uLdr;              // 1 = no float targets: clip to 0..1
uniform float uExposure;
uniform float uTime;
uniform float uGrain;
uniform vec2 uRes;
uniform mat4 uInvVP;
uniform vec3 uEye;
uniform float uFloorOn;
uniform float uFloorY;
uniform vec3 uSkyLow;
uniform vec3 uSkyHigh;
uniform vec3 uFloorTint;
out vec4 o;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 a = uInvVP * vec4(ndc, -1.0, 1.0); a /= a.w;
  vec4 b = uInvVP * vec4(ndc, 1.0, 1.0); b /= b.w;
  vec3 dir = normalize(b.xyz - a.xyz);

  vec3 bg = mix(uSkyLow, uSkyHigh, smoothstep(-0.05, 0.55, dir.y));
  bg += uSkyLow * 0.6 * exp(-abs(dir.y) * 22.0);          // soft horizon band
  if (uFloorOn > 0.5 && dir.y < -1e-4) {
    float t = (uFloorY - uEye.y) / dir.y;
    vec3 hit = uEye + dir * t;
    float r = length(hit.xz);
    vec3 fl = uFloorTint * (0.35 + 1.6 * exp(-r * r * 0.22));
    vec2 gp = hit.xz * 1.25;
    vec2 g = abs(fract(gp - 0.5) - 0.5) / max(fwidth(gp), vec2(1e-4));
    float line = 1.0 - min(min(g.x, g.y), 1.0);
    fl += uFloorTint * line * 0.9 * exp(-r * 0.45);
    float fog = exp(-max(t - 2.0, 0.0) * 0.12);
    bg = mix(bg, fl, fog);
  }

  vec3 hdr = texture(uTex, vUv).rgb * uExposure;
  if (uBloomOn > 0.5) hdr += texture(uBloom, vUv).rgb * uBloomStrength;
  if (uLdr > 0.5) hdr = min(hdr, vec3(1.0));
  vec2 q = vUv - 0.5; q.x *= uRes.x / uRes.y;
  float vig = clamp(1.0 - dot(q, q) * 1.1, 0.0, 1.0);
  vec3 col = aces(hdr) + bg;
  col = pow(col, vec3(1.0 / 2.2));
  col *= mix(0.72, 1.0, vig);
  col += (hash(gl_FragCoord.xy + fract(uTime * 7.13) * 311.0) - 0.5) * uGrain;
  o = vec4(col, 1.0);
}`;

// ---------------------------------------------------------------------------

function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
function srgbToLinear(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
function hexLin(h) { return hexToRgb(h).map(srgbToLinear); }

function mat4Invert(out, m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3], a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11], a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out;
  det = 1 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', {
      antialias: false, alpha: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.canvas = canvas;
    this.hdr = !!gl.getExtension('EXT_color_buffer_float');
    const shadeU = ['uPal', 'uColorAnim', 'uColorDir', 'uColorScale', 'uColorPhase', 'uFlash'];
    this.progFade = this.program(GLSL_FULLSCREEN_VS, GLSL_FADE_FS);
    this.progPts = this.program(GLSL_POINTS_VS, GLSL_POINTS_FS);
    this.progLines = this.program(GLSL_LINES_VS, GLSL_LINES_FS);
    this.progTone = this.program(GLSL_FULLSCREEN_VS, GLSL_TONEMAP_FS);
    this.progPre = this.program(GLSL_FULLSCREEN_VS, GLSL_BLOOM_PRE_FS);
    this.progDown = this.program(GLSL_FULLSCREEN_VS, GLSL_BLOOM_DOWN_FS);
    this.progUp = this.program(GLSL_FULLSCREEN_VS, GLSL_BLOOM_UP_FS);
    this.u = {
      pre: this.uniforms(this.progPre, ['uTex', 'uTexel', 'uExposure', 'uThreshold']),
      down: this.uniforms(this.progDown, ['uTex', 'uTexel']),
      up: this.uniforms(this.progUp, ['uTex', 'uTexel', 'uWeight']),
      fade: this.uniforms(this.progFade, ['uDecay']),
      pts: this.uniforms(this.progPts, ['uVP', 'uSize', 'uGain', 'uFog', 'uSoft', 'uReflect', 'uFloorY', 'uReflFall', ...shadeU]),
      lines: this.uniforms(this.progLines, ['uVP', 'uGain', 'uFog', ...shadeU]),
      tone: this.uniforms(this.progTone, ['uTex', 'uBloom', 'uBloomOn', 'uBloomStrength', 'uLdr', 'uExposure', 'uTime', 'uGrain', 'uRes', 'uInvVP', 'uEye',
        'uFloorOn', 'uFloorY', 'uSkyLow', 'uSkyHigh', 'uFloorTint']),
    };

    // particles
    this.vao = gl.createVertexArray();
    this.bufP = gl.createBuffer();
    this.bufT = gl.createBuffer();
    this.capacity = 0;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufP);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 16, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufT);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 4, 0);

    // links
    this.lineVao = gl.createVertexArray();
    this.bufL = gl.createBuffer();
    this.lineCapacity = 0;
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufL);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 16, 0);
    gl.bindVertexArray(null);

    this.emptyVao = gl.createVertexArray();
    this.fbo = null; this.tex = null; this.w = 0; this.h = 0;
    this.palette = new Float32Array(15);
    this.paletteTarget = new Float32Array(15);
    this.invVP = new Float32Array(16);
    this.bloomLevels = [];
  }

  get bloomSupported() { return this.hdr; }

  program(vsSrc, fsSrc) {
    const gl = this.gl;
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  uniforms(p, names) {
    const out = {};
    for (const n of names) out[n] = this.gl.getUniformLocation(p, n);
    return out;
  }

  resize(w, h) {
    if (w === this.w && h === this.h) return;
    const gl = this.gl;
    this.w = w; this.h = h;
    this.canvas.width = w; this.canvas.height = h;
    if (this.tex) gl.deleteTexture(this.tex);
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    const make = (internal, type, tw = w, th = h) => {
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, tw, th, 0, gl.RGBA, type, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
      return { tex, fbo, ok };
    };
    let r = this.hdr ? make(gl.RGBA16F, gl.HALF_FLOAT) : { ok: false };
    if (!r.ok) { this.hdr = false; r = make(gl.RGBA8, gl.UNSIGNED_BYTE); }
    this.tex = r.tex; this.fbo = r.fbo;
    // bloom mip chain: 1/2, 1/4, ... 1/64 of the canvas
    for (const l of this.bloomLevels) { gl.deleteTexture(l.tex); gl.deleteFramebuffer(l.fbo); }
    this.bloomLevels = [];
    if (this.hdr) {
      let bw = w, bh = h;
      for (let i = 0; i < 6; i++) {
        bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
        const l = make(gl.RGBA16F, gl.HALF_FLOAT, bw, bh);
        if (!l.ok) { this.bloomLevels = []; break; }
        this.bloomLevels.push({ tex: l.tex, fbo: l.fbo, w: bw, h: bh });
      }
    }
    this.clear();
  }

  clear() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  upload(P, T, count) {
    const gl = this.gl;
    if (count > this.capacity) {
      this.capacity = count;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufP);
      gl.bufferData(gl.ARRAY_BUFFER, count * 16, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufT);
      gl.bufferData(gl.ARRAY_BUFFER, count * 4, gl.DYNAMIC_DRAW);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufP);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, P, 0, count * 4);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufT);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, T, 0, count);
  }

  uploadLines(L, lineCount) {
    if (!lineCount) return;
    const gl = this.gl;
    const verts = lineCount * 2;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufL);
    if (verts > this.lineCapacity) {
      this.lineCapacity = Math.max(verts, 4096);
      gl.bufferData(gl.ARRAY_BUFFER, this.lineCapacity * 16, gl.DYNAMIC_DRAW);
    }
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, L, 0, verts * 4);
  }

  // Palettes crossfade: setPalette sets a target, tickPalette eases toward it.
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

  bloom(o) {
    const gl = this.gl, L = this.bloomLevels;
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.emptyVao);
    gl.activeTexture(gl.TEXTURE0);
    // prefilter: accum -> level 0 (bright parts only)
    gl.useProgram(this.progPre);
    gl.bindFramebuffer(gl.FRAMEBUFFER, L[0].fbo);
    gl.viewport(0, 0, L[0].w, L[0].h);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this.u.pre.uTex, 0);
    gl.uniform2f(this.u.pre.uTexel, 1 / this.w, 1 / this.h);
    gl.uniform1f(this.u.pre.uExposure, o.exposure);
    gl.uniform1f(this.u.pre.uThreshold, o.bloom.threshold);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // downsample
    gl.useProgram(this.progDown);
    gl.uniform1i(this.u.down.uTex, 0);
    for (let i = 1; i < L.length; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, L[i].fbo);
      gl.viewport(0, 0, L[i].w, L[i].h);
      gl.bindTexture(gl.TEXTURE_2D, L[i - 1].tex);
      gl.uniform2f(this.u.down.uTexel, 1 / L[i - 1].w, 1 / L[i - 1].h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    // upsample, accumulating into the next larger level
    gl.useProgram(this.progUp);
    gl.uniform1i(this.u.up.uTex, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = L.length - 1; i > 0; i--) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, L[i - 1].fbo);
      gl.viewport(0, 0, L[i - 1].w, L[i - 1].h);
      gl.bindTexture(gl.TEXTURE_2D, L[i].tex);
      gl.uniform2f(this.u.up.uTexel, 0.5 / L[i].w, 0.5 / L[i].h);
      gl.uniform1f(this.u.up.uWeight, 1.0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.BLEND);
  }

  setShade(u, ca, flash) {
    const gl = this.gl;
    gl.uniform1f(u.uFlash, flash || 0);
    gl.uniform3fv(u.uPal, this.palette);
    gl.uniform1f(u.uColorAnim, ca ? 1 : 0);
    gl.uniform3fv(u.uColorDir, ca ? ca.dir : [0, 0, 0]);
    gl.uniform1f(u.uColorScale, ca ? ca.scale : 0);
    gl.uniform1f(u.uColorPhase, ca ? ca.phase : 0);
  }

  draw(o) {
    const gl = this.gl;
    gl.viewport(0, 0, this.w, this.h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.enable(gl.BLEND);

    // 1) fade previous frame (trails)
    if (o.decay <= 0) {
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    } else if (o.decay < 1) {
      gl.useProgram(this.progFade);
      gl.uniform1f(this.u.fade.uDecay, o.decay);
      gl.blendFunc(gl.ZERO, gl.SRC_ALPHA);
      gl.bindVertexArray(this.emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // 2) additive points, then their mirror image under the floor
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.progPts);
    const up = this.u.pts;
    gl.uniformMatrix4fv(up.uVP, false, o.vp);
    gl.uniform2f(up.uFog, o.fogNear, o.fogFar);
    this.setShade(up, o.colorAnim, o.flash);
    gl.bindVertexArray(this.vao);
    gl.uniform1f(up.uSize, o.size);
    gl.uniform1f(up.uGain, o.gain);
    gl.uniform1f(up.uSoft, o.size > 1.6 ? 1 : 0);
    gl.uniform1f(up.uReflect, 0);
    gl.drawArrays(gl.POINTS, 0, o.count);
    if (o.floor) {
      // Reflection: bigger soft points at low gain read as a slightly blurred, glossy mirror.
      const rs = o.size * 2.2;
      gl.uniform1f(up.uSize, rs);
      gl.uniform1f(up.uGain, o.gain * o.floor.reflGain / (rs * rs * 0.35));
      gl.uniform1f(up.uSoft, 1);
      gl.uniform1f(up.uReflect, 1);
      gl.uniform1f(up.uFloorY, o.floor.y);
      gl.uniform1f(up.uReflFall, o.floor.reflFall);
      gl.drawArrays(gl.POINTS, 0, o.count);
    }

    // 3) links
    if (o.lineCount > 0) {
      gl.useProgram(this.progLines);
      const ul = this.u.lines;
      gl.uniformMatrix4fv(ul.uVP, false, o.vp);
      gl.uniform2f(ul.uFog, o.fogNear, o.fogFar);
      gl.uniform1f(ul.uGain, o.lineGain);
      this.setShade(ul, o.colorAnim, o.flash);
      gl.bindVertexArray(this.lineVao);
      gl.drawArrays(gl.LINES, 0, o.lineCount * 2);
    }

    // 4) optional bloom chain
    const bloomOn = !!(o.bloom && o.bloom.on && this.bloomLevels.length);
    if (bloomOn) this.bloom(o);

    // 5) tone map + background to screen
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.w, this.h);
    gl.useProgram(this.progTone);
    const ut = this.u.tone;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(ut.uTex, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bloomOn ? this.bloomLevels[0].tex : this.tex);
    gl.uniform1i(ut.uBloom, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(ut.uBloomOn, bloomOn ? 1 : 0);
    gl.uniform1f(ut.uLdr, this.hdr ? 0 : 1);
    gl.uniform1f(ut.uBloomStrength, bloomOn ? o.bloom.strength : 0);
    gl.uniform1f(ut.uExposure, o.exposure);
    gl.uniform1f(ut.uTime, o.time);
    gl.uniform1f(ut.uGrain, o.grain);
    gl.uniform2f(ut.uRes, this.w, this.h);
    mat4Invert(this.invVP, o.vp);
    gl.uniformMatrix4fv(ut.uInvVP, false, this.invVP);
    gl.uniform3fv(ut.uEye, o.eye);
    gl.uniform1f(ut.uFloorOn, o.floor ? 1 : 0);
    gl.uniform1f(ut.uFloorY, o.floor ? o.floor.y : 0);
    gl.uniform3fv(ut.uSkyLow, o.sky.low);
    gl.uniform3fv(ut.uSkyHigh, o.sky.high);
    gl.uniform3fv(ut.uFloorTint, o.sky.floor);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}

// ---------------------------------------------------------------------------

class Engine {
  // opts.renderer: a WebGL Renderer or a GpuBackend; opts.simBackend: 'js' | 'wasm' | 'gpu'
  constructor(mode, opts) {
    this.mode = mode;
    this.canvas = opts.canvas;
    this.renderer = opts.renderer;
    this.simBackend = opts.simBackend;
    this.backendLabel = opts.label;
    this.camera = new OrbitCamera();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssW = 1; this.cssH = 1;
    this.time = 0;
    this.wall = 0;
    this.frozen = false;
    this.stats = { sim: 0, draw: 0, frame: 16.7 };
    this.lastFrame = performance.now();
    this.lastHud = 0;
    this.reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.pointer = null;
    this.pressed = false;
    this.orbiting = false;
    this.demoActive = true;
    this.demoTime = 0;
    this.demoDown = false;
    this.lastUserInput = -1e9;
    this.ghostEl = document.getElementById('ghost');
    this.toastEl = document.getElementById('toast');

    this.bindControls();
    this.bindInput();
    this.bindPad();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  // ---- UI -----------------------------------------------------------------
  bindControls() {
    document.getElementById('less').addEventListener('click', () => this.changeCount(-1));
    document.getElementById('more').addEventListener('click', () => this.changeCount(+1));
    document.getElementById('freeze').addEventListener('click', () => this.toggleFreeze());
    const turb = document.getElementById('turb');
    turb.addEventListener('input', () => {
      this.userInput();
      this.mode.setTurb(parseFloat(turb.value));
      this.syncTurb(false);
    });
  }

  syncTurb(showToast) {
    const v = this.mode.turb;
    const el = document.getElementById('turb');
    if (document.activeElement !== el) el.value = String(v);
    document.getElementById('turb-v').textContent = v.toFixed(2);
    if (showToast) this.toast(`turbulence ${v.toFixed(2)}`);
  }

  toast(text) {
    const t = this.toastEl;
    t.textContent = text;
    t.classList.add('on');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => t.classList.remove('on'), 900);
  }

  renderOptions() {
    const box = document.getElementById('options');
    box.innerHTML = '';
    (this.mode.optionGroups || []).forEach((o, g) => {
      const row = document.createElement('div');
      row.className = 'chips';
      const l = document.createElement('span');
      l.className = 'chips-label'; l.textContent = o.label;
      row.appendChild(l);
      o.items.forEach((label, i) => {
        const b = document.createElement('button');
        b.className = 'chip'; b.type = 'button'; b.id = `opt-${g}-${i}`;
        b.textContent = label;
        b.setAttribute('aria-pressed', String(i === o.index));
        if (o.current === i && i !== o.index) { b.classList.add('current'); b.setAttribute('aria-current', 'true'); }
        b.addEventListener('click', () => { this.userInput(); o.select(i); this.renderOptions(); });
        row.appendChild(b);
      });
      box.appendChild(row);
    });
  }

  bindPad() {
    const pad = document.getElementById('pad');
    const fromEvent = (e) => {
      const r = pad.getBoundingClientRect();
      return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height];
    };
    let dragging = false;
    pad.addEventListener('pointerdown', (e) => {
      this.userInput();
      dragging = true; pad.setPointerCapture(e.pointerId);
      pad.classList.add('used');
      const [u, v] = fromEvent(e); this.mode.pad.set(u, v);
    });
    pad.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const [u, v] = fromEvent(e); this.mode.pad.set(u, v);
    });
    const end = () => { dragging = false; };
    pad.addEventListener('pointerup', end);
    pad.addEventListener('pointercancel', end);
    document.getElementById('pad-reset').addEventListener('click', () => { this.userInput(); this.mode.pad.reset(); });
    pad.addEventListener('keydown', (e) => {
      const [u, v] = this.mode.pad.get();
      const s = e.shiftKey ? 0.1 : 0.02;
      const m = { ArrowLeft: [-s, 0], ArrowRight: [s, 0], ArrowUp: [0, s], ArrowDown: [0, -s] }[e.key];
      if (m) { e.preventDefault(); e.stopPropagation(); this.userInput(); this.mode.pad.set(u + m[0], v + m[1]); }
    });
    this.padEl = pad;
    this.padKnob = pad.querySelector('.pad-knob');
  }

  updatePad() {
    const p = this.mode.pad;
    const [u, v] = p.get();
    this.padKnob.style.left = (u * 100).toFixed(2) + '%';
    this.padKnob.style.top = ((1 - v) * 100).toFixed(2) + '%';
    const [lx, ly] = p.labels();
    if (lx + ly !== this._padLabels) {
      this._padLabels = lx + ly;
      document.getElementById('pad-x').textContent = lx;
      document.getElementById('pad-y').textContent = ly;
    }
  }

  reset() {
    const m = this.mode;
    document.documentElement.style.setProperty('--accent', m.accent);
    document.getElementById('hint').innerHTML = m.hint;
    this.camera.set(m.camera);
    this.renderer.setPalette(m.palette, true);
    this.sky = { low: hexLin(m.sky.low), high: hexLin(m.sky.high), floor: hexLin(m.sky.floor) };
    m.init(this, m.counts[m.countIndex]);
    this.renderer.clear();
    this.renderOptions();
    this.updatePad();
    this.syncTurb(false);
    this.syncCountButtons();
    this.demoActive = !this.reducedMotion;
    this.auto = window.__MAX_FRAMES ? null : { on: true, settle: 0.6, ms: [], work: [], vsync: Infinity, steps: 0, last: 0 };
    this.demoTime = 0;
    this.demoDown = false;
    this.time = 0;
    this.stats.sim = 0;
  }

  changeCount(dir, auto) {
    const m = this.mode;
    if (!auto && this.auto) { this.auto.on = false; this.auto.watch = false; }   // a manual choice always wins
    const k = Math.max(0, Math.min(m.counts.length - 1, m.countIndex + dir));
    if (k === m.countIndex) return;
    m.countIndex = k;
    m.resize(this, m.counts[k]);
    this.stats.sim = 0;
    this.syncCountButtons();
  }

  // ---- automatic particle count ---------------------------------------------
  // After start-up, measure the median frame interval over AUTO_WINDOW frames. While it
  // holds the display's refresh (the fastest median seen), step the count up; on the
  // first sign of dropped frames step back down once and stop. CPU backends also
  // predict the next step from their own per-frame work, so they don't try a count
  // that would obviously miss. Any manual −/+ turns this off.
  autoCount(rawMs, workMs) {
    const a = this.auto;
    if (!a) return;
    if (!a.on) {
      // After calibration: step down (never up) when frames stay slow, e.g. a phone heating up.
      if (!a.watch || rawMs > 250 || document.hidden) return;
      a.ms.push(rawMs);
      if (a.ms.length < 90) return;
      a.ms.sort((x, y) => x - y);
      const slow = a.ms[a.ms.length >> 1] > Math.max(a.vsync, AUTO_MIN_FRAME_MS) * 1.3 + 0.5;
      a.ms.length = 0;
      if (slow && this.mode.countIndex > 0) { this.changeCount(-1, true); this.toast(`Auto: ${fmtCount(this.mode.count)} particles`); }
      return;
    }
    if (rawMs > 250 || document.hidden) { a.ms.length = 0; a.settle = 0.5; return; }   // tab switch, stall
    if (a.settle > 0) { a.settle -= rawMs / 1000; return; }
    a.ms.push(rawMs); a.work.push(workMs);
    if (a.ms.length < AUTO_WINDOW) return;
    a.ms.sort((x, y) => x - y); a.work.sort((x, y) => x - y);
    const frame = a.ms[a.ms.length >> 1], work = a.work[a.work.length >> 1];
    // Refresh interval: the 10th percentile, so a count that is already dropping frames
    // (intervals alternating 1x / 2x refresh) can't pass itself off as the refresh rate.
    // Capped at 60 Hz: a count that is slow on every frame would otherwise pass for the refresh rate.
    a.vsync = Math.min(a.vsync, a.ms[Math.floor(a.ms.length * 0.1)], 1000 / 60);
    a.ms.length = 0; a.work.length = 0;
    // Aim for the display's refresh, but never above 120 fps: on a 240 Hz screen twice
    // the particles at 120 fps is the better picture.
    const target = Math.max(a.vsync, AUTO_MIN_FRAME_MS);
    const m = this.mode, i = m.countIndex;
    const done = (msg) => { a.on = false; a.watch = true; if (msg) this.toast(msg); };
    if (frame > target * 1.2 + 0.5) {                     // missing frames at this count
      if (a.last === 1 && i > 0) { this.changeCount(-1, true); return done(`Auto: ${fmtCount(m.counts[m.countIndex])} particles`); }
      if (i > 0 && a.steps < 3) { a.steps++; a.last = -1; this.changeCount(-1, true); a.settle = 0.35; return; }
      return done();
    }
    const next = m.counts[i + 1];
    const cpuOk = this.simBackend === 'gpu' || work * (next / m.counts[i]) < target * 0.7;
    if (a.last !== -1 && next && cpuOk && a.steps < 6) { a.steps++; a.last = 1; this.changeCount(1, true); a.settle = 0.35; return; }
    done(a.steps ? `Auto: ${fmtCount(m.counts[i])} particles` : null);
  }

  // − / + are disabled at the ends of the particle-count list.
  syncCountButtons() {
    const m = this.mode, less = document.getElementById('less'), more = document.getElementById('more');
    less.disabled = m.countIndex <= 0;
    more.disabled = m.countIndex >= m.counts.length - 1;
    less.title = less.disabled ? 'Already at the minimum' : 'Fewer particles';
    more.title = more.disabled ? 'Already at the maximum' : 'More particles';
  }

  toggleFreeze() {
    this.frozen = !this.frozen;
    document.getElementById('freeze').setAttribute('aria-pressed', String(this.frozen));
  }

  resize() {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssW = Math.max(1, window.innerWidth);
    this.cssH = Math.max(1, window.innerHeight);
    this.renderer.resize(Math.round(this.cssW * this.dpr), Math.round(this.cssH * this.dpr));
    this.camera.update(this.cssW, this.cssH);
  }

  // ---- input --------------------------------------------------------------
  userInput() {
    this.lastUserInput = this.wall;
    if (this.demoActive) this.stopDemo();
  }

  stopDemo() {
    if (this.demoDown) this.dispatch('up', this.pointer || this.makePointer(this.cssW / 2, this.cssH / 2));
    this.demoActive = false;
    this.demoDown = false;
    this.hideGhost();
  }

  // The ghost circle and its cursor links go together: no visible demo pointer, no hover.
  hideGhost() {
    this.ghostEl.classList.remove('on', 'down');
    if (this.pointer && this.pointer.synthetic) this.dispatch('leave', null);
  }

  makePointer(x, y, extra) {
    const nx = (x / this.cssW) * 2 - 1, ny = 1 - (y / this.cssH) * 2;
    return Object.assign({
      x, y, nx, ny, t: performance.now(), synthetic: false,
    }, extra || {});
  }

  dispatch(type, p) {
    this.pointer = p;
    if (this.mode.pointer) this.mode.pointer(type, p, this);
  }

  bindInput() {
    const c = this.canvas;
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    // Two fingers: pinch = zoom, drag = orbit. A second finger cancels a tear, and the gesture
    // holds until every finger is up so the last one doesn't start tearing.
    const touches = new Map();
    const pair = () => {
      const [a, b] = [...touches.values()];
      return { d: Math.hypot(b.x - a.x, b.y - a.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
    };
    c.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'touch') {
        touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (touches.size === 2) {
          this.userInput();
          if (this.pressed) { this.pressed = false; this.dispatch('up', this.makePointer(e.clientX, e.clientY)); }
          this._pinch = { ...pair(), dist: this.camera.dist, yaw: this.camera.yaw, pitch: this.camera.pitch };
          return;
        }
        if (this._pinch || touches.size > 2) return;
      }
      this.userInput();
      c.setPointerCapture(e.pointerId);
      const orbit = e.button === 2 || e.altKey || e.ctrlKey || e.metaKey || this.frozen;
      if (orbit) {
        this.orbiting = true; c.classList.add('orbiting');
        this._orbitFrom = { x: e.clientX, y: e.clientY, yaw: this.camera.yaw, pitch: this.camera.pitch };
        return;
      }
      if (e.button !== 0) return;
      this.pressed = true;
      this.dispatch('down', this.makePointer(e.clientX, e.clientY));
    });
    c.addEventListener('pointermove', (e) => {
      if (touches.has(e.pointerId)) {
        touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (this._pinch) {
          if (touches.size === 2) {
            const p = pair(), g = this._pinch, lo = this.mode.minPitch !== undefined ? this.mode.minPitch : -1.35;
            this.camera.dist = Math.max(1.5, Math.min(12, g.dist * g.d / Math.max(1, p.d)));
            this.camera.yaw = g.yaw - (p.mx - g.mx) * 0.006;
            this.camera.pitch = Math.max(lo, Math.min(1.35, g.pitch + (p.my - g.my) * 0.006));
          }
          return;
        }
      }
      if (this.orbiting) {
        const o = this._orbitFrom;
        const lo = this.mode.minPitch !== undefined ? this.mode.minPitch : -1.35;
        this.camera.yaw = o.yaw - (e.clientX - o.x) * 0.006;
        this.camera.pitch = Math.max(lo, Math.min(1.35, o.pitch + (e.clientY - o.y) * 0.006));
        return;
      }
      if (this.demoActive && !this.pressed) return;
      this.dispatch(this.pressed ? 'move' : 'hover', this.makePointer(e.clientX, e.clientY));
    });
    const end = (e) => {
      if (touches.delete(e.pointerId) && this._pinch) { if (touches.size === 0) this._pinch = null; return; }
      if (this.orbiting) { this.orbiting = false; c.classList.remove('orbiting'); return; }
      if (!this.pressed) return;
      this.pressed = false;
      this.dispatch('up', this.makePointer(e.clientX, e.clientY));
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('pointerleave', () => { if (!this.pressed && !this.demoActive) this.dispatch('leave', null); });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.userInput();
      const d = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      if (e.shiftKey) {
        // Shift+wheel: many browsers turn it into horizontal scroll, so read either axis.
        const t = d || (e.deltaMode === 1 ? e.deltaX * 16 : e.deltaX);
        this.mode.setTurb(this.mode.turb - t * 0.0012);
        this.syncTurb(true);
      } else {
        this.camera.dist = Math.max(1.5, Math.min(12, this.camera.dist * Math.exp(d * 0.001)));
      }
    }, { passive: false });

    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const j = document.getElementById('journey');
      if (j && !j.hidden) return;                 // the article is a modal dialog
      const k = e.key;
      if (k === ' ') { e.preventDefault(); this.userInput(); this.toggleFreeze(); return; }
      if (k === '[') { this.changeCount(-1); return; }
      if (k === ']') { this.changeCount(+1); return; }
      if (k === 'h' || k === 'H') { document.body.classList.toggle('ui-hidden'); return; }
      if (k === 'r' || k === 'R') { this.reset(); return; }
      if (this.mode.key) { this.userInput(); this.mode.key(k, this); }
    });
  }

  // ---- ghost demo ---------------------------------------------------------
  runDemo(dt) {
    if (!this.demoActive) {
      if (!this.reducedMotion && this.wall - this.lastUserInput > 25 && !this.pressed) {
        this.demoActive = true; this.demoTime = 0;
      } else return;
    }
    if (this.frozen || !this.mode.demo) return;
    this.demoTime += dt;
    if (this.mode.demoDist && !this.orbiting) {
      const d = this.mode.demoDist(this.demoTime);
      this.camera.dist += (d - this.camera.dist) * (1 - Math.exp(-dt * 0.8));
    }
    const g = this.mode.demo(this.demoTime, this);
    if (!g) {
      if (this.demoDown) { this.demoDown = false; this.dispatch('up', this.pointer); }
      this.hideGhost();
      return;
    }
    if (g.turb !== undefined) { this.mode.setTurb(g.turb); this.syncTurb(false); }
    if (g.padU !== undefined) {
      if (this.demoDown) { this.demoDown = false; this.dispatch('up', this.pointer); }
      if (this.pointer && this.pointer.synthetic) this.dispatch('leave', null);   // the ghost is on the pad now
      this.mode.pad.set(g.padU, g.padV);
      const [pu, pv] = this.mode.pad.get();
      const r = this.padEl.getBoundingClientRect();
      this.ghostEl.classList.add('on', 'down');
      this.ghostEl.style.transform = `translate(${r.left + pu * r.width}px, ${r.top + (1 - pv) * r.height}px)`;
      return;
    }
    if (g.u === undefined) { this.hideGhost(); return; }
    const x = g.u * this.cssW, y = g.v * this.cssH;
    const p = this.makePointer(x, y, { synthetic: true });
    if (g.down && !this.demoDown) { this.demoDown = true; this.dispatch('down', p); }
    else if (!g.down && this.demoDown) { this.demoDown = false; this.dispatch('up', p); }
    else this.dispatch(g.down ? 'move' : 'hover', p);
    this.ghostEl.classList.add('on');
    this.ghostEl.classList.toggle('down', !!g.down);
    this.ghostEl.style.transform = `translate(${x}px, ${y}px)`;
  }

  // ---- loop ---------------------------------------------------------------
  start() {
    this.reset();
    let frames = 0;
    const tick = (now) => {
      this.frame(now);
      if (window.__MAX_FRAMES && ++frames >= window.__MAX_FRAMES) return;   // test hook
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  frame(now) {
    const rawDt = (now - this.lastFrame) / 1000;
    this.lastFrame = now;
    const dt = Math.min(1 / 30, Math.max(0, rawDt));
    this.wall += dt;
    this.stats.frame = this.stats.frame * 0.92 + rawDt * 1000 * 0.08;

    const m = this.mode;
    if (m.autoRotate && !this.orbiting && !this.reducedMotion) this.camera.yaw += m.autoRotate * dt;
    this.camera.update(this.cssW, this.cssH);

    this.runDemo(dt);

    const t0 = performance.now();
    if (!this.frozen) {
      this.time += dt;
      m.step(dt, this.time, this);
    }
    const t1 = performance.now();
    if (m.tickColor) m.tickColor(dt, this);
    this.renderer.tickPalette(dt, m.paletteRate || 1.5);
    const trailComp = m.trails === false ? 1 / (1 - m.decay) : 1;
    // A soft (1 - r²)² disc of diameter px delivers π/12·px² of light; gain divides by that per CSS
    // pixel, so every display gets the light of a hard 1-px point at 1×, just in a bigger dot.
    const px = m.pointSize * this.dpr, cssPx = m.pointSize;
    const gain = trailComp * m.gain * Math.pow(m.refCount / m.count, 0.8) / (Math.PI / 12 * cssPx * cssPx) * (this.renderer.hdr ? 1 : 0.5);
    this.renderer.upload(m.P, m.T, m.count);
    const lines = m.lines || { count: 0 };
    this.renderer.uploadLines(lines.data, lines.count);
    this.renderer.draw({
      vp: this.camera.vp, eye: this.camera.eye, count: m.count,
      size: px, gain,
      // Motion blur = the fade pass: last frame is multiplied by `decay`, not cleared.
      // With it off we clear, and scale gain by 1/(1-decay) to keep the same brightness.
      decay: m.trails === false ? 0 : (this.frozen ? Math.min(m.decay, 0.6) : m.decay),
      exposure: m.exposure, grain: m.grain, time: this.wall,
      fogNear: this.camera.dist * 0.6, fogFar: this.camera.dist * 2.4,
      colorAnim: m.colorAnim ? m.colorAnim(this.wall) : null,
      floor: m.floor, sky: this.sky,
      lineCount: lines.count, lineGain: trailComp * m.lineGain / this.dpr,
      bloom: m.bloom, flash: m.flash || 0,
      dpr: this.dpr, cssW: this.cssW, cssH: this.cssH, fwd: this.camera.fwd,
      cursor: (this.simBackend === 'gpu' && m.linksOn && m.hover) ? { x: m.hover.x, y: m.hover.y, ray: this.camera.ray(m.hover.nx, m.hover.ny).d } : null,
    });
    const t2 = performance.now();
    if (!this.frozen) this.stats.sim = this.stats.sim * 0.9 + (t1 - t0) * 0.1;
    this.stats.draw = this.stats.draw * 0.9 + (t2 - t1) * 0.1;

    this.autoCount(rawDt * 1000, t2 - t0);
    this.updatePad();
    if (now - this.lastHud > 250) { this.lastHud = now; this.hud(); }
  }

  hud() {
    document.getElementById('n').textContent = fmtCount(this.mode.count);
    if (!this._labelled) { this._labelled = true; document.getElementById('backend').textContent = this.backendLabel; }
    document.getElementById('simms').textContent = this.stats.sim.toFixed(1) + ' ms';
    document.getElementById('drawms').textContent = this.stats.draw.toFixed(1) + ' ms';
    const fps = 1000 / Math.max(1, this.stats.frame);
    document.getElementById('framems').textContent = this.stats.frame.toFixed(1) + ' ms · ' + Math.round(fps) + ' fps';
    document.getElementById('readout').textContent = this.mode.readout ? this.mode.readout(this) : '';
  }
}
