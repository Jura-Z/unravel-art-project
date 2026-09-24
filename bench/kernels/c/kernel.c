// Unravel kernel in C. Same spec as kernels/js-ref.js, bit for bit.
// Build with -ffp-contract=off (no FMA) and no fast-math.
//
// Two layouts, one source:
//   * scalar (default): canonical AoS arrays, one fused sweep per particle.
//   * SIMD (__wasm_simd128__): S and V stored SoA so 4 particles load with plain
//     v128 loads; P stays AoS xyzw (the renderer's vertex format) via a 4x4 transpose.
// Both do ODE -> warp -> tear -> smoke per particle in ONE pass over memory
// (the reference does four passes, re-streaming ~70 bytes/particle each time).

#include <stdint.h>
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#ifdef __wasm__
#define EXPORT(name) __attribute__((export_name(#name)))
#else
#define EXPORT(name)
#endif
#define SQRT(x) __builtin_sqrtf(x)
#define FLOOR(x) __builtin_floorf(x)

// ---- uniform layout (must match core.js UNI) ----
enum { U_KIND = 0, U_FRAME = 1, U_FLAGS = 2, U_A = 4, U_B, U_C, U_D, U_E, U_FP, U_K, U_R, U_SG,
  U_H = 13, U_BOUND, U_JIT, U_INIT, U_SC = 19, U_CX, U_CY, U_CZ, U_INV_SPEED, U_INV_HOT, U_ORIENT = 25,
  U_WARP_AMP = 34, U_WOX, U_WOY, U_WOZ, U_TX = 39, U_TY, U_TR2, U_WVX, U_WVY, U_WVZ,
  U_BX = 45, U_BY, U_BR2, U_RIGHT = 48, U_UP = 51, U_FWD = 54, U_SW = 57, U_SH = 58, U_VP = 59,
  U_FREQ = 75, U_AMP, U_SOX, U_SOY, U_SOZ, U_RELAX, U_BACK, U_BACK_MORPH, U_DT, U_SIZE = 96 };
enum { FLAG_WARP = 1, FLAG_TEAR = 2, FLAG_BURST = 4 };
#define WG 12
#define W_LO (-2.2f)
#define W_NF 0.9f
#define MORPH_EASE 1.4f

// ---- memory (bump allocator; JS sizes linear memory before setup) ----
#ifdef __wasm__
extern unsigned char __heap_base;
#define HEAP_BASE (&__heap_base)
#else
#include <stdlib.h>
static unsigned char* native_heap;
#define HEAP_BASE native_heap
#endif
static unsigned char* heap_top;
static void* bump(uint32_t bytes) { uintptr_t p = ((uintptr_t)heap_top + 15) & ~(uintptr_t)15; heap_top = (unsigned char*)(p + bytes); return (void*)p; }

static int N;
static float *S, *P, *V, *F, *T, *HEAL, *SS, *W, *U;   // canonical (AoS) arrays
#ifdef __wasm_simd128__
static float *SX, *SY, *SZ, *VX, *VY, *VZ;             // SoA state for the SIMD path
static float *W4, *HB;                                 // padded xyz_ warp grid; home+emission rows of torn particles
static int32_t *TORN;                                  // compacted indices of torn particles (per-range slices)
#endif
static float W_CELL, W_INV, W_GMAX;

typedef struct { int32_t S, P, V, F, T, heal, SS, W, U, end; } Ptrs;   // wasm32 offsets (unused natively)
static Ptrs ptrs;

static void links_alloc(int32_t n);
EXPORT(setup) int32_t setup(int32_t n) {
  N = n;
#ifndef __wasm__
  native_heap = malloc((size_t)n * 128 + (1 << 20));
#endif
  heap_top = HEAP_BASE;
  S = bump(n * 12); P = bump(n * 16); V = bump(n * 12); F = bump(n * 4); T = bump(n * 4); HEAL = bump(n * 4);
  SS = bump(((n + 63) / 64) * 20); W = bump(WG * WG * WG * 12); U = bump(U_SIZE * 4);
#ifdef __wasm_simd128__
  SX = bump(n * 4); SY = bump(n * 4); SZ = bump(n * 4); VX = bump(n * 4); VY = bump(n * 4); VZ = bump(n * 4);
  W4 = bump(WG * WG * WG * 16); HB = bump(n * 16); TORN = bump(n * 4);
#endif
  links_alloc(n);
  W_CELL = 4.4f / 11.0f; W_INV = 1.0f / W_CELL; W_GMAX = 12.0f - 1.001f;
  ptrs = (Ptrs){ (int32_t)(uintptr_t)S, (int32_t)(uintptr_t)P, (int32_t)(uintptr_t)V, (int32_t)(uintptr_t)F,
    (int32_t)(uintptr_t)T, (int32_t)(uintptr_t)HEAL, (int32_t)(uintptr_t)SS, (int32_t)(uintptr_t)W,
    (int32_t)(uintptr_t)U, (int32_t)(uintptr_t)heap_top };
  return (int32_t)(uintptr_t)&ptrs;
}
EXPORT(bytes_needed) int32_t bytes_needed(int32_t n) {
  return (int32_t)((uintptr_t)HEAP_BASE + (uint32_t)n * (12 + 16 + 12 + 4 + 4 + 4 + 24 + 20) + (uint32_t)n / 3 + (uint32_t)n * 6 + 2 * 65536 + 16 * 16 + WG * WG * WG * 16);
}

// ---- RNG ----
static inline uint32_t hash3(uint32_t i, uint32_t frame, uint32_t salt) {
  uint32_t x = (i * 0x9E3779B1u) ^ (frame * 0x85EBCA77u) ^ (salt * 0xC2B2AE3Du);
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
static inline float rnd(uint32_t i, uint32_t frame, uint32_t salt) { return (float)(hash3(i, frame, salt) >> 8) * (1.0f / 16777216.0f); }

// ---- fsin (see core.js) ----
#define INV_PI 0.31830987f
#define PI_A 3.140625f
#define PI_B 0.00096765358979f
static inline float fsin(float x) {
  float q = FLOOR(x * INV_PI + 0.5f);
  float r = (x - q * PI_A) - q * PI_B;
  float r2 = r * r;
  float p = (1.0f / 362880.0f) + r2 * (-1.0f / 39916800.0f);
  p = (-1.0f / 5040.0f) + r2 * p;
  p = (1.0f / 120.0f) + r2 * p;
  p = (-1.0f / 6.0f) + r2 * p;
  p = r + (r * r2) * p;
  return (((int32_t)q) & 1) ? -p : p;
}

// ---- simplex noise gradient + curl ----
static uint8_t PERM[512];
static const float GRAD[48] = { 1,1,0, -1,1,0, 1,-1,0, -1,-1,0, 1,0,1, -1,0,1, 1,0,-1, -1,0,-1,
  0,1,1, 0,-1,1, 0,1,-1, 0,-1,-1, 1,1,0, 0,-1,1, -1,1,0, 0,-1,-1 };
static int perm_ready = 0;
static void perm_init(void) {
  uint8_t p[256];
  for (int i = 0; i < 256; i++) p[i] = (uint8_t)i;
  uint32_t s = 1337u;
  for (int i = 255; i > 0; i--) { s = s * 1664525u + 1013904223u; uint32_t j = s % (uint32_t)(i + 1); uint8_t t = p[i]; p[i] = p[j]; p[j] = t; }
  for (int i = 0; i < 512; i++) PERM[i] = p[i & 255];
  perm_ready = 1;
}

#define F3 (1.0f / 3.0f)
#define G3 (1.0f / 6.0f)

static inline void corner(float x, float y, float z, int gi, float* acc) {
  float tt = ((0.5f - x * x) - y * y) - z * z;
  if (tt > 0.0f) {
    float gx = GRAD[gi], gy = GRAD[gi + 1], gz = GRAD[gi + 2];
    float t2 = tt * tt, t4 = t2 * t2;
    float gd = (gx * x + gy * y) + gz * z;
    acc[0] = acc[0] + t4 * gd;
    float m = ((-8.0f * t2) * tt) * gd;
    acc[1] = acc[1] + (m * x + t4 * gx);
    acc[2] = acc[2] + (m * y + t4 * gy);
    acc[3] = acc[3] + (m * z + t4 * gz);
  }
}

static inline void snoise_grad(float x, float y, float z, float* out) {
  float s = ((x + y) + z) * F3;
  float fi = FLOOR(x + s), fj = FLOOR(y + s), fk = FLOOR(z + s);
  float t = ((fi + fj) + fk) * G3;
  float x0 = x - (fi - t), y0 = y - (fj - t), z0 = z - (fk - t);
  int i1, j1, k1, i2, j2, k2;
  if (x0 >= y0) {
    if (y0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
    else if (x0 >= z0) { i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; }
    else { i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; }
  } else {
    if (y0 < z0) { i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; }
    else if (x0 < z0) { i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; }
    else { i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; }
  }
  const float G3x2 = 2.0f * G3, G3x3 = 3.0f * G3;
  float x1 = (x0 - (float)i1) + G3, y1 = (y0 - (float)j1) + G3, z1 = (z0 - (float)k1) + G3;
  float x2 = (x0 - (float)i2) + G3x2, y2 = (y0 - (float)j2) + G3x2, z2 = (z0 - (float)k2) + G3x2;
  float x3 = (x0 - 1.0f) + G3x3, y3 = (y0 - 1.0f) + G3x3, z3 = (z0 - 1.0f) + G3x3;
  int ii = ((int)fi) & 255, jj = ((int)fj) & 255, kk = ((int)fk) & 255;
  float acc[4] = { 0, 0, 0, 0 };
  corner(x0, y0, z0, (PERM[ii + PERM[jj + PERM[kk]]] & 15) * 3, acc);
  corner(x1, y1, z1, (PERM[ii + i1 + PERM[jj + j1 + PERM[kk + k1]]] & 15) * 3, acc);
  corner(x2, y2, z2, (PERM[ii + i2 + PERM[jj + j2 + PERM[kk + k2]]] & 15) * 3, acc);
  corner(x3, y3, z3, (PERM[ii + 1 + PERM[jj + 1 + PERM[kk + 1]]] & 15) * 3, acc);
  out[0] = 32.0f * acc[1]; out[1] = 32.0f * acc[2]; out[2] = 32.0f * acc[3];
}

static inline void curl(float x, float y, float z, float* out) {
  float g[9];
  snoise_grad(x, y, z, g);
  snoise_grad(x + 31.416f, y - 17.23f, z + 5.71f, g + 3);
  snoise_grad(x - 12.87f, y + 47.31f, z - 23.9f, g + 6);
  out[0] = g[7] - g[5];
  out[1] = g[2] - g[6];
  out[2] = g[3] - g[1];
}

EXPORT(grid) void grid(void) {
  if (!perm_ready) perm_init();
  if (!(((int32_t*)U)[U_FLAGS] & FLAG_WARP)) return;
  float ox = U[U_WOX], oy = U[U_WOY], oz = U[U_WOZ];
  int q = 0;
  for (int gz = 0; gz < WG; gz++) {
    float pz = ((W_LO + (float)gz * W_CELL) * W_NF) + oz;
    for (int gy = 0; gy < WG; gy++) {
      float py = ((W_LO + (float)gy * W_CELL) * W_NF) + oy;
      for (int gx = 0; gx < WG; gx++, q += 3) {
        float px = ((W_LO + (float)gx * W_CELL) * W_NF) + ox;
        curl(px, py, pz, W + q);
#ifdef __wasm_simd128__
        float* w4 = W4 + (q / 3) * 4; w4[0] = W[q]; w4[1] = W[q + 1]; w4[2] = W[q + 2]; w4[3] = 0.0f;
#endif
      }
    }
  }
}

// ---- per-particle pieces shared by both paths ----

static inline void warp_one(float* h, float amp) {
  float fx = (h[0] - W_LO) * W_INV, fy = (h[1] - W_LO) * W_INV, fz = (h[2] - W_LO) * W_INV;
  fx = fx < 0.0f ? 0.0f : fx > W_GMAX ? W_GMAX : fx;
  fy = fy < 0.0f ? 0.0f : fy > W_GMAX ? W_GMAX : fy;
  fz = fz < 0.0f ? 0.0f : fz > W_GMAX ? W_GMAX : fz;
  int ix = (int)fx, iy = (int)fy, iz = (int)fz;
  float ax = fx - (float)ix, ay = fy - (float)iy, az = fz - (float)iz;
  float bx = 1.0f - ax, by = 1.0f - ay, bz = 1.0f - az;
  float w0 = (bx * by) * bz, w1 = (ax * by) * bz, w2 = (bx * ay) * bz, w3 = (ax * ay) * bz;
  float w4 = (bx * by) * az, w5 = (ax * by) * az, w6 = (bx * ay) * az, w7 = (ax * ay) * az;
  const int G = WG, G2 = WG * WG;
  int b0 = (iz * G2 + iy * G + ix) * 3;
  int b1 = b0 + 3, b2 = b0 + G * 3, b3 = b2 + 3, b4 = b0 + G2 * 3, b5 = b4 + 3, b6 = b4 + G * 3, b7 = b6 + 3;
  for (int d = 0; d < 3; d++) {
    float s = ((((((W[b0 + d] * w0 + W[b1 + d] * w1) + W[b2 + d] * w2) + W[b3 + d] * w3) + W[b4 + d] * w4) + W[b5 + d] * w5) + W[b6 + d] * w6) + W[b7 + d] * w7;
    h[d] = h[d] + amp * s;
  }
}

// Tear + burst for one particle. p = its current (last frame) position; v = velocity (in/out); f = tear state.
static inline void tear_one(uint32_t i, float x, float y, float z, float* v, float* f, int flags, uint32_t frame) {
  const float* vp = U + U_VP;
  float w = ((vp[3] * x + vp[7] * y) + vp[11] * z) + vp[15];
  if (w <= 0.05f) return;
  float cxp = ((vp[0] * x + vp[4] * y) + vp[8] * z) + vp[12];
  float cyp = ((vp[1] * x + vp[5] * y) + vp[9] * z) + vp[13];
  float qx = ((cxp / w) * 0.5f + 0.5f) * U[U_SW];
  float qy = (0.5f - (cyp / w) * 0.5f) * U[U_SH];
  if (flags & FLAG_TEAR) {
    float ex = qx - U[U_TX], ey = qy - U[U_TY], d2 = ex * ex + ey * ey, tr2 = U[U_TR2];
    if (d2 < tr2) {
      float ff = 1.0f - d2 / tr2; ff = ff * ff;
      if (*f <= 0.0f) { v[0] = 0; v[1] = 0; v[2] = 0; }
      float pull = ff * 0.35f, jit = 0.06f * ff;
      v[0] = (v[0] + (U[U_WVX] * 1.2f - v[0]) * pull) + (rnd(i, frame, 11) - 0.5f) * jit;
      v[1] = (v[1] + (U[U_WVY] * 1.2f - v[1]) * pull) + (rnd(i, frame, 12) - 0.5f) * jit;
      v[2] = (v[2] + (U[U_WVZ] * 1.2f - v[2]) * pull) + (rnd(i, frame, 13) - 0.5f) * jit;
      if (ff > *f) *f = ff;
    }
  }
  if (flags & FLAG_BURST) {
    float ex = qx - U[U_BX], ey = qy - U[U_BY], d2 = ex * ex + ey * ey, br2 = U[U_BR2];
    if (d2 < br2) {
      float t = 1.0f - d2 / br2, ff = t * SQRT(t);
      float l = SQRT(d2) + 1e-3f, ux = ex / l, uy = -ey / l, s = 1.6f * ff;
      float zr = (rnd(i, frame, 14) - 0.5f) * 0.8f;
      if (*f <= 0.0f) { v[0] = 0; v[1] = 0; v[2] = 0; }
      const float *R = U + U_RIGHT, *Up = U + U_UP, *Fw = U + U_FWD;
      v[0] = v[0] + ((R[0] * ux + Up[0] * uy) + Fw[0] * zr) * s;
      v[1] = v[1] + ((R[1] * ux + Up[1] * uy) + Fw[1] * zr) * s;
      v[2] = v[2] + ((R[2] * ux + Up[2] * uy) + Fw[2] * zr) * s;
      if (ff > *f) *f = ff;
    }
  }
}

// Smoke / return / attach for one particle. Returns 1 if torn (counted).
static inline int smoke_one(uint32_t i, float* p4, float* v, float* fptr, const float* h, float b) {
  float f = *fptr;
  if (f == 0.0f) { p4[0] = h[0]; p4[1] = h[1]; p4[2] = h[2]; p4[3] = b; return 0; }
  float x = p4[0], y = p4[1], z = p4[2];
  float dt = U[U_DT];
  if (f < 0.0f) {
    if (f < -MORPH_EASE) { *fptr = f + dt; p4[3] = b; return 0; }
    float rr = f < -0.36f ? U[U_BACK_MORPH] : U[U_BACK];
    x = x + (h[0] - x) * rr; y = y + (h[1] - y) * rr; z = z + (h[2] - z) * rr;
    f = f + dt; if (f > 0.0f) f = 0.0f;
    p4[0] = x; p4[1] = y; p4[2] = z; p4[3] = b; *fptr = f;
    return 0;
  }
  float c[3];
  float freq = U[U_FREQ];
  curl(x * freq + U[U_SOX], y * freq + U[U_SOY], z * freq + U[U_SOZ], c);
  float g = 1.0f - f, pull = ((7.0f * g) * g) * g;
  float fa = U[U_AMP] * (0.35f + 0.65f * f);
  float relax = U[U_RELAX];
  float vx = v[0], vy = v[1], vz = v[2];
  vx = vx + (((c[0] * fa) + (h[0] - x) * pull) - vx) * relax;
  vy = vy + (((c[1] * fa) + (h[1] - y) * pull) - vy) * relax;
  vz = vz + (((c[2] * fa) + (h[2] - z) * pull) - vz) * relax;
  v[0] = vx; v[1] = vy; v[2] = vz;
  p4[0] = x + vx * dt; p4[1] = y + vy * dt; p4[2] = z + vz * dt;
  p4[3] = 0.7f + (4.5f * f) * f;
  f = f - HEAL[i] * dt;
  *fptr = f > 0.0f ? f : -0.35f;
  return 1;
}

// ODE step for one particle (scalar). Writes new state into s, returns derivative in d.
static inline void ode_one(uint32_t i, float* s, float* d, int kind, uint32_t frame) {
  float x = s[0], y = s[1], z = s[2], dx, dy, dz;
  float a = U[U_A], b = U[U_B], k = U[U_K];
  if (kind == 0) {
    float c = U[U_C], dd = U[U_D], e = U[U_E], fp = U[U_FP];
    float zb = z - b;
    dx = zb * x - dd * y;
    dy = dd * x + zb * y;
    dz = (((c + a * z) - ((z * z) * z) / 3.0f) - (x * x + y * y) * (1.0f + e * z)) + (((fp * z) * x) * x) * x;
  } else if (kind == 1) {
    dx = fsin(k * y) - b * x; dy = fsin(k * z) - b * y; dz = fsin(k * x) - b * z;
  } else if (kind == 2) {
    dx = ((-a * x - k * y) - k * z) - y * y;
    dy = ((-a * y - k * z) - k * x) - z * z;
    dz = ((-a * z - k * x) - k * y) - x * x;
  } else {
    float r = U[U_R], sg = U[U_SG];
    dx = sg * (y - x); dy = x * (r - z) - y; dz = x * y - b * z;
  }
  float h = U[U_H], bound = U[U_BOUND];
  x = x + dx * h; y = y + dy * h; z = z + dz * h;
  if (!(x > -bound && x < bound && y > -bound && y < bound && z > -bound && z < bound)) {
    float jit = U[U_JIT];
    x = U[U_INIT] + (rnd(i, frame, 1) - 0.5f) * jit;
    y = U[U_INIT + 1] + (rnd(i, frame, 2) - 0.5f) * jit;
    z = U[U_INIT + 2] + (rnd(i, frame, 3) - 0.5f) * jit;
  }
  s[0] = x; s[1] = y; s[2] = z; d[0] = dx; d[1] = dy; d[2] = dz;
}

// Home, colour, emission, sample for one particle from its new state s and derivative d.
static inline void home_one(uint32_t i, const float* s, const float* d, float* h, float* bOut, float f) {
  float speed = SQRT((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]);
  const float* o = U + U_ORIENT;
  float sc = U[U_SC];
  float rx = (s[0] - U[U_CX]) * sc, ry = (s[1] - U[U_CY]) * sc, rz = (s[2] - U[U_CZ]) * sc;
  h[0] = (o[0] * rx + o[1] * ry) + o[2] * rz;
  h[1] = (o[3] * rx + o[4] * ry) + o[5] * rz;
  h[2] = (o[6] * rx + o[7] * ry) + o[8] * rz;
  T[i] = speed * U[U_INV_SPEED] + (f > 0.0f ? f * 0.3f : 0.0f);
  float hs = speed * U[U_INV_HOT] - 1.05f;
  float q = (3.2f * hs) * hs + 1.2f * hs;
  *bOut = hs > 0.0f ? 0.55f + (q < 5.5f ? q : 5.5f) : 0.55f;
  if ((i & 63) == 0) { float* ss = SS + (i >> 6) * 5; ss[0] = s[0]; ss[1] = s[1]; ss[2] = s[2]; ss[3] = speed; ss[4] = h[1]; }
}

// ======================= scalar fused path (canonical AoS) =======================
EXPORT(run_scalar) int32_t run_scalar(int32_t i0, int32_t i1) {
  int flags = ((int32_t*)U)[U_FLAGS], kind = ((int32_t*)U)[U_KIND];
  uint32_t frame = (uint32_t)((int32_t*)U)[U_FRAME];
  float amp = U[U_WARP_AMP];
  int torn = 0;
  for (int32_t i = i0; i < i1; i++) {
    float* s = S + i * 3; float* v = V + i * 3; float* p4 = P + i * 4;
    float d[3], h[3], b;
    float f = F[i];
    ode_one(i, s, d, kind, frame);
    home_one(i, s, d, h, &b, f);           // T uses F before this frame's tear, as in the reference
    if (flags & FLAG_WARP) warp_one(h, amp);
    if (flags & (FLAG_TEAR | FLAG_BURST)) tear_one(i, p4[0], p4[1], p4[2], v, F + i, flags, frame);
    torn += smoke_one(i, p4, v, F + i, h, b);
  }
  return torn;
}

#ifdef __wasm_simd128__
// ======================= SIMD path (SoA S/V, 4 particles per step) =======================
EXPORT(load_soa) void load_soa(void) {           // canonical AoS -> SoA (once, after loading init)
  for (int i = 0; i < N; i++) { SX[i] = S[i * 3]; SY[i] = S[i * 3 + 1]; SZ[i] = S[i * 3 + 2]; VX[i] = V[i * 3]; VY[i] = V[i * 3 + 1]; VZ[i] = V[i * 3 + 2]; }
}
EXPORT(load_s_soa) void load_s_soa(void) {       // after the host reseeds S (shape morph): S only, V untouched
  for (int i = 0; i < N; i++) { SX[i] = S[i * 3]; SY[i] = S[i * 3 + 1]; SZ[i] = S[i * 3 + 2]; }
}
EXPORT(store_soa) void store_soa(void) {         // SoA -> canonical AoS (only for checksums)
  for (int i = 0; i < N; i++) { S[i * 3] = SX[i]; S[i * 3 + 1] = SY[i]; S[i * 3 + 2] = SZ[i]; V[i * 3] = VX[i]; V[i * 3 + 1] = VY[i]; V[i * 3 + 2] = VZ[i]; }
}

static inline v128_t vfsin(v128_t x) {
  v128_t q = wasm_f32x4_floor(wasm_f32x4_add(wasm_f32x4_mul(x, wasm_f32x4_splat(INV_PI)), wasm_f32x4_splat(0.5f)));
  v128_t r = wasm_f32x4_sub(wasm_f32x4_sub(x, wasm_f32x4_mul(q, wasm_f32x4_splat(PI_A))), wasm_f32x4_mul(q, wasm_f32x4_splat(PI_B)));
  v128_t r2 = wasm_f32x4_mul(r, r);
  v128_t p = wasm_f32x4_add(wasm_f32x4_splat(1.0f / 362880.0f), wasm_f32x4_mul(r2, wasm_f32x4_splat(-1.0f / 39916800.0f)));
  p = wasm_f32x4_add(wasm_f32x4_splat(-1.0f / 5040.0f), wasm_f32x4_mul(r2, p));
  p = wasm_f32x4_add(wasm_f32x4_splat(1.0f / 120.0f), wasm_f32x4_mul(r2, p));
  p = wasm_f32x4_add(wasm_f32x4_splat(-1.0f / 6.0f), wasm_f32x4_mul(r2, p));
  p = wasm_f32x4_add(r, wasm_f32x4_mul(wasm_f32x4_mul(r, r2), p));
  v128_t odd = wasm_v128_and(wasm_i32x4_trunc_sat_f32x4(q), wasm_i32x4_splat(1));
  v128_t m = wasm_i32x4_eq(odd, wasm_i32x4_splat(1));
  return wasm_v128_bitselect(wasm_f32x4_neg(p), p, m);
}

#define LD(a, i) wasm_v128_load((a) + (i))
#define ST(a, i, v) wasm_v128_store((a) + (i), (v))
#define SPL(x) wasm_f32x4_splat(x)
#define ADD wasm_f32x4_add
#define SUB wasm_f32x4_sub
#define MUL wasm_f32x4_mul
#define DIV wasm_f32x4_div

// v1: vectorises ODE + home only; warp, tear and smoke run per lane.
EXPORT(run_simd1) int32_t run_simd1(int32_t i0, int32_t i1) {   // i0, i1 multiples of 4
  const int32_t* UI = (const int32_t*)U;
  int flags = UI[U_FLAGS], kind = UI[U_KIND];
  uint32_t frame = (uint32_t)UI[U_FRAME];
  const v128_t a = SPL(U[U_A]), b = SPL(U[U_B]), k = SPL(U[U_K]), c = SPL(U[U_C]), dd = SPL(U[U_D]), e = SPL(U[U_E]), fp = SPL(U[U_FP]);
  const v128_t r = SPL(U[U_R]), sg = SPL(U[U_SG]), hh = SPL(U[U_H]), bound = SPL(U[U_BOUND]), nbound = SPL(-U[U_BOUND]);
  const v128_t sc = SPL(U[U_SC]), cx = SPL(U[U_CX]), cy = SPL(U[U_CY]), cz = SPL(U[U_CZ]);
  const v128_t invS = SPL(U[U_INV_SPEED]), invH = SPL(U[U_INV_HOT]);
  const float* o = U + U_ORIENT;
  const v128_t o0 = SPL(o[0]), o1 = SPL(o[1]), o2 = SPL(o[2]), o3 = SPL(o[3]), o4 = SPL(o[4]), o5 = SPL(o[5]), o6 = SPL(o[6]), o7 = SPL(o[7]), o8 = SPL(o[8]);
  const v128_t zero = SPL(0.0f), third = SPL(3.0f), one = SPL(1.0f);
  const float amp = U[U_WARP_AMP];
  int torn = 0;
  float H[12], Bv[4];

  for (int32_t i = i0; i < i1; i += 4) {
    v128_t x = LD(SX, i), y = LD(SY, i), z = LD(SZ, i), dx, dy, dz;
    if (kind == 2) {
      dx = SUB(SUB(SUB(MUL(wasm_f32x4_neg(a), x), MUL(k, y)), MUL(k, z)), MUL(y, y));
      dy = SUB(SUB(SUB(MUL(wasm_f32x4_neg(a), y), MUL(k, z)), MUL(k, x)), MUL(z, z));
      dz = SUB(SUB(SUB(MUL(wasm_f32x4_neg(a), z), MUL(k, x)), MUL(k, y)), MUL(x, x));
    } else if (kind == 1) {
      dx = SUB(vfsin(MUL(k, y)), MUL(b, x)); dy = SUB(vfsin(MUL(k, z)), MUL(b, y)); dz = SUB(vfsin(MUL(k, x)), MUL(b, z));
    } else if (kind == 0) {
      v128_t zb = SUB(z, b);
      dx = SUB(MUL(zb, x), MUL(dd, y));
      dy = ADD(MUL(dd, x), MUL(zb, y));
      dz = ADD(SUB(SUB(ADD(c, MUL(a, z)), DIV(MUL(MUL(z, z), z), third)), MUL(ADD(MUL(x, x), MUL(y, y)), ADD(one, MUL(e, z)))), MUL(MUL(MUL(MUL(fp, z), x), x), x));
    } else {
      dx = MUL(sg, SUB(y, x)); dy = SUB(MUL(x, SUB(r, z)), y); dz = SUB(MUL(x, y), MUL(b, z));
    }
    x = ADD(x, MUL(dx, hh)); y = ADD(y, MUL(dy, hh)); z = ADD(z, MUL(dz, hh));
    v128_t ok = wasm_v128_and(wasm_v128_and(wasm_v128_and(wasm_f32x4_gt(x, nbound), wasm_f32x4_lt(x, bound)),
      wasm_v128_and(wasm_f32x4_gt(y, nbound), wasm_f32x4_lt(y, bound))), wasm_v128_and(wasm_f32x4_gt(z, nbound), wasm_f32x4_lt(z, bound)));
    ST(SX, i, x); ST(SY, i, y); ST(SZ, i, z);
    if (!wasm_i32x4_all_true(ok)) {                   // rare: respawn escaped lanes (scalar, same maths)
      float jit = U[U_JIT];
      for (int l = 0; l < 4; l++) {
        float xs = SX[i + l], ys = SY[i + l], zs = SZ[i + l], bb = U[U_BOUND];
        if (!(xs > -bb && xs < bb && ys > -bb && ys < bb && zs > -bb && zs < bb)) {
          SX[i + l] = U[U_INIT] + (rnd(i + l, frame, 1) - 0.5f) * jit;
          SY[i + l] = U[U_INIT + 1] + (rnd(i + l, frame, 2) - 0.5f) * jit;
          SZ[i + l] = U[U_INIT + 2] + (rnd(i + l, frame, 3) - 0.5f) * jit;
        }
      }
      x = LD(SX, i); y = LD(SY, i); z = LD(SZ, i);
    }
    v128_t speed = wasm_f32x4_sqrt(ADD(ADD(MUL(dx, dx), MUL(dy, dy)), MUL(dz, dz)));
    v128_t rx = MUL(SUB(x, cx), sc), ry = MUL(SUB(y, cy), sc), rz = MUL(SUB(z, cz), sc);
    v128_t hx = ADD(ADD(MUL(o0, rx), MUL(o1, ry)), MUL(o2, rz));
    v128_t hy = ADD(ADD(MUL(o3, rx), MUL(o4, ry)), MUL(o5, rz));
    v128_t hz = ADD(ADD(MUL(o6, rx), MUL(o7, ry)), MUL(o8, rz));
    v128_t fv = LD(F, i);
    v128_t tpos = wasm_f32x4_gt(fv, zero);
    ST(T, i, ADD(MUL(speed, invS), wasm_v128_and(MUL(fv, SPL(0.3f)), tpos)));
    v128_t hs = SUB(MUL(speed, invH), SPL(1.05f));
    v128_t q = ADD(MUL(MUL(SPL(3.2f), hs), hs), MUL(SPL(1.2f), hs));
    v128_t qm = wasm_v128_bitselect(q, SPL(5.5f), wasm_f32x4_lt(q, SPL(5.5f)));
    v128_t bv = wasm_v128_bitselect(ADD(SPL(0.55f), qm), SPL(0.55f), wasm_f32x4_gt(hs, zero));
    if ((i & 63) == 0) {
      float* ss = SS + (i >> 6) * 5;
      ss[0] = wasm_f32x4_extract_lane(x, 0); ss[1] = wasm_f32x4_extract_lane(y, 0); ss[2] = wasm_f32x4_extract_lane(z, 0);
      ss[3] = wasm_f32x4_extract_lane(speed, 0); ss[4] = wasm_f32x4_extract_lane(hy, 0);
    }
    // homes to scratch (xyz per lane) for warp + smoke
    wasm_v128_store(Bv, bv);
    {
      float tx[4], ty[4], tz[4];
      wasm_v128_store(tx, hx); wasm_v128_store(ty, hy); wasm_v128_store(tz, hz);
      for (int l = 0; l < 4; l++) { H[l * 3] = tx[l]; H[l * 3 + 1] = ty[l]; H[l * 3 + 2] = tz[l]; }
    }
    if (flags & FLAG_WARP) for (int l = 0; l < 4; l++) warp_one(H + l * 3, amp);

    // tear / burst (scalar per lane; rarely active)
    if (flags & (FLAG_TEAR | FLAG_BURST)) {
      for (int l = 0; l < 4; l++) {
        float vv[3] = { VX[i + l], VY[i + l], VZ[i + l] };
        float* p4 = P + (i + l) * 4;
        tear_one(i + l, p4[0], p4[1], p4[2], vv, F + i + l, flags, frame);
        VX[i + l] = vv[0]; VY[i + l] = vv[1]; VZ[i + l] = vv[2];
      }
      fv = LD(F, i);
    }

    // smoke / attach: all-attached groups take the vector store path
    if (wasm_i32x4_all_true(wasm_f32x4_eq(fv, zero))) {
      for (int l = 0; l < 4; l++) {
        v128_t row = wasm_f32x4_make(H[l * 3], H[l * 3 + 1], H[l * 3 + 2], Bv[l]);
        wasm_v128_store(P + (i + l) * 4, row);
      }
    } else {
      for (int l = 0; l < 4; l++) {
        float vv[3] = { VX[i + l], VY[i + l], VZ[i + l] };
        torn += smoke_one(i + l, P + (i + l) * 4, vv, F + i + l, H + l * 3, Bv[l]);
        VX[i + l] = vv[0]; VY[i + l] = vv[1]; VZ[i + l] = vv[2];
      }
    }
  }
  return torn;
}

// ---------------------------------------------------------------------------------
// v2: everything that is common is vectorised, and torn particles are compacted.
//  * Homes are kept as per-particle xyzw rows (x, y, z, emission) = the P vertex
//    format, so an attached particle's output is one 16-byte store.
//  * Warp: the grid is padded to xyz_ so one v128 load fetches a whole cell; the
//    trilinear sum runs across xyz in one vector, per particle (same order per axis).
//  * Smoke: torn particles are pushed to a list and processed 4-at-a-time with a
//    fully vectorised simplex/curl (table lookups stay scalar: wasm has no gather).
//    Consecutive particles are rarely all torn, so without compaction most lanes idle.

static inline v128_t lane(v128_t v, int l) {
  switch (l) { case 0: return wasm_i32x4_shuffle(v, v, 0, 0, 0, 0); case 1: return wasm_i32x4_shuffle(v, v, 1, 1, 1, 1);
    case 2: return wasm_i32x4_shuffle(v, v, 2, 2, 2, 2); default: return wasm_i32x4_shuffle(v, v, 3, 3, 3, 3); }
}
#define SEL(c, a, b) wasm_v128_bitselect((a), (b), (c))

static inline void vcorner(v128_t x, v128_t y, v128_t z, const int* gi, v128_t* ax, v128_t* ay, v128_t* az) {
  v128_t tt = SUB(SUB(SUB(SPL(0.5f), MUL(x, x)), MUL(y, y)), MUL(z, z));
  v128_t m = wasm_f32x4_gt(tt, SPL(0.0f));
  if (!wasm_v128_any_true(m)) return;
  v128_t gx = wasm_f32x4_make(GRAD[gi[0]], GRAD[gi[1]], GRAD[gi[2]], GRAD[gi[3]]);
  v128_t gy = wasm_f32x4_make(GRAD[gi[0] + 1], GRAD[gi[1] + 1], GRAD[gi[2] + 1], GRAD[gi[3] + 1]);
  v128_t gz = wasm_f32x4_make(GRAD[gi[0] + 2], GRAD[gi[1] + 2], GRAD[gi[2] + 2], GRAD[gi[3] + 2]);
  v128_t t2 = MUL(tt, tt), t4 = MUL(t2, t2);
  v128_t gd = ADD(ADD(MUL(gx, x), MUL(gy, y)), MUL(gz, z));
  v128_t mm = MUL(MUL(MUL(SPL(-8.0f), t2), tt), gd);
  v128_t zero = SPL(0.0f);
  *ax = ADD(*ax, SEL(m, ADD(MUL(mm, x), MUL(t4, gx)), zero));
  *ay = ADD(*ay, SEL(m, ADD(MUL(mm, y), MUL(t4, gy)), zero));
  *az = ADD(*az, SEL(m, ADD(MUL(mm, z), MUL(t4, gz)), zero));
}

static inline void vsnoise_grad(v128_t x, v128_t y, v128_t z, v128_t* ox, v128_t* oy, v128_t* oz) {
  v128_t s = MUL(ADD(ADD(x, y), z), SPL(F3));
  v128_t fi = wasm_f32x4_floor(ADD(x, s)), fj = wasm_f32x4_floor(ADD(y, s)), fk = wasm_f32x4_floor(ADD(z, s));
  v128_t t = MUL(ADD(ADD(fi, fj), fk), SPL(G3));
  v128_t x0 = SUB(x, SUB(fi, t)), y0 = SUB(y, SUB(fj, t)), z0 = SUB(z, SUB(fk, t));
  v128_t xy = wasm_f32x4_ge(x0, y0), yz = wasm_f32x4_ge(y0, z0), xz = wasm_f32x4_ge(x0, z0);
  v128_t one = SPL(1.0f), zero = SPL(0.0f);
  v128_t i1 = wasm_v128_and(xy, xz), j1 = wasm_v128_andnot(yz, xy), k1 = wasm_v128_not(wasm_v128_or(xz, yz));
  v128_t i2 = wasm_v128_or(xy, xz), j2 = wasm_v128_or(wasm_v128_not(xy), yz), k2 = wasm_v128_not(wasm_v128_and(xz, yz));
  v128_t i1f = SEL(i1, one, zero), j1f = SEL(j1, one, zero), k1f = SEL(k1, one, zero);
  v128_t i2f = SEL(i2, one, zero), j2f = SEL(j2, one, zero), k2f = SEL(k2, one, zero);
  const v128_t g1 = SPL(G3), g2 = SPL(2.0f * G3), g3 = SPL(3.0f * G3);
  v128_t x1 = ADD(SUB(x0, i1f), g1), y1 = ADD(SUB(y0, j1f), g1), z1 = ADD(SUB(z0, k1f), g1);
  v128_t x2 = ADD(SUB(x0, i2f), g2), y2 = ADD(SUB(y0, j2f), g2), z2 = ADD(SUB(z0, k2f), g2);
  v128_t x3 = ADD(SUB(x0, one), g3), y3 = ADD(SUB(y0, one), g3), z3 = ADD(SUB(z0, one), g3);
  // table lookups per lane (scalar)
  int32_t ii[4], jj[4], kk[4], a1[4], b1[4], c1[4], a2[4], b2[4], c2[4];
  wasm_v128_store(ii, wasm_v128_and(wasm_i32x4_trunc_sat_f32x4(fi), wasm_i32x4_splat(255)));
  wasm_v128_store(jj, wasm_v128_and(wasm_i32x4_trunc_sat_f32x4(fj), wasm_i32x4_splat(255)));
  wasm_v128_store(kk, wasm_v128_and(wasm_i32x4_trunc_sat_f32x4(fk), wasm_i32x4_splat(255)));
  wasm_v128_store(a1, wasm_i32x4_neg(i1)); wasm_v128_store(b1, wasm_i32x4_neg(j1)); wasm_v128_store(c1, wasm_i32x4_neg(k1));
  wasm_v128_store(a2, wasm_i32x4_neg(i2)); wasm_v128_store(b2, wasm_i32x4_neg(j2)); wasm_v128_store(c2, wasm_i32x4_neg(k2));
  int g0[4], gA[4], gB[4], gC[4];
  for (int l = 0; l < 4; l++) {
    int i = ii[l], j = jj[l], k = kk[l];
    g0[l] = (PERM[i + PERM[j + PERM[k]]] & 15) * 3;
    gA[l] = (PERM[i + a1[l] + PERM[j + b1[l] + PERM[k + c1[l]]]] & 15) * 3;
    gB[l] = (PERM[i + a2[l] + PERM[j + b2[l] + PERM[k + c2[l]]]] & 15) * 3;
    gC[l] = (PERM[i + 1 + PERM[j + 1 + PERM[k + 1]]] & 15) * 3;
  }
  v128_t ax = zero, ay = zero, az = zero;
  vcorner(x0, y0, z0, g0, &ax, &ay, &az);
  vcorner(x1, y1, z1, gA, &ax, &ay, &az);
  vcorner(x2, y2, z2, gB, &ax, &ay, &az);
  vcorner(x3, y3, z3, gC, &ax, &ay, &az);
  *ox = MUL(SPL(32.0f), ax); *oy = MUL(SPL(32.0f), ay); *oz = MUL(SPL(32.0f), az);
}

static inline void vcurl(v128_t x, v128_t y, v128_t z, v128_t* cx, v128_t* cy, v128_t* cz) {
  v128_t g0x, g0y, g0z, g1x, g1y, g1z, g2x, g2y, g2z;
  vsnoise_grad(x, y, z, &g0x, &g0y, &g0z);
  vsnoise_grad(ADD(x, SPL(31.416f)), SUB(y, SPL(17.23f)), ADD(z, SPL(5.71f)), &g1x, &g1y, &g1z);
  vsnoise_grad(SUB(x, SPL(12.87f)), ADD(y, SPL(47.31f)), SUB(z, SPL(23.9f)), &g2x, &g2y, &g2z);
  *cx = SUB(g2y, g1z);    // g[7] - g[5]
  *cy = SUB(g0z, g2x);    // g[2] - g[6]
  *cz = SUB(g1x, g0y);    // g[3] - g[1]
}

// Smoke for 4 torn particles (indices idx[0..3]); same maths as smoke_one's torn branch.
static inline void vsmoke4(const int32_t* idx) {
  float px[4], py[4], pz[4], vx0[4], vy0[4], vz0[4], hx[4], hy[4], hz[4], fv[4], hl[4];
  for (int l = 0; l < 4; l++) {
    int i = idx[l]; const float* p4 = P + i * 4; const float* hb = HB + i * 4;
    px[l] = p4[0]; py[l] = p4[1]; pz[l] = p4[2]; vx0[l] = VX[i]; vy0[l] = VY[i]; vz0[l] = VZ[i];
    hx[l] = hb[0]; hy[l] = hb[1]; hz[l] = hb[2]; fv[l] = F[i]; hl[l] = HEAL[i];
  }
  v128_t x = LD(px, 0), y = LD(py, 0), z = LD(pz, 0), f = LD(fv, 0);
  v128_t freq = SPL(U[U_FREQ]);
  v128_t cx, cy, cz;
  vcurl(ADD(MUL(x, freq), SPL(U[U_SOX])), ADD(MUL(y, freq), SPL(U[U_SOY])), ADD(MUL(z, freq), SPL(U[U_SOZ])), &cx, &cy, &cz);
  v128_t g = SUB(SPL(1.0f), f), pull = MUL(MUL(MUL(SPL(7.0f), g), g), g);
  v128_t fa = MUL(SPL(U[U_AMP]), ADD(SPL(0.35f), MUL(SPL(0.65f), f)));
  v128_t relax = SPL(U[U_RELAX]), dt = SPL(U[U_DT]);
  v128_t vx = LD(vx0, 0), vy = LD(vy0, 0), vz = LD(vz0, 0);
  vx = ADD(vx, MUL(SUB(ADD(MUL(cx, fa), MUL(SUB(LD(hx, 0), x), pull)), vx), relax));
  vy = ADD(vy, MUL(SUB(ADD(MUL(cy, fa), MUL(SUB(LD(hy, 0), y), pull)), vy), relax));
  vz = ADD(vz, MUL(SUB(ADD(MUL(cz, fa), MUL(SUB(LD(hz, 0), z), pull)), vz), relax));
  v128_t nx = ADD(x, MUL(vx, dt)), ny = ADD(y, MUL(vy, dt)), nz = ADD(z, MUL(vz, dt));
  v128_t w = ADD(SPL(0.7f), MUL(MUL(SPL(4.5f), f), f));
  v128_t nf = SUB(f, MUL(LD(hl, 0), dt));
  nf = SEL(wasm_f32x4_gt(nf, SPL(0.0f)), nf, SPL(-0.35f));
  float ox[4], oy[4], oz[4], ow[4], ovx[4], ovy[4], ovz[4], of[4];
  wasm_v128_store(ox, nx); wasm_v128_store(oy, ny); wasm_v128_store(oz, nz); wasm_v128_store(ow, w);
  wasm_v128_store(ovx, vx); wasm_v128_store(ovy, vy); wasm_v128_store(ovz, vz); wasm_v128_store(of, nf);
  for (int l = 0; l < 4; l++) {
    int i = idx[l]; float* p4 = P + i * 4;
    p4[0] = ox[l]; p4[1] = oy[l]; p4[2] = oz[l]; p4[3] = ow[l];
    VX[i] = ovx[l]; VY[i] = ovy[l]; VZ[i] = ovz[l]; F[i] = of[l];
  }
}

EXPORT(run_simd) int32_t run_simd(int32_t i0, int32_t i1) {   // i0, i1 multiples of 4
  const int32_t* UI = (const int32_t*)U;
  int flags = UI[U_FLAGS], kind = UI[U_KIND];
  uint32_t frame = (uint32_t)UI[U_FRAME];
  const v128_t a = SPL(U[U_A]), b = SPL(U[U_B]), k = SPL(U[U_K]), c = SPL(U[U_C]), dd = SPL(U[U_D]), e = SPL(U[U_E]), fp = SPL(U[U_FP]);
  const v128_t r = SPL(U[U_R]), sg = SPL(U[U_SG]), hh = SPL(U[U_H]), bound = SPL(U[U_BOUND]), nbound = SPL(-U[U_BOUND]);
  const v128_t sc = SPL(U[U_SC]), cx = SPL(U[U_CX]), cy = SPL(U[U_CY]), cz = SPL(U[U_CZ]);
  const v128_t invS = SPL(U[U_INV_SPEED]), invH = SPL(U[U_INV_HOT]);
  const float* o = U + U_ORIENT;
  const v128_t o0 = SPL(o[0]), o1 = SPL(o[1]), o2 = SPL(o[2]), o3 = SPL(o[3]), o4 = SPL(o[4]), o5 = SPL(o[5]), o6 = SPL(o[6]), o7 = SPL(o[7]), o8 = SPL(o[8]);
  const v128_t zero = SPL(0.0f), third = SPL(3.0f), one = SPL(1.0f);
  const v128_t wamp = SPL(U[U_WARP_AMP]), wlo = SPL(W_LO), winv = SPL(W_INV), wgmax = SPL(W_GMAX);
  const int warp = flags & FLAG_WARP, tearing = flags & (FLAG_TEAR | FLAG_BURST);
  int32_t* list = TORN + i0; int nt = 0;

  for (int32_t i = i0; i < i1; i += 4) {
    v128_t x = LD(SX, i), y = LD(SY, i), z = LD(SZ, i), dx, dy, dz;
    if (kind == 2) {
      v128_t na = wasm_f32x4_neg(a);
      dx = SUB(SUB(SUB(MUL(na, x), MUL(k, y)), MUL(k, z)), MUL(y, y));
      dy = SUB(SUB(SUB(MUL(na, y), MUL(k, z)), MUL(k, x)), MUL(z, z));
      dz = SUB(SUB(SUB(MUL(na, z), MUL(k, x)), MUL(k, y)), MUL(x, x));
    } else if (kind == 1) {
      dx = SUB(vfsin(MUL(k, y)), MUL(b, x)); dy = SUB(vfsin(MUL(k, z)), MUL(b, y)); dz = SUB(vfsin(MUL(k, x)), MUL(b, z));
    } else if (kind == 0) {
      v128_t zb = SUB(z, b);
      dx = SUB(MUL(zb, x), MUL(dd, y));
      dy = ADD(MUL(dd, x), MUL(zb, y));
      dz = ADD(SUB(SUB(ADD(c, MUL(a, z)), DIV(MUL(MUL(z, z), z), third)), MUL(ADD(MUL(x, x), MUL(y, y)), ADD(one, MUL(e, z)))), MUL(MUL(MUL(MUL(fp, z), x), x), x));
    } else {
      dx = MUL(sg, SUB(y, x)); dy = SUB(MUL(x, SUB(r, z)), y); dz = SUB(MUL(x, y), MUL(b, z));
    }
    x = ADD(x, MUL(dx, hh)); y = ADD(y, MUL(dy, hh)); z = ADD(z, MUL(dz, hh));
    v128_t ok = wasm_v128_and(wasm_v128_and(wasm_v128_and(wasm_f32x4_gt(x, nbound), wasm_f32x4_lt(x, bound)),
      wasm_v128_and(wasm_f32x4_gt(y, nbound), wasm_f32x4_lt(y, bound))), wasm_v128_and(wasm_f32x4_gt(z, nbound), wasm_f32x4_lt(z, bound)));
    ST(SX, i, x); ST(SY, i, y); ST(SZ, i, z);
    if (!wasm_i32x4_all_true(ok)) {
      float jit = U[U_JIT], bb = U[U_BOUND];
      for (int l = 0; l < 4; l++) {
        float xs = SX[i + l], ys = SY[i + l], zs = SZ[i + l];
        if (!(xs > -bb && xs < bb && ys > -bb && ys < bb && zs > -bb && zs < bb)) {
          SX[i + l] = U[U_INIT] + (rnd(i + l, frame, 1) - 0.5f) * jit;
          SY[i + l] = U[U_INIT + 1] + (rnd(i + l, frame, 2) - 0.5f) * jit;
          SZ[i + l] = U[U_INIT + 2] + (rnd(i + l, frame, 3) - 0.5f) * jit;
        }
      }
      x = LD(SX, i); y = LD(SY, i); z = LD(SZ, i);
    }
    v128_t speed = wasm_f32x4_sqrt(ADD(ADD(MUL(dx, dx), MUL(dy, dy)), MUL(dz, dz)));
    v128_t rx = MUL(SUB(x, cx), sc), ry = MUL(SUB(y, cy), sc), rz = MUL(SUB(z, cz), sc);
    v128_t hx = ADD(ADD(MUL(o0, rx), MUL(o1, ry)), MUL(o2, rz));
    v128_t hy = ADD(ADD(MUL(o3, rx), MUL(o4, ry)), MUL(o5, rz));
    v128_t hz = ADD(ADD(MUL(o6, rx), MUL(o7, ry)), MUL(o8, rz));
    v128_t fv = LD(F, i);
    ST(T, i, ADD(MUL(speed, invS), wasm_v128_and(MUL(fv, SPL(0.3f)), wasm_f32x4_gt(fv, zero))));
    v128_t hs = SUB(MUL(speed, invH), SPL(1.05f));
    v128_t q = ADD(MUL(MUL(SPL(3.2f), hs), hs), MUL(SPL(1.2f), hs));
    v128_t bv = SEL(wasm_f32x4_gt(hs, zero), ADD(SPL(0.55f), SEL(wasm_f32x4_lt(q, SPL(5.5f)), q, SPL(5.5f))), SPL(0.55f));
    if ((i & 63) == 0) {
      float* ss = SS + (i >> 6) * 5;
      ss[0] = wasm_f32x4_extract_lane(x, 0); ss[1] = wasm_f32x4_extract_lane(y, 0); ss[2] = wasm_f32x4_extract_lane(z, 0);
      ss[3] = wasm_f32x4_extract_lane(speed, 0); ss[4] = wasm_f32x4_extract_lane(hy, 0);
    }

    // rows[l] = (hx, hy, hz, b) of lane l: 4x4 transpose
    v128_t t0 = wasm_i32x4_shuffle(hx, hy, 0, 4, 1, 5), t1 = wasm_i32x4_shuffle(hx, hy, 2, 6, 3, 7);
    v128_t t2 = wasm_i32x4_shuffle(hz, bv, 0, 4, 1, 5), t3 = wasm_i32x4_shuffle(hz, bv, 2, 6, 3, 7);
    v128_t row[4] = { wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5), wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7),
                      wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5), wasm_i32x4_shuffle(t1, t3, 2, 3, 6, 7) };

    if (warp) {
      v128_t fx = MUL(SUB(hx, wlo), winv), fy = MUL(SUB(hy, wlo), winv), fz = MUL(SUB(hz, wlo), winv);
      fx = SEL(wasm_f32x4_lt(fx, zero), zero, SEL(wasm_f32x4_gt(fx, wgmax), wgmax, fx));
      fy = SEL(wasm_f32x4_lt(fy, zero), zero, SEL(wasm_f32x4_gt(fy, wgmax), wgmax, fy));
      fz = SEL(wasm_f32x4_lt(fz, zero), zero, SEL(wasm_f32x4_gt(fz, wgmax), wgmax, fz));
      v128_t ix = wasm_i32x4_trunc_sat_f32x4(fx), iy = wasm_i32x4_trunc_sat_f32x4(fy), iz = wasm_i32x4_trunc_sat_f32x4(fz);
      v128_t ax = SUB(fx, wasm_f32x4_convert_i32x4(ix)), ay = SUB(fy, wasm_f32x4_convert_i32x4(iy)), az = SUB(fz, wasm_f32x4_convert_i32x4(iz));
      v128_t bx = SUB(one, ax), by = SUB(one, ay), bz = SUB(one, az);
      v128_t w[8] = { MUL(MUL(bx, by), bz), MUL(MUL(ax, by), bz), MUL(MUL(bx, ay), bz), MUL(MUL(ax, ay), bz),
                      MUL(MUL(bx, by), az), MUL(MUL(ax, by), az), MUL(MUL(bx, ay), az), MUL(MUL(ax, ay), az) };
      int32_t cell[4];
      wasm_v128_store(cell, wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_mul(iz, wasm_i32x4_splat(WG * WG)), wasm_i32x4_mul(iy, wasm_i32x4_splat(WG))), ix));
      const int off[8] = { 0, 1, WG, WG + 1, WG * WG, WG * WG + 1, WG * WG + WG, WG * WG + WG + 1 };
      for (int l = 0; l < 4; l++) {
        const float* base = W4 + cell[l] * 4;
        v128_t s = MUL(LD(base, off[0] * 4), lane(w[0], l));
        s = ADD(s, MUL(LD(base, off[1] * 4), lane(w[1], l)));
        s = ADD(s, MUL(LD(base, off[2] * 4), lane(w[2], l)));
        s = ADD(s, MUL(LD(base, off[3] * 4), lane(w[3], l)));
        s = ADD(s, MUL(LD(base, off[4] * 4), lane(w[4], l)));
        s = ADD(s, MUL(LD(base, off[5] * 4), lane(w[5], l)));
        s = ADD(s, MUL(LD(base, off[6] * 4), lane(w[6], l)));
        s = ADD(s, MUL(LD(base, off[7] * 4), lane(w[7], l)));
        row[l] = ADD(row[l], MUL(wamp, s));      // w lane: b + amp*0 = b
      }
    }

    if (tearing) {
      for (int l = 0; l < 4; l++) {
        float vv[3] = { VX[i + l], VY[i + l], VZ[i + l] };
        float* p4 = P + (i + l) * 4;
        tear_one(i + l, p4[0], p4[1], p4[2], vv, F + i + l, flags, frame);
        VX[i + l] = vv[0]; VY[i + l] = vv[1]; VZ[i + l] = vv[2];
      }
      fv = LD(F, i);
    }

    if (wasm_i32x4_all_true(wasm_f32x4_eq(fv, zero))) {
      for (int l = 0; l < 4; l++) wasm_v128_store(P + (i + l) * 4, row[l]);
    } else {
      float fl[4]; wasm_v128_store(fl, fv);
      for (int l = 0; l < 4; l++) {
        int idx = i + l; float f = fl[l]; float* p4 = P + idx * 4;
        if (f == 0.0f) { wasm_v128_store(p4, row[l]); continue; }
        if (f > 0.0f) { wasm_v128_store(HB + idx * 4, row[l]); list[nt++] = idx; continue; }
        float hrow[4]; wasm_v128_store(hrow, row[l]);          // returning / morph wait: cheap, scalar
        float vv[3] = { VX[idx], VY[idx], VZ[idx] };
        smoke_one(idx, p4, vv, F + idx, hrow, hrow[3]);
      }
    }
  }

  // Phase B: smoke for the compacted torn list, 4 at a time.
  int j = 0;
  for (; j + 4 <= nt; j += 4) vsmoke4(list + j);
  for (; j < nt; j++) {
    int idx = list[j]; float hrow[4];
    for (int d = 0; d < 4; d++) hrow[d] = HB[idx * 4 + d];
    float vv[3] = { VX[idx], VY[idx], VZ[idx] };
    smoke_one(idx, P + idx * 4, vv, F + idx, hrow, hrow[3]);
    VX[idx] = vv[0]; VY[idx] = vv[1]; VZ[idx] = vv[2];
  }
  return nt;
}
#endif

// Accessor for the native runner (wasm callers use the setup() pointer table instead).
#ifdef __wasm__
__attribute__((no_builtin("memset"), used)) void* memset(void* d, int c, unsigned long n) {
  unsigned char* p = d; while (n--) *p++ = (unsigned char)c; return d;
}
__attribute__((no_builtin("memcpy"), used)) void* memcpy(void* d, const void* s, unsigned long n) {
  unsigned char* p = d; const unsigned char* q = s; while (n--) *p++ = *q++; return d;
}
#endif
#include "links.c"

EXPORT(ptr) float* ptr(int32_t which) {
  float* t[9] = { S, P, V, F, T, HEAL, SS, W, U };
  return t[which];
}
