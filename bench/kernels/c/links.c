// Cursor-link search: "the K lowest-priority particles, among every 4th, whose screen
// projection falls inside the cursor circle". priority(i) = i * 2654435761 (mod 2^32),
// a bijection, so the answer is a unique set. Projection is exact f32 in the same
// order as the JS variants in bench/links.mjs, so every variant returns the identical set.
// Included from kernel.c (shares P, U, N and the bump allocator).

#define LK 40
#define LSTRIDE 4
#define LSLOTS 64
enum { U_LCX = 86, U_LCY = 87, U_LR2 = 88 };

static int32_t *LOUT;          // per slot: LK indices then LK priorities
static uint32_t *ORDER;        // stride-4 indices sorted by priority (built once per N)
static int32_t NORDER;
static float *GSX, *GSY;       // grid variant: projected positions of stride-4 particles
static int32_t *GCELL, *GSTART, *GITEMS;
#define GRID_CELL 150.0f
#define GRID_MAXC 4096

static inline uint32_t lprio(uint32_t i) { return i * 2654435761u; }

static void links_alloc(int32_t n) {
  int32_t m = (n + LSTRIDE - 1) / LSTRIDE;
  LOUT = bump(LSLOTS * LK * 8); ORDER = bump(m * 4); NORDER = 0;
  GSX = bump(m * 4); GSY = bump(m * 4); GCELL = bump(m * 4); GSTART = bump((GRID_MAXC + 1) * 4); GITEMS = bump(m * 4);
}

// Projects particle i; returns 1 and its squared screen distance if inside the circle.
static inline int lhit(int32_t i, float* d2out) {
  const float* vp = U + U_VP; const float* p = P + i * 4;
  float x = p[0], y = p[1], z = p[2];
  float w = ((vp[3] * x + vp[7] * y) + vp[11] * z) + vp[15];
  if (!(w > 0.05f)) return 0;
  float cxp = ((vp[0] * x + vp[4] * y) + vp[8] * z) + vp[12];
  float cyp = ((vp[1] * x + vp[5] * y) + vp[9] * z) + vp[13];
  float ex = ((cxp / w) * 0.5f + 0.5f) * U[U_SW] - U[U_LCX];
  float ey = (0.5f - (cyp / w) * 0.5f) * U[U_SH] - U[U_LCY];
  float d2 = ex * ex + ey * ey;
  *d2out = d2;
  return d2 < U[U_LR2];
}

// Top-K by lowest priority. Returns the new count.
static inline int topk_push(int32_t* idx, uint32_t* pri, int m, int32_t i, uint32_t pr) {
  if (m < LK) { idx[m] = i; pri[m] = pr; return m + 1; }
  int worst = 0;
  for (int q = 1; q < LK; q++) if (pri[q] > pri[worst]) worst = q;
  if (pr < pri[worst]) { idx[worst] = i; pri[worst] = pr; }
  return m;
}

// ---- 1. brute force: scan every 4th particle in [i0, i1) (thread-sliceable) ----
EXPORT(links_brute) int32_t links_brute(int32_t i0, int32_t i1, int32_t slot) {
  int32_t* idx = LOUT + slot * LK * 2; uint32_t* pri = (uint32_t*)(idx + LK);
  int m = 0; float d2;
  i0 = (i0 + LSTRIDE - 1) / LSTRIDE * LSTRIDE;
  for (int32_t i = i0; i < i1; i += LSTRIDE) if (lhit(i, &d2)) m = topk_push(idx, pri, m, i, lprio((uint32_t)i));
  return m;
}

#ifdef __wasm_simd128__
// ---- 2. brute force, SIMD: 4 particles per step (4x4 transpose of their xyzw rows) ----
EXPORT(links_brute_simd) int32_t links_brute_simd(int32_t i0, int32_t i1, int32_t slot) {
  int32_t* idx = LOUT + slot * LK * 2; uint32_t* pri = (uint32_t*)(idx + LK);
  const float* vp = U + U_VP;
  const v128_t v0 = SPL(vp[0]), v1 = SPL(vp[1]), v3 = SPL(vp[3]), v4 = SPL(vp[4]), v5 = SPL(vp[5]), v7 = SPL(vp[7]);
  const v128_t v8 = SPL(vp[8]), v9 = SPL(vp[9]), v11 = SPL(vp[11]), v12 = SPL(vp[12]), v13 = SPL(vp[13]), v15 = SPL(vp[15]);
  const v128_t sw = SPL(U[U_SW]), sh = SPL(U[U_SH]), lcx = SPL(U[U_LCX]), lcy = SPL(U[U_LCY]), r2 = SPL(U[U_LR2]);
  const v128_t half = SPL(0.5f), wmin = SPL(0.05f);
  int m = 0;
  i0 = (i0 + 15) / 16 * 16;
  int32_t i = i0;
  for (; i + 16 <= i1; i += 16) {             // particles i, i+4, i+8, i+12
    v128_t r0 = LD(P, i * 4), r1 = LD(P, (i + 4) * 4), r2v = LD(P, (i + 8) * 4), r3 = LD(P, (i + 12) * 4);
    v128_t t0 = wasm_i32x4_shuffle(r0, r1, 0, 4, 1, 5), t1 = wasm_i32x4_shuffle(r0, r1, 2, 6, 3, 7);
    v128_t t2 = wasm_i32x4_shuffle(r2v, r3, 0, 4, 1, 5), t3 = wasm_i32x4_shuffle(r2v, r3, 2, 6, 3, 7);
    v128_t x = wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5), y = wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7), z = wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5);
    v128_t w = ADD(ADD(ADD(MUL(v3, x), MUL(v7, y)), MUL(v11, z)), v15);
    v128_t cxp = ADD(ADD(ADD(MUL(v0, x), MUL(v4, y)), MUL(v8, z)), v12);
    v128_t cyp = ADD(ADD(ADD(MUL(v1, x), MUL(v5, y)), MUL(v9, z)), v13);
    v128_t ex = SUB(MUL(ADD(MUL(DIV(cxp, w), half), half), sw), lcx);
    v128_t ey = SUB(MUL(SUB(half, MUL(DIV(cyp, w), half)), sh), lcy);
    v128_t d2 = ADD(MUL(ex, ex), MUL(ey, ey));
    v128_t hit = wasm_v128_and(wasm_f32x4_gt(w, wmin), wasm_f32x4_lt(d2, r2));
    int mask = wasm_i32x4_bitmask(hit);
    while (mask) {
      int l = __builtin_ctz(mask); mask &= mask - 1;
      int32_t j = i + l * 4;
      m = topk_push(idx, pri, m, j, lprio((uint32_t)j));
    }
  }
  float d; for (; i < i1; i += LSTRIDE) if (lhit(i, &d)) m = topk_push(idx, pri, m, i, lprio((uint32_t)i));
  return m;
}
#endif

// ---- 3. screen-space uniform grid (counting sort), then query the cells under the circle ----
EXPORT(links_grid) int32_t links_grid(void) {
  int32_t* idx = LOUT; uint32_t* pri = (uint32_t*)(idx + LK);
  const float* vp = U + U_VP;
  int32_t m = 0, cnt = 0;
  int gw = (int)(U[U_SW] / GRID_CELL) + 1, gh = (int)(U[U_SH] / GRID_CELL) + 1;
  if (gw * gh > GRID_MAXC) return 0;
  for (int c = 0; c <= gw * gh; c++) GSTART[c] = 0;
  // pass 1: project + count per cell
  for (int32_t i = 0; i < N; i += LSTRIDE, cnt++) {
    const float* p = P + i * 4;
    float x = p[0], y = p[1], z = p[2];
    float w = ((vp[3] * x + vp[7] * y) + vp[11] * z) + vp[15];
    int cell = -1;
    if (w > 0.05f) {
      float cxp = ((vp[0] * x + vp[4] * y) + vp[8] * z) + vp[12];
      float cyp = ((vp[1] * x + vp[5] * y) + vp[9] * z) + vp[13];
      float sx = ((cxp / w) * 0.5f + 0.5f) * U[U_SW], sy = (0.5f - (cyp / w) * 0.5f) * U[U_SH];
      GSX[cnt] = sx; GSY[cnt] = sy;
      // off-screen particles are clamped into border cells (a border cursor circle still sees them)
      float fx = sx / GRID_CELL, fy = sy / GRID_CELL;
      if (fx == fx && fy == fy) {
        fx = fx < 0.0f ? 0.0f : fx > (float)(gw - 1) ? (float)(gw - 1) : fx;
        fy = fy < 0.0f ? 0.0f : fy > (float)(gh - 1) ? (float)(gh - 1) : fy;
        cell = (int)fy * gw + (int)fx; GSTART[cell + 1]++;
      }
    }
    GCELL[cnt] = cell;
  }
  for (int c = 0; c < gw * gh; c++) GSTART[c + 1] += GSTART[c];
  // pass 2: scatter
  static int32_t fill[GRID_MAXC];
  for (int c = 0; c < gw * gh; c++) fill[c] = GSTART[c];
  for (int32_t q = 0; q < cnt; q++) if (GCELL[q] >= 0) GITEMS[fill[GCELL[q]]++] = q;
  // query cells overlapping the circle's bounding box (clamped the same way)
  float r = __builtin_sqrtf(U[U_LR2]), cx = U[U_LCX], cy = U[U_LCY];
  float fx0 = (cx - r) / GRID_CELL, fx1 = (cx + r) / GRID_CELL, fy0 = (cy - r) / GRID_CELL, fy1 = (cy + r) / GRID_CELL;
  #define CL(v, hi) ((v) < 0.0f ? 0 : (v) > (float)(hi) ? (hi) : (int)(v))
  int x0 = CL(fx0, gw - 1), x1 = CL(fx1, gw - 1), y0 = CL(fy0, gh - 1), y1 = CL(fy1, gh - 1);
  for (int gy = y0; gy <= y1; gy++)
    for (int gx = x0; gx <= x1; gx++) {
      int c = gy * gw + gx;
      for (int32_t s = GSTART[c]; s < GSTART[c + 1]; s++) {
        int32_t q = GITEMS[s];
        float ex = GSX[q] - cx, ey = GSY[q] - cy;
        if (ex * ex + ey * ey < U[U_LR2]) m = topk_push(idx, pri, m, q * LSTRIDE, lprio((uint32_t)(q * LSTRIDE)));
      }
    }
  return m;
}

// ---- 4. priority-ordered early exit: walk particles in increasing priority, stop at the K-th hit ----
static void radix_sort_by_prio(uint32_t* keys, uint32_t* vals, uint32_t* tk, uint32_t* tv, int32_t n) {
  for (int shift = 0; shift < 32; shift += 8) {
    uint32_t count[257] = { 0 };
    for (int32_t i = 0; i < n; i++) count[((keys[i] >> shift) & 255) + 1]++;
    for (int b = 0; b < 256; b++) count[b + 1] += count[b];
    for (int32_t i = 0; i < n; i++) { uint32_t d = count[(keys[i] >> shift) & 255]++; tk[d] = keys[i]; tv[d] = vals[i]; }
    for (int32_t i = 0; i < n; i++) { keys[i] = tk[i]; vals[i] = tv[i]; }
  }
}
EXPORT(links_setup_order) void links_setup_order(void) {    // O(N) once per particle count
  int32_t m = (N + LSTRIDE - 1) / LSTRIDE;
  uint32_t *keys = (uint32_t*)GSX, *tk = (uint32_t*)GSY, *tv = (uint32_t*)GCELL;   // reuse grid scratch
  for (int32_t q = 0; q < m; q++) { keys[q] = lprio((uint32_t)(q * LSTRIDE)); ORDER[q] = (uint32_t)(q * LSTRIDE); }
  radix_sort_by_prio(keys, ORDER, tk, tv, m);
  NORDER = m;
}
EXPORT(links_order) int32_t links_order(void) {
  int32_t* idx = LOUT; uint32_t* pri = (uint32_t*)(idx + LK);
  int m = 0; float d2;
  for (int32_t q = 0; q < NORDER && m < LK; q++) {
    int32_t i = (int32_t)ORDER[q];
    if (lhit(i, &d2)) { idx[m] = i; pri[m] = lprio((uint32_t)i); m++; }   // already in priority order
  }
  return m;
}
#ifdef __wasm_simd128__
// ---- 5. priority-ordered early exit, 4 at a time (gathered rows, same transpose as 2) ----
EXPORT(links_order_simd) int32_t links_order_simd(void) {
  int32_t* idx = LOUT; uint32_t* pri = (uint32_t*)(idx + LK);
  const float* vp = U + U_VP;
  const v128_t v0 = SPL(vp[0]), v1 = SPL(vp[1]), v3 = SPL(vp[3]), v4 = SPL(vp[4]), v5 = SPL(vp[5]), v7 = SPL(vp[7]);
  const v128_t v8 = SPL(vp[8]), v9 = SPL(vp[9]), v11 = SPL(vp[11]), v12 = SPL(vp[12]), v13 = SPL(vp[13]), v15 = SPL(vp[15]);
  const v128_t sw = SPL(U[U_SW]), sh = SPL(U[U_SH]), lcx = SPL(U[U_LCX]), lcy = SPL(U[U_LCY]), r2 = SPL(U[U_LR2]);
  const v128_t half = SPL(0.5f), wmin = SPL(0.05f);
  int m = 0, q = 0;
  for (; q + 4 <= NORDER; q += 4) {
    const uint32_t* o = ORDER + q;
    v128_t r0 = LD(P, o[0] * 4), r1 = LD(P, o[1] * 4), r2v = LD(P, o[2] * 4), r3 = LD(P, o[3] * 4);
    v128_t t0 = wasm_i32x4_shuffle(r0, r1, 0, 4, 1, 5), t1 = wasm_i32x4_shuffle(r0, r1, 2, 6, 3, 7);
    v128_t t2 = wasm_i32x4_shuffle(r2v, r3, 0, 4, 1, 5), t3 = wasm_i32x4_shuffle(r2v, r3, 2, 6, 3, 7);
    v128_t x = wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5), y = wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7), z = wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5);
    v128_t w = ADD(ADD(ADD(MUL(v3, x), MUL(v7, y)), MUL(v11, z)), v15);
    v128_t cxp = ADD(ADD(ADD(MUL(v0, x), MUL(v4, y)), MUL(v8, z)), v12);
    v128_t cyp = ADD(ADD(ADD(MUL(v1, x), MUL(v5, y)), MUL(v9, z)), v13);
    v128_t ex = SUB(MUL(ADD(MUL(DIV(cxp, w), half), half), sw), lcx);
    v128_t ey = SUB(MUL(SUB(half, MUL(DIV(cyp, w), half)), sh), lcy);
    v128_t d2 = ADD(MUL(ex, ex), MUL(ey, ey));
    int mask = wasm_i32x4_bitmask(wasm_v128_and(wasm_f32x4_gt(w, wmin), wasm_f32x4_lt(d2, r2)));
    while (mask) {
      int l = __builtin_ctz(mask); mask &= mask - 1;
      idx[m] = (int32_t)o[l]; pri[m] = lprio(o[l]);
      if (++m == LK) return m;
    }
  }
  float d; for (; q < NORDER && m < LK; q++) { int32_t i = (int32_t)ORDER[q]; if (lhit(i, &d)) { idx[m] = i; pri[m] = lprio((uint32_t)i); m++; } }
  return m;
}
#endif
EXPORT(links_out) int32_t* links_out(void) { return LOUT; }
