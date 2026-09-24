// Native replay runner: same kernel, same trace, same checksums. Prints one JSON line.
//   native <init.bin> <trace.bin> <n> <frames> <runs> <check frames...>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
int32_t setup(int32_t n); void grid(void); int32_t run_scalar(int32_t i0, int32_t i1); float* ptr(int32_t which);

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static uint32_t fnv(uint32_t h, const float* a, size_t count) {
  const uint32_t* u = (const uint32_t*)a;
  for (size_t i = 0; i < count; i++) { h ^= u[i]; h *= 0x01000193u; }
  return h;
}
static int cmp(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static float* readall(const char* p, size_t* bytes) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
  fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
  float* b = malloc(*bytes); if (fread(b, 1, *bytes, f) != *bytes) exit(1); fclose(f); return b;
}
int main(int argc, char** argv) {
  size_t ib, tb;
  float* init = readall(argv[1], &ib); float* trace = readall(argv[2], &tb);
  int n = atoi(argv[3]), frames = atoi(argv[4]), runs = atoi(argv[5]);
  int nchk = argc - 6; int chk[8]; for (int c = 0; c < nchk; c++) chk[c] = atoi(argv[6 + c]);
  setup(n);
  float *S = ptr(0), *P = ptr(1), *V = ptr(2), *F = ptr(3), *T = ptr(4), *HEAL = ptr(5), *U = ptr(8);
  double* times = malloc(sizeof(double) * frames);
  double best = 1e30, bestMed = 0, bestP95 = 0; uint32_t sums[8];
  for (int r = 0; r < runs; r++) {
    size_t off = 0;
    memcpy(S, init + off, n * 12); off += n * 3; memcpy(P, init + off, n * 16); off += n * 4;
    memcpy(V, init + off, n * 12); off += n * 3; memcpy(F, init + off, n * 4); off += n;
    memcpy(T, init + off, n * 4); off += n; memcpy(HEAL, init + off, n * 4);
    double total = 0; uint32_t rs[8];
    for (int f = 0; f < frames; f++) {
      memcpy(U, trace + (size_t)f * 96, 96 * 4);
      double t0 = now_ms();
      grid(); run_scalar(0, n);
      times[f] = now_ms() - t0; total += times[f];
      for (int c = 0; c < nchk; c++) if (chk[c] == f) {
        uint32_t h = 0x811c9dc5u;
        h = fnv(h, S, n * 3); h = fnv(h, P, n * 4); h = fnv(h, V, n * 3); h = fnv(h, F, n); h = fnv(h, T, n);
        rs[c] = h;
      }
    }
    qsort(times, frames, sizeof(double), cmp);
    if (total / frames < best) { best = total / frames; bestMed = times[frames / 2]; bestP95 = times[(int)(frames * 0.95)]; memcpy(sums, rs, sizeof rs); }
  }
  printf("{\"mean\":%.4f,\"median\":%.4f,\"p95\":%.4f,\"checks\":{", best, bestMed, bestP95);
  for (int c = 0; c < nchk; c++) printf("%s\"%d\":\"%08x\"", c ? "," : "", chk[c], sums[c]);
  printf("}}\n");
  return 0;
}
