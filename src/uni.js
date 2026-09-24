// Uniform block shared by the WASM (C) and WebGPU (WGSL) kernels. Must match
// bench/core.js UNI, kernel.c and kernel.wgsl.
const UNI = {
  KIND: 0, FRAME: 1, FLAGS: 2,
  A: 4, B: 5, C: 6, D: 7, E: 8, FP: 9, K: 10, R: 11, SG: 12,
  H: 13, BOUND: 14, JIT: 15, INIT: 16,
  SC: 19, CX: 20, CY: 21, CZ: 22, INV_SPEED: 23, INV_HOT: 24, ORIENT: 25,
  WARP_AMP: 34, WOX: 35, WOY: 36, WOZ: 37,
  TX: 39, TY: 40, TR2: 41, WVX: 42, WVY: 43, WVZ: 44,
  BX: 45, BY: 46, BR2: 47, RIGHT: 48, UP: 51, FWD: 54, SW: 57, SH: 58, VP: 59,
  FREQ: 75, AMP: 76, SOX: 77, SOY: 78, SOZ: 79, RELAX: 80, BACK: 81, BACK_MORPH: 82, DT: 83,
  MORPH_TOP: 84, MORPH_SPAN: 85,
  LCX: 86, LCY: 87, LR2: 88,                 // cursor-link search (links.c)
  SIZE: 96,
};
const UFLAG_WARP = 1, UFLAG_TEAR = 2, UFLAG_BURST = 4;
// Cursor links, shared by every backend: up to LINK_K particles within LINK_RADIUS_PX of the cursor.
const LINK_K = 40, LINK_RADIUS_PX = 150;
// Turbulence grid of the kernels: WARP_G³ cells over [-2.2, 2.2]³ (bench/kernels/c/kernel.c: WG, W_CELL).
const WARP_KERNEL_G = 12;
const KIND_OF = { Aizawa: 0, Thomas: 1, Halvorsen: 2, Lorenz: 3 };
const b64bytes = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
