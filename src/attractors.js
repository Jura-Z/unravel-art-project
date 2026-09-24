// Strange attractors used by Unravel. Pad ranges were chosen by sweeping for
// escape and collapse (see test/attractor_sweep.cjs). `axes` are the names shown on the pad.

const ATTRACTORS = [
  {
    name: 'Halvorsen', h: 0.0035, spread: 2.5, init: [-1.48, -1.51, 2.04],
    def: { a: 1.89, k: 4 },
    pad: [['a', 1.7, 2.2], ['k', 3.6, 4.3]], axes: ['damping', 'coupling'],
    orient: null
  },
  {
    name: 'Thomas', h: 0.05, spread: 3, init: [1, 0.5, 0],
    def: { b: 0.19, k: 1.0 },
    pad: [['b', 0.12, 0.2], ['k', 0.95, 1.3]], axes: ['damping', 'frequency'],
    orient: null, // (1,1,1) diagonal up
  },
  {
    name: 'Lorenz', h: 0.003, spread: 10, init: [1, 1, 20],
    def: { s: 10, r: 28, b: 8 / 3 },
    pad: [['r', 24, 45], ['s', 7, 14]], axes: ['drive', 'mixing'],
    orient: [1, 0, 0, 0, 0, 1, 0, 1, 0],
  },
  {
    name: 'Aizawa', h: 0.009, spread: 1.5, init: [0.1, 0, 0.3],
    def: { a: 0.95, b: 0.7, c: 0.6, d: 3.5, e: 0.25, f: 0.1 },
    pad: [['d', 2.0, 4.2], ['a', 0.65, 0.95]], axes: ['twist', 'lift'],
    // rows of a 3x3 matrix taking raw coords to world (z-axis up)
    orient: [1, 0, 0, 0, 0, 1, 0, 1, 0]
  },
];

// Rotation that maps the (1,1,1) diagonal to +y, for the cyclically symmetric systems.
(function diagonalOrient() {
  const up = [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)];
  const r = [1 / Math.sqrt(2), -1 / Math.sqrt(2), 0];
  const f = [up[1] * r[2] - up[2] * r[1], up[2] * r[0] - up[0] * r[2], up[0] * r[1] - up[1] * r[0]];
  const m = [r[0], r[1], r[2], up[0], up[1], up[2], f[0], f[1], f[2]];
  ATTRACTORS.forEach((s) => { if (!s.orient) s.orient = m; });
})();
