// Orbit camera + the few matrix helpers the renderers need (column-major, GL style).

function mat4Perspective(out, fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  out.fill(0);
  out[0] = f / aspect; out[5] = f;
  out[10] = (far + near) * nf; out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function mat4Mul(out, a, b) {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      out[c * 4 + r] =
        a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

class OrbitCamera {
  constructor() {
    this.yaw = 0; this.pitch = 0; this.dist = 3.2;
    this.target = [0, 0, 0];
    this.fov = 45 * Math.PI / 180;
    this.view = new Float32Array(16);
    this.proj = new Float32Array(16);
    this.vp = new Float32Array(16);
    this.eye = [0, 0, 0];
    this.right = [1, 0, 0]; this.up = [0, 1, 0]; this.fwd = [0, 0, -1];
    this.aspect = 1; this.heightPx = 1; this.widthPx = 1;
  }

  set(o) {
    if (o.yaw !== undefined) this.yaw = o.yaw;
    if (o.pitch !== undefined) this.pitch = o.pitch;
    if (o.dist !== undefined) this.dist = o.dist;
    if (o.target) this.target = o.target.slice();
  }

  update(widthPx, heightPx) {
    this.widthPx = widthPx; this.heightPx = heightPx;
    this.aspect = widthPx / heightPx;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const t = this.target;
    const e = this.eye;
    e[0] = t[0] + this.dist * cp * sy;
    e[1] = t[1] + this.dist * sp;
    e[2] = t[2] + this.dist * cp * cy;

    // Forward, right, up basis
    let fx = t[0] - e[0], fy = t[1] - e[1], fz = t[2] - e[2];
    let l = Math.hypot(fx, fy, fz); fx /= l; fy /= l; fz /= l;
    let rx = fy * 0 - fz * 1, ry = fz * 0 - fx * 0, rz = fx * 1 - fy * 0; // f x worldUp
    l = Math.hypot(rx, ry, rz) || 1; rx /= l; ry /= l; rz /= l;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    this.fwd = [fx, fy, fz]; this.right = [rx, ry, rz]; this.up = [ux, uy, uz];

    const v = this.view;
    v[0] = rx; v[1] = ux; v[2] = -fx; v[3] = 0;
    v[4] = ry; v[5] = uy; v[6] = -fy; v[7] = 0;
    v[8] = rz; v[9] = uz; v[10] = -fz; v[11] = 0;
    v[12] = -(rx * e[0] + ry * e[1] + rz * e[2]);
    v[13] = -(ux * e[0] + uy * e[1] + uz * e[2]);
    v[14] = (fx * e[0] + fy * e[1] + fz * e[2]);
    v[15] = 1;

    mat4Perspective(this.proj, this.fov, this.aspect, 0.05, 100);
    mat4Mul(this.vp, this.proj, this.view);
  }

  // World units per CSS pixel on the plane through the target.
  worldPerPx() {
    return 2 * this.dist * Math.tan(this.fov / 2) / this.heightPx;
  }



  // Ray through NDC as {o, d} (d normalized).
  ray(nx, ny) {
    const th = Math.tan(this.fov / 2);
    const r = this.right, u = this.up, f = this.fwd;
    let dx = f[0] + nx * th * this.aspect * r[0] + ny * th * u[0];
    let dy = f[1] + nx * th * this.aspect * r[1] + ny * th * u[1];
    let dz = f[2] + nx * th * this.aspect * r[2] + ny * th * u[2];
    const l = Math.hypot(dx, dy, dz);
    return { o: this.eye.slice(), d: [dx / l, dy / l, dz / l] };
  }
}
