// Unravel kernel as WebGPU compute. One invocation per particle, the same fused
// order as the C scalar path (ODE -> home -> warp -> tear -> smoke). All maths is
// f32 in the reference's operation order, but WGSL does not promise correctly
// rounded division/sqrt, and it may contract a*b+c into FMA, so results are
// validated with a tolerance, not bit-for-bit.

struct Tables { perm: array<u32, 512>, grad: array<f32, 48> };

@group(0) @binding(0) var<storage, read> U: array<f32, 96>;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(0) @binding(2) var<storage, read_write> P: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> F: array<f32>;
@group(0) @binding(5) var<storage, read_write> T: array<f32>;
@group(0) @binding(6) var<storage, read> HEAL: array<f32>;
@group(0) @binding(7) var<storage, read_write> SS: array<f32>;
@group(0) @binding(8) var<storage, read_write> W: array<f32>;
@group(0) @binding(9) var<storage, read> TB: Tables;

const FLAG_WARP = 1u; const FLAG_TEAR = 2u; const FLAG_BURST = 4u;
const WG = 12u;
const W_LO = -2.2f; const W_NF = 0.9f; const MORPH_EASE = 1.4f;
override W_CELL: f32; override W_INV: f32; override W_GMAX: f32; override N: u32;

fn ui(i: u32) -> u32 { return bitcast<u32>(U[i]); }

fn hash3(i: u32, frame: u32, salt: u32) -> u32 {
  var x = (i * 0x9E3779B1u) ^ (frame * 0x85EBCA77u) ^ (salt * 0xC2B2AE3Du);
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return x;
}
fn rnd(i: u32, frame: u32, salt: u32) -> f32 { return f32(hash3(i, frame, salt) >> 8u) * (1.0f / 16777216.0f); }

fn fsin(x: f32) -> f32 {
  let q = floor(x * 0.31830987f + 0.5f);
  let r = (x - q * 3.140625f) - q * 0.00096765358979f;
  let r2 = r * r;
  var p = (1.0f / 362880.0f) + r2 * (-1.0f / 39916800.0f);
  p = (-1.0f / 5040.0f) + r2 * p;
  p = (1.0f / 120.0f) + r2 * p;
  p = (-1.0f / 6.0f) + r2 * p;
  p = r + (r * r2) * p;
  if ((i32(q) & 1) != 0) { return -p; }
  return p;
}

fn corner(x: f32, y: f32, z: f32, gi: u32, acc: ptr<function, vec3<f32>>) {
  let tt = ((0.5f - x * x) - y * y) - z * z;
  if (tt > 0.0f) {
    let gx = TB.grad[gi]; let gy = TB.grad[gi + 1u]; let gz = TB.grad[gi + 2u];
    let t2 = tt * tt; let t4 = t2 * t2;
    let gd = (gx * x + gy * y) + gz * z;
    let m = ((-8.0f * t2) * tt) * gd;
    (*acc).x = (*acc).x + (m * x + t4 * gx);
    (*acc).y = (*acc).y + (m * y + t4 * gy);
    (*acc).z = (*acc).z + (m * z + t4 * gz);
  }
}

fn pm(i: u32) -> u32 { return TB.perm[i]; }

fn snoise_grad(x: f32, y: f32, z: f32) -> vec3<f32> {
  let F3 = 1.0f / 3.0f; let G3 = 1.0f / 6.0f;
  let s = ((x + y) + z) * F3;
  let fi = floor(x + s); let fj = floor(y + s); let fk = floor(z + s);
  let t = ((fi + fj) + fk) * G3;
  let x0 = x - (fi - t); let y0 = y - (fj - t); let z0 = z - (fk - t);
  var i1 = 0u; var j1 = 0u; var k1 = 0u; var i2 = 0u; var j2 = 0u; var k2 = 0u;
  if (x0 >= y0) {
    if (y0 >= z0) { i1 = 1u; i2 = 1u; j2 = 1u; }
    else if (x0 >= z0) { i1 = 1u; i2 = 1u; k2 = 1u; }
    else { k1 = 1u; i2 = 1u; k2 = 1u; }
  } else {
    if (y0 < z0) { k1 = 1u; j2 = 1u; k2 = 1u; }
    else if (x0 < z0) { j1 = 1u; j2 = 1u; k2 = 1u; }
    else { j1 = 1u; i2 = 1u; j2 = 1u; }
  }
  let G3x2 = 2.0f * G3; let G3x3 = 3.0f * G3;
  let x1 = (x0 - f32(i1)) + G3; let y1 = (y0 - f32(j1)) + G3; let z1 = (z0 - f32(k1)) + G3;
  let x2 = (x0 - f32(i2)) + G3x2; let y2 = (y0 - f32(j2)) + G3x2; let z2 = (z0 - f32(k2)) + G3x2;
  let x3 = (x0 - 1.0f) + G3x3; let y3 = (y0 - 1.0f) + G3x3; let z3 = (z0 - 1.0f) + G3x3;
  let ii = u32(i32(fi) & 255); let jj = u32(i32(fj) & 255); let kk = u32(i32(fk) & 255);
  var acc = vec3<f32>(0.0f);
  corner(x0, y0, z0, (pm(ii + pm(jj + pm(kk))) & 15u) * 3u, &acc);
  corner(x1, y1, z1, (pm(ii + i1 + pm(jj + j1 + pm(kk + k1))) & 15u) * 3u, &acc);
  corner(x2, y2, z2, (pm(ii + i2 + pm(jj + j2 + pm(kk + k2))) & 15u) * 3u, &acc);
  corner(x3, y3, z3, (pm(ii + 1u + pm(jj + 1u + pm(kk + 1u))) & 15u) * 3u, &acc);
  return vec3<f32>(32.0f * acc.x, 32.0f * acc.y, 32.0f * acc.z);
}

fn curl(x: f32, y: f32, z: f32) -> vec3<f32> {
  let g0 = snoise_grad(x, y, z);
  let g1 = snoise_grad(x + 31.416f, y - 17.23f, z + 5.71f);
  let g2 = snoise_grad(x - 12.87f, y + 47.31f, z - 23.9f);
  return vec3<f32>(g2.y - g1.z, g0.z - g2.x, g1.x - g0.y);
}

@compute @workgroup_size(64)
fn grid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x;
  if (q >= WG * WG * WG || (ui(2u) & FLAG_WARP) == 0u) { return; }
  let gx = q % WG; let gy = (q / WG) % WG; let gz = q / (WG * WG);
  let px = ((W_LO + f32(gx) * W_CELL) * W_NF) + U[35];
  let py = ((W_LO + f32(gy) * W_CELL) * W_NF) + U[36];
  let pz = ((W_LO + f32(gz) * W_CELL) * W_NF) + U[37];
  let c = curl(px, py, pz);
  W[q * 3u] = c.x; W[q * 3u + 1u] = c.y; W[q * 3u + 2u] = c.z;
}

@compute @workgroup_size(64)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let kind = ui(0u); let frame = ui(1u); let flags = ui(2u);
  // ---- ODE ----
  var x = S[i * 3u]; var y = S[i * 3u + 1u]; var z = S[i * 3u + 2u];
  var dx: f32; var dy: f32; var dz: f32;
  let a = U[4]; let b = U[5]; let k = U[10];
  if (kind == 0u) {
    let zb = z - b;
    dx = zb * x - U[7] * y;
    dy = U[7] * x + zb * y;
    dz = (((U[6] + a * z) - ((z * z) * z) / 3.0f) - (x * x + y * y) * (1.0f + U[8] * z)) + (((U[9] * z) * x) * x) * x;
  } else if (kind == 1u) {
    dx = fsin(k * y) - b * x; dy = fsin(k * z) - b * y; dz = fsin(k * x) - b * z;
  } else if (kind == 2u) {
    dx = ((-a * x - k * y) - k * z) - y * y;
    dy = ((-a * y - k * z) - k * x) - z * z;
    dz = ((-a * z - k * x) - k * y) - x * x;
  } else {
    dx = U[12] * (y - x); dy = x * (U[11] - z) - y; dz = x * y - b * z;
  }
  let h = U[13]; let bound = U[14];
  x = x + dx * h; y = y + dy * h; z = z + dz * h;
  if (!(x > -bound && x < bound && y > -bound && y < bound && z > -bound && z < bound)) {
    let jit = U[15];
    x = U[16] + (rnd(i, frame, 1u) - 0.5f) * jit;
    y = U[17] + (rnd(i, frame, 2u) - 0.5f) * jit;
    z = U[18] + (rnd(i, frame, 3u) - 0.5f) * jit;
  }
  S[i * 3u] = x; S[i * 3u + 1u] = y; S[i * 3u + 2u] = z;
  // ---- home, colour, emission, samples ----
  let speed = sqrt((dx * dx + dy * dy) + dz * dz);
  let sc = U[19];
  let rx = (x - U[20]) * sc; let ry = (y - U[21]) * sc; let rz = (z - U[22]) * sc;
  var hx = (U[25] * rx + U[26] * ry) + U[27] * rz;
  var hy = (U[28] * rx + U[29] * ry) + U[30] * rz;
  var hz = (U[31] * rx + U[32] * ry) + U[33] * rz;
  var f = F[i];
  var tadd = 0.0f; if (f > 0.0f) { tadd = f * 0.3f; }
  T[i] = speed * U[23] + tadd;
  let hs = speed * U[24] - 1.05f;
  let q = (3.2f * hs) * hs + 1.2f * hs;
  var bb = 0.55f; if (hs > 0.0f) { bb = 0.55f + select(5.5f, q, q < 5.5f); }
  if ((i & 63u) == 0u) { let s = (i >> 6u) * 5u; SS[s] = x; SS[s + 1u] = y; SS[s + 2u] = z; SS[s + 3u] = speed; SS[s + 4u] = hy; }
  // ---- warp ----
  if ((flags & FLAG_WARP) != 0u) {
    var fx = (hx - W_LO) * W_INV; var fy = (hy - W_LO) * W_INV; var fz = (hz - W_LO) * W_INV;
    fx = select(select(fx, W_GMAX, fx > W_GMAX), 0.0f, fx < 0.0f);
    fy = select(select(fy, W_GMAX, fy > W_GMAX), 0.0f, fy < 0.0f);
    fz = select(select(fz, W_GMAX, fz > W_GMAX), 0.0f, fz < 0.0f);
    let ix = u32(fx); let iy = u32(fy); let iz = u32(fz);
    let ax = fx - f32(ix); let ay = fy - f32(iy); let az = fz - f32(iz);
    let bx = 1.0f - ax; let by = 1.0f - ay; let bz = 1.0f - az;
    let w0 = (bx * by) * bz; let w1 = (ax * by) * bz; let w2 = (bx * ay) * bz; let w3 = (ax * ay) * bz;
    let w4 = (bx * by) * az; let w5 = (ax * by) * az; let w6 = (bx * ay) * az; let w7 = (ax * ay) * az;
    let b0 = (iz * WG * WG + iy * WG + ix) * 3u;
    let b1 = b0 + 3u; let b2 = b0 + WG * 3u; let b3 = b2 + 3u; let b4 = b0 + WG * WG * 3u; let b5 = b4 + 3u; let b6 = b4 + WG * 3u; let b7 = b6 + 3u;
    let amp = U[34];
    var hv = vec3<f32>(hx, hy, hz);
    for (var d = 0u; d < 3u; d++) {
      let s = ((((((W[b0 + d] * w0 + W[b1 + d] * w1) + W[b2 + d] * w2) + W[b3 + d] * w3) + W[b4 + d] * w4) + W[b5 + d] * w5) + W[b6 + d] * w6) + W[b7 + d] * w7;
      hv[d] = hv[d] + amp * s;
    }
    hx = hv.x; hy = hv.y; hz = hv.z;
  }
  var p4 = P[i];
  var vx = V[i * 3u]; var vy = V[i * 3u + 1u]; var vz = V[i * 3u + 2u];
  // ---- tear / burst (old position) ----
  if ((flags & (FLAG_TEAR | FLAG_BURST)) != 0u) {
    let w = ((U[62] * p4.x + U[66] * p4.y) + U[70] * p4.z) + U[74];
    if (w > 0.05f) {
      let cxp = ((U[59] * p4.x + U[63] * p4.y) + U[67] * p4.z) + U[71];
      let cyp = ((U[60] * p4.x + U[64] * p4.y) + U[68] * p4.z) + U[72];
      let qx = ((cxp / w) * 0.5f + 0.5f) * U[57];
      let qy = (0.5f - (cyp / w) * 0.5f) * U[58];
      if ((flags & FLAG_TEAR) != 0u) {
        let ex = qx - U[39]; let ey = qy - U[40]; let d2 = ex * ex + ey * ey; let tr2 = U[41];
        if (d2 < tr2) {
          var ff = 1.0f - d2 / tr2; ff = ff * ff;
          if (f <= 0.0f) { vx = 0.0f; vy = 0.0f; vz = 0.0f; }
          let pull = ff * 0.35f; let jt = 0.06f * ff;
          vx = (vx + (U[42] * 1.2f - vx) * pull) + (rnd(i, frame, 11u) - 0.5f) * jt;
          vy = (vy + (U[43] * 1.2f - vy) * pull) + (rnd(i, frame, 12u) - 0.5f) * jt;
          vz = (vz + (U[44] * 1.2f - vz) * pull) + (rnd(i, frame, 13u) - 0.5f) * jt;
          if (ff > f) { f = ff; }
        }
      }
      if ((flags & FLAG_BURST) != 0u) {
        let ex = qx - U[45]; let ey = qy - U[46]; let d2 = ex * ex + ey * ey; let br2 = U[47];
        if (d2 < br2) {
          let t = 1.0f - d2 / br2; let ff = t * sqrt(t);
          let l = sqrt(d2) + 1e-3f; let ux = ex / l; let uy = -ey / l; let s = 1.6f * ff;
          let zr = (rnd(i, frame, 14u) - 0.5f) * 0.8f;
          if (f <= 0.0f) { vx = 0.0f; vy = 0.0f; vz = 0.0f; }
          vx = vx + ((U[48] * ux + U[51] * uy) + U[54] * zr) * s;
          vy = vy + ((U[49] * ux + U[52] * uy) + U[55] * zr) * s;
          vz = vz + ((U[50] * ux + U[53] * uy) + U[56] * zr) * s;
          if (ff > f) { f = ff; }
        }
      }
    }
  }
  // ---- smoke / return / attach ----
  let dt = U[83];
  if (f == 0.0f) {
    P[i] = vec4<f32>(hx, hy, hz, bb);
  } else if (f < 0.0f) {
    if (f < -MORPH_EASE) { F[i] = f + dt; P[i] = vec4<f32>(p4.xyz, bb); }
    else {
      let rr = select(U[81], U[82], f < -0.36f);
      let nx = p4.x + (hx - p4.x) * rr; let ny = p4.y + (hy - p4.y) * rr; let nz = p4.z + (hz - p4.z) * rr;
      var nf = f + dt; if (nf > 0.0f) { nf = 0.0f; }
      P[i] = vec4<f32>(nx, ny, nz, bb); F[i] = nf;
    }
  } else {
    let freq = U[75];
    let c = curl(p4.x * freq + U[77], p4.y * freq + U[78], p4.z * freq + U[79]);
    let g = 1.0f - f; let pull = ((7.0f * g) * g) * g;
    let fa = U[76] * (0.35f + 0.65f * f);
    let relax = U[80];
    vx = vx + (((c.x * fa) + (hx - p4.x) * pull) - vx) * relax;
    vy = vy + (((c.y * fa) + (hy - p4.y) * pull) - vy) * relax;
    vz = vz + (((c.z * fa) + (hz - p4.z) * pull) - vz) * relax;
    P[i] = vec4<f32>(p4.x + vx * dt, p4.y + vy * dt, p4.z + vz * dt, 0.7f + (4.5f * f) * f);
    let nf = f - HEAL[i] * dt;
    F[i] = select(-0.35f, nf, nf > 0.0f);
  }
  V[i * 3u] = vx; V[i * 3u + 1u] = vy; V[i * 3u + 2u] = vz;
}

// Shape morph (app only): every particle keeps its position and waits for a top-down
// wave before easing onto its home on the new attractor. U[84] = top y, U[85] = span.
@compute @workgroup_size(64)
fn morph(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let top = clamp((U[84] - P[i].y) / U[85], 0.0f, 1.0f);
  F[i] = -(MORPH_EASE + 1.6f * top + rnd(i, ui(1u), 31u) * 0.15f);
  V[i * 3u] = 0.0f; V[i * 3u + 1u] = 0.0f; V[i * 3u + 2u] = 0.0f;
}
