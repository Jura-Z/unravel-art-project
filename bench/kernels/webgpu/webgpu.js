// WebGPU compute backend (browser). State lives in GPU storage buffers; per frame the
// CPU writes only the 384-byte uniform block and records two dispatches. P is laid out
// as vec4 xyzw so the renderer can bind it directly as a vertex buffer (zero copy).
import { UNI, PERM, GRAD, W_CELL, W_INV, W_GMAX, WG } from '../../core.js';

const UNI_STRIDE = 512;          // uniform slot size (dynamic offsets must be 256-aligned)
const RING = 512;                // slots: a whole recorded session fits, so it uploads once

export async function makeWebGPU(n, wgslSource) {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const hasTs = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTs ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30), maxStorageBuffersPerShaderStage: 10 },
  });
  const module = device.createShaderModule({ code: wgslSource });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === 'error');
  if (errs.length) throw new Error('WGSL: ' + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; '));
  const constants = { W_CELL, W_INV, W_GMAX, N: n };
  const layout = device.createBindGroupLayout({
    entries: Array.from({ length: 10 }, (_, b) => ({
      binding: b, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: [0, 6, 9].includes(b) ? 'read-only-storage' : 'storage', hasDynamicOffset: b === 0 },
    })),
  });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const gridPipe = device.createComputePipeline({ layout: pl, compute: { module, entryPoint: 'grid', constants } });
  const stepPipe = device.createComputePipeline({ layout: pl, compute: { module, entryPoint: 'step', constants } });

  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const mk = (bytes) => device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: S });
  const buf = {
    U: mk(RING * UNI_STRIDE), S: mk(n * 12), P: mk(n * 16), V: mk(n * 12), F: mk(n * 4), T: mk(n * 4),
    heal: mk(n * 4), SS: mk(Math.ceil(n / 64) * 20), W: mk(WG * WG * WG * 12), TB: mk(512 * 4 + 48 * 4),
  };
  const tables = new Uint32Array(512 + 48);
  tables.set(PERM, 0);
  new Float32Array(tables.buffer, 512 * 4, 48).set(GRAD);
  device.queue.writeBuffer(buf.TB, 0, tables);
  const order = ['U', 'S', 'P', 'V', 'F', 'T', 'heal', 'SS', 'W', 'TB'];
  const bind = device.createBindGroup({ layout, entries: order.map((k, b) => ({ binding: b, resource: b === 0 ? { buffer: buf.U, size: UNI.SIZE * 4 } : { buffer: buf[k] } })) });
  const staged = new Float32Array(RING * UNI_STRIDE / 4);

  // readback staging (checkpoints only)
  const sizes = { S: n * 12, P: n * 16, V: n * 12, F: n * 4, T: n * 4 };
  const staging = Object.fromEntries(Object.entries(sizes).map(([k, s]) => [k, device.createBuffer({ size: s, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })]));
  const out = { S: new Float32Array(n * 3), P: new Float32Array(n * 4), V: new Float32Array(n * 3), F: new Float32Array(n), T: new Float32Array(n) };

  let qs = null, tsBuf = null, tsRead = null;
  if (hasTs) {
    qs = device.createQuerySet({ type: 'timestamp', count: 2 });
    tsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    tsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
  const groupsP = Math.ceil(n / 64), groupsG = Math.ceil(WG * WG * WG / 64);

  return {
    name: 'webgpu',
    gpuName: [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.description].filter(Boolean).join(' '),
    device,
    load(init) {
      device.queue.writeBuffer(buf.S, 0, init.S); device.queue.writeBuffer(buf.P, 0, init.P);
      device.queue.writeBuffer(buf.V, 0, init.V); device.queue.writeBuffer(buf.F, 0, init.F);
      device.queue.writeBuffer(buf.T, 0, init.T); device.queue.writeBuffer(buf.heal, 0, init.heal);
    },
    // Upload a whole trace of uniform blocks once (one write, then frames just pick an offset).
    uploadTrace(trace, frames) {
      for (let f = 0; f < frames; f++) staged.set(trace.subarray(f * UNI.SIZE, (f + 1) * UNI.SIZE), f * UNI_STRIDE / 4);
      device.queue.writeBuffer(buf.U, 0, staged, 0, frames * UNI_STRIDE / 4);
      this.ring = true;
    },
    encodeFrame(pass, slot) {
      pass.setBindGroup(0, bind, [slot * UNI_STRIDE]);
      pass.setPipeline(gridPipe); pass.dispatchWorkgroups(groupsG);
      pass.setPipeline(stepPipe); pass.dispatchWorkgroups(groupsP);
    },
    // Realistic mode: one submission per frame. U is written into slot 0 unless a trace is uploaded.
    step(U, _UI, timestamps, frame) {
      let slot = 0;
      if (this.ring && frame !== undefined) slot = frame; else device.queue.writeBuffer(buf.U, 0, U);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(timestamps && qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
      this.encodeFrame(pass, slot);
      pass.end();
      if (timestamps && qs) { enc.resolveQuerySet(qs, 0, 2, tsBuf, 0); enc.copyBufferToBuffer(tsBuf, 0, tsRead, 0, 16); }
      device.queue.submit([enc.finish()]);
    },
    // Throughput mode: frames [f0, f1) in ONE pass/submission, bracketed by two timestamps.
    // (Chrome quantises timestamps to ~100 µs, so single fast frames read as 0.)
    runBatch(f0, f1) {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
      for (let f = f0; f < f1; f++) this.encodeFrame(pass, f);
      pass.end();
      if (qs) { enc.resolveQuerySet(qs, 0, 2, tsBuf, 0); enc.copyBufferToBuffer(tsBuf, 0, tsRead, 0, 16); }
      device.queue.submit([enc.finish()]);
    },
    async gpuTimeMs() {
      if (!tsRead) return null;
      await tsRead.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(tsRead.getMappedRange().slice(0));
      tsRead.unmap();
      return Number(t[1] - t[0]) / 1e6;
    },
    async finish() { await device.queue.onSubmittedWorkDone(); },
    async state() {
      const enc = device.createCommandEncoder();
      for (const k of Object.keys(sizes)) enc.copyBufferToBuffer(buf[k], 0, staging[k], 0, sizes[k]);
      device.queue.submit([enc.finish()]);
      for (const k of Object.keys(sizes)) {
        await staging[k].mapAsync(GPUMapMode.READ);
        out[k].set(new Float32Array(staging[k].getMappedRange()));
        staging[k].unmap();
      }
      return out;
    },
    buffers: buf,
    close() { device.destroy(); },
  };
}
