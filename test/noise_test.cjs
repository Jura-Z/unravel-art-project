const fs = require('fs');
require('vm').runInThisContext(fs.readFileSync('src/noise.js','utf8'));
// gradient check
const g = new Float64Array(3), g2 = new Float64Array(3); let maxErr = 0, maxV = 0;
for (let k = 0; k < 2000; k++) {
  const x = Math.random()*10, y = Math.random()*10, z = Math.random()*10, h = 1e-5;
  const v = snoise3d(x,y,z,g,0); maxV = Math.max(maxV, Math.abs(v));
  const fx = (snoise3d(x+h,y,z,g2,0)-snoise3d(x-h,y,z,g2,0))/(2*h);
  const fy = (snoise3d(x,y+h,z,g2,0)-snoise3d(x,y-h,z,g2,0))/(2*h);
  const fz = (snoise3d(x,y,z+h,g2,0)-snoise3d(x,y,z-h,g2,0))/(2*h);
  maxErr = Math.max(maxErr, Math.abs(fx-g[0]), Math.abs(fy-g[1]), Math.abs(fz-g[2]));
}
console.log('grad max abs err', maxErr.toExponential(2), ' max |n|', maxV.toFixed(3));
// divergence check
const c = new Float64Array(3); let maxDiv = 0, meanSpeed = 0;
for (let k = 0; k < 2000; k++) {
  const x = Math.random()*10, y = Math.random()*10, z = Math.random()*10, h = 1e-4;
  curlNoise(x+h,y,z,c); const a=c[0]; curlNoise(x-h,y,z,c); const b=c[0];
  curlNoise(x,y+h,z,c); const d=c[1]; curlNoise(x,y-h,z,c); const e=c[1];
  curlNoise(x,y,z+h,c); const f=c[2]; curlNoise(x,y,z-h,c); const q=c[2];
  maxDiv = Math.max(maxDiv, Math.abs((a-b+d-e+f-q)/(2*h)));
  curlNoise(x,y,z,c); meanSpeed += Math.hypot(c[0],c[1],c[2]);
}
console.log('max divergence', maxDiv.toExponential(2), ' mean speed', (meanSpeed/2000).toFixed(3));
// timing
const N = 100000, P = new Float32Array(N*3); for (let i=0;i<N*3;i++) P[i]=Math.random()*4;
for (let rep=0; rep<3; rep++) {
  const t0 = performance.now();
  for (let i=0;i<N;i++){ curlNoise(P[i*3],P[i*3+1],P[i*3+2],c); P[i*3]+=c[0]*1e-3; P[i*3+1]+=c[1]*1e-3; P[i*3+2]+=c[2]*1e-3; }
  const dt = performance.now()-t0;
  console.log(`curl ${N} particles: ${dt.toFixed(1)} ms  (${(dt*1e6/N).toFixed(0)} ns/particle)`);
}
// mean divergence
let sd=0; for (let k=0;k<5000;k++){const x=Math.random()*10,y=Math.random()*10,z=Math.random()*10,h=1e-4;
curlNoise(x+h,y,z,c); const a=c[0]; curlNoise(x-h,y,z,c); const b=c[0];
curlNoise(x,y+h,z,c); const d=c[1]; curlNoise(x,y-h,z,c); const e=c[1];
curlNoise(x,y,z+h,c); const f=c[2]; curlNoise(x,y,z-h,c); const q=c[2]; sd+=Math.abs((a-b+d-e+f-q)/(2*h));}
console.log('mean |div|', (sd/5000).toExponential(2));
