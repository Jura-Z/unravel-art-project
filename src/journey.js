// "How it got fast": the performance investigation, told for people using the page.
// Opens as an overlay from the footer link or a #journey link; the instrument keeps running behind it.

const JOURNEY_DATA = {
  // Dev container (Xeon 2.8 GHz, 2 cores, Node 22), 150k particles, scripted 300-frame session
  steps: [
    { name: 'JavaScript, exact float32', ms: 71.1, note: 'The reference: every operation rounded with Math.fround' },
    { name: 'JavaScript, one pass over memory', ms: 73.8, note: 'Fusing four passes into one did nothing' },
    { name: 'C → WebAssembly, scalar', ms: 17.3, note: 'Native float32 maths, same operation order' },
    { name: 'WASM SIMD, first attempt', ms: 14.6, note: 'Vectorised the cheap part (the attractor)' },
    { name: 'WASM SIMD, rebuilt', ms: 6.8, note: 'Compaction, vector noise, one-load grid cells' },
    { name: 'WASM SIMD + 2 threads', ms: 3.75, note: 'Shared memory, no copies, one Atomics hand-off per frame' },
  ],
  // A desktop browser (NVIDIA Ada Lovelace GPU, 32 threads, Chrome 152)
  scale: [
    { n: '150k', wasm: 2.16, gpu: 0.0131 },
    { n: '1M', wasm: 14.7, gpu: 0.129 },
    { n: '4M', wasm: 58.9, gpu: 0.421 },
  ],
};

function journeyBars(rows, key, fmt, opts = {}) {
  const max = Math.max(...rows.map((r) => r[key]));
  const min = Math.min(...rows.map((r) => r[key]));
  const pos = (v) => opts.log
    ? (Math.log10(v) - Math.log10(min / 2)) / (Math.log10(max) - Math.log10(min / 2))
    : v / max;
  return `<div class="jbars" role="table" aria-label="${opts.label}">` + rows.map((r, i) => {
    const last = opts.highlight === i || r.gpu;
    return `<div class="jrow${last ? ' hl' : ''}" role="row" title="${r.note ? r.note.replace(/"/g, '') : ''}">
      <span class="jname" role="rowheader">${r.name || r.label}</span>
      <span class="jtrack" role="cell"><span class="jbar" style="width:${Math.max(0.6, pos(r[key]) * 100).toFixed(1)}%"></span></span>
      <span class="jval" role="cell">${fmt(r[key], r)}</span>
    </div>`;
  }).join('') + '</div>';
}

function buildJourney() {
  const d = JOURNEY_DATA, base = d.steps[0].ms;
  const steps = journeyBars(d.steps, 'ms', (v) => `${v.toFixed(v < 10 ? 2 : 1)} ms <em>${(base / v).toFixed(base / v < 1.5 ? 2 : 1)}×</em>`, { label: 'Milliseconds per frame by implementation', highlight: d.steps.length - 1 });
  const scaleRows = [];
  for (const s of d.scale) {
    scaleRows.push({ label: `${s.n} · WASM SIMD`, v: s.wasm });
    scaleRows.push({ label: `${s.n} · WebGPU`, v: s.gpu, gpu: true });
  }
  const scale = journeyBars(scaleRows, 'v', (v, r) => `${v < 1 ? v.toFixed(3) : v.toFixed(1)} ms${r.gpu ? ` <em>${(scaleRows[scaleRows.indexOf(r) - 1].v / v).toFixed(0)}×</em>` : ''}`, { log: true, label: 'Milliseconds per frame, WASM SIMD vs WebGPU, log scale' });

  const el = document.createElement('div');
  el.id = 'journey';
  el.className = 'journey';
  el.hidden = true;
  el.setAttribute('role', 'dialog');
  el.setAttribute('aria-modal', 'true');
  el.setAttribute('aria-labelledby', 'journey-title');
  el.innerHTML = `
  <article class="jcard">
    <button class="jclose" type="button" aria-label="Close">×</button>
    <p class="jkicker">Behind the instrument</p>
    <h2 id="journey-title">How it got fast</h2>
    <p class="jlead">Every frame, each particle follows a strange attractor, gets bent by turbulence and your brush, and, once torn, drifts through 3D curl noise. At 150,000 particles the first JavaScript version needed <strong>71 ms</strong> a frame, four times over budget. On a desktop GPU the same work takes <strong>0.013 ms</strong>. This is how it got from one to the other.</p>

    <h3>Rule one: faster must mean identical</h3>
    <p>A speed-up only counts if it computes the same thing. So before optimising anything, we pinned down what "the same" means:</p>
    <ul>
      <li><strong>One arithmetic.</strong> Every step is float32 in a fixed order, with no fused multiply-add and no fast-math. <code>sin</code> is our own polynomial.</li>
      <li><strong>Randomness is a hash of particle and frame,</strong> so threads and GPUs draw the same numbers without sharing state.</li>
      <li><strong>Record once, replay everywhere.</strong> A scripted session (a tear, a burst, a turbulence swell) is replayed by every version. The checksums of all particle state must match bit for bit.</li>
    </ul>
    <p>Writing the spec found bugs before optimising did. The textbook simplex-noise kernel turned out to be slightly discontinuous, and the particle respawn was a data race waiting for threads.</p>

    <h3>JavaScript → WebAssembly → SIMD</h3>
    <p class="jcap">Milliseconds per frame, 150k particles, the same session. Every row is bit-exact with the first.</p>
    ${steps}
    <ul>
      <li><strong>In JavaScript, exactness is the cost.</strong> Rounding every operation to float32 costs about 4×. Fusing four passes into one cache-friendly sweep changed nothing, because arithmetic dominates, not memory.</li>
      <li><strong>C compiled to WebAssembly gave 4× for free,</strong> by doing float32 natively.</li>
      <li><strong>The first SIMD attempt gave 1.15×.</strong> It vectorised the part that was already cheap. The profile said the time was in the smoke's noise.</li>
      <li><strong>Rebuilt around that profile, SIMD gave 2.5× more.</strong> Torn particles are packed together so the noise runs four at a time, and each turbulence cell is one 16-byte load.</li>
      <li><strong>Threads halve it again on two cores,</strong> sharing memory with one hand-off per frame. The page can't rely on them, though: they need cross-origin isolation, which many hosts can't turn on.</li>
    </ul>

    <h3>Then the GPU</h3>
    <p class="jcap">Milliseconds per frame (log scale). Desktop browser, NVIDIA Ada Lovelace GPU.</p>
    ${scale}
    <ul>
      <li><strong>The same kernel in WGSL runs 115–165× faster than WASM SIMD,</strong> and stays under half a millisecond at 4 million particles.</li>
      <li><strong>So the renderer moved to WebGPU too.</strong> WebGL can't read WebGPU buffers, and copying particles back every frame would undo the win. Particles are born, moved and drawn without leaving the GPU. The CPU sends under a kilobyte of settings per frame.</li>
      <li><strong>Timing it needed a trick.</strong> Browsers round GPU timestamps to about 0.1 ms, so one frame read as zero. The benchmark times a batch of frames instead.</li>
      <li><strong>The GPU can't be bit-exact.</strong> WGSL allows looser division and fused multiply-adds. 98.8% of values match exactly after one frame, and the rest differ by a few millionths. Because the motion is chaotic, it is checked at frames 1 and 10.</li>
    </ul>

    <h3>Second round: the details you feel</h3>
    <p>With the simulation solved, the remaining costs were in the interaction.</p>
    <ul>
      <li><strong>Lines to the cursor.</strong> Each frame the app needs the 40 particles near the cursor with the lowest fixed hash (a stable choice, so the lines don't flicker). The answer is unique, so every variant was checked against it over 240 recorded queries. The winner asks in the answer's order: particles are pre-sorted by hash once, and the walk stops at the 40th hit.</li>
    </ul>
    <table class="jtable">
      <thead><tr><th>Cursor search, 600k particles</th><th>mean</th><th>worst</th></tr></thead>
      <tbody>
        <tr><td>JavaScript, scan every 4th particle</td><td>3.05 ms</td><td>8.0 ms</td></tr>
        <tr><td>JavaScript, screen-space grid</td><td>7.8 ms</td><td>13 ms</td></tr>
        <tr class="win"><td>JavaScript, ordered walk, then scan</td><td>1.1 ms</td><td>3.8 ms</td></tr>
        <tr><td>WASM SIMD scan (2 threads)</td><td>1.6 (1.0) ms</td><td>6.1 (5.6) ms</td></tr>
        <tr class="win"><td>WASM SIMD ordered walk</td><td>0.44 ms</td><td>1.7 ms</td></tr>
      </tbody>
    </table>
    <ul>
      <li><strong>The spatial grid lost.</strong> There is one cursor per frame and the particles always move. Rebuilding the grid costs as much as simply looking at every particle.</li>
      <li><strong>Changing shape stalled for 0.3 s at 2.4M particles.</strong> Every particle re-integrated the attractor in JavaScript, then 29 MB went to the GPU. Now each shape keeps one cached orbit and the GPU seeds particles from it: 14 ms, nothing uploaded. The page also prepares the next shape while idle.</li>
      <li><strong>The particle count picks itself.</strong> The page measures its own frame time and adds particles while it keeps up with your display, up to 120 fps. On the first dropped frame it steps back. Press − or + to choose yourself.</li>
    </ul>

    <h3>How it was made</h3>
    <p>One person and one AI, in short loops. I (Jura) set the direction and judged every result by eye. Claude, Anthropic's model, wrote the code and checked its own work with screenshots and numbers before showing me.</p>
    <ul>
      <li><strong>Critique before code.</strong> The first request was to critique the idea. Two of its claims were overstated; I pushed back, and it agreed and revised.</li>
      <li><strong>Four prototypes to calibrate taste,</strong> each built around one verb. I picked by feel: tearing, plus the attractor.</li>
      <li><strong>A benchmark we could trust,</strong> built before any optimisation, so each experiment took seconds to judge.</li>
      <li><strong>Closing the loop on real hardware.</strong> The AI's sandbox has no GPU. The benchmark page saves every run where Claude can read it, and later Claude tested in my own browser.</li>
    </ul>

    <h3>What went wrong</h3>
    <ul>
      <li><strong>A benchmark that lied:</strong> code loaded through <code>eval</code> ran 5× slower in V8.</li>
      <li><strong>Threads that hung:</strong> reserving 2 GB of shared memory failed silently inside a worker.</li>
      <li><strong>A test browser that couldn't show WebGPU:</strong> headless Chromium on Linux loses the device when a WebGPU canvas presents, so tests render offscreen.</li>
      <li><strong>A brush that launched particles</strong> at 10× hand speed. Torn threads now ease toward your hand's velocity.</li>
      <li><strong>Nearest-to-cursor looked wrong:</strong> the 40 closest particles clumped into one tight fan, so selection is by hash inside the radius.</li>
      <li><strong>A demo cursor that vanished</strong> while its lines stayed behind.</li>
    </ul>

    <h3>What runs on your machine</h3>
    <p>WebGPU when your browser has it. Otherwise WebAssembly SIMD with WebGL2, which is bit-exact and uploads straight from WebAssembly memory. Plain JavaScript is the last resort. The footer switches between them. Nothing allocates per frame on any path.</p>
    <p class="jfoot">The benchmark, the kernels (C, WGSL, JS), the recorded session and the checksums are in the source.</p>
  </article>`;
  document.body.appendChild(el);
  const close = () => { el.hidden = true; const l = document.getElementById('journey-link'); if (l) l.focus(); if (location.hash === '#journey') history.replaceState(null, '', location.pathname + location.search); };
  el.querySelector('.jclose').addEventListener('click', close);
  el.addEventListener('click', (e) => { if (e.target === el) close(); });
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !el.hidden) close(); }, true);
  return el;
}

function openJourney() {
  const el = document.getElementById('journey') || buildJourney();
  el.hidden = false;
  el.scrollTop = 0;
  el.querySelector('.jclose').focus();
}

(function wireJourney() {
  const link = document.getElementById('journey-link');
  if (link) link.addEventListener('click', (e) => { e.preventDefault(); openJourney(); });
  if (location.hash === '#journey') setTimeout(openJourney, 0);
})();
