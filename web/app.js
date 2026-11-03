// web/engine.ts
import createModule from "./strike.mjs";
var V_C = 350 * 1852 / 3600;
var BIRD_MASS = 4 * 0.45359237;
var RHO_BIRD = 950;
var C0_BIRD = 1480;
var GAMMA_TAIT = 7;
var T_SKIN = 0.0016;
var PANEL = 0;
var RIGID_WALL = 1;
var BYTES_PER_I64 = 8;
var SLICE_BYTES = 12;
var WGSL_URL = "strike.wgsl";
function common(m) {
  const lenPtr = m._malloc(BYTES_PER_I64);
  function bytes(fn) {
    const ptr = fn(lenPtr);
    const len = Number(new BigInt64Array(m.HEAPU8.buffer, lenPtr, 1)[0]);
    return m.HEAPU8.buffer.slice(ptr, ptr + len);
  }
  return {
    bytes,
    setSkin: (t) => m._set_skin(t),
    time: () => m._get_time(),
    dt: () => m._get_dt(),
    steps: () => m._get_steps(),
    tContact: () => m._get_t_contact(),
    np: () => m._get_np(),
    nn: () => m._get_nn(),
    ne: () => m._get_ne(),
    particleMass: () => m._get_particle_mass(),
    spacing: () => m._get_spacing(),
    smoothing: () => m._get_smoothing(),
    birdD: () => m._get_bird_d(),
    birdL: () => m._get_bird_l(),
    birdV: () => m._get_bird_v(),
    contactForce: () => m._get_contact_force(),
    centerForce: () => m._get_center_force(),
    sensorR: () => m._get_sensor_r(),
    impulse: () => m._get_impulse(),
    panelWork: () => m._get_panel_work(),
    hourglassWork: () => m._get_hg_work(),
    birdWork: () => m._get_bird_work(),
    failed: () => m._get_failed(),
    failedSkin: () => m._get_failed_skin(),
    maxPlasticStrain: () => m._get_max_ep(),
    maxDeflection: () => m._get_max_defl(),
    failStrain: (mat) => m._get_eps_fail(mat),
    birdKE: () => m._get_bird_ke(),
    panelKE: () => m._get_panel_ke(),
    panelMass: () => m._get_panel_mass(),
    fillRender: () => void m._fill_render(),
    nodes: () => new Float32Array(bytes(m._buf_nodes), 0, 3 * m._get_nn()),
    particles: () => new Float32Array(bytes(m._buf_particles), 0, 3 * m._get_np()),
    pressure: () => new Float32Array(bytes(m._buf_pressure), 0, m._get_np()),
    strain: () => new Float32Array(bytes(m._buf_strain), 0, m._get_ne()),
    elements: () => new Int32Array(bytes(m._buf_elements), 0, 4 * m._get_ne())
  };
}
async function loadEngine() {
  const m = await createModule();
  m._goose_init(0, 0);
  return {
    ...common(m),
    gpu: false,
    setup: (mode, cells, spacing, speed) => m._setup(mode, cells, spacing, speed),
    advance: (steps, tEnd) => m._advance(steps, tEnd),
    stepPhase: (tEnd) => m._step_phase(tEnd)
  };
}
async function text(name) {
  const url = new URL(name, import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return readFile(url, "utf8");
  }
  const r = await fetch(url);
  if (!r.ok)
    throw new Error(`${name}: ${r.status}`);
  return r.text();
}
async function loadGpuEngine(gpu = globalThis.navigator?.gpu) {
  const adapter = await gpu?.requestAdapter();
  if (!gpu || !adapter)
    return null;
  const device = await adapter.requestDevice();
  const wgsl = new TextEncoder().encode(await text(WGSL_URL));
  const { default: createGpuModule } = await import("./strike-gpu.mjs");
  const m = await createGpuModule({ preinitializedWebGPUDevice: device });
  m._goose_init(0, 0);
  const src = m._malloc(wgsl.length), sl = m._malloc(SLICE_BYTES);
  m.HEAPU8.set(wgsl, src);
  const dv = new DataView(m.HEAPU8.buffer, sl, SLICE_BYTES);
  dv.setUint32(0, src, true);
  dv.setBigInt64(4, BigInt(wgsl.length), true);
  const errors = [];
  device.addEventListener("uncapturederror", (e) => errors.push(e.error.message));
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  const ok = m._gpu_init(sl) && m._gpu_setup(0, 56, 0.009, V_C);
  m._free(sl);
  m._free(src);
  const internal = await device.popErrorScope(), validation = await device.popErrorScope();
  const failure = internal || validation;
  if (!ok || failure)
    throw new Error(`the WebGPU solver did not start${failure ? `: ${failure.message}` : ""}`);
  const base = common(m);
  return {
    ...base,
    gpu: true,
    errors,
    setup: (mode, cells, spacing, speed) => m._gpu_setup(mode, cells, spacing, speed),
    run: (steps, tEnd) => m._gpu_run(steps, tEnd),
    fillRender: () => {},
    poll: () => m._gpu_poll(),
    pending: () => m._gpu_pending(),
    running: () => m._gpu_running() !== 0,
    history: () => new Float32Array(base.bytes(m._gpu_buf_hist), 0, 4 * Math.min(m._get_steps(), m._gpu_hist_cap())),
    birdKE: () => m._gpu_bird_ke(),
    panelKE: () => m._gpu_panel_ke(),
    gpuObject: gpu
  };
}
function hugoniot(v) {
  const b = RHO_BIRD * C0_BIRD * C0_BIRD / GAMMA_TAIT;
  let lo = RHO_BIRD * 1.000001;
  let hi = RHO_BIRD * 2;
  for (let i = 0;i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    const f = b * (Math.pow(mid / RHO_BIRD, GAMMA_TAIT) - 1) - RHO_BIRD * v * v * mid / (mid - RHO_BIRD);
    if (f > 0)
      hi = mid;
    else
      lo = mid;
  }
  const rho1 = 0.5 * (lo + hi);
  const us = v * rho1 / (rho1 - RHO_BIRD);
  return [RHO_BIRD * us * v, us];
}
function stagnation(v) {
  return 0.5 * RHO_BIRD * v * v;
}

// web/checked.ts
function at(a, i) {
  if (!(i >= 0 && i < a.length))
    throw new RangeError(`index ${i} is outside 0..${a.length}`);
  return a[i];
}
function must(v, what) {
  if (v === null || v === undefined)
    throw new Error(`${what} is missing`);
  return v;
}

// web/gif.ts
var COLOURS_15_BIT = 32768;
var PALETTE_SIZE = 256;
var LZW_CLEAR = 256;
var LZW_END = 257;
var LZW_FIRST_FREE = 258;
var LZW_MAX_CODES = 4096;
var LZW_MIN_SIZE = 9;
var LZW_MAX_SIZE = 12;
var SUB_BLOCK = 255;

class GifRecorder {
  w;
  h;
  frames = [];
  hist = new Uint32Array(COLOURS_15_BIT);
  constructor(width, height) {
    this.w = width;
    this.h = height;
  }
  add(image, delay) {
    const d = image.data, n = this.w * this.h, px = new Uint16Array(n);
    for (let i = 0;i < n; i++) {
      const c = at(d, 4 * i) >> 3 << 10 | at(d, 4 * i + 1) >> 3 << 5 | at(d, 4 * i + 2) >> 3;
      px[i] = c;
      this.hist[c] = at(this.hist, c) + 1;
    }
    this.frames.push({ px, delay });
  }
  encode() {
    const pal = medianCut(this.hist, PALETTE_SIZE);
    const map = new Int16Array(COLOURS_15_BIT).fill(-1);
    const out = new ByteWriter;
    out.str("GIF89a");
    out.u16(this.w);
    out.u16(this.h);
    out.byte(247);
    out.byte(0);
    out.byte(0);
    for (let i = 0;i < PALETTE_SIZE; i++) {
      const c = pal[i] ?? [0, 0, 0];
      out.byte(c[0]);
      out.byte(c[1]);
      out.byte(c[2]);
    }
    out.bytes([33, 255, 11]);
    out.str("NETSCAPE2.0");
    out.bytes([3, 1, 0, 0, 0]);
    const idx = new Uint8Array(this.w * this.h);
    for (const f of this.frames) {
      for (let i = 0;i < idx.length; i++) {
        const c = at(f.px, i);
        let k = at(map, c);
        if (k < 0) {
          k = nearest(pal, c);
          map[c] = k;
        }
        idx[i] = k;
      }
      out.bytes([33, 249, 4, 0]);
      out.u16(f.delay);
      out.bytes([0, 0]);
      out.byte(44);
      out.u16(0);
      out.u16(0);
      out.u16(this.w);
      out.u16(this.h);
      out.byte(0);
      lzw(idx, out);
    }
    out.byte(59);
    return out.result();
  }
}
var channel = (c, k) => c >> 10 - 5 * k & 31;
function medianCut(hist, n) {
  const cols = [];
  for (let c = 0;c < COLOURS_15_BIT; c++)
    if (at(hist, c))
      cols.push(c);
  const boxes = [cols];
  while (boxes.length < n) {
    let bi = -1, best = 0, axis = 0;
    boxes.forEach((b, i) => {
      if (b.length < 2)
        return;
      for (let k = 0;k < 3; k++) {
        let lo = 31, hi = 0, w = 0;
        for (const c of b) {
          const v = channel(c, k);
          if (v < lo)
            lo = v;
          if (v > hi)
            hi = v;
          w += at(hist, c);
        }
        const score = (hi - lo) * Math.sqrt(w);
        if (score > best) {
          best = score;
          bi = i;
          axis = k;
        }
      }
    });
    if (bi < 0)
      break;
    const b = at(boxes, bi).sort((p, q) => channel(p, axis) - channel(q, axis));
    let total = 0, half = 0, cut = b.length - 1;
    for (const c of b)
      total += at(hist, c);
    for (let i = 0;i < b.length - 1; i++) {
      half += at(hist, at(b, i));
      if (half >= total / 2) {
        cut = i + 1;
        break;
      }
    }
    boxes.splice(bi, 1, b.slice(0, cut), b.slice(cut));
  }
  return boxes.map((b) => {
    let r = 0, g = 0, bl = 0, w = 0;
    for (const c of b) {
      const k = at(hist, c);
      r += channel(c, 0) * k;
      g += channel(c, 1) * k;
      bl += channel(c, 2) * k;
      w += k;
    }
    const mean = (v) => Math.round(v / w * 255 / 31);
    return [mean(r), mean(g), mean(bl)];
  });
}
function nearest(pal, c) {
  const r = (c >> 10 & 31) * 255 / 31, g = (c >> 5 & 31) * 255 / 31, b = (c & 31) * 255 / 31;
  let best = 0, bd = Infinity;
  for (let i = 0;i < pal.length; i++) {
    const p = at(pal, i), d = 2 * (p[0] - r) ** 2 + 4 * (p[1] - g) ** 2 + 3 * (p[2] - b) ** 2;
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}
function lzw(idx, out) {
  out.byte(8);
  let size = LZW_MIN_SIZE, next = LZW_FIRST_FREE, acc = 0, nbits = 0;
  const block = [];
  const emit = (code) => {
    acc |= code << nbits;
    nbits += size;
    while (nbits >= 8) {
      block.push(acc & 255);
      acc >>>= 8;
      nbits -= 8;
      if (block.length === SUB_BLOCK) {
        out.byte(SUB_BLOCK);
        out.bytes(block);
        block.length = 0;
      }
    }
  };
  const dict = new Map;
  emit(LZW_CLEAR);
  let prefix = at(idx, 0);
  for (let i = 1;i < idx.length; i++) {
    const k = at(idx, i), key = prefix * 256 + k, code = dict.get(key);
    if (code !== undefined) {
      prefix = code;
      continue;
    }
    emit(prefix);
    if (next < LZW_MAX_CODES) {
      dict.set(key, next++);
      if (next > 1 << size && size < LZW_MAX_SIZE)
        size++;
    } else {
      emit(LZW_CLEAR);
      dict.clear();
      size = LZW_MIN_SIZE;
      next = LZW_FIRST_FREE;
    }
    prefix = k;
  }
  emit(prefix);
  emit(LZW_END);
  if (nbits > 0)
    block.push(acc & 255);
  if (block.length) {
    out.byte(block.length);
    out.bytes(block);
  }
  out.byte(0);
}

class ByteWriter {
  buf = new Uint8Array(1 << 20);
  n = 0;
  byte(b) {
    if (this.n === this.buf.length) {
      const nb = new Uint8Array(2 * this.n);
      nb.set(this.buf);
      this.buf = nb;
    }
    this.buf[this.n++] = b;
  }
  bytes(a) {
    for (const b of a)
      this.byte(b);
  }
  u16(v) {
    this.byte(v & 255);
    this.byte(v >> 8 & 255);
  }
  str(s) {
    for (let i = 0;i < s.length; i++)
      this.byte(s.charCodeAt(i));
  }
  result() {
    return this.buf.subarray(0, this.n);
  }
}

// web/app.ts
var KNOT_MS = 1852 / 3600;
var T_END = 0.003;
var BUDGET_MS = 12;
var APPROACH_S = 1.6;
var APPROACH_M = 1.2;
var PRESSURE_LOW_PA = 1e5;
var PRESSURE_HIGH_PA = 400000000;
var EP_MAX = 0.18;
var GIF_W = 480;
var GIF_SIM_US = 25;
var GIF_CAPTION_PX = 28;
var GIF_APPROACH_PERIOD_MS = 100;
var GIF_APPROACH_DELAY_CS = 10;
var GIF_SIM_DELAY_CS = 4;
var GIF_LAST_DELAY_CS = 300;
var GPU_BATCH_MAX = 512;
var GPU_BATCH_MIN = 8;
var STRUCTURE_FAIL_PHASE = 4;
var MIN_DEPTH = 0.01;
var SKIN_COLUMN_FRACTION = 0.76 / 1.0668;
var SKIN_ROW_MULTIPLE = 8;
var WALL_RUN_LENGTHS = 1.5;
var MUTED_KEY = "goosestrike-muted";
var SOURCE_URL = "strike.goose";
var DEFAULT_YAW = 1;
var DEFAULT_PITCH = 0.25;
var DEFAULT_DIST = 1.3;
var DIST_MIN = 0.4;
var DIST_MAX = 5;
var PITCH_LIMIT = 1.45;
function el(id, kind) {
  const found = document.getElementById(id);
  if (!(found instanceof kind))
    throw new Error(`#${id} is missing or has the wrong type`);
  return found;
}
var $ = (id) => el(id, HTMLElement);
var button = (id) => el(id, HTMLButtonElement);
var input = (id) => el(id, HTMLInputElement);
var canvasOf = (id) => el(id, HTMLCanvasElement);
var context2d = (cv) => must(cv.getContext("2d"), "the 2D context");
var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
var fmt = (x, d = 1) => x.toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
var sign = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
async function loadSolver() {
  try {
    const cpu = new URLSearchParams(location.search).get("solver") === "cpu";
    const gpu = cpu ? null : await loadGpuEngine().catch((error) => {
      console.warn("WebGPU solver:", error);
      return null;
    });
    return gpu ?? await loadEngine();
  } catch (error) {
    $("fire").textContent = "Goose unavailable";
    $("banner").setAttribute("role", "alert");
    $("banner").textContent = "The solver could not load. Reload the page to try again.";
    throw error;
  }
}
var eng = await loadSolver();
var PRESETS = [
  { name: "Potato goose", spacing: 0.009, cells: 56, peak: -42.6, plateau: 25.8, defl: 150.8, skin: 0, stiff: 36, run: "10 s", gpuRun: "1 to 2 s", gpuOnly: false },
  { name: "Good goose", spacing: 0.006, cells: 88, peak: 0.9, plateau: 27.7, defl: 152.1, skin: 8, stiff: 62, run: "35 s", gpuRun: "4 to 6 s", gpuOnly: false },
  { name: "Very good goose", spacing: 0.0045, cells: 112, peak: 2.7, plateau: 13.1, defl: 152.9, skin: 25, stiff: 152, run: "2 min", gpuRun: "10 to 12 s", gpuOnly: false },
  { name: "Excellent goose", spacing: 0.0035, cells: 144, peak: 11.3, plateau: 3.2, defl: 228.1, skin: 249, stiff: 191, run: null, gpuRun: "20 to 60 s", gpuOnly: true }
].filter((p) => eng.gpu || !p.gpuOnly);
function preset(i) {
  return at(PRESETS, i);
}
var st = {
  phase: "idle",
  mode: PANEL,
  preset: 0,
  speed: V_C,
  tSkin: T_SKIN,
  costume: true,
  pov: false,
  muted: false,
  yaw: DEFAULT_YAW,
  pitch: DEFAULT_PITCH,
  dist: DEFAULT_DIST,
  t0: 0,
  ke0: 1,
  peakF: 0,
  scene: null,
  hist: [],
  wall: null,
  frameSim: 0,
  frameReal: 16.7,
  msPerStep: 0,
  fps: 60,
  batch: { seen: 0, steps: 16 },
  splatted: false,
  solverPhase: -1,
  gif: null,
  gifHold: false
};
try {
  st.muted = localStorage.getItem(MUTED_KEY) === "1";
} catch {
  st.muted = false;
}
var audio = null;
function ac() {
  try {
    audio ??= new (window.AudioContext ?? window.webkitAudioContext ?? AudioContext);
  } catch {
    st.muted = true;
    showMute();
    return null;
  }
  return audio;
}
function honk(pitch = 1, when = 0) {
  if (st.muted)
    return;
  const a = ac();
  if (!a)
    return;
  const t = a.currentTime + when;
  const osc = a.createOscillator(), osc2 = a.createOscillator();
  const bp = a.createBiquadFilter(), g = a.createGain();
  osc.type = "sawtooth";
  osc2.type = "square";
  osc.frequency.setValueAtTime(410 * pitch, t);
  osc.frequency.exponentialRampToValueAtTime(300 * pitch, t + 0.22);
  osc2.frequency.setValueAtTime(205 * pitch, t);
  osc2.frequency.exponentialRampToValueAtTime(150 * pitch, t + 0.22);
  bp.type = "bandpass";
  bp.frequency.value = 1100;
  bp.Q.value = 2.5;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.35, t + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.26);
  osc.connect(bp);
  osc2.connect(bp);
  bp.connect(g);
  g.connect(a.destination);
  osc.start(t);
  osc2.start(t);
  osc.stop(t + 0.3);
  osc2.stop(t + 0.3);
}
function splat() {
  if (st.muted)
    return;
  const a = ac();
  if (!a)
    return;
  const t = a.currentTime;
  const n = Math.floor(a.sampleRate * 0.5);
  const buf = a.createBuffer(1, n, a.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0;i < n; i++)
    d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 2);
  const src = a.createBufferSource(), lp = a.createBiquadFilter(), g = a.createGain();
  src.buffer = buf;
  lp.type = "lowpass";
  lp.frequency.setValueAtTime(4000, t);
  lp.frequency.exponentialRampToValueAtTime(150, t + 0.45);
  g.gain.value = 0.6;
  src.connect(lp);
  lp.connect(g);
  g.connect(a.destination);
  src.start(t);
}
function readRender() {
  eng.fillRender();
  return { nodes: eng.nodes(), parts: eng.particles(), press: eng.pressure(), strain: eng.strain() };
}
function nearestParticle(px, x, y, z) {
  let best = 0, bd = Infinity;
  for (let i = 0;i < px.length / 3; i++) {
    const dx = at(px, 3 * i) - x, dy = at(px, 3 * i + 1) - y, dz = at(px, 3 * i + 2) - z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}
var GOOSE = {
  belly: [252, 250, 244],
  back: [234, 228, 216],
  orange: "#e8913a",
  ink: "#2b2925",
  blush: "rgba(226, 120, 110, 0.5)",
  star: "#e9b640"
};
function paintGoose(p) {
  const n = p.length / 3;
  const D = eng.birdD(), L = eng.birdL(), r = 0.5 * D;
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0;i < n; i++) {
    zmin = Math.min(zmin, at(p, 3 * i + 2));
    zmax = Math.max(zmax, at(p, 3 * i + 2));
  }
  const head = zmin + r;
  const mid = zmin + 0.5 * L;
  const skin = [];
  for (let i = 0;i < n; i++) {
    const y = at(p, 3 * i + 1) / r;
    const s = Math.round(32 * smoothstep(-0.6, 1.2, y)) / 32;
    skin.push(rgb(GOOSE.belly.map((c, k) => c + s * (at(GOOSE.back, k) - c))));
  }
  const near = (x, y, z) => nearestParticle(p, x, y, z);
  const onHead = (x, y, z) => {
    const d = norm([x, y, -z]);
    return near(r * d[0], r * d[1], head + r * d[2]);
  };
  const anchors = {
    nose: onHead(0, -0.1, 1),
    eyes: [onHead(0.5, 0.42, 0.75), onHead(-0.5, 0.42, 0.75)],
    cheeks: [onHead(0.72, 0.02, 0.62), onHead(-0.72, 0.02, 0.62)],
    tuft: onHead(0, 0.95, 0.3),
    wings: [near(0.92 * r, 0.3 * r, mid + 0.25 * r), near(-0.92 * r, 0.3 * r, mid + 0.25 * r)],
    feet: [near(0.4 * r, -0.92 * r, mid + 0.3 * r), near(-0.4 * r, -0.92 * r, mid + 0.3 * r)],
    tail: near(0, 0, zmax)
  };
  return { skin, anchors };
}
function skinElements(cells) {
  const gs = Math.floor(cells * SKIN_COLUMN_FRACTION / SKIN_ROW_MULTIPLE + 0.5) * SKIN_ROW_MULTIPLE;
  return cells * gs;
}
function featureEdges(el) {
  const ne = el.length / 4, count = new Map;
  const key = (a, b) => a < b ? a * 1e6 + b : b * 1e6 + a;
  const side = (i) => key(at(el, i), at(el, i - i % 4 + (i + 1) % 4));
  for (let i = 0;i < 4 * ne; i++)
    count.set(side(i), (count.get(side(i)) ?? 0) + 1);
  const edges = new Uint8Array(ne);
  for (let i = 0;i < 4 * ne; i++)
    if (count.get(side(i)) !== 2)
      edges[i >> 2] = at(edges, i >> 2) | 1 << i % 4;
  return edges;
}
function sceneOf() {
  const render = readRender();
  const elems = eng.elements();
  const init = render.parts.slice();
  return { ...render, elems, edges: featureEdges(elems), skinNe: 0, init, ...paintGoose(init) };
}
function setupRun(mode) {
  const pr = preset(st.preset);
  eng.setSkin(st.tSkin);
  eng.setup(mode, pr.cells, pr.spacing, st.speed);
  st.batch = { seen: 0, steps: 16 };
  st.mode = mode;
  legend();
  const scene = sceneOf();
  scene.skinNe = mode === PANEL ? skinElements(pr.cells) : 0;
  st.scene = scene;
  st.ke0 = eng.birdKE();
  $("energyTitle").textContent = `Goose energy at start: ${fmt(st.ke0 / 1000, 1)} kJ`;
  st.peakF = 0;
  st.frameSim = 0;
  st.msPerStep = 0;
  st.hist = [];
  st.splatted = false;
  st.solverPhase = -1;
  drawForceChart();
  telemetry();
  return scene;
}
var canvas = canvasOf("view");
var ctx = context2d(canvas);
var W = 0;
var H = 0;
var DPR = 1;
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  W = Math.max(1, Math.round(r.width * DPR));
  H = Math.max(1, Math.round(r.height * DPR));
  canvas.width = W;
  canvas.height = H;
}
new ResizeObserver(resize).observe(canvas);
var cam = { eye: [0, 0, 0], r: [1, 0, 0], u: [0, 1, 0], f: [0, 0, 1], foc: 1 };
function norm(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
var dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function setCamera() {
  let eye, target;
  const upHint = [0, 1, 0];
  if (st.pov) {
    eye = [0.12, 0.06, -0.7];
    target = [0, 0, 0.02];
  } else {
    target = [0, 0, 0.1];
    const c = Math.cos(st.pitch);
    eye = [target[0] + st.dist * c * Math.sin(st.yaw), target[1] + st.dist * Math.sin(st.pitch), target[2] + st.dist * c * Math.cos(st.yaw)];
  }
  cam.eye = eye;
  cam.f = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  cam.r = norm(cross(cam.f, upHint));
  cam.u = cross(cam.r, cam.f);
  cam.foc = 0.5 * H / Math.tan((st.pov ? 34 : 20) * Math.PI / 180);
}
function proj(x, y, z, out, o) {
  const d = [x - cam.eye[0], y - cam.eye[1], z - cam.eye[2]];
  const cz = dot(d, cam.f), cx = dot(d, cam.r), cy = dot(d, cam.u);
  out[o] = 0.5 * W + cam.foc * cx / cz;
  out[o + 1] = 0.5 * H - cam.foc * cy / cz;
  out[o + 2] = cz;
}
var DEPTH_BUCKETS = 4096;
var sNodes = new Float32Array(0);
var sParts = new Float32Array(0);
var keys = new Float32Array(0);
var order = new Int32Array(0);
var tmp = new Int32Array(0);
var counts = new Int32Array(DEPTH_BUCKETS + 1);
function depthSort(n) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0;i < n; i++) {
    const k = at(keys, i);
    if (k < lo)
      lo = k;
    if (k > hi)
      hi = k;
  }
  const s = (DEPTH_BUCKETS - 1) / Math.max(hi - lo, 0.000000001);
  counts.fill(0);
  for (let i = 0;i < n; i++)
    tmp[i] = DEPTH_BUCKETS - 1 - Math.floor((at(keys, i) - lo) * s);
  for (let i = 0;i < n; i++) {
    const b = at(tmp, i) + 1;
    counts[b] = at(counts, b) + 1;
  }
  for (let b = 0;b < DEPTH_BUCKETS; b++)
    counts[b + 1] = at(counts, b + 1) + at(counts, b);
  for (let i = 0;i < n; i++) {
    const b = at(tmp, i), slot = at(counts, b);
    order[slot] = i;
    counts[b] = slot + 1;
  }
}
function grow(n) {
  if (keys.length < n) {
    keys = new Float32Array(n);
    order = new Int32Array(n);
    tmp = new Int32Array(n);
  }
}
function smoothstep(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}
function rgb(c) {
  return `rgb(${Math.round(at(c, 0))} ${Math.round(at(c, 1))} ${Math.round(at(c, 2))})`;
}
var map3 = (v, f) => [f(v[0]), f(v[1]), f(v[2])];
function toLab(hex) {
  const lin = (k) => {
    const v = parseInt(hex.slice(k, k + 2), 16) / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const l = [lin(1), lin(3), lin(5)];
  const m = map3([
    0.4122214708 * l[0] + 0.5363325363 * l[1] + 0.0514459929 * l[2],
    0.2119034982 * l[0] + 0.6806995451 * l[1] + 0.1073969566 * l[2],
    0.0883024619 * l[0] + 0.2817188376 * l[1] + 0.6299787005 * l[2]
  ], Math.cbrt);
  return [
    0.2104542553 * m[0] + 0.793617785 * m[1] - 0.0040720468 * m[2],
    1.9779984951 * m[0] - 2.428592205 * m[1] + 0.4505937099 * m[2],
    0.0259040371 * m[0] + 0.7827717662 * m[1] - 0.808675766 * m[2]
  ];
}
function fromLab(c) {
  const m = map3([
    c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2],
    c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2],
    c[0] - 0.0894841775 * c[1] - 1.291485548 * c[2]
  ], (v) => v * v * v);
  const l = [
    4.0767416621 * m[0] - 3.3077115913 * m[1] + 0.2309699292 * m[2],
    -1.2684380046 * m[0] + 2.6097574011 * m[1] - 0.3413193965 * m[2],
    -0.0041960863 * m[0] - 0.7034186147 * m[1] + 1.707614701 * m[2]
  ];
  return map3(l, (v) => 255 * Math.max(0, Math.min(1, v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055)));
}
function ramp(stops) {
  const lab = stops.map(toLab), out = [];
  for (let i = 0;i < 256; i++) {
    const x = i / 255 * (lab.length - 1), k = Math.min(lab.length - 2, Math.floor(x)), f = x - k;
    const a = at(lab, k), b = at(lab, k + 1);
    out.push(fromLab([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])]));
  }
  return out;
}
var HOT = ["#e2b54f", "#c4613f", "#6e2a22"];
var MAPS = {
  metal: ramp(["#c7c3ba", ...HOT]),
  primer: ramp(["#9da27f", ...HOT]),
  pressure: ramp(["#243552", "#3f6f86", "#94b49c", "#e8c766", "#c4613f"])
};
var GRID = "rgba(43, 41, 37, 0.35)";
var SHADES = 64;
var mapCache = new Map;
function mapColour(name, s, shade = 1) {
  const i = Math.round(255 * Math.max(0, Math.min(1, s))), j = Math.round(SHADES * shade);
  const key = name + (i * 32 + j);
  let c = mapCache.get(key);
  if (!c) {
    c = rgb(map3(at(MAPS[name], i), (v) => v * j / SHADES));
    mapCache.set(key, c);
  }
  return c;
}
var logP = (p) => (Math.log10(Math.max(p, PRESSURE_LOW_PA)) - Math.log10(PRESSURE_LOW_PA)) / (Math.log10(PRESSURE_HIGH_PA) - Math.log10(PRESSURE_LOW_PA));
function pressureColour(p) {
  return mapColour("pressure", logP(p));
}
function strainColour(ep, shade, inside) {
  return mapColour(inside ? "primer" : "metal", Math.sqrt(ep / EP_MAX), shade);
}
function cssRamp(name) {
  return `linear-gradient(90deg, ${Array.from({ length: 17 }, (_, k) => mapColour(name, k / 16)).join(",")})`;
}
function render(scene, offsetZ, flap) {
  setCamera();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const nn = scene.nodes.length / 3, np = scene.parts.length / 3, ne = scene.elems.length / 4;
  if (sNodes.length < 3 * nn)
    sNodes = new Float32Array(3 * nn);
  if (sParts.length < 3 * np)
    sParts = new Float32Array(3 * np);
  for (let i = 0;i < nn; i++)
    proj(at(scene.nodes, 3 * i), at(scene.nodes, 3 * i + 1), at(scene.nodes, 3 * i + 2), sNodes, 3 * i);
  for (let i = 0;i < np; i++)
    proj(at(scene.parts, 3 * i), at(scene.parts, 3 * i + 1), at(scene.parts, 3 * i + 2) + offsetZ, sParts, 3 * i);
  if (st.mode === RIGID_WALL)
    drawWall();
  const costume = st.costume && st.mode === PANEL;
  const decor = costume ? decorations(scene, offsetZ, flap) : [];
  const n = ne + np + decor.length;
  grow(n);
  for (let e = 0;e < ne; e++) {
    const a = at(scene.elems, 4 * e), b = at(scene.elems, 4 * e + 1), c = at(scene.elems, 4 * e + 2), d = at(scene.elems, 4 * e + 3);
    keys[e] = at(scene.strain, e) < 0 ? -1 : 0.25 * (at(sNodes, 3 * a + 2) + at(sNodes, 3 * b + 2) + at(sNodes, 3 * c + 2) + at(sNodes, 3 * d + 2));
  }
  for (let i = 0;i < np; i++)
    keys[ne + i] = at(sParts, 3 * i + 2);
  for (let k = 0;k < decor.length; k++)
    keys[ne + np + k] = at(decor, k).depth;
  depthSort(n);
  const pr = eng.spacing() * 0.62 * cam.foc;
  for (let k = 0;k < n; k++) {
    const id = at(order, k);
    if (id < ne)
      drawShell(scene, id);
    else if (id < ne + np) {
      const i = id - ne, z = at(sParts, 3 * i + 2);
      if (z <= MIN_DEPTH)
        continue;
      ctx.fillStyle = costume ? at(scene.skin, i) : pressureColour(at(scene.press, i));
      ctx.beginPath();
      ctx.arc(at(sParts, 3 * i), at(sParts, 3 * i + 1), Math.max(0.8, pr / z), 0, 2 * Math.PI);
      ctx.fill();
    } else
      at(decor, id - ne - np).draw();
  }
  if (costume && contactStarted())
    drawStars(scene, offsetZ);
}
function drawShell(scene, e) {
  if (at(scene.strain, e) < 0)
    return;
  const { elems: el, nodes: P } = scene;
  const a = at(el, 4 * e), b = at(el, 4 * e + 1), c = at(el, 4 * e + 2), d = at(el, 4 * e + 3);
  if (at(sNodes, 3 * a + 2) <= MIN_DEPTH || at(sNodes, 3 * c + 2) <= MIN_DEPTH)
    return;
  const node = (i) => [at(P, 3 * i), at(P, 3 * i + 1), at(P, 3 * i + 2)];
  const sub = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
  const nrm = norm(cross(sub(node(c), node(a)), sub(node(d), node(b))));
  const toEye = norm(sub(cam.eye, node(a)));
  const facing = dot(nrm, toEye);
  const shade = 0.72 + 0.28 * Math.abs(facing);
  const inside = e >= scene.skinNe || facing < 0;
  ctx.fillStyle = strainColour(at(scene.strain, e), shade, inside);
  ctx.strokeStyle = ctx.fillStyle;
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.moveTo(at(sNodes, 3 * a), at(sNodes, 3 * a + 1));
  ctx.lineTo(at(sNodes, 3 * b), at(sNodes, 3 * b + 1));
  ctx.lineTo(at(sNodes, 3 * c), at(sNodes, 3 * c + 1));
  ctx.lineTo(at(sNodes, 3 * d), at(sNodes, 3 * d + 1));
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = GRID;
  ctx.lineWidth = 0.5 * DPR;
  ctx.stroke();
  const f = at(scene.edges, e);
  if (!f)
    return;
  const q = [a, b, c, d];
  ctx.strokeStyle = GOOSE.ink;
  ctx.lineWidth = 1.2 * DPR;
  ctx.beginPath();
  for (let k = 0;k < 4; k++) {
    if (!(f >> k & 1))
      continue;
    const i = at(q, k), j = at(q, (k + 1) % 4);
    ctx.moveTo(at(sNodes, 3 * i), at(sNodes, 3 * i + 1));
    ctx.lineTo(at(sNodes, 3 * j), at(sNodes, 3 * j + 1));
  }
  ctx.stroke();
}
function drawWall() {
  const o = [0, 0, 0];
  ctx.strokeStyle = "#7a8794";
  ctx.lineWidth = 1;
  for (let k = -5;k <= 5; k++) {
    const lines = [[k * 0.05, -0.25, k * 0.05, 0.25], [-0.25, k * 0.05, 0.25, k * 0.05]];
    for (const [x0, y0, x1, y1] of lines) {
      proj(x0, y0, 0, o, 0);
      const sx = at(o, 0), sy = at(o, 1);
      proj(x1, y1, 0, o, 0);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(at(o, 0), at(o, 1));
      ctx.stroke();
    }
  }
}
function contactStarted() {
  return eng.tContact() >= 0 && st.phase !== "approach";
}
function sp(i) {
  return [at(sParts, 3 * i), at(sParts, 3 * i + 1), at(sParts, 3 * i + 2)];
}
function blob(pts) {
  const m = (p, q) => [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
  const n = pts.length;
  let s = m(at(pts, n - 1), at(pts, 0));
  ctx.beginPath();
  ctx.moveTo(s[0], s[1]);
  for (let i = 0;i < n; i++) {
    const p = at(pts, i);
    s = m(p, at(pts, (i + 1) % n));
    ctx.quadraticCurveTo(p[0], p[1], s[0], s[1]);
  }
  ctx.closePath();
}
function decorations(scene, offsetZ, flap) {
  const now = reducedMotion.matches ? 0 : performance.now();
  const A = scene.anchors, D = eng.birdD(), P = scene.parts;
  const wp = (i) => [at(P, 3 * i), at(P, 3 * i + 1), at(P, 3 * i + 2) + offsetZ];
  const pt = (w) => {
    const o = [0, 0, 0];
    proj(w[0], w[1], w[2], o, 0);
    return o;
  };
  const nose = wp(A.nose), tail = wp(A.tail);
  const n = pt(nose), t = pt(tail);
  let ux = n[0] - t[0], uy = n[1] - t[1];
  const ul = Math.hypot(ux, uy);
  if (ul < 0.000001) {
    ux = 1;
    uy = 0;
  } else {
    ux /= ul;
    uy /= ul;
  }
  const top = pt([nose[0], nose[1] + 0.1, nose[2]]);
  const side = (top[0] - n[0]) * -uy + (top[1] - n[1]) * ux > 0 ? 1 : -1;
  const vx = -uy * side, vy = ux * side;
  const at2 = (c, k, u, v) => [at(c, 0) + k * (u * ux + v * vx), at(c, 1) + k * (u * uy + v * vy)];
  const dizzy = contactStarted();
  const list = [];
  const scale = (z) => D * cam.foc / z;
  const outline = (z) => {
    ctx.strokeStyle = GOOSE.ink;
    ctx.lineWidth = Math.max(DPR, 0.012 * scale(z));
    ctx.lineJoin = "round";
    ctx.stroke();
  };
  const part = (c, fill, pts, dz) => {
    list.push({ depth: c[2] + dz, draw() {
      ctx.fillStyle = fill;
      blob(pts);
      ctx.fill();
      outline(c[2]);
    } });
  };
  const feather = rgb(GOOSE.belly);
  for (const i of A.wings) {
    const c = sp(i), k = scale(c[2]);
    const up = dizzy ? -0.05 + 0.03 * Math.sin(now / 40) : 0.25 + 0.2 * flap;
    part(c, feather, [at2(c, k, 0.18, 0.02), at2(c, k, -0.2, 0.12 + up), at2(c, k, -0.5, 0.1 + 0.8 * up), at2(c, k, -0.3, -0.12)], -0.03);
  }
  {
    const c = sp(A.tuft), k = scale(c[2]);
    part(c, feather, [at2(c, k, 0.05, -0.02), at2(c, k, 0.04, 0.14), at2(c, k, -0.12, 0.24), at2(c, k, -0.07, 0.04)], -0.02);
  }
  {
    const k = scale(n[2]);
    part(n, GOOSE.orange, [at2(n, k, -0.04, 0.09), at2(n, k, 0.26, 0.05), at2(n, k, 0.26, -0.03), at2(n, k, -0.04, -0.08)], -0.03);
  }
  for (const i of A.feet) {
    const c = sp(i), k = scale(c[2]);
    part(c, GOOSE.orange, [at2(c, k, 0, 0), at2(c, k, -0.06, -0.2), at2(c, k, 0.06, -0.26), at2(c, k, 0.22, -0.2), at2(c, k, 0.1, -0.08)], -0.01);
  }
  for (const i of A.cheeks) {
    const c = sp(i);
    const rr = 0.07 * scale(c[2]);
    list.push({ depth: c[2] - 0.008, draw() {
      ctx.fillStyle = GOOSE.blush;
      ctx.beginPath();
      ctx.ellipse(c[0], c[1], rr * 1.3, rr * 0.85, Math.atan2(uy, ux), 0, 2 * Math.PI);
      ctx.fill();
    } });
  }
  for (const i of A.eyes) {
    const c = sp(i);
    const rr = 0.11 * scale(c[2]);
    list.push({ depth: c[2] - 0.01, draw() {
      ctx.fillStyle = GOOSE.ink;
      ctx.beginPath();
      ctx.arc(c[0], c[1], rr, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "#fff";
      if (dizzy) {
        ctx.beginPath();
        ctx.arc(c[0], c[1], rr * 0.82, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = GOOSE.ink;
        ctx.lineWidth = Math.max(DPR, 0.16 * rr);
        ctx.beginPath();
        const turn = now / 120;
        for (let k = 0;k <= 40; k++) {
          const a = turn + k * 0.45, q = rr * 0.7 * k / 40;
          const x = c[0] + q * Math.cos(a), y = c[1] + q * Math.sin(a);
          if (k === 0)
            ctx.moveTo(x, y);
          else
            ctx.lineTo(x, y);
        }
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(c[0] - rr * 0.32, c[1] - rr * 0.32, rr * 0.34, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(c[0] + rr * 0.3, c[1] + rr * 0.3, rr * 0.14, 0, 2 * Math.PI);
        ctx.fill();
      }
    } });
  }
  return list;
}
function drawStars(scene, offsetZ) {
  const P = scene.parts, np = P.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0;i < np; i++) {
    cx += at(P, 3 * i);
    cy += at(P, 3 * i + 1);
    cz += at(P, 3 * i + 2);
  }
  cx /= np;
  cy /= np;
  cz /= np;
  const D = eng.birdD(), t = reducedMotion.matches ? 0 : performance.now() / 500, o = [0, 0, 0];
  for (let k = 0;k < 3; k++) {
    const a = t + k * 2.094;
    proj(cx + 0.6 * D * Math.cos(a), cy + 0.9 * D, cz + offsetZ + 0.6 * D * Math.sin(a), o, 0);
    if (o[2] <= MIN_DEPTH)
      continue;
    const r = 0.1 * D * cam.foc / o[2];
    const q = [];
    for (let j = 0;j < 10; j++) {
      const ang = -Math.PI / 2 + j * Math.PI / 5, rr = j % 2 ? r * 0.5 : r;
      q.push([o[0] + rr * Math.cos(ang), o[1] + rr * Math.sin(ang)]);
    }
    ctx.fillStyle = GOOSE.star;
    ctx.beginPath();
    q.forEach((p, j) => j ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = GOOSE.ink;
    ctx.lineWidth = Math.max(DPR, 0.012 * D * cam.foc / o[2]);
    ctx.lineJoin = "round";
    ctx.stroke();
  }
}
var chartData = new WeakMap;
var chartResize = new ResizeObserver((entries) => {
  for (const { target } of entries) {
    if (!(target instanceof HTMLCanvasElement))
      continue;
    const data = chartData.get(target);
    if (data)
      chart(target, data.series, data.opt);
  }
});
function chart(cv, series, opt) {
  if (!chartData.has(cv))
    chartResize.observe(cv);
  chartData.set(cv, { series, opt });
  const dpr = DPR, r = cv.getBoundingClientRect();
  cv.width = Math.round(r.width * dpr);
  cv.height = Math.round(r.height * dpr);
  const c = context2d(cv);
  const css = getComputedStyle(document.body);
  const ink = css.getPropertyValue("--ink"), muted = css.getPropertyValue("--muted"), line = css.getPropertyValue("--line");
  c.scale(dpr, dpr);
  const { width: w, height: h } = r, L = 46, R = 16, T = 8, B = 30;
  const fy = opt.log ? (y) => Math.log10(Math.max(y, opt.ymin)) : (y) => y;
  const y0 = fy(opt.ymin), y1 = fy(opt.ymax);
  const X = (x) => L + (w - L - R) * (x - opt.xmin) / (opt.xmax - opt.xmin);
  const Y = (y) => T + (h - T - B) * (1 - (fy(y) - y0) / (y1 - y0));
  c.font = `11px ${css.getPropertyValue("--mono")}`;
  c.strokeStyle = line;
  c.fillStyle = muted;
  c.lineWidth = 1;
  for (const yt of opt.yticks) {
    c.beginPath();
    c.moveTo(L, Y(yt));
    c.lineTo(w - R, Y(yt));
    c.stroke();
    c.textAlign = "right";
    c.fillText(String(yt), L - 4, Y(yt) + 4);
  }
  for (const xt of opt.xticks) {
    c.textAlign = "center";
    c.fillText(String(xt), X(xt), h - B + 14);
  }
  c.textAlign = "left";
  c.fillText(opt.ylabel, L + 4, T + 10);
  c.textAlign = "right";
  c.fillText(opt.xlabel, w - R, h - 4);
  for (const hl of opt.hlines ?? []) {
    c.strokeStyle = hl.color;
    c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(L, Y(hl.y));
    c.lineTo(w - R, Y(hl.y));
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = hl.color;
    c.textAlign = "right";
    c.fillText(hl.label, w - R - 2, Y(hl.y) - 4);
  }
  for (const s of series) {
    c.strokeStyle = s.color || ink;
    c.lineWidth = 1.6;
    c.beginPath();
    s.xs.forEach((x, i) => {
      const px = X(Math.min(Math.max(x, opt.xmin), opt.xmax)), py = Y(Math.min(Math.max(at(s.ys, i), opt.ymin), opt.ymax));
      if (i)
        c.lineTo(px, py);
      else
        c.moveTo(px, py);
    });
    c.stroke();
  }
}
function ticks(max, n) {
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const out = [];
  for (let v = 0;v <= max + 0.000000001; v += step)
    out.push(+v.toFixed(6));
  return out;
}
function drawForceChart() {
  const xs = st.hist.map((h) => h.t * 1e6), ys = st.hist.map((h) => h.f / 1000);
  const ymax = Math.max(50, ...ys) * 1.1;
  chart(canvasOf("forceChart"), [{ xs, ys }], {
    xmin: 0,
    xmax: T_END * 1e6,
    ymin: 0,
    ymax,
    xticks: ticks(T_END * 1e6, 6),
    yticks: ticks(ymax, 4),
    xlabel: "time (µs)",
    ylabel: "contact force (kN)"
  });
}
var isRunning = () => st.phase === "approach" || st.phase === "sim" || st.phase === "wall";
function fire() {
  if (isRunning())
    return;
  setupRun(PANEL);
  st.phase = "approach";
  st.t0 = performance.now();
  startGif();
  $("verdict").hidden = true;
  setBusy(true);
  const view = canvas.getBoundingClientRect();
  if (view.top < 0 || view.bottom > window.innerHeight) {
    canvas.scrollIntoView({ behavior: "instant", block: "nearest" });
  }
  honk(Math.sqrt(st.speed / V_C));
  honk(Math.sqrt(st.speed / V_C) * 1.06, 0.3);
}
function setBusy(b) {
  const focusStop = b && (document.activeElement === $("fire") || document.activeElement === $("wilbeck"));
  const focusFire = !b && document.activeElement === $("stop");
  button("fire").disabled = b;
  button("wilbeck").disabled = b;
  button("stop").disabled = !b;
  button("gif").disabled = b || !st.gif;
  $("fire").textContent = b ? "Goose in flight" : "Fire the goose";
  const runName = st.mode === RIGID_WALL ? "Wall check" : "Panel run";
  $("runStatus").textContent = b ? `${runName} active. Press Escape or Stop and reset to stop.` : st.phase === "done" ? `${runName} complete. Results appear below.` : "";
  for (const control of document.querySelectorAll("#presets input, #speed, #speedNumber, #skin"))
    control.disabled = b;
  const snap = document.getElementById("snap");
  if (snap instanceof HTMLButtonElement)
    snap.disabled = b;
  if (focusStop)
    button("stop").focus({ preventScroll: true });
  if (focusFire)
    button("fire").focus({ preventScroll: true });
}
function clearWallChart() {
  const wallChart = canvasOf("wallChart");
  chartResize.unobserve(wallChart);
  chartData.delete(wallChart);
  context2d(wallChart).clearRect(0, 0, wallChart.width, wallChart.height);
}
function stopRun() {
  if (!isRunning())
    return;
  if (st.mode === RIGID_WALL) {
    st.wall = null;
    clearWallChart();
    $("wilbeckOut").textContent = "Wall check stopped. No result.";
  }
  st.gif = null;
  st.gifHold = false;
  $("bubble").hidden = true;
  preview();
  setBusy(false);
  $("runStatus").textContent = "Run stopped. Goose reset.";
  $("banner").textContent = "Run stopped. Goose reset. Select a speed or resolution, then press Fire the goose.";
}
function stepSolver(cpu, tEnd, perStep) {
  const t0 = performance.now(), s0 = cpu.time();
  let steps = 0, t = t0;
  while (t - t0 < BUDGET_MS) {
    const k = cpu.stepPhase(tEnd);
    if (k < 0)
      break;
    t = performance.now();
    st.solverPhase = k;
    if (k === STRUCTURE_FAIL_PHASE) {
      steps++;
      perStep(cpu.time(), cpu.contactForce(), cpu.centerForce());
    }
  }
  const ms = t - t0;
  st.msPerStep = steps ? ms / steps : 0;
  st.frameSim = cpu.time() - s0;
  return cpu.time() >= tEnd;
}
function gpuStep(gpu, tEnd, perStep) {
  const b = st.batch, s0 = gpu.time();
  let got = 0, steps = 0, r;
  while ((r = gpu.poll()) === 1)
    got++;
  if (r < 0)
    throw new Error("the GPU read-back failed");
  if (got) {
    const h = gpu.history();
    steps = h.length / 4 - b.seen;
    for (;b.seen < h.length / 4; b.seen++)
      perStep(at(h, 4 * b.seen), at(h, 4 * b.seen + 1), at(h, 4 * b.seen + 2));
    if (steps) {
      st.msPerStep = st.frameReal / steps;
      st.frameSim = gpu.time() - s0;
    }
  }
  if (got === 2)
    b.steps = Math.min(GPU_BATCH_MAX, Math.ceil(b.steps * 1.5));
  if (got === 0 && gpu.pending() === 2)
    b.steps = Math.max(GPU_BATCH_MIN, Math.floor(b.steps * 0.8));
  if (!gpu.running())
    return gpu.pending() === 0;
  while (gpu.run(b.steps, tEnd))
    ;
  return false;
}
var solve = (tEnd, perStep) => eng.gpu ? gpuStep(eng, tEnd, perStep) : stepSolver(eng, tEnd, perStep);
function telemetry() {
  const tc = eng.tContact();
  const rows = [
    ["sim time", `${fmt(eng.time() * 1e6, 1)} µs`],
    ["since contact", tc >= 0 ? `${fmt((eng.time() - tc) * 1e6, 1)} µs` : "goose airborne"],
    ["time step dt", `${fmt(eng.dt() * 1e9, 1)} ns`],
    ["steps", fmt(eng.steps(), 0)],
    ["slow motion", st.frameSim > 0 ? `1 : ${fmt(st.frameReal / 1000 / st.frameSim, 0)}` : "—"],
    ["solver", `${eng.gpu ? "GPU" : "CPU"}, ${fmt(st.msPerStep, 2)} ms/step`],
    ["frame rate", `${fmt(st.fps, 0)} fps`],
    ["contact force", `${fmt(eng.contactForce() / 1000, 1)} kN`],
    ["peak force", `${fmt(st.peakF / 1000, 1)} kN`],
    ["max deflection", `${fmt(eng.maxDeflection() * 1000, 1)} mm`],
    ["max plastic strain", `${fmt(eng.maxPlasticStrain() * 100, 1)} %`],
    ["fails at (skin / stiffeners)", `${fmt(eng.failStrain(0) * 100, 1)} / ${fmt(eng.failStrain(1) * 100, 1)} %`],
    ["eroded skin shells", fmt(eng.failedSkin(), 0)],
    ["eroded stiffener shells", fmt(eng.failed() - eng.failedSkin(), 0)],
    ["bird", `${fmt(eng.np(), 0)} particles, ${fmt(BIRD_MASS, 3)} kg`],
    ["panel", `${fmt(eng.nn(), 0)} nodes, ${fmt(eng.ne(), 0)} shells`],
    ["skin", `${fmt(st.tSkin * 1000, 1)} mm 2024-T3`]
  ];
  const tele = $("tele");
  if (!tele.children.length)
    tele.innerHTML = rows.map(([k]) => `<dt>${k}</dt><dd></dd>`).join("");
  rows.forEach(([, v], i) => {
    const cell = at(tele.children, 2 * i + 1);
    if (cell.textContent !== v)
      cell.textContent = v;
  });
  $("honkBar").style.transform = `scaleX(${Math.min(1, eng.contactForce() / 300000)})`;
  const parts = [
    ["bird kinetic", eng.birdKE(), "#3f6f86"],
    ["bird squish (internal)", eng.birdWork(), "#94b49c"],
    ["panel kinetic", eng.panelKE(), "#e2b54f"],
    ["panel strain + plastic", eng.panelWork(), "#c4613f"]
  ];
  const sum = parts.reduce((s, p) => s + p[1], 0);
  $("ebar").innerHTML = parts.map(([, v, c]) => `<div style="width:${Math.max(0, v / st.ke0 * 100)}%;background:${c}"></div>`).join("");
  $("ekey").innerHTML = parts.map(([k, v, c]) => `<li><span><i style="background:${c}"></i>${k}</span><span>${fmt(v / 1000, 2)} kJ</span></li>`).join("") + `<li><span>balance error</span><span>${sign((sum / st.ke0 - 1) * 100, 2)} %</span></li>`;
}
function verdict() {
  const v = $("verdict"), pr = preset(st.preset);
  const skin = eng.failedSkin(), stiff = eng.failed() - skin;
  const breached = skin > 0;
  v.className = "frame panel verdict " + (breached ? "breached" : "held");
  v.hidden = false;
  v.innerHTML = `
    <p class="stamp">${breached ? "Skin breached" : "Skin held"}</p>
    <h2>${breached ? "The goose tore the skin." : "The skin held. The goose did not."}</h2>
    <p>This run: ${pr.name} at ${fmt(st.speed, 2)} m/s (${fmt(st.speed / KNOT_MS, 1)} kt), ${fmt(pr.spacing * 1000, 1)} mm bird lattice, ${fmt(st.tSkin * 1000, 1)} mm skin, on the ${eng.gpu ? "GPU" : "CPU"}.</p>
    <p>Eroded shell elements: ${fmt(skin, 0)} in the 2024-T3 skin, ${fmt(stiff, 0)} in the 7075-T6 stringers and frames.
      An element erodes when its plastic strain reaches the failure strain of its material. Maximum deflection ${fmt(eng.maxDeflection() * 1000, 1)} mm.
      Peak contact force ${fmt(st.peakF / 1000, 1)} kN. Maximum plastic strain ${fmt(eng.maxPlasticStrain() * 100, 1)} %.
      The panel took ${fmt(eng.panelWork() / 1000, 1)} kJ of the goose's ${fmt(st.ke0 / 1000, 1)} kJ.</p>
    <p>Does this fail 25.571(e)(1)? Not by itself. The rule accepts damage: the aeroplane must complete the flight with it.
      That needs a residual strength analysis of the damaged structure, which this page does not do.
      This is not a certification analysis.</p>
    <p class="hint">Measured at V<sub>C</sub> with the 1.6 mm skin only: maximum deflection ${PRESETS.map((p) => fmt(p.defl, 1)).join(", ")} mm
      and ${PRESETS.map((p) => p.skin).join(", ")} eroded skin elements at ${PRESETS.map((p) => fmt(p.spacing * 1000, 1)).join(", ")} mm.
      From 9 mm to 4.5 mm, the deflection changes by less than 2 %.${eng.gpu ? " At 3.5 mm, many more skin elements erode, a hole opens in the skin, and the deflection is 50 % larger." : ""}
      The skin erosion does not converge: erosion depends on the element size, so a tear prediction needs a mesh-calibrated failure model.</p>`;
}
function wallCheck() {
  if (isRunning())
    return;
  setupRun(RIGID_WALL);
  const v = st.speed;
  const [ph, us] = hugoniot(v);
  st.wall = { ph, us, ps: stagnation(v), tl: eng.birdL() / v, area: Math.PI * eng.sensorR() ** 2, xs: [], ys: [], peak: 0, plat: 0, platN: 0 };
  st.phase = "wall";
  clearWallChart();
  $("wilbeckOut").textContent = "Wall check active. No result yet.";
  startGif();
  $("verdict").hidden = true;
  $("wallChart").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
  setBusy(true);
  honk(0.8);
}
function wallSample(time, f, fc) {
  st.peakF = Math.max(st.peakF, f);
  const w = must(st.wall, "the wall check"), tc = eng.tContact(), t = time - tc;
  if (tc < 0 || t < 0)
    return;
  const p = fc / w.area;
  w.xs.push(t * 1e6);
  w.ys.push(p / 1e6);
  if (p > w.peak)
    w.peak = p;
  if (t > 0.25 * w.tl && t < 0.75 * w.tl) {
    w.plat += p;
    w.platN++;
  }
  if (!st.splatted) {
    st.splatted = true;
    splat();
  }
}
function drawWallChart() {
  const w = must(st.wall, "the wall check");
  chart(canvasOf("wallChart"), [{ xs: w.xs, ys: w.ys, color: "#3f6f86" }], {
    xmin: 0,
    xmax: WALL_RUN_LENGTHS * w.tl * 1e6,
    ymin: 1,
    ymax: 1000,
    log: true,
    xticks: ticks(WALL_RUN_LENGTHS * w.tl * 1e6, 6),
    yticks: [1, 10, 100, 1000],
    xlabel: "time since contact (µs)",
    ylabel: "centre pressure (MPa, log)",
    hlines: [
      { y: w.ph / 1e6, label: `P_H ${fmt(w.ph / 1e6, 1)} MPa`, color: "#a8432a" },
      { y: w.ps / 1e6, label: `P_s ${fmt(w.ps / 1e6, 2)} MPa`, color: "#4f7a4a" }
    ]
  });
  const pm = w.platN ? w.plat / w.platN : 0;
  $("wilbeckOut").textContent = `v = ${fmt(st.speed, 2)} m/s, bird lattice ${fmt(preset(st.preset).spacing * 1000, 1)} mm, ${fmt(eng.np(), 0)} particles, sensor r = ${fmt(eng.sensorR() * 1000, 2)} mm
` + `Hugoniot  P_H = ${fmt(w.ph / 1e6, 1)} MPa (U_s = ${fmt(w.us, 0)} m/s)   peak    ${fmt(w.peak / 1e6, 1)} MPa   error ${sign((w.peak - w.ph) / w.ph * 100)} %
` + `Stagnation P_s = ${fmt(w.ps / 1e6, 2)} MPa                plateau ${fmt(pm / 1e6, 2)} MPa   error ${w.platN ? sign((pm - w.ps) / w.ps * 100) : "—"} %
` + `impulse / (m v) = ${fmt(eng.impulse() / (BIRD_MASS * st.speed), 4)}   (plateau: mean over 0.25 to 0.75 of L/v = ${fmt(w.tl * 1e6, 0)} µs)`;
}
var lastFrame = performance.now();
function frame(now) {
  try {
    drawFrame(now);
  } catch (error) {
    console.error(error);
    stopRun();
    $("banner").textContent = "The solver failed. Run stopped. Reload the page to try again.";
  }
  requestAnimationFrame(frame);
}
function finishRun() {
  st.phase = "done";
  st.solverPhase = -1;
  st.gifHold = true;
  setBusy(false);
}
function drawFrame(now) {
  st.frameReal = now - lastFrame;
  st.fps += 0.05 * (1000 / Math.max(st.frameReal, 1) - st.fps);
  lastFrame = now;
  let offsetZ = 0, flap = 0;
  const banner = $("banner");
  let bannerText;
  if (st.phase === "approach") {
    const s = Math.min(1, (now - st.t0) / 1000 / APPROACH_S);
    offsetZ = APPROACH_M * (1 - s);
    flap = reducedMotion.matches ? 0 : Math.sin(now / 70);
    bannerText = `APPROACH: cartoon, not simulated (the real ${fmt(APPROACH_M, 1)} m takes ${fmt(APPROACH_M / st.speed * 1000, 1)} ms at ${fmt(st.speed, 1)} m/s)`;
    if (s >= 1)
      st.phase = "sim";
  } else if (st.phase === "sim") {
    const done = solve(T_END, (t, f) => {
      st.peakF = Math.max(st.peakF, f);
      st.hist.push({ t, f });
    });
    refreshScene();
    if (eng.tContact() >= 0 && !st.splatted) {
      st.splatted = true;
      splat();
      honk(1.35);
      honk(1.5, 0.18);
      const b = $("bubble");
      b.hidden = false;
      b.classList.remove("out");
      b.style.animation = "none";
      b.offsetWidth;
      b.style.animation = "";
      setTimeout(() => b.classList.add("out"), 700);
      setTimeout(() => {
        b.hidden = true;
      }, 860);
    }
    bannerText = `SIMULATING ${eng.gpu ? "on the GPU (WebGPU)" : "in Goose → WebAssembly"} · t = ${fmt(eng.time() * 1e6, 0)} µs of ${fmt(T_END * 1e6, 0)} µs · goose is now a fluid`;
    telemetry();
    drawForceChart();
    if (done) {
      finishRun();
      verdict();
      honk(0.7, 0.1);
    }
  } else if (st.phase === "wall") {
    const done = solve(WALL_RUN_LENGTHS * must(st.wall, "the wall check").tl, wallSample);
    refreshScene();
    telemetry();
    bannerText = `CONTROL EXPERIMENT: goose vs. rigid wall · ${fmt(eng.time() * 1e6, 0)} µs · colour = pressure`;
    drawWallChart();
    if (done)
      finishRun();
  } else if (st.phase === "done") {
    bannerText = "Done. Drag to orbit. Scroll to zoom. Fire again for another goose.";
  }
  if (bannerText !== undefined && banner.textContent !== bannerText)
    banner.textContent = bannerText;
  if (st.scene)
    render(st.scene, offsetZ, flap);
  captureGif(now);
  codeView(now);
}
function refreshScene() {
  if (st.scene)
    Object.assign(st.scene, readRender());
}
function legend() {
  const showP = !st.costume || st.mode === RIGID_WALL;
  $("legend").innerHTML = (showP ? `bird pressure (log)<div class="ramp" style="background:${cssRamp("pressure")}"></div><span class="ticks"><span>0.1</span><span>6.3</span><span>400 MPa</span></span>` : "") + (st.mode === PANEL ? `panel plastic strain (square-root scale)<div class="ramp" style="background:${cssRamp("metal")}"></div><span class="ticks">${[0, 1, 2, 3].map((k) => `<span>${fmt(EP_MAX * 100 * k * k / 9, 0)}${k === 3 ? " %" : ""}</span>`).join("")}</span>` : "");
}
function startGif() {
  const h = Math.round(GIF_W * H / W) & ~1;
  const cv = document.createElement("canvas");
  cv.width = GIF_W;
  cv.height = h + GIF_CAPTION_PX;
  st.gif = {
    rec: new GifRecorder(GIF_W, h + GIF_CAPTION_PX),
    h,
    cv,
    next: 0,
    simNext: 0,
    speed: st.speed,
    done: false,
    title: st.mode === PANEL ? preset(st.preset).name : "Rigid wall check"
  };
  st.gifHold = false;
  button("gif").disabled = true;
  $("gif").textContent = "Export GIF";
}
function captureGif(now) {
  const g = st.gif;
  if (!g || g.done)
    return;
  let delay;
  if (st.phase === "approach") {
    if (now < g.next)
      return;
    g.next = now + GIF_APPROACH_PERIOD_MS;
    delay = GIF_APPROACH_DELAY_CS;
  } else if (st.phase === "sim" || st.phase === "wall") {
    const tUs = eng.time() * 1e6;
    if (tUs < g.simNext)
      return;
    g.simNext = (Math.floor(tUs / GIF_SIM_US) + 1) * GIF_SIM_US;
    delay = GIF_SIM_DELAY_CS;
  } else if (st.gifHold) {
    g.done = true;
    delay = GIF_LAST_DELAY_CS;
  } else
    return;
  const c = context2d(g.cv);
  const css = getComputedStyle(document.body);
  c.fillStyle = css.getPropertyValue("--stage");
  c.fillRect(0, 0, GIF_W, g.h);
  const scale = Math.min(GIF_W / canvas.width, g.h / canvas.height);
  const w = canvas.width * scale, h = canvas.height * scale;
  c.drawImage(canvas, (GIF_W - w) / 2, (g.h - h) / 2, w, h);
  c.fillStyle = css.getPropertyValue("--bg");
  c.fillRect(0, g.h, GIF_W, GIF_CAPTION_PX);
  c.fillStyle = css.getPropertyValue("--ink");
  c.font = `12px ${css.getPropertyValue("--sans")}`;
  c.textBaseline = "middle";
  const y = g.h + GIF_CAPTION_PX / 2;
  const t = st.phase === "approach" ? "approach (cartoon)" : `t = ${fmt(eng.time() * 1e6, 0)} µs`;
  c.fillText(`${g.title} · ${t}`, 8, y);
  const pressure = !st.costume || st.mode === RIGID_WALL, name = pressure ? "pressure" : "metal";
  const x0 = GIF_W - 220;
  for (let k = 0;k < 90; k++) {
    c.fillStyle = mapColour(name, k / 89);
    c.fillRect(x0 + k, y - 5, 1, 10);
  }
  c.fillStyle = css.getPropertyValue("--ink");
  c.textAlign = "right";
  c.fillText(pressure ? "0.1" : "0", x0 - 4, y);
  c.textAlign = "left";
  c.fillText(pressure ? "400 MPa, log" : `${fmt(EP_MAX * 100, 0)} % plastic, √`, x0 + 94, y);
  c.textAlign = "left";
  g.rec.add(c.getImageData(0, 0, GIF_W, g.h + GIF_CAPTION_PX), delay);
}
async function exportGif() {
  const g = st.gif;
  if (!g || !g.rec.frames.length)
    return;
  const b = button("gif");
  b.disabled = true;
  b.textContent = "Making the GIF…";
  await new Promise((r) => setTimeout(r, 30));
  try {
    const bytes = g.rec.encode();
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/gif" }));
    setTimeout(() => URL.revokeObjectURL(url), 1e4);
    const a = document.createElement("a");
    a.href = url;
    a.download = `goosestrike-${g.title.toLowerCase().replace(/ /g, "-")}-${g.speed.toFixed(2)}mps.gif`;
    a.click();
    if (st.gif === g) {
      b.textContent = `Export GIF (${fmt(bytes.length / 1e6, 1)} MB)`;
      b.disabled = false;
    }
  } catch {
    if (st.gif === g) {
      b.textContent = "GIF export failed. Retry";
      b.disabled = false;
    }
  }
}
var PHASES = [
  { name: "time step", fns: ["stable_dt", "shell_dt"] },
  { name: "SPH bird", fns: ["sph_forces", "bin_particles"] },
  { name: "shells, plasticity, erosion", fns: ["shell_forces", "shell_element", "plane_stress_return"] },
  { name: "contact", fns: ["contact", "bin_elements", "closest_tri"] },
  { name: "integration", fns: ["integrate", "tait"] }
];
var code = { lines: null, step: null, ranges: [], shown: -2, last: 0, scrolled: 0, scrolledK: -1, chips: [] };
async function loadCode() {
  const src = await (await fetch(SOURCE_URL)).text();
  const lines = src.replace(/\n$/, "").split(`
`);
  const fnRange = (name) => {
    let a = lines.findIndex((l) => new RegExp(`^(export )?fn ${name}\\(`).test(l));
    if (a < 0)
      return null;
    let b = a;
    while (b < lines.length && lines[b] !== "}")
      b++;
    while (a > 0 && at(lines, a - 1).startsWith("//"))
      a--;
    return [a, b];
  };
  const [r0, r1] = must(fnRange("run_phase"), "run_phase in strike.goose");
  code.ranges = PHASES.map((ph, k) => {
    const a = lines.findIndex((l, i) => i > r0 && i < r1 && (l.includes(`k == ${k} {`) || k === 4 && l.trim() === "} else {"));
    let b = a + 1;
    while (b < r1 && !at(lines, b).trim().startsWith("}"))
      b++;
    const functions = ph.fns.map(fnRange).filter((r) => r !== null);
    return [[a, b - 1], ...functions];
  });
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const kw = /\b(fn|let|var|if|else|for|in|while|return|export|import|continue)\b/g;
  const html = (i) => {
    const l = at(lines, i), c = l.indexOf("//");
    const body = c < 0 ? l : l.slice(0, c), note = c < 0 ? "" : l.slice(c);
    return `<span class="ln" data-n="${i + 1}">${esc(body).replace(kw, "<b>$1</b>")}${note ? `<i>${esc(note)}</i>` : ""}
</span>`;
  };
  $("step").innerHTML = lines.slice(r0, r1 + 1).map((_, j) => html(r0 + j)).join("");
  $("code").innerHTML = lines.map((_, i) => html(i)).join("");
  code.step = { first: r0, lines: $("step").children };
  code.lines = $("code").children;
  $("phases").innerHTML = PHASES.map((ph, k) => `<li><button class="button" data-k="${k}"><span>${k}</span> ${ph.name}</button></li>`).join("");
  code.chips = [...$("phases").querySelectorAll("button")];
  for (const b of code.chips)
    b.onclick = () => scrollCode(Number(b.dataset["k"]));
}
function scrollCode(k) {
  const ranges = at(code.ranges, k);
  const first = ranges[1]?.[0] ?? at(ranges, 0)[0];
  const el = at(must(code.lines, "the code lines"), first);
  $("code").scrollTo({ top: el.offsetTop - $("code").offsetTop - 8, behavior: reducedMotion.matches ? "instant" : "smooth" });
}
function codeView(now) {
  if (!code.lines || now - code.last < 120)
    return;
  code.last = now;
  const running = st.phase === "sim" || st.phase === "wall";
  const k = running ? st.solverPhase : -1;
  const focused = document.activeElement;
  const sourceFocused = focused === $("code") || focused === $("step") || code.chips.some((c) => c === focused);
  if (!sourceFocused && k >= 0 && k !== code.scrolledK && now - code.scrolled > 1000) {
    scrollCode(k);
    code.scrolled = now;
    code.scrolledK = k;
  }
  const state = k >= 0 ? `Phase ${k} (${at(PHASES, k).name}) ran last.` : running && eng.gpu ? "On the GPU, the kernels of web/strike.wgsl do all five phases of each step. No line is lit." : "The solver is idle.";
  if ($("codeState").textContent !== state)
    $("codeState").textContent = state;
  if (k === code.shown)
    return;
  code.chips.forEach((b, i) => {
    b.dataset["active"] = String(i === k);
  });
  for (const l of $("code").querySelectorAll(".on"))
    l.classList.remove("on");
  for (const l of $("step").querySelectorAll(".on"))
    l.classList.remove("on");
  if (k >= 0) {
    const stepPane = must(code.step, "the step pane");
    for (const [a, b] of at(code.ranges, k))
      for (let i = a;i <= b; i++)
        at(code.lines, i).classList.add("on");
    const [a, b] = at(at(code.ranges, k), 0);
    for (let i = a;i <= b; i++)
      at(stepPane.lines, i - stepPane.first).classList.add("on");
  }
  code.shown = k;
}
function showSpeed() {
  const v = st.speed;
  input("speedNumber").value = v.toFixed(2);
  $("speedEntryStatus").textContent = "";
  $("speedOut").textContent = `${fmt(v, 2)} m/s = ${fmt(v / KNOT_MS, 1)} kt`;
  const on = Math.abs(v - V_C) < 0.005;
  const sc = $("specCheck");
  sc.className = "hint " + (on ? "ok" : "off");
  sc.innerHTML = on ? "On spec: 4 lb at V<sub>C</sub> = 350 KCAS, sea level. 14 CFR 25.571(e)(1)." : `Off spec: the rule asks for V<sub>C</sub> = ${fmt(V_C, 2)} m/s. <button type="button" id="snap" class="snap">Snap to V<sub>C</sub></button>. Science continues anyway.`;
  const snap = document.getElementById("snap");
  if (snap instanceof HTMLButtonElement)
    snap.onclick = () => {
      const focusSpeed = document.activeElement === snap;
      st.speed = V_C;
      input("speed").value = String(V_C);
      showSpeed();
      if (focusSpeed)
        input("speedNumber").focus({ preventScroll: true });
      if (st.phase === "idle")
        preview();
    };
}
function buildPresets() {
  $("presets").innerHTML = PRESETS.map((p, i) => `
    <label class="preset"><input type="radio" name="preset" value="${i}" ${i === st.preset ? "checked" : ""}>
      <b>${p.name}</b> · ${fmt(p.spacing * 1000, 1)} mm bird, ${p.cells} skin cells
      <small>wall check at V<sub>C</sub> only: peak ${sign(p.peak)} %, plateau ${sign(p.plateau)} % · solver about ${eng.gpu ? p.gpuRun : p.run}</small></label>`).join("");
  for (const el of document.querySelectorAll("#presets input")) {
    el.onchange = () => {
      st.preset = Number(el.value);
      preview();
    };
  }
}
function preview() {
  setupRun(PANEL);
  st.phase = "idle";
  $("runStatus").textContent = "";
  $("verdict").hidden = true;
  $("banner").textContent = "Goose loaded. Goose is unaware. Press Fire the goose.";
}
input("speed").value = String(V_C);
for (const attr of ["min", "max", "step"])
  input("speedNumber")[attr] = input("speed")[attr];
input("speedNumber").onchange = () => {
  const entry = input("speedNumber");
  if (!entry.checkValidity()) {
    entry.value = st.speed.toFixed(2);
    $("speedEntryStatus").textContent = `Enter a speed from ${entry.min} to ${entry.max} m/s in steps of ${entry.step} m/s. The control kept the previous speed.`;
    return;
  }
  st.speed = entry.valueAsNumber;
  input("speed").value = String(st.speed);
  showSpeed();
  if (st.phase === "idle")
    preview();
};
input("speed").oninput = () => {
  st.speed = Number(input("speed").value);
  showSpeed();
};
input("speed").onchange = () => {
  if (st.phase === "idle")
    preview();
};
function showSkin() {
  const t = st.tSkin * 1000;
  $("skinOut").textContent = `${fmt(t, 1)} mm = ${fmt(t / 25.4, 3)} in`;
}
input("skin").value = String(T_SKIN * 1000);
input("skin").oninput = () => {
  st.tSkin = Number(input("skin").value) / 1000;
  showSkin();
};
input("skin").onchange = () => {
  if (st.phase === "idle")
    preview();
};
button("fire").onclick = fire;
button("stop").onclick = stopRun;
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || button("stop").disabled)
    return;
  e.preventDefault();
  stopRun();
});
button("wilbeck").onclick = wallCheck;
button("gif").onclick = () => {
  exportGif();
};
function setPressed(b, on) {
  b.setAttribute("aria-pressed", String(on));
  b.dataset["active"] = String(on);
}
button("mute").onclick = () => {
  st.muted = !st.muted;
  try {
    localStorage.setItem(MUTED_KEY, st.muted ? "1" : "0");
  } catch {
    showMute();
  }
  showMute();
};
function showMute() {
  setPressed($("mute"), st.muted);
  $("mute").textContent = st.muted ? "\uD83D\uDD07 Honks off" : "\uD83D\uDD0A Honks on";
}
button("btnCostume").onclick = () => {
  st.costume = !st.costume;
  setPressed($("btnCostume"), st.costume);
  legend();
};
var drag = null;
button("btnPov").onclick = () => {
  drag = null;
  st.pov = !st.pov;
  setPressed($("btnPov"), st.pov);
  showZoom();
};
function resetCamera() {
  drag = null;
  st.yaw = DEFAULT_YAW;
  st.pitch = DEFAULT_PITCH;
  st.dist = DEFAULT_DIST;
  st.pov = false;
  setPressed($("btnPov"), false);
  showZoom();
}
button("btnReset").onclick = resetCamera;
button("btnZoomIn").onclick = () => zoom(1 / 1.2);
button("btnZoomOut").onclick = () => zoom(1.2);
function showZoom() {
  const focused = document.activeElement;
  const zoomIn = button("btnZoomIn"), zoomOut = button("btnZoomOut");
  canvas.style.touchAction = st.pov ? "pan-y pinch-zoom" : "pinch-zoom";
  zoomIn.disabled = st.pov || st.dist <= DIST_MIN;
  zoomOut.disabled = st.pov || st.dist >= DIST_MAX;
  if ((focused === zoomIn || focused === zoomOut) && focused.disabled) {
    canvas.focus({ preventScroll: true });
  }
}
function zoom(factor) {
  if (st.pov)
    return;
  st.dist = Math.max(DIST_MIN, Math.min(DIST_MAX, st.dist * factor));
  showZoom();
}
window.addEventListener("blur", () => {
  drag = null;
});
function rotate(dx, dy) {
  if (st.pov)
    return;
  st.yaw += dx;
  st.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, st.pitch + dy));
}
var TURNS = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] };
canvas.addEventListener("keydown", (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey)
    return;
  if (e.key === "Home") {
    e.preventDefault();
    resetCamera();
    return;
  }
  if (st.pov)
    return;
  if (e.key === "+" || e.key === "-") {
    e.preventDefault();
    zoom(e.key === "+" ? 1 / 1.2 : 1.2);
    return;
  }
  const turn = TURNS[e.key];
  if (!turn)
    return;
  e.preventDefault();
  rotate(turn[0], turn[1]);
});
canvas.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || drag || st.pov)
    return;
  drag = { pointerId: e.pointerId, x: e.clientX, y: e.clientY };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag || drag.pointerId !== e.pointerId || st.pov)
    return;
  rotate((e.clientX - drag.x) * 0.008, (e.clientY - drag.y) * 0.008);
  drag.x = e.clientX;
  drag.y = e.clientY;
});
for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
  canvas.addEventListener(event, (e) => {
    if (drag?.pointerId === e.pointerId)
      drag = null;
  });
}
var WHEEL_LINE_PX = 16;
canvas.addEventListener("wheel", (e) => {
  if (st.pov || e.ctrlKey || e.metaKey)
    return;
  e.preventDefault();
  const unit = e.deltaMode === 1 ? WHEEL_LINE_PX : e.deltaMode === 2 ? canvas.clientHeight : 1;
  zoom(Math.exp(e.deltaY * unit * 0.001));
}, { passive: false });
loadCode().catch(() => {
  $("codeState").textContent = "The source code could not load. The solver can still run. Reload the page to try again.";
});
showSpeed();
showSkin();
showMute();
buildPresets();
resize();
preview();
setBusy(false);
requestAnimationFrame(frame);
