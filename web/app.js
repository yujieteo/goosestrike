// The page: runs the Goose solver (WebAssembly) frame by frame and draws it.
import { loadEngine, hugoniot, stagnation, V_C, BIRD_MASS } from "./engine.mjs";
import { GifRecorder } from "./gif.mjs";

const KT = 1852 / 3600;            // m/s per knot
const T_END = 3.0e-3;              // panel run length, s
const BUDGET_MS = 12;              // solver time per animation frame
const APPROACH_S = 1.6;            // cartoon approach, real seconds
const APPROACH_M = 1.2;            // cartoon approach distance, m
const P_LO = 1e5, P_HI = 4e8;      // pressure colour scale, Pa (log)
const EP_MAX = 0.18;               // plastic strain colour scale
const GIF_W = 480;                 // GIF width, px
const GIF_SIM_US = 25;             // sim time between GIF frames, µs

// Bird lattice and skin cells per preset, and what validate.goose measured
// with them natively (rigid wall check and the panel at V_C).
const PRESETS = [
  { name: "Potato goose", spacing: 0.009, cells: 56, peak: -42.6, plateau: 25.8, defl: 150.8, skin: 0, stiff: 36, run: "10 s" },
  { name: "Good goose", spacing: 0.006, cells: 88, peak: 0.9, plateau: 27.7, defl: 152.1, skin: 8, stiff: 62, run: "35 s" },
  { name: "Very good goose", spacing: 0.0045, cells: 112, peak: 2.7, plateau: 13.1, defl: 152.9, skin: 25, stiff: 152, run: "2 min" },
];

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 1) => x.toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
const sign = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);

const eng = await loadEngine();

const st = {
  phase: "idle",          // idle | approach | sim | done | wall
  mode: 0,                // 0: panel, 1: rigid wall
  preset: 0,
  speed: V_C,
  costume: true,
  pov: false,
  muted: false,
  yaw: 1.0, pitch: 0.25, dist: 1.3,
  t0: 0,
  ke0: 1,
  peakF: 0,
  elems: null, skinNe: 0,
  nodes: null, parts: null, press: null, strain: null,
  init: null,             // particle positions at set-up
  skin: null,             // costume colour per particle
  anchors: null,
  hist: [],
  wall: null,
  frameSim: 0, frameReal: 16.7, msPerStep: 0,
  splatted: false,
  solverPhase: -1,        // the solver phase that ran last (run_phase in strike.goose)
  gif: null,              // the GIF recorder of the last run
  gifHold: false,         // the run ended: the next GIF frame is the last
};

try { st.muted = localStorage.getItem("goosestrike-muted") === "1"; } catch { /* no storage */ }

// ---------------------------------------------------------------------------
// sound: a synthesized honk and splat

let audio = null;
function ac() {
  if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
  return audio;
}

function honk(pitch = 1, when = 0) {
  if (st.muted) return;
  const a = ac(), t = a.currentTime + when;
  const osc = a.createOscillator(), osc2 = a.createOscillator();
  const bp = a.createBiquadFilter(), g = a.createGain();
  osc.type = "sawtooth"; osc2.type = "square";
  osc.frequency.setValueAtTime(410 * pitch, t);
  osc.frequency.exponentialRampToValueAtTime(300 * pitch, t + 0.22);
  osc2.frequency.setValueAtTime(205 * pitch, t);
  osc2.frequency.exponentialRampToValueAtTime(150 * pitch, t + 0.22);
  bp.type = "bandpass"; bp.frequency.value = 1100; bp.Q.value = 2.5;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.35, t + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.26);
  osc.connect(bp); osc2.connect(bp); bp.connect(g); g.connect(a.destination);
  osc.start(t); osc2.start(t); osc.stop(t + 0.3); osc2.stop(t + 0.3);
}

function splat() {
  if (st.muted) return;
  const a = ac(), t = a.currentTime;
  const n = Math.floor(a.sampleRate * 0.5);
  const buf = a.createBuffer(1, n, a.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 2);
  const src = a.createBufferSource(), lp = a.createBiquadFilter(), g = a.createGain();
  src.buffer = buf;
  lp.type = "lowpass";
  lp.frequency.setValueAtTime(4000, t);
  lp.frequency.exponentialRampToValueAtTime(150, t + 0.45);
  g.gain.value = 0.6;
  src.connect(lp); lp.connect(g); g.connect(a.destination);
  src.start(t);
}

// ---------------------------------------------------------------------------
// set-up of a run

function readState() {
  eng.fillRender();
  st.nodes = eng.nodes();
  st.parts = eng.particles();
  st.press = eng.pressure();
  st.strain = eng.strain();
}

function nearest(px, x, y, z) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < px.length / 3; i++) {
    const dx = px[3 * i] - x, dy = px[3 * i + 1] - y, dz = px[3 * i + 2] - z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

// Costume: a cartoon goose painted on the capsule. The capsule flies toward
// -z, so the head is at the low-z end, and +y is the goose's back. All the
// paint uses the one palette GOOSE: soft round shapes with one ink outline.
const GOOSE = {
  belly: [252, 250, 244], back: [234, 228, 216],   // feathers, belly to back
  orange: "#e8913a", ink: "#2b2925", blush: "rgba(226, 120, 110, 0.5)", star: "#e9b640",
};

function paintGoose() {
  const p = st.init, n = p.length / 3;
  const D = eng.birdD(), L = eng.birdL(), r = 0.5 * D;
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < n; i++) { zmin = Math.min(zmin, p[3 * i + 2]); zmax = Math.max(zmax, p[3 * i + 2]); }
  const head = zmin + r;              // centre of the front hemisphere
  const mid = zmin + 0.5 * L;
  // feathers: a smooth blend from the belly to the back, in 32 tones
  st.skin = new Array(n);
  for (let i = 0; i < n; i++) {
    const y = p[3 * i + 1] / r;
    const s = Math.round(32 * smoothstep(-0.6, 1.2, y)) / 32;
    st.skin[i] = rgb(GOOSE.belly.map((c, k) => c + s * (GOOSE.back[k] - c)));
  }
  // a point on the front hemisphere in the direction (x, y, -z)
  const onHead = (x, y, z) => { const d = norm([x, y, -z]); return nearest(p, r * d[0], r * d[1], head + r * d[2]); };
  st.anchors = {
    nose: onHead(0, -0.1, 1),
    eyes: [onHead(0.5, 0.42, 0.75), onHead(-0.5, 0.42, 0.75)],
    cheeks: [onHead(0.72, 0.02, 0.62), onHead(-0.72, 0.02, 0.62)],
    tuft: onHead(0, 0.95, 0.3),
    wings: [nearest(p, 0.92 * r, 0.3 * r, mid + 0.25 * r), nearest(p, -0.92 * r, 0.3 * r, mid + 0.25 * r)],
    feet: [nearest(p, 0.4 * r, -0.92 * r, mid + 0.3 * r), nearest(p, -0.4 * r, -0.92 * r, mid + 0.3 * r)],
    tail: nearest(p, 0, 0, zmax),
  };
}

function skinElements(cells) {
  // build_panel puts the skin first: cells x gs elements, gs rounded to 8
  const gs = Math.floor(cells * 0.760 / 1.0668 / 8 + 0.5) * 8;
  return cells * gs;
}

function setupRun(mode) {
  const pr = PRESETS[st.preset];
  eng.setup(mode, pr.cells, pr.spacing, st.speed);
  st.mode = mode;
  readState();
  st.elems = eng.elements();
  st.skinNe = mode === 0 ? skinElements(pr.cells) : 0;
  st.init = st.parts.slice();
  st.ke0 = eng.birdKE();
  $("energyTitle").textContent = `Goose energy at start: ${fmt(st.ke0 / 1e3, 1)} kJ`;
  st.peakF = 0;
  st.frameSim = 0;
  st.msPerStep = 0;
  st.hist = [];
  st.splatted = false;
  st.solverPhase = -1;
  paintGoose();
  drawForceChart();
  telemetry();
}

// ---------------------------------------------------------------------------
// the camera and the painter's algorithm

const canvas = $("view");
const ctx = canvas.getContext("2d");
let W = 0, H = 0, DPR = 1;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  W = Math.max(1, Math.round(r.width * DPR));
  H = Math.max(1, Math.round(r.height * DPR));
  canvas.width = W; canvas.height = H;
}
new ResizeObserver(resize).observe(canvas);

const cam = { eye: [0, 0, 0], r: [1, 0, 0], u: [0, 1, 0], f: [0, 0, 1], foc: 1 };

function norm(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

function setCamera() {
  let eye, target, upHint;
  if (st.pov) {
    // a passenger at the window, inside the fuselage, looking out
    eye = [0.12, 0.06, -0.7]; target = [0, 0, 0.02]; upHint = [0, 1, 0];
  } else {
    // The fuselage side stands upright: x (along the fuselage) is
    // horizontal, y (around the fuselage) is up, the goose flies along -z.
    target = [0, 0, 0.1];
    const c = Math.cos(st.pitch);
    eye = [target[0] + st.dist * c * Math.sin(st.yaw), target[1] + st.dist * Math.sin(st.pitch), target[2] + st.dist * c * Math.cos(st.yaw)];
    upHint = [0, 1, 0];
  }
  cam.eye = eye;
  cam.f = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
  cam.r = norm(cross(cam.f, upHint));
  cam.u = cross(cam.r, cam.f);
  cam.foc = 0.5 * H / Math.tan((st.pov ? 34 : 20) * Math.PI / 180);
}

// world -> [screen x, screen y, depth]
function proj(x, y, z, out, o) {
  const dx = x - cam.eye[0], dy = y - cam.eye[1], dz = z - cam.eye[2];
  const cz = dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2];
  const cx = dx * cam.r[0] + dy * cam.r[1] + dz * cam.r[2];
  const cy = dx * cam.u[0] + dy * cam.u[1] + dz * cam.u[2];
  out[o] = 0.5 * W + cam.foc * cx / cz;
  out[o + 1] = 0.5 * H - cam.foc * cy / cz;
  out[o + 2] = cz;
}

// buckets from far to near
const NB = 4096;
let sNodes = new Float32Array(0), sParts = new Float32Array(0);
let keys = new Float32Array(0), order = new Int32Array(0), tmp = new Int32Array(0);
const counts = new Int32Array(NB + 1);

function depthSort(n) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { const k = keys[i]; if (k < lo) lo = k; if (k > hi) hi = k; }
  const s = (NB - 1) / Math.max(hi - lo, 1e-9);
  counts.fill(0);
  for (let i = 0; i < n; i++) tmp[i] = NB - 1 - Math.floor((keys[i] - lo) * s);
  for (let i = 0; i < n; i++) counts[tmp[i] + 1]++;
  for (let b = 0; b < NB; b++) counts[b + 1] += counts[b];
  for (let i = 0; i < n; i++) order[counts[tmp[i]]++] = i;
}

function grow(n) {
  if (keys.length < n) { keys = new Float32Array(n); order = new Int32Array(n); tmp = new Int32Array(n); }
}

// Colour maps: smooth ramps, interpolated in OKLab between the stops, in
// 256 levels. The strain map starts at the colour of the bare metal (the
// primer inside), so the colour moves smoothly from zero plastic strain.
function smoothstep(a, b, x) { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
function rgb(c) { return `rgb(${Math.round(c[0])} ${Math.round(c[1])} ${Math.round(c[2])})`; }

function toLab(hex) {
  const l = [1, 3, 5].map((k) => { const v = parseInt(hex.slice(k, k + 2), 16) / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; });
  const m = [0.4122214708 * l[0] + 0.5363325363 * l[1] + 0.0514459929 * l[2],
    0.2119034982 * l[0] + 0.6806995451 * l[1] + 0.1073969566 * l[2],
    0.0883024619 * l[0] + 0.2817188376 * l[1] + 0.6299787005 * l[2]].map(Math.cbrt);
  return [0.2104542553 * m[0] + 0.793617785 * m[1] - 0.0040720468 * m[2],
    1.9779984951 * m[0] - 2.428592205 * m[1] + 0.4505937099 * m[2],
    0.0259040371 * m[0] + 0.7827717662 * m[1] - 0.808675766 * m[2]];
}

function fromLab(c) {
  const m = [c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2],
    c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2],
    c[0] - 0.0894841775 * c[1] - 1.291485548 * c[2]].map((v) => v * v * v);
  const l = [4.0767416621 * m[0] - 3.3077115913 * m[1] + 0.2309699292 * m[2],
    -1.2684380046 * m[0] + 2.6097574011 * m[1] - 0.3413193965 * m[2],
    -0.0041960863 * m[0] - 0.7034186147 * m[1] + 1.707614701 * m[2]];
  return l.map((v) => 255 * Math.max(0, Math.min(1, v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055)));
}

function ramp(stops) {
  const lab = stops.map(toLab), out = [];
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (lab.length - 1), k = Math.min(lab.length - 2, Math.floor(x)), f = x - k;
    out.push(fromLab(lab[k].map((v, j) => v + f * (lab[k + 1][j] - v))));
  }
  return out;
}

const HOT = ["#e2b54f", "#c4613f", "#6e2a22"];             // amber, terracotta, oxblood
const MAPS = {
  metal: ramp(["#c7c3ba", ...HOT]),                        // the skin outside
  primer: ramp(["#9da27f", ...HOT]),                       // inside, stringers and frames
  pressure: ramp(["#243552", "#3f6f86", "#94b49c", "#e8c766", "#c4613f"]),
};
const SHADES = 64;
const mapCache = new Map();

// colour of level s (0 to 1) of a map, darkened by shade (0 to 1)
function mapColour(name, s, shade = 1) {
  const i = Math.round(255 * Math.max(0, Math.min(1, s))), j = Math.round(SHADES * shade);
  const key = name + i * 32 + j;
  let c = mapCache.get(key);
  if (!c) { c = rgb(MAPS[name][i].map((v) => v * j / SHADES)); mapCache.set(key, c); }
  return c;
}

const logP = (p) => (Math.log10(Math.max(p, P_LO)) - Math.log10(P_LO)) / (Math.log10(P_HI) - Math.log10(P_LO));
function pressureColour(p) { return mapColour("pressure", logP(p)); }
// square-root scale, so that the first yield shows: 2 % and 8 % are at 1/3 and 2/3
function strainColour(ep, shade, inside) { return mapColour(inside ? "primer" : "metal", Math.sqrt(ep / EP_MAX), shade); }

// a CSS gradient of a map, for the legend
function cssRamp(name) {
  return `linear-gradient(90deg, ${Array.from({ length: 17 }, (_, k) => mapColour(name, k / 16)).join(",")})`;
}

function render(offsetZ, flap) {
  setCamera();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const nn = st.nodes ? st.nodes.length / 3 : 0;
  const np = st.parts ? st.parts.length / 3 : 0;
  const ne = st.elems ? st.elems.length / 4 : 0;
  if (sNodes.length < 3 * nn) sNodes = new Float32Array(3 * nn);
  if (sParts.length < 3 * np) sParts = new Float32Array(3 * np);
  for (let i = 0; i < nn; i++) proj(st.nodes[3 * i], st.nodes[3 * i + 1], st.nodes[3 * i + 2], sNodes, 3 * i);
  for (let i = 0; i < np; i++) proj(st.parts[3 * i], st.parts[3 * i + 1], st.parts[3 * i + 2] + offsetZ, sParts, 3 * i);

  if (st.mode === 1) drawWall();

  // decorations follow the particles they are painted on
  const costume = st.costume && st.mode === 0;
  const decor = costume ? decorations(offsetZ, flap) : [];
  const n = ne + np + decor.length;
  grow(n);
  for (let e = 0; e < ne; e++) {
    const a = st.elems[4 * e], b = st.elems[4 * e + 1], c = st.elems[4 * e + 2], d = st.elems[4 * e + 3];
    keys[e] = st.strain[e] < 0 ? -1 : 0.25 * (sNodes[3 * a + 2] + sNodes[3 * b + 2] + sNodes[3 * c + 2] + sNodes[3 * d + 2]);
  }
  for (let i = 0; i < np; i++) keys[ne + i] = sParts[3 * i + 2];
  for (let k = 0; k < decor.length; k++) keys[ne + np + k] = decor[k].depth;
  depthSort(n);

  const pr = eng.spacing() * 0.62 * cam.foc;
  for (let k = 0; k < n; k++) {
    const id = order[k];
    if (id < ne) drawShell(id);
    else if (id < ne + np) {
      const i = id - ne, z = sParts[3 * i + 2];
      if (z <= 0.01) continue;
      ctx.fillStyle = costume ? st.skin[i] : pressureColour(st.press[i]);
      ctx.beginPath();
      ctx.arc(sParts[3 * i], sParts[3 * i + 1], Math.max(0.8, pr / z), 0, 2 * Math.PI);
      ctx.fill();
    } else decor[id - ne - np].draw();
  }
  if (costume && contactStarted()) drawStars(offsetZ);
}

function drawShell(e) {
  if (st.strain[e] < 0) return;             // eroded
  const el = st.elems, P = st.nodes;
  const a = el[4 * e], b = el[4 * e + 1], c = el[4 * e + 2], d = el[4 * e + 3];
  if (sNodes[3 * a + 2] <= 0.01 || sNodes[3 * c + 2] <= 0.01) return;
  // normal from the diagonals, for shading and for the side in view
  const r1 = [P[3 * c] - P[3 * a], P[3 * c + 1] - P[3 * a + 1], P[3 * c + 2] - P[3 * a + 2]];
  const r2 = [P[3 * d] - P[3 * b], P[3 * d + 1] - P[3 * b + 1], P[3 * d + 2] - P[3 * b + 2]];
  const nrm = norm(cross(r1, r2));
  const toEye = norm([cam.eye[0] - P[3 * a], cam.eye[1] - P[3 * a + 1], cam.eye[2] - P[3 * a + 2]]);
  const facing = nrm[0] * toEye[0] + nrm[1] * toEye[1] + nrm[2] * toEye[2];
  const shade = 0.72 + 0.28 * Math.abs(facing);
  const inside = e >= st.skinNe || facing < 0;
  ctx.fillStyle = strainColour(st.strain[e], shade, inside);
  ctx.strokeStyle = ctx.fillStyle;
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  ctx.moveTo(sNodes[3 * a], sNodes[3 * a + 1]);
  ctx.lineTo(sNodes[3 * b], sNodes[3 * b + 1]);
  ctx.lineTo(sNodes[3 * c], sNodes[3 * c + 1]);
  ctx.lineTo(sNodes[3 * d], sNodes[3 * d + 1]);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawWall() {
  const pts = [];
  const o = [0, 0, 0];
  ctx.strokeStyle = "#7a8794";
  ctx.lineWidth = 1;
  for (let k = -5; k <= 5; k++) {
    for (const [x0, y0, x1, y1] of [[k * 0.05, -0.25, k * 0.05, 0.25], [-0.25, k * 0.05, 0.25, k * 0.05]]) {
      proj(x0, y0, 0, o, 0); pts[0] = o[0]; pts[1] = o[1];
      proj(x1, y1, 0, o, 0);
      ctx.beginPath(); ctx.moveTo(pts[0], pts[1]); ctx.lineTo(o[0], o[1]); ctx.stroke();
    }
  }
}

function contactStarted() { return eng.tContact() >= 0 && st.phase !== "approach"; }

// screen position and depth of particle i
function sp(i) { return [sParts[3 * i], sParts[3 * i + 1], sParts[3 * i + 2]]; }

// a closed smooth curve with the points as control points: the one shape
// of all the costume parts
function blob(pts) {
  const m = (p, q) => [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
  const n = pts.length;
  let s = m(pts[n - 1], pts[0]);
  ctx.beginPath(); ctx.moveTo(s[0], s[1]);
  for (let i = 0; i < n; i++) { s = m(pts[i], pts[(i + 1) % n]); ctx.quadraticCurveTo(pts[i][0], pts[i][1], s[0], s[1]); }
  ctx.closePath();
}

// The parts are drawn in screen space, in a frame of the goose: u along the
// body toward the beak, v toward the back. Thus they keep their shape from
// all sides. Sizes are in bird diameters.
function decorations(offsetZ, flap) {
  const A = st.anchors, D = eng.birdD(), P = st.parts;
  const wp = (i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2] + offsetZ];
  const pt = (w) => { const o = [0, 0, 0]; proj(w[0], w[1], w[2], o, 0); return o; };
  const nose = wp(A.nose), tail = wp(A.tail);
  const n = pt(nose), t = pt(tail);
  let ux = n[0] - t[0], uy = n[1] - t[1];
  const ul = Math.hypot(ux, uy);
  if (ul < 1e-6) { ux = 1; uy = 0; } else { ux /= ul; uy /= ul; }
  // v: perpendicular to u, on the side of the goose's back (+y)
  const top = pt([nose[0], nose[1] + 0.1, nose[2]]);
  const side = (top[0] - n[0]) * -uy + (top[1] - n[1]) * ux > 0 ? 1 : -1;
  const vx = -uy * side, vy = ux * side;
  // a point at (u, v) bird diameters from the screen point c
  const at = (c, k, u, v) => [c[0] + k * (u * ux + v * vx), c[1] + k * (u * uy + v * vy)];
  const dizzy = contactStarted();
  const list = [];
  const scale = (z) => D * cam.foc / z;
  // the one outline rule: ink, 1.2 % of the bird diameter, at least 1 px
  const outline = (z) => { ctx.strokeStyle = GOOSE.ink; ctx.lineWidth = Math.max(DPR, 0.012 * scale(z)); ctx.lineJoin = "round"; ctx.stroke(); };
  const part = (c, fill, pts, dz) => list.push({ depth: c[2] + dz, draw() { ctx.fillStyle = fill; blob(pts); ctx.fill(); outline(c[2]); } });
  const feather = rgb(GOOSE.belly);

  // wings: soft feathers on the sides, flapping in the approach, limp after it
  for (const i of A.wings) {
    const c = sp(i), k = scale(c[2]);
    const up = dizzy ? -0.05 + 0.03 * Math.sin(performance.now() / 40) : 0.25 + 0.2 * flap;
    part(c, feather, [at(c, k, 0.18, 0.02), at(c, k, -0.2, 0.12 + up), at(c, k, -0.5, 0.1 + 0.8 * up), at(c, k, -0.3, -0.12)], -0.03);
  }
  // a tuft of feathers on the head
  {
    const c = sp(A.tuft), k = scale(c[2]);
    part(c, feather, [at(c, k, 0.05, -0.02), at(c, k, 0.04, 0.14), at(c, k, -0.12, 0.24), at(c, k, -0.07, 0.04)], -0.02);
  }
  // beak: a short round bean
  {
    const k = scale(n[2]);
    part(n, GOOSE.orange, [at(n, k, -0.04, 0.09), at(n, k, 0.26, 0.05), at(n, k, 0.26, -0.03), at(n, k, -0.04, -0.08)], -0.03);
  }
  // orange webbed feet, tucked under the belly
  for (const i of A.feet) {
    const c = sp(i), k = scale(c[2]);
    part(c, GOOSE.orange, [at(c, k, 0, 0), at(c, k, -0.06, -0.2), at(c, k, 0.06, -0.26), at(c, k, 0.22, -0.2), at(c, k, 0.1, -0.08)], -0.01);
  }
  // blush
  for (const i of A.cheeks) {
    const c = sp(i);
    const rr = 0.07 * scale(c[2]);
    list.push({ depth: c[2] - 0.008, draw() {
      ctx.fillStyle = GOOSE.blush;
      ctx.beginPath(); ctx.ellipse(c[0], c[1], rr * 1.3, rr * 0.85, Math.atan2(uy, ux), 0, 2 * Math.PI); ctx.fill();
    } });
  }
  // eyes: big and round before contact, spirals after
  for (const i of A.eyes) {
    const c = sp(i);
    const rr = 0.11 * scale(c[2]);
    list.push({ depth: c[2] - 0.01, draw() {
      ctx.fillStyle = GOOSE.ink;
      ctx.beginPath(); ctx.arc(c[0], c[1], rr, 0, 2 * Math.PI); ctx.fill();
      ctx.fillStyle = "#fff";
      if (dizzy) {
        ctx.beginPath(); ctx.arc(c[0], c[1], rr * 0.82, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = GOOSE.ink; ctx.lineWidth = Math.max(DPR, 0.16 * rr);
        ctx.beginPath();
        const turn = performance.now() / 120;
        for (let k = 0; k <= 40; k++) {
          const a = turn + k * 0.45, q = rr * 0.7 * k / 40;
          const x = c[0] + q * Math.cos(a), y = c[1] + q * Math.sin(a);
          if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(c[0] - rr * 0.32, c[1] - rr * 0.32, rr * 0.34, 0, 2 * Math.PI); ctx.fill();
        ctx.beginPath(); ctx.arc(c[0] + rr * 0.3, c[1] + rr * 0.3, rr * 0.14, 0, 2 * Math.PI); ctx.fill();
      }
    } });
  }
  return list;
}

function drawStars(offsetZ) {
  const P = st.parts, np = P.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < np; i++) { cx += P[3 * i]; cy += P[3 * i + 1]; cz += P[3 * i + 2]; }
  cx /= np; cy /= np; cz /= np;
  const D = eng.birdD(), t = performance.now() / 500, o = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const a = t + k * 2.094;
    proj(cx + 0.6 * D * Math.cos(a), cy + 0.9 * D, cz + offsetZ + 0.6 * D * Math.sin(a), o, 0);
    if (o[2] <= 0.01) continue;
    const r = 0.1 * D * cam.foc / o[2];
    const q = [];
    for (let j = 0; j < 10; j++) {
      const ang = -Math.PI / 2 + j * Math.PI / 5, rr = j % 2 ? r * 0.5 : r;
      q.push([o[0] + rr * Math.cos(ang), o[1] + rr * Math.sin(ang)]);
    }
    ctx.fillStyle = GOOSE.star;
    ctx.beginPath(); q.forEach((p, j) => (j ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = GOOSE.ink; ctx.lineWidth = Math.max(DPR, 0.012 * D * cam.foc / o[2]); ctx.lineJoin = "round"; ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// charts

const chartData = new WeakMap();
const chartResize = new ResizeObserver((entries) => {
  for (const { target } of entries) {
    const data = chartData.get(target);
    if (data) chart(target, data.series, data.opt);
  }
});

function chart(cv, series, opt) {
  if (!chartData.has(cv)) chartResize.observe(cv);
  chartData.set(cv, { series, opt });
  const dpr = DPR, r = cv.getBoundingClientRect();
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const c = cv.getContext("2d");
  const css = getComputedStyle(document.body);
  const ink = css.getPropertyValue("--ink"), muted = css.getPropertyValue("--muted"), line = css.getPropertyValue("--line");
  c.scale(dpr, dpr);
  const w = r.width, h = r.height, L = 46, R = 8, T = 8, B = 30;
  const fy = opt.log ? (y) => Math.log10(Math.max(y, opt.ymin)) : (y) => y;
  const y0 = fy(opt.ymin), y1 = fy(opt.ymax);
  const X = (x) => L + (w - L - R) * (x - opt.xmin) / (opt.xmax - opt.xmin);
  const Y = (y) => T + (h - T - B) * (1 - (fy(y) - y0) / (y1 - y0));
  c.font = `11px ${css.getPropertyValue("--mono")}`;
  c.strokeStyle = line; c.fillStyle = muted; c.lineWidth = 1;
  for (const yt of opt.yticks) {
    c.beginPath(); c.moveTo(L, Y(yt)); c.lineTo(w - R, Y(yt)); c.stroke();
    c.textAlign = "right"; c.fillText(String(yt), L - 4, Y(yt) + 4);
  }
  for (const xt of opt.xticks) { c.textAlign = "center"; c.fillText(String(xt), X(xt), h - B + 14); }
  c.textAlign = "left"; c.fillText(opt.ylabel, L + 4, T + 10);
  c.textAlign = "right"; c.fillText(opt.xlabel, w - R, h - 4);
  for (const hl of opt.hlines || []) {
    c.strokeStyle = hl.color; c.setLineDash([5, 4]);
    c.beginPath(); c.moveTo(L, Y(hl.y)); c.lineTo(w - R, Y(hl.y)); c.stroke();
    c.setLineDash([]); c.fillStyle = hl.color; c.textAlign = "right"; c.fillText(hl.label, w - R - 2, Y(hl.y) - 4);
  }
  for (const s of series) {
    c.strokeStyle = s.color || ink; c.lineWidth = 1.6; c.beginPath();
    s.xs.forEach((x, i) => { const px = X(Math.min(Math.max(x, opt.xmin), opt.xmax)), py = Y(Math.min(Math.max(s.ys[i], opt.ymin), opt.ymax)); i ? c.lineTo(px, py) : c.moveTo(px, py); });
    c.stroke();
  }
}

function ticks(max, n) {
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

function drawForceChart() {
  const xs = st.hist.map((h) => h.t * 1e6), ys = st.hist.map((h) => h.f / 1e3);
  const ymax = Math.max(50, ...ys) * 1.1;
  chart($("forceChart"), [{ xs, ys, color: "#c4613f" }], {
    xmin: 0, xmax: T_END * 1e6, ymin: 0, ymax, xticks: ticks(T_END * 1e6, 6), yticks: ticks(ymax, 4),
    xlabel: "time (µs)", ylabel: "contact force (kN)",
  });
}

// ---------------------------------------------------------------------------
// the panel run

function fire() {
  if (st.phase === "approach" || st.phase === "sim" || st.phase === "wall") return;
  setupRun(0);
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
  $("fire").disabled = b;
  $("wilbeck").disabled = b;
  $("stop").disabled = !b;
  $("gif").disabled = b || !st.gif;
  $("fire").textContent = b ? "GOOSE IN FLIGHT" : "FIRE THE GOOSE";
  for (const el of document.querySelectorAll("#presets input, #speed, #speedNumber, #snap")) el.disabled = b;
}

function stopRun() {
  if (st.phase !== "approach" && st.phase !== "sim" && st.phase !== "wall") return;
  if (st.mode === 1) {
    st.wall = null;
    const chart = $("wallChart");
    chartResize.unobserve(chart);
    chartData.delete(chart);
    chart.getContext("2d").clearRect(0, 0, chart.width, chart.height);
    $("wilbeckOut").textContent = "Wall check stopped. No result.";
  }
  st.gif = null;
  st.gifHold = false;
  $("bubble").hidden = true;
  preview();
  setBusy(false);
  $("banner").textContent = "Run stopped. Goose reset. Select a speed or resolution, then press FIRE.";
}

// Runs the solver one phase at a time for BUDGET_MS. A frame can end in the
// middle of a step: the code view shows the phase that ran last.
function stepSolver(tEnd, perStep) {
  const t0 = performance.now(), s0 = eng.time();
  let steps = 0, t = t0;
  while (t - t0 < BUDGET_MS) {
    const k = eng.stepPhase(tEnd);
    if (k < 0) break;
    t = performance.now();
    st.solverPhase = k;
    if (k === 4) { steps++; if (perStep) perStep(); }
  }
  const ms = t - t0;
  st.msPerStep = steps ? ms / steps : 0;
  st.frameSim = eng.time() - s0;
}

function telemetry() {
  const tc = eng.tContact();
  const rows = [
    ["sim time", `${fmt(eng.time() * 1e6, 1)} µs`],
    ["since contact", tc >= 0 ? `${fmt((eng.time() - tc) * 1e6, 1)} µs` : "goose airborne"],
    ["time step dt", `${fmt(eng.dt() * 1e9, 1)} ns`],
    ["steps", fmt(eng.steps(), 0)],
    ["slow motion", st.frameSim > 0 ? `1 : ${fmt(st.frameReal / 1000 / st.frameSim, 0)}` : "—"],
    ["solver", `${fmt(st.msPerStep, 2)} ms/step`],
    ["contact force", `${fmt(eng.contactForce() / 1e3, 1)} kN`],
    ["peak force", `${fmt(st.peakF / 1e3, 1)} kN`],
    ["max deflection", `${fmt(eng.maxDeflection() * 1e3, 1)} mm`],
    ["max plastic strain", `${fmt(eng.maxPlasticStrain() * 100, 1)} %`],
    ["fails at (skin / stiffeners)", `${fmt(eng.failStrain(0) * 100, 1)} / ${fmt(eng.failStrain(1) * 100, 1)} %`],
    ["eroded skin shells", fmt(eng.failedSkin(), 0)],
    ["eroded stiffener shells", fmt(eng.failed() - eng.failedSkin(), 0)],
    ["bird", `${fmt(eng.np(), 0)} particles, ${fmt(BIRD_MASS, 3)} kg`],
    ["panel", `${fmt(eng.nn(), 0)} nodes, ${fmt(eng.ne(), 0)} shells`],
  ];
  $("tele").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("honkBar").style.width = `${Math.min(100, eng.contactForce() / 3e5 * 100)}%`;

  const parts = [
    ["bird kinetic", eng.birdKE(), "#3f6f86"],
    ["bird squish (internal)", eng.birdWork(), "#94b49c"],
    ["panel kinetic", eng.panelKE(), "#e2b54f"],
    ["panel strain + plastic", eng.panelWork(), "#c4613f"],
  ];
  const sum = parts.reduce((s, p) => s + p[1], 0);
  $("ebar").innerHTML = parts.map(([, v, c]) => `<div style="width:${Math.max(0, v / st.ke0 * 100)}%;background:${c}"></div>`).join("");
  $("ekey").innerHTML = parts.map(([k, v, c]) => `<li><span><i style="background:${c}"></i>${k}</span><span>${fmt(v / 1e3, 2)} kJ</span></li>`).join("")
    + `<li><span>balance error</span><span>${sign((sum / st.ke0 - 1) * 100, 2)} %</span></li>`;
}

function verdict() {
  const v = $("verdict"), pr = PRESETS[st.preset];
  const skin = eng.failedSkin(), stiff = eng.failed() - skin;
  const breached = skin > 0;
  v.className = "verdict " + (breached ? "breached" : "held");
  v.hidden = false;
  v.innerHTML = `
    <div class="stamp">${breached ? "SKIN BREACHED" : "SKIN HELD"}</div>
    <h2>${breached ? "The goose tore the skin." : "The skin held. The goose did not."}</h2>
    <p>This run: ${pr.name} at ${fmt(st.speed, 2)} m/s (${fmt(st.speed / KT, 1)} kt), ${fmt(pr.spacing * 1e3, 1)} mm bird lattice.</p>
    <p>Eroded shell elements: ${fmt(skin, 0)} in the 2024-T3 skin, ${fmt(stiff, 0)} in the 7075-T6 stringers and frames.
      An element erodes when its plastic strain reaches the failure strain of its material. Maximum deflection ${fmt(eng.maxDeflection() * 1e3, 1)} mm.
      Peak contact force ${fmt(st.peakF / 1e3, 1)} kN. Maximum plastic strain ${fmt(eng.maxPlasticStrain() * 100, 1)} %.
      The panel took ${fmt(eng.panelWork() / 1e3, 1)} kJ of the goose's ${fmt(st.ke0 / 1e3, 1)} kJ.</p>
    <p>Does this fail 25.571(e)(1)? Not by itself. The rule accepts damage: the aeroplane must complete the flight with it.
      That needs a residual strength analysis of the damaged structure, which this page does not do.
      This is not a certification analysis.</p>
    <p class="hint">Measured at V<sub>C</sub> only: the deflection converges with the mesh (${PRESETS.map((p) => fmt(p.defl, 1)).join(", ")} mm at ${PRESETS.map((p) => fmt(p.spacing * 1e3, 1)).join(", ")} mm).
      The skin erosion does not (${PRESETS.map((p) => p.skin).join(", ")} skin elements): erosion depends on the element size, so a tear prediction needs a mesh-calibrated failure model.</p>`;
}

// ---------------------------------------------------------------------------
// the rigid wall check

function wallCheck() {
  if (st.phase === "approach" || st.phase === "sim" || st.phase === "wall") return;
  setupRun(1);
  const v = st.speed;
  const [ph, us] = hugoniot(v);
  st.wall = { ph, us, ps: stagnation(v), tl: eng.birdL() / v, area: Math.PI * eng.sensorR() ** 2, xs: [], ys: [], peak: 0, plat: 0, platN: 0 };
  st.phase = "wall";
  startGif();
  $("verdict").hidden = true;
  $("wallChart").scrollIntoView({ behavior: "smooth", block: "center" });
  setBusy(true);
  honk(0.8);
}

function wallSample() {
  st.peakF = Math.max(st.peakF, eng.contactForce());
  const w = st.wall, tc = eng.tContact();
  if (tc < 0) return;
  const t = eng.time() - tc, p = eng.centerForce() / w.area;
  w.xs.push(t * 1e6); w.ys.push(p / 1e6);
  if (p > w.peak) w.peak = p;
  if (t > 0.25 * w.tl && t < 0.75 * w.tl) { w.plat += p; w.platN++; }
  if (!st.splatted) { st.splatted = true; splat(); }
}

function drawWallChart() {
  const w = st.wall;
  chart($("wallChart"), [{ xs: w.xs, ys: w.ys, color: "#3f6f86" }], {
    xmin: 0, xmax: 1.5 * w.tl * 1e6, ymin: 1, ymax: 1000, log: true,
    xticks: ticks(1.5 * w.tl * 1e6, 6), yticks: [1, 10, 100, 1000],
    xlabel: "time since contact (µs)", ylabel: "centre pressure (MPa, log)",
    hlines: [{ y: w.ph / 1e6, label: `P_H ${fmt(w.ph / 1e6, 1)} MPa`, color: "#a8432a" },
             { y: w.ps / 1e6, label: `P_s ${fmt(w.ps / 1e6, 2)} MPa`, color: "#4f7a4a" }],
  });
  const pm = w.platN ? w.plat / w.platN : 0;
  $("wilbeckOut").textContent =
    `v = ${fmt(st.speed, 2)} m/s, bird lattice ${fmt(PRESETS[st.preset].spacing * 1e3, 1)} mm, ${eng.np()} particles, sensor r = ${fmt(eng.sensorR() * 1e3, 2)} mm\n` +
    `Hugoniot  P_H = ${fmt(w.ph / 1e6, 1)} MPa (U_s = ${fmt(w.us, 0)} m/s)   peak    ${fmt(w.peak / 1e6, 1)} MPa   error ${sign((w.peak - w.ph) / w.ph * 100)} %\n` +
    `Stagnation P_s = ${fmt(w.ps / 1e6, 2)} MPa                plateau ${fmt(pm / 1e6, 2)} MPa   error ${w.platN ? sign((pm - w.ps) / w.ps * 100) : "—"} %\n` +
    `impulse / (m v) = ${fmt(eng.impulse() / (BIRD_MASS * st.speed), 4)}   (plateau: mean over 0.25 to 0.75 of L/v = ${fmt(w.tl * 1e6, 0)} µs)`;
}

// ---------------------------------------------------------------------------
// the frame loop

let lastFrame = performance.now();
function frame(now) {
  st.frameReal = now - lastFrame;
  lastFrame = now;
  let offsetZ = 0, flap = 0;
  const banner = $("banner");
  if (st.phase === "approach") {
    const s = Math.min(1, (now - st.t0) / 1000 / APPROACH_S);
    offsetZ = APPROACH_M * (1 - s);
    flap = Math.sin(now / 70);
    banner.textContent = `APPROACH: cartoon, not simulated (the real ${fmt(APPROACH_M, 1)} m takes ${fmt(APPROACH_M / st.speed * 1e3, 1)} ms at ${fmt(st.speed, 1)} m/s)`;
    if (s >= 1) st.phase = "sim";
  } else if (st.phase === "sim") {
    stepSolver(T_END, () => {
      const f = eng.contactForce();
      st.peakF = Math.max(st.peakF, f);
      st.hist.push({ t: eng.time(), f });
    });
    readState();
    if (eng.tContact() >= 0 && !st.splatted) {
      st.splatted = true;
      splat(); honk(1.35); honk(1.5, 0.18);
      const b = $("bubble"); b.hidden = false; b.style.animation = "none"; void b.offsetWidth; b.style.animation = "";
      setTimeout(() => { b.hidden = true; }, 900);
    }
    banner.textContent = `SIMULATING in Goose → WebAssembly · t = ${fmt(eng.time() * 1e6, 0)} µs of ${fmt(T_END * 1e6, 0)} µs · goose is now a fluid`;
    telemetry();
    drawForceChart();
    if (eng.time() >= T_END) {
      st.phase = "done"; st.solverPhase = -1; st.gifHold = true; setBusy(false); verdict(); honk(0.7, 0.1);
    }
  } else if (st.phase === "wall") {
    stepSolver(1.5 * st.wall.tl, wallSample);
    readState();
    telemetry();
    banner.textContent = `CONTROL EXPERIMENT: goose vs. rigid wall · ${fmt(eng.time() * 1e6, 0)} µs · colour = pressure`;
    drawWallChart();
    if (eng.time() >= 1.5 * st.wall.tl) { st.phase = "done"; st.solverPhase = -1; st.gifHold = true; setBusy(false); }
  } else if (st.phase === "done") {
    banner.textContent = "Done. Drag to orbit. Scroll to zoom. Fire again for another goose.";
  }
  legend();
  if (st.nodes) render(offsetZ, flap);
  captureGif(now);
  codeView(now);
  requestAnimationFrame(frame);
}

function legend() {
  const showP = !st.costume || st.mode === 1;
  $("legend").innerHTML =
    (showP ? `bird pressure (log)<div class="ramp" style="background:${cssRamp("pressure")}"></div><span class="ticks"><span>0.1</span><span>6.3</span><span>400 MPa</span></span>` : "") +
    (st.mode === 0 ? `panel plastic strain (square-root scale)<div class="ramp" style="background:${cssRamp("metal")}"></div><span class="ticks">${[0, 1, 2, 3].map((k) => `<span>${fmt(EP_MAX * 100 * k * k / 9, 0)}${k === 3 ? " %" : ""}</span>`).join("")}</span>` : "");
}

// ---------------------------------------------------------------------------
// the GIF export: one frame each GIF_SIM_US of simulated time, and about
// ten per second of the cartoon approach

function startGif() {
  const h = Math.round(GIF_W * H / W) & ~1;
  st.gif = { rec: new GifRecorder(GIF_W, h + 28), h, cv: document.createElement("canvas"), next: 0, simNext: 0, speed: st.speed, title: st.mode === 0 ? PRESETS[st.preset].name : "Rigid wall check" };
  st.gif.cv.width = GIF_W; st.gif.cv.height = h + 28;
  st.gifHold = false;
  $("gif").disabled = true;
  $("gif").textContent = "Export GIF";
}

function captureGif(now) {
  const g = st.gif;
  if (!g || g.done) return;
  let delay;
  if (st.phase === "approach") {
    if (now < g.next) return;
    g.next = now + 100; delay = 10;
  } else if (st.phase === "sim" || st.phase === "wall") {
    const tUs = eng.time() * 1e6;
    if (tUs < g.simNext) return;
    g.simNext = (Math.floor(tUs / GIF_SIM_US) + 1) * GIF_SIM_US; delay = 4;
  } else if (st.gifHold) {
    g.done = true; delay = 300;     // the last frame stays 3 s
  } else return;
  const c = g.cv.getContext("2d");
  const css = getComputedStyle(document.body);
  c.fillStyle = css.getPropertyValue("--stage");
  c.fillRect(0, 0, GIF_W, g.h);
  c.drawImage(canvas, 0, 0, GIF_W, g.h);
  // the caption: what it is, the time, and the colour bar with its numbers
  c.fillStyle = css.getPropertyValue("--bg");
  c.fillRect(0, g.h, GIF_W, 28);
  c.fillStyle = css.getPropertyValue("--ink");
  c.font = `12px ${css.getPropertyValue("--sans")}`;
  c.textBaseline = "middle";
  const y = g.h + 14;
  const t = st.phase === "approach" ? "approach (cartoon)" : `t = ${fmt(eng.time() * 1e6, 0)} µs`;
  c.fillText(`${g.title} · ${t}`, 8, y);
  const pressure = !st.costume || st.mode === 1, name = pressure ? "pressure" : "metal";
  const x0 = GIF_W - 220;
  for (let k = 0; k < 90; k++) { c.fillStyle = mapColour(name, k / 89); c.fillRect(x0 + k, y - 5, 1, 10); }
  c.fillStyle = css.getPropertyValue("--ink");
  c.textAlign = "right"; c.fillText(pressure ? "0.1" : "0", x0 - 4, y);
  c.textAlign = "left"; c.fillText(pressure ? "400 MPa, log" : `${fmt(EP_MAX * 100, 0)} % plastic, √`, x0 + 94, y);
  c.textAlign = "left";
  g.rec.add(c.getImageData(0, 0, GIF_W, g.h + 28), delay);
}

async function exportGif() {
  const g = st.gif;
  if (!g || !g.rec.frames.length) return;
  const b = $("gif");
  b.disabled = true; b.textContent = "Making the GIF…";
  await new Promise((r) => setTimeout(r, 30));      // let the button text show
  const bytes = g.rec.encode();
  const url = URL.createObjectURL(new Blob([bytes], { type: "image/gif" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `goosestrike-${g.title.toLowerCase().replace(/ /g, "-")}-${fmt(g.speed, 0)}ms.gif`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  b.textContent = `Export GIF (${fmt(bytes.length / 1e6, 1)} MB)`;
  b.disabled = false;
}

// ---------------------------------------------------------------------------
// the code view: sim/strike.goose, with the phase of the step that ran last.
// The phase comes from the solver (step_phase), not from a clock.

const PHASES = [
  { name: "time step", fns: ["stable_dt", "shell_dt"] },
  { name: "SPH bird", fns: ["sph_forces", "bin_particles"] },
  { name: "shells, plasticity, erosion", fns: ["shell_forces", "shell_element", "plane_stress_return"] },
  { name: "contact", fns: ["contact", "bin_elements", "closest_tri"] },
  { name: "integration", fns: ["integrate", "tait"] },
];
const code = { lines: null, step: null, ranges: [], shown: -2, last: 0, scrolled: 0, scrolledK: -1, chips: [] };

async function loadCode() {
  const src = await (await fetch("strike.goose")).text();
  const lines = src.replace(/\n$/, "").split("\n");
  // a function: from its comment lines to the first "}" at the start of a line
  const fnRange = (name) => {
    let a = lines.findIndex((l) => new RegExp(`^(export )?fn ${name}\\(`).test(l));
    if (a < 0) return null;
    let b = a;
    while (b < lines.length && lines[b] !== "}") b++;
    while (a > 0 && lines[a - 1].startsWith("//")) a--;
    return [a, b];
  };
  const [r0, r1] = fnRange("run_phase");
  code.ranges = PHASES.map((ph, k) => {
    // the branch of run_phase for phase k, and the functions it calls
    let a = lines.findIndex((l, i) => i > r0 && i < r1 && (l.includes(`k == ${k} {`) || (k === 4 && l.trim() === "} else {")));
    let b = a + 1;
    while (b < r1 && !lines[b].trim().startsWith("}")) b++;
    return [[a, b - 1], ...ph.fns.map(fnRange).filter(Boolean)];
  });
  const esc = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const kw = /\b(fn|let|var|if|else|for|in|while|return|export|import|continue)\b/g;
  const html = (i) => {
    const l = lines[i], c = l.indexOf("//");
    const body = c < 0 ? l : l.slice(0, c), note = c < 0 ? "" : l.slice(c);
    return `<span class="ln" data-n="${i + 1}">${esc(body).replace(kw, "<b>$1</b>")}${note ? `<i>${esc(note)}</i>` : ""}\n</span>`;
  };
  // the step pane: run_phase, which calls one phase; the source pane: all of it
  $("step").innerHTML = lines.slice(r0, r1 + 1).map((_, j) => html(r0 + j)).join("");
  $("code").innerHTML = lines.map((_, i) => html(i)).join("");
  code.step = { first: r0, lines: $("step").children };
  code.lines = $("code").children;
  $("phases").innerHTML = PHASES.map((ph, k) => `<li><button data-k="${k}"><span>${k}</span> ${ph.name}</button></li>`).join("");
  code.chips = [...$("phases").querySelectorAll("button")];
  for (const b of code.chips) b.onclick = () => scrollCode(+b.dataset.k);
}

// scrolls the source pane to the first function of phase k
function scrollCode(k) {
  const el = code.lines[code.ranges[k][1]?.[0] ?? code.ranges[k][0][0]];
  $("code").scrollTo({ top: el.offsetTop - $("code").offsetTop - 8, behavior: "smooth" });
}

function codeView(now) {
  if (!code.lines || now - code.last < 120) return;   // a readable rate
  code.last = now;
  const k = st.phase === "sim" || st.phase === "wall" ? st.solverPhase : -1;
  code.chips.forEach((b, i) => b.classList.toggle("on", i === k));
  // the source pane moves at most once a second, so that the eye can follow
  if (k >= 0 && k !== code.scrolledK && now - code.scrolled > 1000) {
    scrollCode(k); code.scrolled = now; code.scrolledK = k;
  }
  if (k === code.shown) return;
  for (const l of code.lines) l.classList.remove("on");
  for (const l of code.step.lines) l.classList.remove("on");
  if (k >= 0) {
    for (const [a, b] of code.ranges[k]) for (let i = a; i <= b; i++) code.lines[i].classList.add("on");
    const [a, b] = code.ranges[k][0];
    for (let i = a; i <= b; i++) code.step.lines[i - code.step.first].classList.add("on");
  }
  code.shown = k;
  $("codeState").textContent = k >= 0 ? `Phase ${k} (${PHASES[k].name}) ran last.` : "The solver is idle.";
}

// ---------------------------------------------------------------------------
// controls

function showSpeed() {
  const v = st.speed;
  $("speedNumber").value = v.toFixed(2);
  $("speedEntryStatus").textContent = "";
  $("speedOut").textContent = `${fmt(v, 2)} m/s = ${fmt(v / KT, 1)} kt`;
  const on = Math.abs(v - V_C) < 0.005;
  const sc = $("specCheck");
  sc.className = "hint " + (on ? "ok" : "off");
  sc.innerHTML = on
    ? "On spec: 4 lb at V<sub>C</sub> = 350 KCAS, sea level. 14 CFR 25.571(e)(1)."
    : `Off spec: the rule asks for V<sub>C</sub> = ${fmt(V_C, 2)} m/s. <button type="button" id="snap">Snap to V<sub>C</sub></button>. Science continues anyway.`;
  const snap = $("snap");
  if (snap) snap.onclick = () => {
    st.speed = V_C; $("speed").value = V_C; showSpeed();
    if (st.phase === "idle") preview();
  };
}

function buildPresets() {
  $("presets").innerHTML = PRESETS.map((p, i) => `
    <label class="preset"><input type="radio" name="preset" value="${i}" ${i === st.preset ? "checked" : ""}>
      <b>${p.name}</b> · ${fmt(p.spacing * 1e3, 1)} mm bird, ${p.cells} skin cells
      <small>wall check at V<sub>C</sub> only: peak ${sign(p.peak)} %, plateau ${sign(p.plateau)} % · solver about ${p.run}</small></label>`).join("");
  for (const el of document.querySelectorAll("#presets input")) {
    el.onchange = () => { st.preset = +el.value; preview(); };
  }
}

function preview() {
  setupRun(0);
  st.phase = "idle";
  $("verdict").hidden = true;
  $("banner").textContent = "Goose loaded. Goose is unaware. Press FIRE.";
}

$("speed").value = V_C;
for (const attr of ["min", "max", "step"]) $("speedNumber")[attr] = $("speed")[attr];
$("speedNumber").onchange = (e) => {
  const input = e.target;
  if (!input.checkValidity()) {
    input.value = st.speed.toFixed(2);
    $("speedEntryStatus").textContent = "Invalid speed. The control kept the previous speed.";
    return;
  }
  st.speed = input.valueAsNumber;
  $("speed").value = st.speed;
  showSpeed();
  if (st.phase === "idle") preview();
};
$("speed").oninput = (e) => { st.speed = +e.target.value; showSpeed(); };
// Update the idle preview on release, so a drag does not repeat solver setup.
$("speed").onchange = () => { if (st.phase === "idle") preview(); };
$("fire").onclick = fire;
$("stop").onclick = stopRun;
$("wilbeck").onclick = wallCheck;
$("gif").onclick = exportGif;
$("mute").onclick = () => {
  st.muted = !st.muted;
  try { localStorage.setItem("goosestrike-muted", st.muted ? "1" : "0"); } catch { /* no storage */ }
  showMute();
};
function showMute() {
  $("mute").setAttribute("aria-pressed", String(st.muted));
  $("mute").textContent = st.muted ? "🔇 Honks off" : "🔊 Honks on";
}
$("btnCostume").onclick = () => { st.costume = !st.costume; $("btnCostume").setAttribute("aria-pressed", String(st.costume)); };
$("btnPov").onclick = () => { st.pov = !st.pov; $("btnPov").setAttribute("aria-pressed", String(st.pov)); showZoom(); };
$("btnReset").onclick = () => { st.yaw = 1.0; st.pitch = 0.25; st.dist = 1.3; st.pov = false; $("btnPov").setAttribute("aria-pressed", "false"); showZoom(); };
$("btnZoomIn").onclick = () => zoom(1 / 1.2);
$("btnZoomOut").onclick = () => zoom(1.2);

function showZoom() {
  $("btnZoomIn").disabled = st.pov || st.dist <= 0.4;
  $("btnZoomOut").disabled = st.pov || st.dist >= 5;
}

function zoom(factor) {
  if (st.pov) return;
  st.dist = Math.max(0.4, Math.min(5, st.dist * factor));
  showZoom();
}

let drag = null;
function rotate(dx, dy) {
  if (st.pov) return;
  st.yaw += dx;
  st.pitch = Math.max(-1.45, Math.min(1.45, st.pitch + dy));
}

canvas.addEventListener("keydown", (e) => {
  if (e.altKey || e.ctrlKey || e.metaKey || st.pov) return;
  const turns = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] };
  const turn = turns[e.key];
  if (!turn) return;
  e.preventDefault();
  rotate(...turn);
});
canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", (e) => {
  if (!drag || st.pov) return;
  rotate((e.clientX - drag.x) * 0.008, (e.clientY - drag.y) * 0.008);
  drag = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("pointerup", () => { drag = null; });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoom(Math.exp(e.deltaY * 0.001));
}, { passive: false });

loadCode();
showSpeed();
showMute();
buildPresets();
resize();
preview();
setBusy(false);
requestAnimationFrame(frame);
