// The page: runs the Goose solver (WebAssembly) frame by frame and draws it.
import { loadEngine, hugoniot, stagnation, V_C, BIRD_MASS } from "./engine.mjs";

const KT = 1852 / 3600;            // m/s per knot
const T_END = 3.0e-3;              // panel run length, s
const BUDGET_MS = 12;              // solver time per animation frame
const APPROACH_S = 1.6;            // cartoon approach, real seconds
const APPROACH_M = 1.2;            // cartoon approach distance, m
const P_LO = 1e5, P_HI = 4e8;      // pressure colour scale, Pa (log)
const EP_MAX = 0.18;               // plastic strain colour scale
const EP_SHOW = 0.002;             // below 0.2 % plastic strain, bare metal colour

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

// Costume: a white cartoon goose painted on the capsule. The capsule flies
// toward -z, so the head is at the low-z end; +y is the goose's back.
function paintGoose() {
  const p = st.init, n = p.length / 3;
  const D = eng.birdD(), L = eng.birdL(), r = 0.5 * D;
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < n; i++) { zmin = Math.min(zmin, p[3 * i + 2]); zmax = Math.max(zmax, p[3 * i + 2]); }
  const head = zmin + r;              // centre of the front hemisphere
  const mid = zmin + 0.5 * L;
  st.skin = new Array(n);
  for (let i = 0; i < n; i++) {
    const y = p[3 * i + 1] / r, u = (p[3 * i + 2] - zmin) / L;
    let c = "#f7f5ef";                                  // white feathers
    if (u > 0.35 && y > 0.25) c = "#dcd8cf";            // a grey back
    if (u > 0.88) c = "#ecebe6";                        // a fluffy tail
    if (u < 0.32 && y < -0.35) c = "#fffdf8";           // a bright neck
    st.skin[i] = c;
  }
  st.anchors = {
    nose: nearest(p, 0, -0.1 * r, zmin),
    eyes: [nearest(p, 0.6 * r, 0.6 * r, head), nearest(p, -0.6 * r, 0.6 * r, head)],
    cheeks: [nearest(p, 0.85 * r, 0.1 * r, head + 0.2 * r), nearest(p, -0.85 * r, 0.1 * r, head + 0.2 * r)],
    wings: [nearest(p, 0.6 * r, 0.75 * r, mid + 0.3 * r), nearest(p, -0.6 * r, 0.75 * r, mid + 0.3 * r)],
    feet: [nearest(p, 0.4 * r, -r, mid + 0.2 * r), nearest(p, -0.4 * r, -r, mid + 0.2 * r)],
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
  st.peakF = 0;
  st.hist = [];
  st.splatted = false;
  paintGoose();
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

function pressureColour(p) {
  const s = Math.max(0, Math.min(1, (Math.log10(Math.max(p, P_LO)) - Math.log10(P_LO)) / (Math.log10(P_HI) - Math.log10(P_LO))));
  return `hsl(${230 - 230 * s} 85% ${40 + 22 * s}%)`;
}

function strainColour(ep, shade, inside) {
  const s = Math.min(1, ep / EP_MAX);
  if (ep < EP_SHOW) {
    const l = Math.round((inside ? 52 : 74) * shade);
    return inside ? `hsl(80 30% ${l}%)` : `hsl(210 12% ${l}%)`;   // primer inside, bare metal outside
  }
  return `hsl(${50 - 50 * s} 95% ${Math.round((58 - 10 * s) * shade)}%)`;
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
  const shade = 0.55 + 0.45 * Math.abs(facing);
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

function decorations(offsetZ, flap) {
  const A = st.anchors, D = eng.birdD(), P = st.parts;
  const wp = (i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2] + offsetZ];
  const nose = wp(A.nose), tail = wp(A.tail);
  const axis = norm([nose[0] - tail[0], nose[1] - tail[1], nose[2] - tail[2]]);
  const pt = (w) => { const o = [0, 0, 0]; proj(w[0], w[1], w[2], o, 0); return o; };
  const add = (w, s, k) => [w[0] + s * k[0], w[1] + s * k[1], w[2] + s * k[2]];
  const dizzy = contactStarted();
  const list = [];
  const scale = (z) => cam.foc / z;

  // wings: paint only, no mass
  for (const [side, i] of [[1, A.wings[0]], [-1, A.wings[1]]]) {
    const s = wp(i);
    // little wings on the back, flapping in the approach, limp after it
    const lift = dizzy ? 0.1 * D + 0.04 * D * Math.sin(performance.now() / 40) : (0.45 + 0.3 * flap) * D;
    const out = dizzy ? 0.55 * D : (0.35 - 0.2 * flap) * D;
    const tip = add([s[0] + side * out, s[1] + lift, s[2]], -0.35 * D, axis);
    const back = add([s[0] + side * 0.1 * D, s[1] + 0.05 * D, s[2]], -0.5 * D, axis);
    const a = pt(s), b = pt(tip), c = pt(back);
    list.push({ depth: a[2] + 0.002, draw() {
      ctx.fillStyle = "#fbfaf6"; ctx.strokeStyle = "#8f8a7d"; ctx.lineWidth = 1.5 * DPR;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]); ctx.closePath();
      ctx.fill(); ctx.stroke();
    } });
  }
  // beak
  {
    const n = pt(nose);
    const tipW = add(nose, 0.42 * D, axis);
    const t = pt(tipW);
    const up = pt([nose[0], nose[1] + 0.13 * D, nose[2]]), dn = pt([nose[0], nose[1] - 0.11 * D, nose[2]]);
    list.push({ depth: n[2] - 0.004, draw() {
      ctx.fillStyle = "#f08c00"; ctx.strokeStyle = "#a35b00"; ctx.lineWidth = 1.2 * DPR;
      ctx.beginPath(); ctx.moveTo(up[0], up[1]); ctx.lineTo(t[0], t[1]); ctx.lineTo(dn[0], dn[1]); ctx.closePath();
      ctx.fill(); ctx.stroke();
    } });
  }
  // orange feet, tucked under the belly
  for (const [side, i] of [[1, A.feet[0]], [-1, A.feet[1]]]) {
    const f = wp(i);
    const k = pt(f);
    const toes = [-0.12, 0, 0.12].map((dx) => pt(add([f[0] + side * 0.05 * D + dx * D, f[1] - 0.28 * D, f[2]], 0.12 * D, axis)));
    list.push({ depth: k[2] + 0.001, draw() {
      ctx.fillStyle = "#f59f00"; ctx.strokeStyle = "#a35b00"; ctx.lineWidth = 1.2 * DPR;
      ctx.beginPath(); ctx.moveTo(k[0], k[1]);
      for (const q of toes) ctx.lineTo(q[0], q[1]);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    } });
  }
  // cheeks
  for (const i of A.cheeks) {
    const c = sp(i);
    const rr = 0.07 * D * scale(c[2]);
    list.push({ depth: c[2] - 0.003, draw() {
      ctx.fillStyle = "rgba(255,120,150,0.55)";
      ctx.beginPath(); ctx.ellipse(c[0], c[1], rr * 1.3, rr * 0.8, 0, 0, 2 * Math.PI); ctx.fill();
    } });
  }
  // eyes: round before contact, spirals after
  for (const i of A.eyes) {
    const c = sp(i);
    const rr = 0.12 * D * scale(c[2]);
    list.push({ depth: c[2] - 0.004, draw() {
      ctx.fillStyle = "#fff"; ctx.strokeStyle = "#111"; ctx.lineWidth = 1.2 * DPR;
      ctx.beginPath(); ctx.arc(c[0], c[1], rr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      if (dizzy) {
        ctx.beginPath();
        const turn = performance.now() / 120;
        for (let k = 0; k <= 40; k++) {
          const a = turn + k * 0.45, q = rr * 0.9 * k / 40;
          const x = c[0] + q * Math.cos(a), y = c[1] + q * Math.sin(a);
          if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      } else {
        ctx.fillStyle = "#111";
        ctx.beginPath(); ctx.arc(c[0] - rr * 0.15, c[1] + rr * 0.1, rr * 0.55, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.beginPath(); ctx.arc(c[0] - rr * 0.32, c[1] - rr * 0.12, rr * 0.18, 0, 2 * Math.PI); ctx.fill();
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
    ctx.fillStyle = "#ffd43b"; ctx.strokeStyle = "#a07800"; ctx.lineWidth = DPR;
    ctx.beginPath();
    for (let j = 0; j < 10; j++) {
      const ang = -Math.PI / 2 + j * Math.PI / 5, q = j % 2 ? r * 0.45 : r;
      const x = o[0] + q * Math.cos(ang), y = o[1] + q * Math.sin(ang);
      if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.fill(); ctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// charts

function chart(cv, series, opt) {
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
  c.font = "11px ui-monospace, Menlo, monospace";
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
  chart($("forceChart"), [{ xs, ys, color: "#e8590c" }], {
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
  $("verdict").hidden = true;
  setBusy(true);
  honk(Math.sqrt(st.speed / V_C));
  honk(Math.sqrt(st.speed / V_C) * 1.06, 0.3);
}

function setBusy(b) {
  $("fire").disabled = b;
  $("wilbeck").disabled = b;
  $("fire").textContent = b ? "GOOSE IN FLIGHT" : "FIRE THE GOOSE";
  for (const el of document.querySelectorAll("#presets input, #speed")) el.disabled = b;
}

function stepSolver(tEnd, perStep) {
  const t0 = performance.now(), s0 = eng.time();
  let steps = 0;
  do {
    steps += eng.advance(1, tEnd);
    if (perStep) perStep();
  } while (performance.now() - t0 < BUDGET_MS && eng.time() < tEnd);
  const ms = performance.now() - t0;
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
    ["bird kinetic", eng.birdKE(), "#4dabf7"],
    ["bird squish (internal)", eng.birdWork(), "#9775fa"],
    ["panel kinetic", eng.panelKE(), "#20c997"],
    ["panel strain + plastic", eng.panelWork(), "#fd7e14"],
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
    <p>Eroded shell elements: ${fmt(skin, 0)} in the 2024-T3 skin, ${fmt(stiff, 0)} in the 7075-T6 stringers and frames.
      An element erodes when its plastic strain reaches the failure strain of its material. Maximum deflection ${fmt(eng.maxDeflection() * 1e3, 1)} mm.
      Peak contact force ${fmt(st.peakF / 1e3, 1)} kN. Maximum plastic strain ${fmt(eng.maxPlasticStrain() * 100, 1)} %.
      The panel took ${fmt(eng.panelWork() / 1e3, 1)} kJ of the goose's ${fmt(st.ke0 / 1e3, 1)} kJ.</p>
    <p>Does this fail 25.571(e)(1)? Not by itself. The rule accepts damage: the aeroplane must complete the flight with it.
      That needs a residual strength analysis of the damaged structure, which this page does not do.
      This is not a certification analysis.</p>
    <p class="hint">Measured at V<sub>C</sub> only: the deflection converges with the mesh (${PRESETS.map((p) => fmt(p.defl, 1)).join(", ")} mm at ${PRESETS.map((p) => fmt(p.spacing * 1e3, 1)).join(", ")} mm).
      The skin erosion does not (${PRESETS.map((p) => p.skin).join(", ")} skin elements): erosion depends on the element size, so a tear prediction needs a mesh-calibrated failure model.
      This run: ${pr.name}, ${fmt(pr.spacing * 1e3, 1)} mm bird lattice.</p>`;
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
  $("verdict").hidden = true;
  $("wallChart").scrollIntoView({ behavior: "smooth", block: "center" });
  setBusy(true);
  honk(0.8);
}

function wallSample() {
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
  chart($("wallChart"), [{ xs: w.xs, ys: w.ys, color: "#4dabf7" }], {
    xmin: 0, xmax: 1.5 * w.tl * 1e6, ymin: 1, ymax: 1000, log: true,
    xticks: ticks(1.5 * w.tl * 1e6, 6), yticks: [1, 10, 100, 1000],
    xlabel: "time since contact (µs)", ylabel: "centre pressure (MPa, log)",
    hlines: [{ y: w.ph / 1e6, label: `P_H ${fmt(w.ph / 1e6, 1)} MPa`, color: "#c92a2a" },
             { y: w.ps / 1e6, label: `P_s ${fmt(w.ps / 1e6, 2)} MPa`, color: "#2b8a3e" }],
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
    stepSolver(T_END, () => { st.peakF = Math.max(st.peakF, eng.contactForce()); });
    readState();
    st.hist.push({ t: eng.time(), f: eng.contactForce() });
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
      st.phase = "done"; setBusy(false); verdict(); honk(0.7, 0.1);
    }
  } else if (st.phase === "wall") {
    stepSolver(1.5 * st.wall.tl, wallSample);
    readState();
    banner.textContent = `CONTROL EXPERIMENT: goose vs. rigid wall · ${fmt(eng.time() * 1e6, 0)} µs · colour = pressure`;
    drawWallChart();
    if (eng.time() >= 1.5 * st.wall.tl) { st.phase = "done"; setBusy(false); }
  } else if (st.phase === "done") {
    banner.textContent = "Done. Drag to orbit. Scroll to zoom. Fire again for another goose.";
  }
  legend();
  if (st.nodes) render(offsetZ, flap);
  requestAnimationFrame(frame);
}

function legend() {
  const showP = !st.costume || st.mode === 1;
  const pRamp = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((s) => `hsl(${230 - 230 * s} 85% ${40 + 22 * s}%)`).join(",")})`;
  const eRamp = `linear-gradient(90deg, hsl(210 12% 74%), hsl(50 95% 58%), hsl(0 95% 48%))`;
  $("legend").innerHTML =
    (showP ? `bird pressure (log)<div class="ramp" style="background:${pRamp}"></div>0.1 MPa … 400 MPa<br>` : "") +
    (st.mode === 0 ? `panel plastic strain<div class="ramp" style="background:${eRamp}"></div>${fmt(EP_SHOW * 100, 1)} … ${fmt(EP_MAX * 100, 0)} %` : "");
}

// ---------------------------------------------------------------------------
// controls

function showSpeed() {
  const v = st.speed;
  $("speedOut").textContent = `${fmt(v, 2)} m/s = ${fmt(v / KT, 1)} kt`;
  const on = Math.abs(v - V_C) < 0.005;
  const sc = $("specCheck");
  sc.className = "hint " + (on ? "ok" : "off");
  sc.innerHTML = on
    ? "On spec: 4 lb at V<sub>C</sub> = 350 KCAS, sea level. 14 CFR 25.571(e)(1)."
    : `Off spec: the rule asks for V<sub>C</sub> = ${fmt(V_C, 2)} m/s. <a href="#" id="snap">Snap to V<sub>C</sub></a>. Science continues anyway.`;
  const snap = $("snap");
  if (snap) snap.onclick = (e) => { e.preventDefault(); st.speed = V_C; $("speed").value = V_C; showSpeed(); };
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
  telemetry();
  $("banner").textContent = "Goose loaded. Goose is unaware. Press FIRE.";
}

$("speed").value = V_C;
$("speed").oninput = (e) => { st.speed = +e.target.value; showSpeed(); };
$("fire").onclick = fire;
$("wilbeck").onclick = wallCheck;
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
$("btnPov").onclick = () => { st.pov = !st.pov; $("btnPov").setAttribute("aria-pressed", String(st.pov)); };
$("btnReset").onclick = () => { st.yaw = 1.0; st.pitch = 0.25; st.dist = 1.3; st.pov = false; $("btnPov").setAttribute("aria-pressed", "false"); };

let drag = null;
canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", (e) => {
  if (!drag || st.pov) return;
  st.yaw += (e.clientX - drag.x) * 0.008;
  st.pitch = Math.max(-1.45, Math.min(1.45, st.pitch + (e.clientY - drag.y) * 0.008));
  drag = { x: e.clientX, y: e.clientY };
});
canvas.addEventListener("pointerup", () => { drag = null; });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  st.dist = Math.max(0.4, Math.min(5, st.dist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });

showSpeed();
showMute();
buildPresets();
resize();
preview();
setBusy(false);
requestAnimationFrame(frame);
