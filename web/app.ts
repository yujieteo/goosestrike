import {
  loadEngine, loadGpuEngine, hugoniot, stagnation, V_C, BIRD_MASS, T_SKIN, PANEL, RIGID_WALL,
  type AnyEngine, type GpuEngine, type CpuEngine, type SolverMode,
} from "./engine.ts";
import { GifRecorder } from "./gif.ts";
import { assert, at, must } from "./checked.ts";
import {
  KNOT_MS, T_END, BUDGET_MS, APPROACH_S, APPROACH_M, PRESSURE_LOW_PA, PRESSURE_HIGH_PA, EP_MAX, GIF_W, GIF_SIM_US,
  GIF_CAPTION_PX, GIF_APPROACH_PERIOD_MS, GIF_APPROACH_DELAY_CS, GIF_SIM_DELAY_CS, GIF_LAST_DELAY_CS, GPU_BATCH_MAX,
  GPU_BATCH_MIN, STRUCTURE_FAIL_PHASE, MIN_DEPTH, SKIN_COLUMN_FRACTION, SKIN_ROW_MULTIPLE, WALL_RUN_LENGTHS, MUTED_KEY,
  SOURCE_URL, DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_DIST, DIST_MIN, DIST_MAX, PITCH_LIMIT, DEPTH_BUCKETS, SHADES, GRID,
  WHEEL_LINE_PX, GOOSE, type Vec3,
} from "./params.ts";
import { ALL_PRESETS, PHASES, type Preset } from "./presets.ts";

declare global {
  interface Window { webkitAudioContext?: typeof AudioContext }
}

type Vec2 = [number, number];
type Phase = "idle" | "approach" | "sim" | "done" | "wall";
type MapName = "metal" | "primer" | "pressure";
type PerStep = (time: number, contactForce: number, centerForce: number) => void;
type Range = [number, number];

function el<T extends HTMLElement>(id: string, kind: new () => T): T {
  assert(id.length > 0, "an element id is given");
  assert(typeof kind === "function", "an element type is given");
  const found = document.getElementById(id);
  if (!(found instanceof kind)) throw new Error(`#${id} is missing or has the wrong type`);
  assert(found.id === id, "the element is the one asked for");
  assert(found.isConnected, "the element is in the page");
  return found;
}
function $(id: string): HTMLElement {
  assert(id.length > 0, "an element id is given");
  const found = el(id, HTMLElement);
  assert(found.id === id, "the element has the requested id");
  assert(found.isConnected, "the element is in the page");
  return found;
}
function button(id: string): HTMLButtonElement {
  assert(id.length > 0, "a button id is given");
  const found = el(id, HTMLButtonElement);
  assert(found.tagName === "BUTTON", "the element is a button");
  assert(found.id === id, "the button has the requested id");
  return found;
}
function input(id: string): HTMLInputElement {
  assert(id.length > 0, "an input id is given");
  const found = el(id, HTMLInputElement);
  assert(found.tagName === "INPUT", "the element is an input");
  assert(found.id === id, "the input has the requested id");
  return found;
}
function canvasOf(id: string): HTMLCanvasElement {
  assert(id.length > 0, "a canvas id is given");
  const found = el(id, HTMLCanvasElement);
  assert(found.tagName === "CANVAS", "the element is a canvas");
  assert(found.id === id, "the canvas has the requested id");
  return found;
}
function context2d(cv: HTMLCanvasElement): CanvasRenderingContext2D {
  assert(cv.tagName === "CANVAS", "the 2D context comes from a canvas");
  assert(cv.width >= 0 && cv.height >= 0, "the canvas has a size");
  const found = must(cv.getContext("2d"), "the 2D context");
  assert(found.canvas === cv, "the context belongs to the canvas");
  return found;
}

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
function fmt(x: number, d = 1): string {
  assert(Number.isInteger(d), "the digit count is a whole number");
  assert(d >= 0 && d <= 20, "the digit count is 0 to 20");
  const text = x.toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
  assert(text.length > 0, "the number has text");
  return text;
}
function sign(x: number, d = 1): string {
  assert(Number.isInteger(d), "the digit count is a whole number");
  assert(d >= 0 && d <= 20, "the digit count is 0 to 20");
  const text = (x > 0 ? "+" : x < 0 ? "−" : "") + fmt(Math.abs(x), d);
  assert(text.length > 0, "the signed number has text");
  return text;
}

async function loadSolver(): Promise<AnyEngine> {
  assert($("fire").isConnected && $("banner").isConnected, "the fire button and the banner exist");
  assert(typeof location.search === "string", "the page has a query string");
  try {
    const cpu = new URLSearchParams(location.search).get("solver") === "cpu";
    const gpu = cpu ? null : await loadGpuEngine().catch((error: unknown) => { console.warn("WebGPU solver:", error); return null; });
    assert(!cpu || gpu === null, "the cpu choice skips the GPU");
    const chosen = gpu ?? await loadEngine();
    assert(chosen.gpu === (gpu !== null), "the engine kind matches the choice");
    return chosen;
  } catch (error) {
    $("fire").textContent = "Goose unavailable";
    $("banner").setAttribute("role", "alert");
    $("banner").textContent = "The solver could not load. Reload the page to try again.";
    throw error;
  }
}
const eng: AnyEngine = await loadSolver();

const PRESETS: Preset[] = ALL_PRESETS.filter((p) => eng.gpu || !p.gpuOnly);

function preset(i: number): Preset {
  assert(Number.isInteger(i), "a preset index is a whole number");
  assert(PRESETS.length > 0, "there is a preset to pick");
  assert(i >= 0 && i < PRESETS.length, "the preset index is in range");
  return at(PRESETS, i);
}

interface Anchors {
  nose: number; eyes: number[]; cheeks: number[]; tuft: number; wings: number[]; feet: number[]; tail: number;
}

interface Scene {
  nodes: Float32Array; parts: Float32Array; press: Float32Array; strain: Float32Array;
  elems: Int32Array; skinNe: number; edges: Uint8Array; init: Float32Array; skin: string[]; anchors: Anchors;
}

interface Wall {
  ph: number; us: number; ps: number; tl: number; area: number; xs: number[]; ys: number[]; peak: number; plat: number; platN: number;
}

interface GifState {
  rec: GifRecorder; h: number; cv: HTMLCanvasElement; next: number; simNext: number; speed: number; title: string; done: boolean;
}

interface Batch { seen: number; steps: number }

interface State {
  phase: Phase; mode: SolverMode; preset: number; speed: number; tSkin: number;
  costume: boolean; pov: boolean; muted: boolean; yaw: number; pitch: number; dist: number;
  t0: number; ke0: number; peakF: number; scene: Scene | null; hist: { t: number; f: number }[];
  wall: Wall | null; frameSim: number; frameReal: number; msPerStep: number; fps: number;
  batch: Batch; splatted: boolean; solverPhase: number; gif: GifState | null; gifHold: boolean;
}

const st: State = {
  phase: "idle", mode: PANEL, preset: 0, speed: V_C, tSkin: T_SKIN,
  costume: true, pov: false, muted: false, yaw: DEFAULT_YAW, pitch: DEFAULT_PITCH, dist: DEFAULT_DIST,
  t0: 0, ke0: 1, peakF: 0, scene: null, hist: [],
  wall: null, frameSim: 0, frameReal: 16.7, msPerStep: 0, fps: 60,
  batch: { seen: 0, steps: 16 }, splatted: false, solverPhase: -1, gif: null, gifHold: false,
};

try { st.muted = localStorage.getItem(MUTED_KEY) === "1"; } catch { st.muted = false; }

let audio: AudioContext | null = null;
function ac(): AudioContext | null {
  assert(audio === null || audio.sampleRate > 0, "an existing audio context has a sample rate");
  assert(typeof st.muted === "boolean", "the mute flag is a boolean");
  try {
    audio ??= new (window.AudioContext ?? window.webkitAudioContext ?? AudioContext)();
  } catch {
    st.muted = true;
    showMute();
    return null;
  }
  assert(audio !== null, "an audio context exists once it was made");
  assert(audio.state !== "closed", "the audio context is open");
  return audio;
}

function honk(pitch = 1, when = 0): void {
  assert(pitch > 0, "the honk pitch is positive");
  assert(when >= 0, "the honk starts now or later");
  assert(Number.isFinite(pitch + when), "the honk timing is finite");
  if (st.muted) return;
  const a = ac();
  if (!a) return;
  const t = a.currentTime + when;
  assert(Number.isFinite(t), "the honk start time is finite");
  const osc = a.createOscillator(), osc2 = a.createOscillator();
  const bp = a.createBiquadFilter(), g = a.createGain();
  assert(a.sampleRate > 0, "the audio context has a sample rate");
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
  assert(osc.type === "sawtooth" && osc2.type === "square", "the honk has two voices");
}

function splat(): void {
  assert(st.splatted, "a splat follows the first contact");
  assert(!st.muted || st.splatted, "a muted splat is silent but still counted");
  if (st.muted) return;
  const a = ac();
  if (!a) return;
  const t = a.currentTime;
  const n = Math.floor(a.sampleRate * 0.5);
  assert(n > 0, "the noise buffer has samples");
  const buf = a.createBuffer(1, n, a.sampleRate);
  const d = buf.getChannelData(0);
  assert(d.length === n, "the noise buffer has the requested length");
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 2);
  const src = a.createBufferSource(), lp = a.createBiquadFilter(), g = a.createGain();
  src.buffer = buf;
  lp.type = "lowpass";
  lp.frequency.setValueAtTime(4000, t);
  lp.frequency.exponentialRampToValueAtTime(150, t + 0.45);
  g.gain.value = 0.6;
  src.connect(lp); lp.connect(g); g.connect(a.destination);
  assert(Number.isFinite(t), "the splat start time is finite");
  src.start(t);
}

function readRender(): Pick<Scene, "nodes" | "parts" | "press" | "strain"> {
  eng.fillRender();
  const out = { nodes: eng.nodes(), parts: eng.particles(), press: eng.pressure(), strain: eng.strain() };
  assert(out.nodes.length % 3 === 0, "node positions come in triples");
  assert(out.parts.length % 3 === 0, "particle positions come in triples");
  assert(out.press.length * 3 === out.parts.length, "there is one pressure per particle");
  return out;
}

function nearestParticle(px: Float32Array, x: number, y: number, z: number): number {
  assert(px.length > 0 && px.length % 3 === 0, "particle positions come in non-empty triples");
  assert(!Number.isNaN(x + y + z), "the query point is a number");
  let best = 0, bd = Infinity;
  for (let i = 0; i < px.length / 3; i++) {
    const dx = at(px, 3 * i) - x, dy = at(px, 3 * i + 1) - y, dz = at(px, 3 * i + 2) - z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bd) { bd = d; best = i; }
  }
  assert(best >= 0 && 3 * best < px.length, "the nearest particle exists");
  assert(bd < Infinity, "some particle was at a finite distance");
  return best;
}

function paintGoose(p: Float32Array): Pick<Scene, "skin" | "anchors"> {
  assert(p.length > 0 && p.length % 3 === 0, "the goose has particles in triples");
  const n = p.length / 3;
  const D = eng.birdD(), L = eng.birdL(), r = 0.5 * D;
  assert(r > 0 && L > 0, "the bird has a size");
  assert(L >= D, "the bird is longer than it is wide");
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < n; i++) { zmin = Math.min(zmin, at(p, 3 * i + 2)); zmax = Math.max(zmax, at(p, 3 * i + 2)); }
  const head = zmin + r;
  assert(zmax >= zmin, "the bird has a length along z");
  const mid = zmin + 0.5 * L;
  assert(mid >= zmin && head >= zmin, "the head and middle lie along the bird");
  const skin: string[] = [];
  for (let i = 0; i < n; i++) {
    const y = at(p, 3 * i + 1) / r;
    const s = Math.round(32 * smoothstep(-0.6, 1.2, y)) / 32;
    skin.push(rgb(GOOSE.belly.map((c, k) => c + s * (at(GOOSE.back, k) - c))));
  }
  const near = (x: number, y: number, z: number): number => nearestParticle(p, x, y, z);
  const onHead = (x: number, y: number, z: number): number => { const d = norm([x, y, -z]); return near(r * d[0], r * d[1], head + r * d[2]); };
  const anchors: Anchors = {
    nose: onHead(0, -0.1, 1),
    eyes: [onHead(0.5, 0.42, 0.75), onHead(-0.5, 0.42, 0.75)],
    cheeks: [onHead(0.72, 0.02, 0.62), onHead(-0.72, 0.02, 0.62)],
    tuft: onHead(0, 0.95, 0.3),
    wings: [near(0.92 * r, 0.3 * r, mid + 0.25 * r), near(-0.92 * r, 0.3 * r, mid + 0.25 * r)],
    feet: [near(0.4 * r, -0.92 * r, mid + 0.3 * r), near(-0.4 * r, -0.92 * r, mid + 0.3 * r)],
    tail: near(0, 0, zmax),
  };
  assert(skin.length === n, "there is one colour per particle");
  assert(anchors.eyes.length === 2 && anchors.cheeks.length === 2, "the goose has two eyes and two cheeks");
  assert(anchors.wings.length === 2 && anchors.feet.length === 2, "the goose has two wings and two feet");
  return { skin, anchors };
}

function skinElements(cells: number): number {
  assert(Number.isInteger(cells) && cells > 0, "the skin has a whole number of cells");
  const gs = Math.floor(cells * SKIN_COLUMN_FRACTION / SKIN_ROW_MULTIPLE + 0.5) * SKIN_ROW_MULTIPLE;
  assert(gs > 0 && gs % SKIN_ROW_MULTIPLE === 0, "the skin rows come in multiples of eight");
  const total = cells * gs;
  assert(Number.isInteger(total) && total > 0, "the skin has a whole number of elements");
  return total;
}

function featureEdges(el: Int32Array): Uint8Array {
  assert(el.length % 4 === 0, "elements have four corners");
  const ne = el.length / 4, count = new Map<number, number>();
  const key = (a: number, b: number): number => a < b ? a * 1e6 + b : b * 1e6 + a;
  const side = (i: number): number => key(at(el, i), at(el, i - i % 4 + (i + 1) % 4));
  for (let i = 0; i < 4 * ne; i++) count.set(side(i), (count.get(side(i)) ?? 0) + 1);
  const edges = new Uint8Array(ne);
  assert(count.size <= 4 * ne, "no side is counted more than once per element");
  assert(Number.isInteger(ne), "the element count is whole");
  for (let i = 0; i < 4 * ne; i++) if (count.get(side(i)) !== 2) edges[i >> 2] = at(edges, i >> 2) | (1 << i % 4);
  assert(edges.length * 4 === el.length, "there is one flag byte per element");
  assert(edges.byteLength === ne, "an element has one byte of edge flags");
  return edges;
}

function sceneOf(): Scene {
  const render = readRender();
  const elems = eng.elements();
  const init = render.parts.slice();
  const scene: Scene = { ...render, elems, edges: featureEdges(elems), skinNe: 0, init, ...paintGoose(init) };
  assert(scene.edges.length * 4 === scene.elems.length, "an edge byte per element");
  assert(scene.skin.length * 3 === scene.init.length, "a skin colour per particle");
  assert(scene.strain.length * 4 === scene.elems.length || scene.strain.length === 0, "a strain per element");
  return scene;
}

function setupRun(mode: SolverMode): Scene {
  assert(mode === PANEL || mode === RIGID_WALL, "the mode is panel or wall");
  assert(st.speed > 0 && st.tSkin > 0, "the speed and the skin are positive");
  const pr = preset(st.preset);
  assert(pr.cells > 0 && pr.spacing > 0, "the preset has a skin grid and a bird lattice");
  eng.setSkin(st.tSkin);
  eng.setup(mode, pr.cells, pr.spacing, st.speed);
  st.batch = { seen: 0, steps: 16 };
  st.mode = mode;
  legend();
  const scene = sceneOf();
  assert(scene.nodes.length % 3 === 0, "node positions come in triples");
  scene.skinNe = mode === PANEL ? skinElements(pr.cells) : 0;
  assert(scene.skinNe >= 0, "the skin element count is not negative");
  st.scene = scene;
  st.ke0 = eng.birdKE();
  $("energyTitle").textContent = `Goose energy at start: ${fmt(st.ke0 / 1e3, 1)} kJ`;
  st.peakF = 0;
  st.frameSim = 0;
  st.msPerStep = 0;
  st.hist = [];
  st.splatted = false;
  st.solverPhase = -1;
  drawForceChart();
  telemetry();
  assert(st.scene === scene, "the new scene is the current one");
  assert(st.hist.length === 0 && st.peakF === 0, "the run history is cleared");
  return scene;
}

const canvas = canvasOf("view");
const ctx = context2d(canvas);
let W = 0, H = 0, DPR = 1;

function resize(): void {
  assert(canvas.isConnected, "the view canvas is in the page");
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  assert(DPR > 0 && DPR <= 2, "the pixel ratio is in range");
  const r = canvas.getBoundingClientRect();
  W = Math.max(1, Math.round(r.width * DPR));
  H = Math.max(1, Math.round(r.height * DPR));
  assert(W >= 1 && H >= 1, "the canvas has a size");
  assert(Number.isInteger(W) && Number.isInteger(H), "the canvas size is whole pixels");
  canvas.width = W; canvas.height = H;
}
new ResizeObserver(resize).observe(canvas);

const cam = { eye: [0, 0, 0] as Vec3, r: [1, 0, 0] as Vec3, u: [0, 1, 0] as Vec3, f: [0, 0, 1] as Vec3, foc: 1 };

function norm(v: Vec3): Vec3 {
  assert(v.length === 3, "a vector has three components");
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  assert(l > 0, "the divisor is not zero");
  return [v[0] / l, v[1] / l, v[2] / l];
}
function cross(a: Vec3, b: Vec3): Vec3 {
  assert(a.length === 3 && b.length === 3, "a cross product takes two vectors");
  assert(a !== b, "a vector is not crossed with itself");
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a: Vec3, b: Vec3): number {
  assert(a.length === 3 && b.length === 3, "a dot product takes two vectors");
  assert(a !== null && b !== null, "both vectors exist");
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function setCamera(): void {
  assert(H > 0, "the canvas has a height");
  assert(st.dist >= DIST_MIN && st.dist <= DIST_MAX, "the camera distance is in range");
  assert(Math.abs(st.pitch) <= PITCH_LIMIT, "the camera pitch is in range");
  let eye: Vec3, target: Vec3;
  const upHint: Vec3 = [0, 1, 0];
  if (st.pov) {
    eye = [0.12, 0.06, -0.7]; target = [0, 0, 0.02];
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
  assert(cam.foc > 0 && Number.isFinite(cam.foc), "the focal length is positive");
  assert(Math.abs(Math.hypot(cam.f[0], cam.f[1], cam.f[2]) - 1) < 1e-6, "the view direction is a unit vector");
}

function proj(x: number, y: number, z: number, out: Float32Array | number[], o: number): void {
  const d: Vec3 = [x - cam.eye[0], y - cam.eye[1], z - cam.eye[2]];
  const cz = dot(d, cam.f), cx = dot(d, cam.r), cy = dot(d, cam.u);
  out[o] = 0.5 * W + cam.foc * cx / cz;
  out[o + 1] = 0.5 * H - cam.foc * cy / cz;
  out[o + 2] = cz;
}

let sNodes = new Float32Array(0), sParts = new Float32Array(0);
let keys = new Float32Array(0), order = new Int32Array(0), tmp = new Int32Array(0);
const counts = new Int32Array(DEPTH_BUCKETS + 1);

function depthSort(n: number): void {
  assert(n >= 0 && n <= keys.length, "every item has a sort key");
  assert(n <= order.length && n <= tmp.length, "every item has a slot");
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { const k = at(keys, i); if (k < lo) lo = k; if (k > hi) hi = k; }
  const s = (DEPTH_BUCKETS - 1) / Math.max(hi - lo, 1e-9);
  counts.fill(0);
  for (let i = 0; i < n; i++) tmp[i] = DEPTH_BUCKETS - 1 - Math.floor((at(keys, i) - lo) * s);
  for (let i = 0; i < n; i++) { const b = at(tmp, i) + 1; counts[b] = at(counts, b) + 1; }
  for (let b = 0; b < DEPTH_BUCKETS; b++) counts[b + 1] = at(counts, b + 1) + at(counts, b);
  assert(at(counts, DEPTH_BUCKETS) === n, "the bucket offsets add up to the item count");
  for (let i = 0; i < n; i++) { const b = at(tmp, i), slot = at(counts, b); order[slot] = i; counts[b] = slot + 1; }
  assert(order.length >= n, "the order covers every item");
}

function grow(n: number): void {
  assert(n >= 0, "the item count is not negative");
  assert(Number.isInteger(n), "the item count is whole");
  if (keys.length < n) { keys = new Float32Array(n); order = new Int32Array(n); tmp = new Int32Array(n); }
  assert(keys.length >= n && order.length >= n && tmp.length >= n, "the sort buffers cover every item");
}

function smoothstep(a: number, b: number, x: number): number {
  assert(a !== b, "the ramp has a width");
  assert(!Number.isNaN(x), "the input is a number");
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  const out = t * t * (3 - 2 * t);
  assert(out >= 0 && out <= 1, "the smooth step stays between 0 and 1");
  return out;
}
function rgb(c: readonly number[]): string {
  assert(c.length >= 3, "a colour has three channels");
  assert(Number.isFinite(at(c, 0) + at(c, 1) + at(c, 2)), "the channels are finite");
  return `rgb(${Math.round(at(c, 0))} ${Math.round(at(c, 1))} ${Math.round(at(c, 2))})`;
}
function map3(v: Vec3, f: (x: number) => number): Vec3 {
  assert(v.length === 3, "a vector has three components");
  assert(typeof f === "function", "a mapping function is given");
  return [f(v[0]), f(v[1]), f(v[2])];
}

function toLab(hex: string): Vec3 {
  assert(/^#[0-9a-f]{6}$/i.test(hex), "a colour is written #rrggbb");
  assert(hex.length === 7, "a colour has seven characters");
  const lin = (k: number): number => { const v = parseInt(hex.slice(k, k + 2), 16) / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  const l: Vec3 = [lin(1), lin(3), lin(5)];
  assert(Math.min(...l) >= 0 && Math.max(...l) <= 1, "linear channels are between 0 and 1");
  const m = map3([0.4122214708 * l[0] + 0.5363325363 * l[1] + 0.0514459929 * l[2],
    0.2119034982 * l[0] + 0.6806995451 * l[1] + 0.1073969566 * l[2],
    0.0883024619 * l[0] + 0.2817188376 * l[1] + 0.6299787005 * l[2]], Math.cbrt);
  assert(Math.min(...m) >= 0, "the cube roots are not negative");
  return [0.2104542553 * m[0] + 0.793617785 * m[1] - 0.0040720468 * m[2],
    1.9779984951 * m[0] - 2.428592205 * m[1] + 0.4505937099 * m[2],
    0.0259040371 * m[0] + 0.7827717662 * m[1] - 0.808675766 * m[2]];
}

function fromLab(c: Vec3): Vec3 {
  assert(c.length === 3, "a Lab colour has three channels");
  assert(Number.isFinite(c[0] + c[1] + c[2]), "the Lab channels are finite");
  const m = map3([c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2],
    c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2],
    c[0] - 0.0894841775 * c[1] - 1.291485548 * c[2]], (v) => v * v * v);
  const l: Vec3 = [4.0767416621 * m[0] - 3.3077115913 * m[1] + 0.2309699292 * m[2],
    -1.2684380046 * m[0] + 2.6097574011 * m[1] - 0.3413193965 * m[2],
    -0.0041960863 * m[0] - 0.7034186147 * m[1] + 1.707614701 * m[2]];
  const out = map3(l, (v) => 255 * Math.max(0, Math.min(1, v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055)));
  assert(Math.min(...out) >= 0 && Math.max(...out) <= 255, "the colour channels are 0 to 255");
  return out;
}

function ramp(stops: string[]): Vec3[] {
  assert(stops.length >= 2, "a ramp needs two stops");
  assert(stops.length < 256, "a ramp has fewer stops than entries");
  const lab = stops.map(toLab), out: Vec3[] = [];
  assert(lab.length === stops.length, "every stop has a Lab colour");
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (lab.length - 1), k = Math.min(lab.length - 2, Math.floor(x)), f = x - k;
    const a = at(lab, k), b = at(lab, k + 1);
    out.push(fromLab([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])]));
  }
  assert(out.length === 256, "a ramp has 256 entries");
  assert(at(at(out, 0), 0) !== at(at(out, 255), 0), "the ends of a ramp differ");
  return out;
}

const HOT = ["#e2b54f", "#c4613f", "#6e2a22"];
const MAPS: Record<MapName, Vec3[]> = {
  metal: ramp(["#c7c3ba", ...HOT]),
  primer: ramp(["#9da27f", ...HOT]),
  pressure: ramp(["#243552", "#3f6f86", "#94b49c", "#e8c766", "#c4613f"]),
};
const mapCache = new Map<string, string>();

function mapColour(name: MapName, s: number, shade = 1): string {
  assert(name in MAPS, "the colour map exists");
  assert(!Number.isNaN(s), "the map position is a number");
  assert(shade > 0 && shade < 2, "the shade is positive");
  const i = Math.round(255 * Math.max(0, Math.min(1, s))), j = Math.round(SHADES * shade);
  const key = name + (i * 32 + j);
  let c = mapCache.get(key);
  if (!c) { c = rgb(map3(at(MAPS[name], i), (v) => v * j / SHADES)); mapCache.set(key, c); }
  assert(c.length > 0, "the colour has text");
  assert(mapCache.size > 0, "the colour was cached");
  return c;
}

function logP(p: number): number {
  assert(PRESSURE_HIGH_PA > PRESSURE_LOW_PA, "the pressure scale has a range");
  assert(!Number.isNaN(p), "the pressure is a number");
  return (Math.log10(Math.max(p, PRESSURE_LOW_PA)) - Math.log10(PRESSURE_LOW_PA)) / (Math.log10(PRESSURE_HIGH_PA) - Math.log10(PRESSURE_LOW_PA));
}
function pressureColour(p: number): string {
  assert(!Number.isNaN(p), "the pressure is a number");
  assert(PRESSURE_LOW_PA > 0, "the pressure scale starts above zero");
  return mapColour("pressure", logP(p));
}
function strainColour(ep: number, shade: number, inside: boolean): string {
  assert(ep >= 0, "plastic strain is not negative");
  assert(shade > 0 && shade < 2, "the shade is positive");
  assert(typeof inside === "boolean", "the side flag is a boolean");
  return mapColour(inside ? "primer" : "metal", Math.sqrt(ep / EP_MAX), shade);
}

function cssRamp(name: MapName): string {
  assert(name in MAPS, "the colour map exists");
  const stops = Array.from({ length: 17 }, (_, k) => mapColour(name, k / 16));
  assert(stops.length === 17, "a gradient has 17 stops");
  assert(stops.join("").length > 0, "the gradient stops have text");
  return `linear-gradient(90deg, ${stops.join(",")})`;
}

interface Decoration { depth: number; draw: () => void }

function render(scene: Scene, offsetZ: number, flap: number): void {
  assert(Number.isFinite(offsetZ + flap), "the approach offset and the flap are finite");
  assert(W > 0 && H > 0, "the canvas has a size");
  setCamera();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const nn = scene.nodes.length / 3, np = scene.parts.length / 3, ne = scene.elems.length / 4;
  assert(Number.isInteger(nn) && Number.isInteger(np) && Number.isInteger(ne), "the counts are whole numbers");
  if (sNodes.length < 3 * nn) sNodes = new Float32Array(3 * nn);
  if (sParts.length < 3 * np) sParts = new Float32Array(3 * np);
  for (let i = 0; i < nn; i++) proj(at(scene.nodes, 3 * i), at(scene.nodes, 3 * i + 1), at(scene.nodes, 3 * i + 2), sNodes, 3 * i);
  for (let i = 0; i < np; i++) proj(at(scene.parts, 3 * i), at(scene.parts, 3 * i + 1), at(scene.parts, 3 * i + 2) + offsetZ, sParts, 3 * i);

  if (st.mode === RIGID_WALL) drawWall();

  const costume = st.costume && st.mode === PANEL;
  assert(st.mode === PANEL || st.mode === RIGID_WALL, "the run has a known mode");
  const decor = costume ? decorations(scene, offsetZ, flap) : [];
  assert(costume || decor.length === 0, "decorations only appear with the costume");
  const n = ne + np + decor.length;
  assert(n >= 0 && decor.length >= 0, "there is a count of items to sort");
  grow(n);
  for (let e = 0; e < ne; e++) {
    const a = at(scene.elems, 4 * e), b = at(scene.elems, 4 * e + 1), c = at(scene.elems, 4 * e + 2), d = at(scene.elems, 4 * e + 3);
    keys[e] = at(scene.strain, e) < 0 ? -1 : 0.25 * (at(sNodes, 3 * a + 2) + at(sNodes, 3 * b + 2) + at(sNodes, 3 * c + 2) + at(sNodes, 3 * d + 2));
  }
  for (let i = 0; i < np; i++) keys[ne + i] = at(sParts, 3 * i + 2);
  for (let k = 0; k < decor.length; k++) keys[ne + np + k] = at(decor, k).depth;
  depthSort(n);

  assert(keys.length >= n, "the sort keys cover every item");
  assert(at(counts, DEPTH_BUCKETS) === n, "every item landed in a depth bucket");
  const pr = eng.spacing() * 0.62 * cam.foc;
  for (let k = 0; k < n; k++) {
    const id = at(order, k);
    if (id < ne) drawShell(scene, id);
    else if (id < ne + np) {
      const i = id - ne, z = at(sParts, 3 * i + 2);
      if (z <= MIN_DEPTH) continue;
      ctx.fillStyle = costume ? at(scene.skin, i) : pressureColour(at(scene.press, i));
      ctx.beginPath();
      ctx.arc(at(sParts, 3 * i), at(sParts, 3 * i + 1), Math.max(0.8, pr / z), 0, 2 * Math.PI);
      ctx.fill();
    } else at(decor, id - ne - np).draw();
  }
  assert(sNodes.length >= 3 * nn && sParts.length >= 3 * np, "every node and particle was projected");
  if (costume && contactStarted()) drawStars(scene, offsetZ);
}

function drawShell(scene: Scene, e: number): void {
  assert(e >= 0 && e < scene.strain.length, "the element exists");
  assert(scene.elems.length === 4 * scene.strain.length, "an element has four corners");
  if (at(scene.strain, e) < 0) return;
  const el = scene.elems, P = scene.nodes;
  assert(sNodes.length >= P.length, "every node was projected");
  const a = at(el, 4 * e), b = at(el, 4 * e + 1), c = at(el, 4 * e + 2), d = at(el, 4 * e + 3);
  if (at(sNodes, 3 * a + 2) <= MIN_DEPTH || at(sNodes, 3 * c + 2) <= MIN_DEPTH) return;
  const node = (i: number): Vec3 => [at(P, 3 * i), at(P, 3 * i + 1), at(P, 3 * i + 2)];
  const sub = (u: Vec3, v: Vec3): Vec3 => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
  const nrm = norm(cross(sub(node(c), node(a)), sub(node(d), node(b))));
  const toEye = norm(sub(cam.eye, node(a)));
  const facing = dot(nrm, toEye);
  assert(Math.abs(facing) <= 1.0001, "the facing is a cosine");
  const shade = 0.72 + 0.28 * Math.abs(facing);
  assert(shade >= 0.72 && shade <= 1.01, "the shade is between 72 and 100 percent");
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
  if (!f) return;
  assert(f > 0 && f < 16, "a flagged element has one to four edge bits");
  const q = [a, b, c, d];
  ctx.strokeStyle = GOOSE.ink;
  ctx.lineWidth = 1.2 * DPR;
  ctx.beginPath();
  for (let k = 0; k < 4; k++) {
    if (!(f >> k & 1)) continue;
    const i = at(q, k), j = at(q, (k + 1) % 4);
    ctx.moveTo(at(sNodes, 3 * i), at(sNodes, 3 * i + 1));
    ctx.lineTo(at(sNodes, 3 * j), at(sNodes, 3 * j + 1));
  }
  ctx.stroke();
}

function drawWall(): void {
  assert(st.mode === RIGID_WALL, "the wall is drawn in the wall run");
  assert(W > 0 && H > 0, "the canvas has a size");
  const o = [0, 0, 0];
  assert(o.length === 3, "a projected point has x, y and depth");
  ctx.strokeStyle = "#7a8794";
  ctx.lineWidth = 1;
  for (let k = -5; k <= 5; k++) {
    const lines: [number, number, number, number][] = [[k * 0.05, -0.25, k * 0.05, 0.25], [-0.25, k * 0.05, 0.25, k * 0.05]];
    for (const [x0, y0, x1, y1] of lines) {
      proj(x0, y0, 0, o, 0);
      const sx = at(o, 0), sy = at(o, 1);
      proj(x1, y1, 0, o, 0);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(at(o, 0), at(o, 1)); ctx.stroke();
    }
  }
  assert(o[2] !== 0, "the wall lies in front of the camera");
}

function contactStarted(): boolean {
  const tc = eng.tContact();
  assert(!Number.isNaN(tc), "the contact time is a number");
  assert(st.phase.length > 0, "the run has a phase");
  return tc >= 0 && st.phase !== "approach";
}

function sp(i: number): Vec3 {
  assert(Number.isInteger(i) && i >= 0, "a particle index is a whole number");
  assert(3 * i + 2 < sParts.length, "the particle was projected");
  return [at(sParts, 3 * i), at(sParts, 3 * i + 1), at(sParts, 3 * i + 2)];
}

function blob(pts: Vec2[]): void {
  assert(pts.length >= 3, "a blob has at least three points");
  assert(at(pts, 0).length === 2, "a point has two coordinates");
  const m = (p: Vec2, q: Vec2): Vec2 => [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
  const n = pts.length;
  assert(n >= 3, "a blob has points to smooth");
  let s = m(at(pts, n - 1), at(pts, 0));
  ctx.beginPath(); ctx.moveTo(s[0], s[1]);
  for (let i = 0; i < n; i++) {
    const p = at(pts, i);
    s = m(p, at(pts, (i + 1) % n));
    ctx.quadraticCurveTo(p[0], p[1], s[0], s[1]);
  }
  ctx.closePath();
  assert(Number.isFinite(s[0] + s[1]), "the blob closes on a finite point");
}

function cheekDecoration(c: Vec3, rr: number, angle: number): Decoration {
  assert(!Number.isNaN(rr), "the cheek radius is a number");
  assert(Number.isFinite(angle), "the cheek angle is finite");
  return { depth: c[2] - 0.008, draw() {
    ctx.fillStyle = GOOSE.blush;
    ctx.beginPath(); ctx.ellipse(c[0], c[1], rr * 1.3, rr * 0.85, angle, 0, 2 * Math.PI); ctx.fill();
  } };
}

function dizzySpiral(c: Vec3, rr: number, now: number): void {
  assert(c.length === 3, "a centre has x, y and depth");
  assert(Number.isFinite(now), "the clock is finite");
  ctx.beginPath(); ctx.arc(c[0], c[1], rr * 0.82, 0, 2 * Math.PI); ctx.fill();
  ctx.strokeStyle = GOOSE.ink; ctx.lineWidth = Math.max(DPR, 0.16 * rr);
  ctx.beginPath();
  const turn = now / 120;
  for (let k = 0; k <= 40; k++) {
    const a = turn + k * 0.45, q = rr * 0.7 * k / 40;
    const x = c[0] + q * Math.cos(a), y = c[1] + q * Math.sin(a);
    if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function eyeDecoration(c: Vec3, rr: number, dizzy: boolean, now: number): Decoration {
  assert(c.length === 3, "a centre has x, y and depth");
  assert(Number.isFinite(now), "the clock is finite");
  assert(!Number.isNaN(rr), "the eye radius is a number");
  return { depth: c[2] - 0.01, draw() {
    ctx.fillStyle = GOOSE.ink;
    ctx.beginPath(); ctx.arc(c[0], c[1], rr, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#fff";
    if (dizzy) {
      dizzySpiral(c, rr, now);
    } else {
      ctx.beginPath(); ctx.arc(c[0] - rr * 0.32, c[1] - rr * 0.32, rr * 0.34, 0, 2 * Math.PI); ctx.fill();
      ctx.beginPath(); ctx.arc(c[0] + rr * 0.3, c[1] + rr * 0.3, rr * 0.14, 0, 2 * Math.PI); ctx.fill();
    }
  } };
}

function decorations(scene: Scene, offsetZ: number, flap: number): Decoration[] {
  assert(scene.anchors.eyes.length === 2, "the goose has two eyes");
  assert(Number.isFinite(offsetZ + flap), "the approach offset and the flap are finite");
  const now = reducedMotion.matches ? 0 : performance.now();
  const A = scene.anchors, D = eng.birdD(), P = scene.parts;
  assert(D > 0 && P.length > 0, "the bird has a size and particles");
  const wp = (i: number): Vec3 => [at(P, 3 * i), at(P, 3 * i + 1), at(P, 3 * i + 2) + offsetZ];
  const pt = (w: Vec3): Vec3 => { const o: Vec3 = [0, 0, 0]; proj(w[0], w[1], w[2], o, 0); return o; };
  const nose = wp(A.nose), tail = wp(A.tail);
  const n = pt(nose), t = pt(tail);
  let ux = n[0] - t[0], uy = n[1] - t[1];
  const ul = Math.hypot(ux, uy);
  if (ul < 1e-6) { ux = 1; uy = 0; } else { ux /= ul; uy /= ul; }
  const top = pt([nose[0], nose[1] + 0.1, nose[2]]);
  const side = (top[0] - n[0]) * -uy + (top[1] - n[1]) * ux > 0 ? 1 : -1;
  const vx = -uy * side, vy = ux * side;
  const at2 = (c: readonly number[], k: number, u: number, v: number): Vec2 => [at(c, 0) + k * (u * ux + v * vx), at(c, 1) + k * (u * uy + v * vy)];
  const dizzy = contactStarted();
  const list: Decoration[] = [];
  const scale = (z: number): number => D * cam.foc / z;
  const outline = (z: number): void => { ctx.strokeStyle = GOOSE.ink; ctx.lineWidth = Math.max(DPR, 0.012 * scale(z)); ctx.lineJoin = "round"; ctx.stroke(); };
  const part = (c: Vec3, fill: string, pts: Vec2[], dz: number): void => {
    list.push({ depth: c[2] + dz, draw() { ctx.fillStyle = fill; blob(pts); ctx.fill(); outline(c[2]); } });
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
  for (const i of A.cheeks) list.push(cheekDecoration(sp(i), 0.07 * scale(sp(i)[2]), Math.atan2(uy, ux)));
  for (const i of A.eyes) list.push(eyeDecoration(sp(i), 0.11 * scale(sp(i)[2]), dizzy, now));
  assert(list.length === A.wings.length + A.feet.length + A.cheeks.length + A.eyes.length + 2, "every part of the goose has a decoration");
  return list;
}

function drawStars(scene: Scene, offsetZ: number): void {
  assert(scene.parts.length > 0 && scene.parts.length % 3 === 0, "the goose has particles in triples");
  assert(Number.isFinite(offsetZ), "the approach offset is finite");
  const P = scene.parts, np = P.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < np; i++) { cx += at(P, 3 * i); cy += at(P, 3 * i + 1); cz += at(P, 3 * i + 2); }
  cx /= np; cy /= np; cz /= np;
  assert(Number.isFinite(cx + cy + cz), "the centroid is finite");
  const D = eng.birdD(), t = reducedMotion.matches ? 0 : performance.now() / 500, o: Vec3 = [0, 0, 0];
  assert(D > 0 && Number.isFinite(t), "the stars orbit a bird of positive size");
  for (let k = 0; k < 3; k++) {
    const a = t + k * 2.094;
    proj(cx + 0.6 * D * Math.cos(a), cy + 0.9 * D, cz + offsetZ + 0.6 * D * Math.sin(a), o, 0);
    if (o[2] <= MIN_DEPTH) continue;
    const r = 0.1 * D * cam.foc / o[2];
    assert(r > 0, "a star has a radius");
    const q: Vec2[] = [];
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

interface HLine { y: number; label: string; color: string }
interface Series { xs: number[]; ys: number[]; color?: string }
interface ChartOptions {
  xmin: number; xmax: number; ymin: number; ymax: number; xticks: number[]; yticks: number[];
  xlabel: string; ylabel: string; log?: boolean; hlines?: HLine[];
}

const chartData = new WeakMap<HTMLCanvasElement, { series: Series[]; opt: ChartOptions }>();
const chartResize = new ResizeObserver((entries) => {
  for (const { target } of entries) {
    if (!(target instanceof HTMLCanvasElement)) continue;
    const data = chartData.get(target);
    if (data) chart(target, data.series, data.opt);
  }
});

function chart(cv: HTMLCanvasElement, series: Series[], opt: ChartOptions): void {
  assert(series.length > 0, "a chart has a series");
  assert(opt.xmax > opt.xmin && opt.ymax > opt.ymin, "the axes have a range");
  assert(opt.xticks.length > 0 && opt.yticks.length > 0, "the axes have ticks");
  assert(!opt.log || opt.ymin > 0, "a log axis starts above zero");
  if (!chartData.has(cv)) chartResize.observe(cv);
  chartData.set(cv, { series, opt });
  const dpr = DPR, r = cv.getBoundingClientRect();
  assert(dpr > 0, "the pixel ratio is positive");
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const c = context2d(cv);
  assert(c.canvas === cv, "the chart draws on its own canvas");
  const css = getComputedStyle(document.body);
  const ink = css.getPropertyValue("--ink"), muted = css.getPropertyValue("--muted"), line = css.getPropertyValue("--line");
  c.scale(dpr, dpr);
  const w = r.width, h = r.height, L = 46, R = 16, T = 8, B = 30;
  const fy = opt.log ? (y: number): number => Math.log10(Math.max(y, opt.ymin)) : (y: number): number => y;
  const y0 = fy(opt.ymin), y1 = fy(opt.ymax);
  const X = (x: number): number => L + (w - L - R) * (x - opt.xmin) / (opt.xmax - opt.xmin);
  const Y = (y: number): number => T + (h - T - B) * (1 - (fy(y) - y0) / (y1 - y0));
  c.font = `11px ${css.getPropertyValue("--mono")}`;
  c.strokeStyle = line; c.fillStyle = muted; c.lineWidth = 1;
  for (const yt of opt.yticks) {
    c.beginPath(); c.moveTo(L, Y(yt)); c.lineTo(w - R, Y(yt)); c.stroke();
    c.textAlign = "right"; c.fillText(String(yt), L - 4, Y(yt) + 4);
  }
  for (const xt of opt.xticks) { c.textAlign = "center"; c.fillText(String(xt), X(xt), h - B + 14); }
  c.textAlign = "left"; c.fillText(opt.ylabel, L + 4, T + 10);
  c.textAlign = "right"; c.fillText(opt.xlabel, w - R, h - 4);
  for (const hl of opt.hlines ?? []) {
    c.strokeStyle = hl.color; c.setLineDash([5, 4]);
    c.beginPath(); c.moveTo(L, Y(hl.y)); c.lineTo(w - R, Y(hl.y)); c.stroke();
    c.setLineDash([]); c.fillStyle = hl.color; c.textAlign = "right"; c.fillText(hl.label, w - R - 2, Y(hl.y) - 4);
  }
  for (const s of series) {
  assert(s.xs.length === s.ys.length, "a series has one y per x");
    c.strokeStyle = s.color || ink; c.lineWidth = 1.6; c.beginPath();
    s.xs.forEach((x, i) => {
      const px = X(Math.min(Math.max(x, opt.xmin), opt.xmax)), py = Y(Math.min(Math.max(at(s.ys, i), opt.ymin), opt.ymax));
      if (i) c.lineTo(px, py); else c.moveTo(px, py);
    });
    c.stroke();
  }
  assert(chartData.get(cv)?.series === series, "the chart data is kept for resizes");
}

function ticks(max: number, n: number): number[] {
  assert(max > 0 && Number.isFinite(max), "the tick range is positive");
  assert(Number.isInteger(n) && n > 0, "the tick count is a whole number");
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  assert(raw > 0 && mag > 0, "the raw step has a magnitude");
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  assert(step >= raw, "the step is at least the raw step");
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
  assert(out.length >= 1 && out[0] === 0, "the ticks start at zero");
  assert(at(out, out.length - 1) <= max + 1e-6, "the ticks stay within the range");
  return out;
}

function drawForceChart(): void {
  assert(canvasOf("forceChart").isConnected, "the force chart canvas exists");
  const xs = st.hist.map((h) => h.t * 1e6), ys = st.hist.map((h) => h.f / 1e3);
  assert(xs.length === ys.length, "there is one force per time");
  const ymax = Math.max(50, ...ys) * 1.1;
  assert(ymax > 0 && Number.isFinite(ymax), "the force axis has a range");
  chart(canvasOf("forceChart"), [{ xs, ys }], {
    xmin: 0, xmax: T_END * 1e6, ymin: 0, ymax, xticks: ticks(T_END * 1e6, 6), yticks: ticks(ymax, 4),
    xlabel: "time (µs)", ylabel: "contact force (kN)",
  });
}

function isRunning(): boolean {
  assert(st.phase.length > 0, "the run has a phase");
  assert(st.scene !== null || st.phase === "idle", "a run has a scene");
  return st.phase === "approach" || st.phase === "sim" || st.phase === "wall";
}

function fire(): void {
  assert(st.speed > 0, "the goose has a speed");
  if (isRunning()) return;
  assert(st.phase === "idle" || st.phase === "done", "a run starts from rest");
  setupRun(PANEL);
  st.phase = "approach";
  assert(st.scene !== null, "set-up built a scene");
  assert(st.mode === PANEL, "the fire button starts a panel run");
  st.t0 = performance.now();
  startGif();
  $("verdict").hidden = true;
  setBusy(true);
  assert(button("stop").disabled === false, "the stop button works during a run");
  const view = canvas.getBoundingClientRect();
  if (view.top < 0 || view.bottom > window.innerHeight) {
    canvas.scrollIntoView({ behavior: "instant", block: "nearest" });
  }
  assert(st.gif !== null, "a GIF recording started with the run");
  honk(Math.sqrt(st.speed / V_C));
  honk(Math.sqrt(st.speed / V_C) * 1.06, 0.3);
}

function setBusy(b: boolean): void {
  assert(typeof b === "boolean", "the busy flag is a boolean");
  assert(button("fire").isConnected && button("stop").isConnected, "the run buttons exist");
  const focusStop = b && (document.activeElement === $("fire") || document.activeElement === $("wilbeck"));
  const focusFire = !b && document.activeElement === $("stop");
  button("fire").disabled = b;
  button("wilbeck").disabled = b;
  button("stop").disabled = !b;
  button("gif").disabled = b || !st.gif;
  $("fire").textContent = b ? "Goose in flight" : "Fire the goose";
  const runName = st.mode === RIGID_WALL ? "Wall check" : "Panel run";
  $("runStatus").textContent = b
    ? `${runName} active. Press Escape or Stop and reset to stop.`
    : st.phase === "done" ? `${runName} complete. Results appear below.` : "";
  for (const control of document.querySelectorAll<HTMLInputElement>("#presets input, #speed, #speedNumber, #skin")) control.disabled = b;
  const snap = document.getElementById("snap");
  if (snap instanceof HTMLButtonElement) snap.disabled = b;
  assert(button("fire").disabled === b, "the fire button follows the busy flag");
  assert(button("stop").disabled === !b, "the stop button follows the busy flag");
  if (focusStop) button("stop").focus({ preventScroll: true });
  if (focusFire) button("fire").focus({ preventScroll: true });
}

function clearWallChart(): void {
  assert(canvasOf("wallChart").isConnected, "the wall chart canvas exists");
  const wallChart = canvasOf("wallChart");
  assert(wallChart.width >= 0 && wallChart.height >= 0, "the wall chart has a size");
  chartResize.unobserve(wallChart);
  chartData.delete(wallChart);
  context2d(wallChart).clearRect(0, 0, wallChart.width, wallChart.height);
  assert(!chartData.has(wallChart), "the wall chart is forgotten");
}

function stopRun(): void {
  assert(button("stop").isConnected, "the stop button exists");
  if (!isRunning()) return;
  assert(st.phase !== "idle" && st.phase !== "done", "only a running run is stopped");
  if (st.mode === RIGID_WALL) {
    st.wall = null;
    clearWallChart();
    $("wilbeckOut").textContent = "Wall check stopped. No result.";
  }
  st.gif = null;
  st.gifHold = false;
  $("bubble").hidden = true;
  preview();
  assert(st.phase === "idle", "the goose is reset to idle");
  assert(st.hist.length === 0, "the force history is cleared");
  setBusy(false);
  assert(!button("fire").disabled, "the fire button works again");
  $("runStatus").textContent = "Run stopped. Goose reset.";
  $("banner").textContent = "Run stopped. Goose reset. Select a speed or resolution, then press Fire the goose.";
  assert(st.gif === null && !st.gifHold, "the recording was dropped");
}

function stepSolver(cpu: CpuEngine, tEnd: number, perStep: PerStep): boolean {
  assert(tEnd > 0, "the run has an end time");
  assert(typeof perStep === "function", "a per-step callback is given");
  const t0 = performance.now(), s0 = cpu.time();
  let steps = 0, t = t0;
  while (t - t0 < BUDGET_MS) {
    const k = cpu.stepPhase(tEnd);
    if (k < 0) break;
    t = performance.now();
    st.solverPhase = k;
    if (k === STRUCTURE_FAIL_PHASE) { steps++; perStep(cpu.time(), cpu.contactForce(), cpu.centerForce()); }
  }
  const ms = t - t0;
  assert(ms >= 0 && steps >= 0, "time and steps run forwards");
  st.msPerStep = steps ? ms / steps : 0;
  st.frameSim = cpu.time() - s0;
  assert(st.msPerStep >= 0 && st.frameSim >= 0, "the step timing is not negative");
  return cpu.time() >= tEnd;
}

function gpuStep(gpu: GpuEngine, tEnd: number, perStep: PerStep): boolean {
  assert(tEnd > 0, "the run has an end time");
  assert(gpu.gpu, "the GPU stepper gets a GPU engine");
  assert(st.batch.steps >= GPU_BATCH_MIN && st.batch.steps <= GPU_BATCH_MAX, "the batch size is in range");
  const b = st.batch, s0 = gpu.time();
  let got = 0, steps = 0, r: number;
  while ((r = gpu.poll()) === 1) got++;
  if (r < 0) throw new Error("the GPU read-back failed");
  assert(got >= 0, "the count of finished batches is not negative");
  if (got) {
    const h = gpu.history();
    steps = h.length / 4 - b.seen;
    for (; b.seen < h.length / 4; b.seen++) perStep(at(h, 4 * b.seen), at(h, 4 * b.seen + 1), at(h, 4 * b.seen + 2));
    if (steps) { st.msPerStep = st.frameReal / steps; st.frameSim = gpu.time() - s0; }
  }
  if (got === 2) b.steps = Math.min(GPU_BATCH_MAX, Math.ceil(b.steps * 1.5));
  if (got === 0 && gpu.pending() === 2) b.steps = Math.max(GPU_BATCH_MIN, Math.floor(b.steps * 0.8));
  assert(b.steps >= GPU_BATCH_MIN && b.steps <= GPU_BATCH_MAX, "the batch size stays in range");
  assert(b.seen >= 0, "the history cursor is not negative");
  if (!gpu.running()) return gpu.pending() === 0;
  while (gpu.run(b.steps, tEnd));
  return false;
}

function solve(tEnd: number, perStep: PerStep): boolean {
  assert(tEnd > 0, "the run has an end time");
  assert(typeof perStep === "function", "a per-step callback is given");
  return eng.gpu ? gpuStep(eng, tEnd, perStep) : stepSolver(eng, tEnd, perStep);
}

function telemetry(): void {
  assert($("tele").isConnected, "the telemetry list exists");
  assert(st.ke0 > 0, "the start energy is positive");
  const tc = eng.tContact();
  assert(!Number.isNaN(tc), "the contact time is a number");
  const rows: [string, string][] = [
    ["sim time", `${fmt(eng.time() * 1e6, 1)} µs`],
    ["since contact", tc >= 0 ? `${fmt((eng.time() - tc) * 1e6, 1)} µs` : "goose airborne"],
    ["time step dt", `${fmt(eng.dt() * 1e9, 1)} ns`],
    ["steps", fmt(eng.steps(), 0)],
    ["slow motion", st.frameSim > 0 ? `1 : ${fmt(st.frameReal / 1000 / st.frameSim, 0)}` : "—"],
    ["solver", `${eng.gpu ? "GPU" : "CPU"}, ${fmt(st.msPerStep, 2)} ms/step`],
    ["frame rate", `${fmt(st.fps, 0)} fps`],
    ["contact force", `${fmt(eng.contactForce() / 1e3, 1)} kN`],
    ["peak force", `${fmt(st.peakF / 1e3, 1)} kN`],
    ["max deflection", `${fmt(eng.maxDeflection() * 1e3, 1)} mm`],
    ["max plastic strain", `${fmt(eng.maxPlasticStrain() * 100, 1)} %`],
    ["fails at (skin / stiffeners)", `${fmt(eng.failStrain(0) * 100, 1)} / ${fmt(eng.failStrain(1) * 100, 1)} %`],
    ["eroded skin shells", fmt(eng.failedSkin(), 0)],
    ["eroded stiffener shells", fmt(eng.failed() - eng.failedSkin(), 0)],
    ["bird", `${fmt(eng.np(), 0)} particles, ${fmt(BIRD_MASS, 3)} kg`],
    ["panel", `${fmt(eng.nn(), 0)} nodes, ${fmt(eng.ne(), 0)} shells`],
    ["skin", `${fmt(st.tSkin * 1e3, 1)} mm 2024-T3`],
  ];
  const tele = $("tele");
  assert(rows.length > 0, "there are telemetry rows");
  if (!tele.children.length) tele.innerHTML = rows.map(([k]) => `<dt>${k}</dt><dd></dd>`).join("");
  assert(tele.children.length === 2 * rows.length, "each row has a term and a definition");
  rows.forEach(([, v], i) => {
    const cell = at(tele.children, 2 * i + 1);
    if (cell.textContent !== v) cell.textContent = v;
  });
  $("honkBar").style.transform = `scaleX(${Math.min(1, eng.contactForce() / 3e5)})`;

  const parts: [string, number, string][] = [
    ["bird kinetic", eng.birdKE(), "#3f6f86"],
    ["bird squish (internal)", eng.birdWork(), "#94b49c"],
    ["panel kinetic", eng.panelKE(), "#e2b54f"],
    ["panel strain + plastic", eng.panelWork(), "#c4613f"],
  ];
  const sum = parts.reduce((s, p) => s + p[1], 0);
  assert(parts.length === 4, "the energy bar has four parts");
  assert(Number.isFinite(sum), "the energy sum is finite");
  $("ebar").innerHTML = parts.map(([, v, c]) => `<div style="width:${Math.max(0, v / st.ke0 * 100)}%;background:${c}"></div>`).join("");
  $("ekey").innerHTML = parts.map(([k, v, c]) => `<li><span><i style="background:${c}"></i>${k}</span><span>${fmt(v / 1e3, 2)} kJ</span></li>`).join("")
    + `<li><span>balance error</span><span>${sign((sum / st.ke0 - 1) * 100, 2)} %</span></li>`;
  assert($("ekey").children.length === parts.length + 1, "the energy key lists each part and the balance");
}

function verdict(): void {
  assert(st.scene !== null, "a run built a scene");
  assert(st.ke0 > 0, "the start energy is positive");
  const v = $("verdict"), pr = preset(st.preset);
  const skin = eng.failedSkin(), stiff = eng.failed() - skin;
  assert(skin >= 0, "the eroded skin count is not negative");
  assert(Number.isFinite(stiff), "the eroded stiffener count is finite");
  const breached = skin > 0;
  v.className = "frame panel verdict " + (breached ? "breached" : "held");
  v.hidden = false;
  v.innerHTML = `
    <p class="stamp">${breached ? "Skin breached" : "Skin held"}</p>
    <h2>${breached ? "The goose tore the skin." : "The skin held. The goose did not."}</h2>
    <p>This run: ${pr.name} at ${fmt(st.speed, 2)} m/s (${fmt(st.speed / KNOT_MS, 1)} kt), ${fmt(pr.spacing * 1e3, 1)} mm bird lattice, ${fmt(st.tSkin * 1e3, 1)} mm skin, on the ${eng.gpu ? "GPU" : "CPU"}.</p>
    <p>Eroded shell elements: ${fmt(skin, 0)} in the 2024-T3 skin, ${fmt(stiff, 0)} in the 7075-T6 stringers and frames.
      An element erodes when its plastic strain reaches the failure strain of its material. Maximum deflection ${fmt(eng.maxDeflection() * 1e3, 1)} mm.
      Peak contact force ${fmt(st.peakF / 1e3, 1)} kN. Maximum plastic strain ${fmt(eng.maxPlasticStrain() * 100, 1)} %.
      The panel took ${fmt(eng.panelWork() / 1e3, 1)} kJ of the goose's ${fmt(st.ke0 / 1e3, 1)} kJ.</p>
    <p>Does this fail 25.571(e)(1)? Not by itself. The rule accepts damage: the aeroplane must complete the flight with it.
      That needs a residual strength analysis of the damaged structure, which this page does not do.
      This is not a certification analysis.</p>
    <p class="hint">Measured at V<sub>C</sub> with the 1.6 mm skin only: maximum deflection ${PRESETS.map((p) => fmt(p.defl, 1)).join(", ")} mm
      and ${PRESETS.map((p) => p.skin).join(", ")} eroded skin elements at ${PRESETS.map((p) => fmt(p.spacing * 1e3, 1)).join(", ")} mm.
      From 9 mm to 4.5 mm, the deflection changes by less than 2 %.${eng.gpu ? " At 3.5 mm, many more skin elements erode, a hole opens in the skin, and the deflection is 50 % larger." : ""}
      The skin erosion does not converge: erosion depends on the element size, so a tear prediction needs a mesh-calibrated failure model.</p>`;
  assert(!v.hidden, "the verdict is shown");
  assert(v.innerHTML.length > 0, "the verdict has text");
}

function wallCheck(): void {
  assert(button("wilbeck").isConnected, "the wall check button exists");
  if (isRunning()) return;
  assert(st.phase === "idle" || st.phase === "done", "a wall check starts from rest");
  setupRun(RIGID_WALL);
  const v = st.speed;
  assert(v > 0, "the goose has a speed");
  const [ph, us] = hugoniot(v);
  assert(ph > 0 && us > v, "the Hugoniot pressure is positive and the shock outruns the bird");
  st.wall = { ph, us, ps: stagnation(v), tl: eng.birdL() / v, area: Math.PI * eng.sensorR() ** 2, xs: [], ys: [], peak: 0, plat: 0, platN: 0 };
  st.phase = "wall";
  assert(st.wall !== null && st.wall.area > 0, "the wall check has a sensor");
  assert(st.wall.tl > 0, "the bird has a length in time");
  clearWallChart();
  $("wilbeckOut").textContent = "Wall check active. No result yet.";
  startGif();
  $("verdict").hidden = true;
  $("wallChart").scrollIntoView({ behavior: reducedMotion.matches ? "instant" : "smooth", block: "center" });
  setBusy(true);
  assert(!button("stop").disabled, "the stop button works during a run");
  assert(st.mode === RIGID_WALL, "the wall check runs in the wall mode");
  honk(0.8);
}

function wallSample(time: number, f: number, fc: number): void {
  assert(time >= 0, "time runs forwards");
  assert(st.phase === "wall", "samples arrive during the wall check");
  assert(!Number.isNaN(f + fc), "the forces are numbers");
  st.peakF = Math.max(st.peakF, f);
  const w = must(st.wall, "the wall check"), tc = eng.tContact(), t = time - tc;
  assert(w.area > 0, "the sensor has an area");
  if (tc < 0 || t < 0) return;
  const p = fc / w.area;
  w.xs.push(t * 1e6); w.ys.push(p / 1e6);
  assert(w.xs.length === w.ys.length, "a pressure was kept for each time");
  if (p > w.peak) w.peak = p;
  if (t > 0.25 * w.tl && t < 0.75 * w.tl) { w.plat += p; w.platN++; }
  if (!st.splatted) { st.splatted = true; splat(); }
}

function drawWallChart(): void {
  assert(st.phase === "wall", "the wall chart is drawn during the wall check");
  const w = must(st.wall, "the wall check");
  assert(w.tl > 0 && w.area > 0, "the wall check has a duration and a sensor");
  assert(w.xs.length === w.ys.length, "the series has one y per x");
  chart(canvasOf("wallChart"), [{ xs: w.xs, ys: w.ys, color: "#3f6f86" }], {
    xmin: 0, xmax: WALL_RUN_LENGTHS * w.tl * 1e6, ymin: 1, ymax: 1000, log: true,
    xticks: ticks(WALL_RUN_LENGTHS * w.tl * 1e6, 6), yticks: [1, 10, 100, 1000],
    xlabel: "time since contact (µs)", ylabel: "centre pressure (MPa, log)",
    hlines: [{ y: w.ph / 1e6, label: `P_H ${fmt(w.ph / 1e6, 1)} MPa`, color: "#a8432a" },
             { y: w.ps / 1e6, label: `P_s ${fmt(w.ps / 1e6, 2)} MPa`, color: "#4f7a4a" }],
  });
  const pm = w.platN ? w.plat / w.platN : 0;
  assert(w.ps > 0 && w.ph > 0, "the reference pressures are positive");
  assert(w.platN >= 0, "the plateau count is not negative");
  $("wilbeckOut").textContent =
    `v = ${fmt(st.speed, 2)} m/s, bird lattice ${fmt(preset(st.preset).spacing * 1e3, 1)} mm, ${fmt(eng.np(), 0)} particles, sensor r = ${fmt(eng.sensorR() * 1e3, 2)} mm\n` +
    `Hugoniot  P_H = ${fmt(w.ph / 1e6, 1)} MPa (U_s = ${fmt(w.us, 0)} m/s)   peak    ${fmt(w.peak / 1e6, 1)} MPa   error ${sign((w.peak - w.ph) / w.ph * 100)} %\n` +
    `Stagnation P_s = ${fmt(w.ps / 1e6, 2)} MPa                plateau ${fmt(pm / 1e6, 2)} MPa   error ${w.platN ? sign((pm - w.ps) / w.ps * 100) : "—"} %\n` +
    `impulse / (m v) = ${fmt(eng.impulse() / (BIRD_MASS * st.speed), 4)}   (plateau: mean over 0.25 to 0.75 of L/v = ${fmt(w.tl * 1e6, 0)} µs)`;
}

let lastFrame = performance.now();
function frame(now: number): void {
  assert(Number.isFinite(now), "the frame time is finite");
  assert(now >= 0, "the frame time is not negative");
  try {
    drawFrame(now);
  } catch (error) {
    console.error(error);
    stopRun();
    $("banner").textContent = "The solver failed. Run stopped. Reload the page to try again.";
  }
  assert(typeof requestAnimationFrame === "function", "the page can ask for another frame");
  requestAnimationFrame(frame);
}

function finishRun(): void {
  assert(isRunning(), "only a running run can finish");
  st.phase = "done"; st.solverPhase = -1; st.gifHold = true; setBusy(false);
  assert(st.phase === "done" && st.solverPhase === -1, "the run is finished");
  assert(!button("fire").disabled, "the fire button works again");
}

function drawFrame(now: number): void {
  assert(Number.isFinite(now), "the frame time is finite");
  assert(typeof st.phase === "string", "the run has a phase");
  st.frameReal = now - lastFrame;
  st.fps += 0.05 * (1000 / Math.max(st.frameReal, 1) - st.fps);
  assert(st.fps > 0 && Number.isFinite(st.fps), "the frame rate is positive");
  lastFrame = now;
  let offsetZ = 0, flap = 0;
  const banner = $("banner");
  let bannerText: string | undefined;
  if (st.phase === "approach") {
    const s = Math.min(1, (now - st.t0) / 1000 / APPROACH_S);
    offsetZ = APPROACH_M * (1 - s);
    flap = reducedMotion.matches ? 0 : Math.sin(now / 70);
    bannerText = `APPROACH: cartoon, not simulated (the real ${fmt(APPROACH_M, 1)} m takes ${fmt(APPROACH_M / st.speed * 1e3, 1)} ms at ${fmt(st.speed, 1)} m/s)`;
    if (s >= 1) st.phase = "sim";
  } else if (st.phase === "sim") {
    const done = solve(T_END, (t, f) => {
      st.peakF = Math.max(st.peakF, f);
      st.hist.push({ t, f });
    });
    refreshScene();
    if (eng.tContact() >= 0 && !st.splatted) {
      st.splatted = true;
      splat(); honk(1.35); honk(1.5, 0.18);
      const b = $("bubble"); b.hidden = false; b.classList.remove("out"); b.style.animation = "none"; void b.offsetWidth; b.style.animation = "";
      setTimeout(() => b.classList.add("out"), 700);
      setTimeout(() => { b.hidden = true; }, 860);
    }
    bannerText = `SIMULATING ${eng.gpu ? "on the GPU (WebGPU)" : "in Goose → WebAssembly"} · t = ${fmt(eng.time() * 1e6, 0)} µs of ${fmt(T_END * 1e6, 0)} µs · goose is now a fluid`;
    telemetry();
    drawForceChart();
    if (done) { finishRun(); verdict(); honk(0.7, 0.1); }
  } else if (st.phase === "wall") {
    const done = solve(WALL_RUN_LENGTHS * must(st.wall, "the wall check").tl, wallSample);
    refreshScene();
    telemetry();
    bannerText = `CONTROL EXPERIMENT: goose vs. rigid wall · ${fmt(eng.time() * 1e6, 0)} µs · colour = pressure`;
    drawWallChart();
    if (done) finishRun();
  } else if (st.phase === "done") {
    bannerText = "Done. Drag to orbit. Scroll to zoom. Fire again for another goose.";
  }
  assert(offsetZ >= 0 && Number.isFinite(offsetZ), "the goose has not passed the panel");
  assert(Math.abs(flap) <= 1, "the wing flap is between -1 and 1");
  if (bannerText !== undefined && banner.textContent !== bannerText) banner.textContent = bannerText;
  if (st.scene) render(st.scene, offsetZ, flap);
  captureGif(now);
  codeView(now);
}

function refreshScene(): void {
  assert(st.phase !== "idle", "the scene only refreshes during a run");
  if (st.scene) Object.assign(st.scene, readRender());
  assert(st.scene === null || st.scene.press.length * 3 === st.scene.parts.length, "the scene has a pressure per particle");
}

function legend(): void {
  assert($("legend").isConnected, "the legend exists");
  const showP = !st.costume || st.mode === RIGID_WALL;
  $("legend").innerHTML =
    (showP ? `bird pressure (log)<div class="ramp" style="background:${cssRamp("pressure")}"></div><span class="ticks"><span>0.1</span><span>6.3</span><span>400 MPa</span></span>` : "") +
    (st.mode === PANEL ? `panel plastic strain (square-root scale)<div class="ramp" style="background:${cssRamp("metal")}"></div><span class="ticks">${[0, 1, 2, 3].map((k) => `<span>${fmt(EP_MAX * 100 * k * k / 9, 0)}${k === 3 ? " %" : ""}</span>`).join("")}</span>` : "");
  assert($("legend").innerHTML.length > 0, "the legend has a colour scale");
}

function startGif(): void {
  assert(W > 0 && H > 0, "the canvas has a size");
  const h = Math.round(GIF_W * H / W) & ~1;
  assert(h > 0 && h % 2 === 0, "the GIF height is even and positive");
  const cv = document.createElement("canvas");
  cv.width = GIF_W; cv.height = h + GIF_CAPTION_PX;
  assert(cv.height > GIF_CAPTION_PX, "the GIF has room for the picture");
  st.gif = {
    rec: new GifRecorder(GIF_W, h + GIF_CAPTION_PX), h, cv, next: 0, simNext: 0, speed: st.speed, done: false,
    title: st.mode === PANEL ? preset(st.preset).name : "Rigid wall check",
  };
  st.gifHold = false;
  assert(st.gif !== null && st.gif.rec.frames.length === 0, "the recorder starts empty");
  button("gif").disabled = true;
  $("gif").textContent = "Export GIF";
}

function captureGif(now: number): void {
  assert(Number.isFinite(now), "the frame time is finite");
  const g = st.gif;
  assert(g === null || g.h > 0, "the GIF picture has a height");
  if (!g || g.done) return;
  let delay: number;
  if (st.phase === "approach") {
    if (now < g.next) return;
    g.next = now + GIF_APPROACH_PERIOD_MS; delay = GIF_APPROACH_DELAY_CS;
  } else if (st.phase === "sim" || st.phase === "wall") {
    const tUs = eng.time() * 1e6;
    if (tUs < g.simNext) return;
    g.simNext = (Math.floor(tUs / GIF_SIM_US) + 1) * GIF_SIM_US; delay = GIF_SIM_DELAY_CS;
  } else if (st.gifHold) {
    g.done = true; delay = GIF_LAST_DELAY_CS;
  } else return;
  const c = context2d(g.cv);
  assert(g.cv.width === GIF_W && g.cv.height === g.h + GIF_CAPTION_PX, "the GIF canvas has the frame size");
  assert(delay > 0, "the frame has a delay");
  const css = getComputedStyle(document.body);
  c.fillStyle = css.getPropertyValue("--stage");
  c.fillRect(0, 0, GIF_W, g.h);
  const scale = Math.min(GIF_W / canvas.width, g.h / canvas.height);
  assert(scale > 0, "the picture is scaled into the frame");
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
  const pressure = !st.costume || st.mode === RIGID_WALL, name: MapName = pressure ? "pressure" : "metal";
  const x0 = GIF_W - 220;
  for (let k = 0; k < 90; k++) { c.fillStyle = mapColour(name, k / 89); c.fillRect(x0 + k, y - 5, 1, 10); }
  c.fillStyle = css.getPropertyValue("--ink");
  c.textAlign = "right"; c.fillText(pressure ? "0.1" : "0", x0 - 4, y);
  c.textAlign = "left"; c.fillText(pressure ? "400 MPa, log" : `${fmt(EP_MAX * 100, 0)} % plastic, √`, x0 + 94, y);
  c.textAlign = "left";
  g.rec.add(c.getImageData(0, 0, GIF_W, g.h + GIF_CAPTION_PX), delay);
}

async function exportGif(): Promise<void> {
  assert(button("gif").isConnected, "the GIF button exists");
  const g = st.gif;
  assert(g === null || g.rec.w === GIF_W, "the recorder has the GIF width");
  if (!g || !g.rec.frames.length) return;
  assert(g.rec.frames.length > 0, "there are frames to export");
  const b = button("gif");
  b.disabled = true; b.textContent = "Making the GIF…";
  await new Promise((r) => setTimeout(r, 30));
  try {
    const bytes = g.rec.encode();
    assert(bytes.length > 0, "the GIF has bytes");
    const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type: "image/gif" }));
    setTimeout(() => URL.revokeObjectURL(url), 10000);
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

interface CodeView {
  lines: HTMLCollection | null; step: { first: number; lines: HTMLCollection } | null; ranges: Range[][];
  shown: number; last: number; scrolled: number; scrolledK: number; chips: HTMLButtonElement[];
}
const code: CodeView = { lines: null, step: null, ranges: [], shown: -2, last: 0, scrolled: 0, scrolledK: -1, chips: [] };

async function loadCode(): Promise<void> {
  assert($("code").isConnected && $("step").isConnected, "the code panes exist");
  const src = await (await fetch(SOURCE_URL)).text();
  assert(src.length > 0, "the source file is not empty");
  const lines = src.replace(/\n$/, "").split("\n");
  assert(lines.length > 0, "the source has lines");
  const fnRange = (name: string): Range | null => {
    let a = lines.findIndex((l) => new RegExp(`^(export )?fn ${name}\\(`).test(l));
    if (a < 0) return null;
    let b = a;
    while (b < lines.length && lines[b] !== "}") b++;
    while (a > 0 && at(lines, a - 1).startsWith("//")) a--;
    return [a, b];
  };
  const [r0, r1] = must(fnRange("run_phase"), "run_phase in strike.goose");
  assert(r0 < r1 && r1 < lines.length, "run_phase spans lines of the source");
  code.ranges = PHASES.map((ph, k) => {
    const a = lines.findIndex((l, i) => i > r0 && i < r1 && (l.includes(`k == ${k} {`) || (k === 4 && l.trim() === "} else {")));
    let b = a + 1;
    while (b < r1 && !at(lines, b).trim().startsWith("}")) b++;
    const functions = ph.fns.map(fnRange).filter((r): r is Range => r !== null);
    return [[a, b - 1], ...functions];
  });
  const esc = (t: string): string => t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  assert(code.ranges.length === PHASES.length, "every phase has line ranges");
  const kw = /\b(fn|let|var|if|else|for|in|while|return|export|import|continue)\b/g;
  const html = (i: number): string => {
    const l = at(lines, i), c = l.indexOf("//");
    const body = c < 0 ? l : l.slice(0, c), note = c < 0 ? "" : l.slice(c);
    return `<span class="ln" data-n="${i + 1}">${esc(body).replace(kw, "<b>$1</b>")}${note ? `<i>${esc(note)}</i>` : ""}\n</span>`;
  };
  $("step").innerHTML = lines.slice(r0, r1 + 1).map((_, j) => html(r0 + j)).join("");
  $("code").innerHTML = lines.map((_, i) => html(i)).join("");
  code.step = { first: r0, lines: $("step").children };
  code.lines = $("code").children;
  $("phases").innerHTML = PHASES.map((ph, k) => `<li><button class="button" data-k="${k}"><span>${k}</span> ${ph.name}</button></li>`).join("");
  code.chips = [...$("phases").querySelectorAll("button")];
  for (const b of code.chips) b.onclick = () => scrollCode(Number(b.dataset["k"]));
  assert(code.chips.length === PHASES.length, "every phase has a chip");
  assert(code.lines !== null && code.step !== null, "the code lines were read");
}

function scrollCode(k: number): void {
  assert(Number.isInteger(k) && k >= 0 && k < PHASES.length, "k is a phase");
  assert($("code").isConnected, "the code pane exists");
  const ranges = at(code.ranges, k);
  assert(ranges.length >= 1, "a phase has at least its step range");
  const first = ranges[1]?.[0] ?? at(ranges, 0)[0];
  const el = at(must(code.lines, "the code lines"), first) as HTMLElement;
  $("code").scrollTo({ top: el.offsetTop - $("code").offsetTop - 8, behavior: reducedMotion.matches ? "instant" : "smooth" });
}

function codeView(now: number): void {
  assert(Number.isFinite(now), "the frame time is finite");
  if (!code.lines || now - code.last < 120) return;
  code.last = now;
  const running = st.phase === "sim" || st.phase === "wall";
  const k = running ? st.solverPhase : -1;
  assert(k >= -1 && k < PHASES.length, "the phase is known or none");
  const focused = document.activeElement;
  const sourceFocused = focused === $("code") || focused === $("step") || code.chips.some((c) => c === focused);
  if (!sourceFocused && k >= 0 && k !== code.scrolledK && now - code.scrolled > 1000) {
    scrollCode(k); code.scrolled = now; code.scrolledK = k;
  }
  const state = k >= 0 ? `Phase ${k} (${at(PHASES, k).name}) ran last.`
    : running && eng.gpu ? "On the GPU, the kernels of web/strike.wgsl do all five phases of each step. No line is lit." : "The solver is idle.";
  if ($("codeState").textContent !== state) $("codeState").textContent = state;
  if (k === code.shown) return;
  assert(code.chips.length === PHASES.length, "every phase has a chip");
  code.chips.forEach((b, i) => { b.dataset["active"] = String(i === k); });
  for (const l of $("code").querySelectorAll(".on")) l.classList.remove("on");
  for (const l of $("step").querySelectorAll(".on")) l.classList.remove("on");
  if (k >= 0) {
    const stepPane = must(code.step, "the step pane");
    for (const [a, b] of at(code.ranges, k)) for (let i = a; i <= b; i++) at(code.lines, i).classList.add("on");
    const [a, b] = at(at(code.ranges, k), 0);
    for (let i = a; i <= b; i++) at(stepPane.lines, i - stepPane.first).classList.add("on");
  }
  code.shown = k;
}

function showSpeed(): void {
  assert(st.speed > 0 && Number.isFinite(st.speed), "the speed is positive");
  const v = st.speed;
  input("speedNumber").value = v.toFixed(2);
  assert(input("speedNumber").value === v.toFixed(2), "the number box shows the speed");
  $("speedEntryStatus").textContent = "";
  $("speedOut").textContent = `${fmt(v, 2)} m/s = ${fmt(v / KNOT_MS, 1)} kt`;
  const on = Math.abs(v - V_C) < 0.005;
  const sc = $("specCheck");
  assert(sc.isConnected, "the spec note exists");
  sc.className = "hint " + (on ? "ok" : "off");
  sc.innerHTML = on
    ? "On spec: 4 lb at V<sub>C</sub> = 350 KCAS, sea level. 14 CFR 25.571(e)(1)."
    : `Off spec: the rule asks for V<sub>C</sub> = ${fmt(V_C, 2)} m/s. <button type="button" id="snap" class="snap">Snap to V<sub>C</sub></button>. Science continues anyway.`;
  const snap = document.getElementById("snap");
  if (snap instanceof HTMLButtonElement) snap.onclick = () => {
    const focusSpeed = document.activeElement === snap;
    st.speed = V_C; input("speed").value = String(V_C); showSpeed();
    if (focusSpeed) input("speedNumber").focus({ preventScroll: true });
    if (st.phase === "idle") preview();
  };
}

function buildPresets(): void {
  assert(PRESETS.length > 0, "there is a preset to build");
  assert(st.preset >= 0 && st.preset < PRESETS.length, "the chosen preset exists");
  $("presets").innerHTML = PRESETS.map((p, i) => `
    <label class="preset"><input type="radio" name="preset" value="${i}" ${i === st.preset ? "checked" : ""}>
      <b>${p.name}</b> · ${fmt(p.spacing * 1e3, 1)} mm bird, ${p.cells} skin cells
      <small>wall check at V<sub>C</sub> only: peak ${sign(p.peak)} %, plateau ${sign(p.plateau)} % · solver about ${eng.gpu ? p.gpuRun : p.run}</small></label>`).join("");
  for (const el of document.querySelectorAll<HTMLInputElement>("#presets input")) {
    el.onchange = () => { st.preset = Number(el.value); preview(); };
  }
  assert(document.querySelectorAll("#presets input").length === PRESETS.length, "there is an input per preset");
}

function preview(): void {
  assert(PRESETS.length > 0, "there is a preset to show");
  setupRun(PANEL);
  assert(st.scene !== null, "set-up built a scene");
  assert(st.mode === PANEL, "the preview is a panel");
  st.phase = "idle";
  $("runStatus").textContent = "";
  $("verdict").hidden = true;
  $("banner").textContent = "Goose loaded. Goose is unaware. Press Fire the goose.";
  assert(st.phase === "idle", "the page is idle after a preview");
}

function showSkin(): void {
  assert(st.tSkin > 0 && st.tSkin < 0.01, "the skin thickness is in metres");
  const t = st.tSkin * 1e3;
  $("skinOut").textContent = `${fmt(t, 1)} mm = ${fmt(t / 25.4, 3)} in`;
  assert($("skinOut").textContent !== "", "the skin thickness is shown");
}

function setPressed(b: HTMLElement, on: boolean): void {
  assert(b.isConnected, "the toggle is in the page");
  b.setAttribute("aria-pressed", String(on));
  b.dataset["active"] = String(on);
  assert(b.getAttribute("aria-pressed") === b.dataset["active"], "the toggle states agree");
}

function showMute(): void {
  assert($("mute").isConnected, "the mute button exists");
  setPressed($("mute"), st.muted);
  $("mute").textContent = st.muted ? "🔇 Honks off" : "🔊 Honks on";
  assert($("mute").textContent !== "", "the mute button has a label");
}

interface Drag { pointerId: number; x: number; y: number }
let drag: Drag | null = null;

function resetCamera(): void {
  assert(canvas.isConnected, "the view canvas is in the page");
  drag = null;
  st.yaw = DEFAULT_YAW; st.pitch = DEFAULT_PITCH; st.dist = DEFAULT_DIST; st.pov = false;
  setPressed($("btnPov"), false);
  showZoom();
  assert(st.dist === DEFAULT_DIST && st.pitch === DEFAULT_PITCH && st.yaw === DEFAULT_YAW, "the camera is back at its default");
  assert(!st.pov && drag === null, "the camera is outside the goose and not dragged");
}

function showZoom(): void {
  assert(st.dist >= DIST_MIN && st.dist <= DIST_MAX, "the camera distance is in range");
  const focused = document.activeElement;
  const zoomIn = button("btnZoomIn"), zoomOut = button("btnZoomOut");
  assert(zoomIn !== zoomOut, "the zoom buttons are different");
  canvas.style.touchAction = st.pov ? "pan-y pinch-zoom" : "pinch-zoom";
  zoomIn.disabled = st.pov || st.dist <= DIST_MIN;
  zoomOut.disabled = st.pov || st.dist >= DIST_MAX;
  assert(canvas.style.touchAction.length > 0, "the canvas has a touch action");
  if ((focused === zoomIn || focused === zoomOut) && (focused as HTMLButtonElement).disabled) {
    canvas.focus({ preventScroll: true });
  }
  assert(zoomIn.disabled === (st.pov || st.dist <= DIST_MIN), "zoom in follows the limit");
  assert(zoomOut.disabled === (st.pov || st.dist >= DIST_MAX), "zoom out follows the limit");
}

function zoom(factor: number): void {
  assert(factor > 0, "a zoom factor is positive");
  if (st.pov) return;
  st.dist = Math.max(DIST_MIN, Math.min(DIST_MAX, st.dist * factor));
  showZoom();
  assert(st.dist >= DIST_MIN && st.dist <= DIST_MAX, "the camera distance stays in range");
}

function rotate(dx: number, dy: number): void {
  assert(Number.isFinite(dx + dy), "a rotation is finite");
  if (st.pov) return;
  st.yaw += dx;
  st.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, st.pitch + dy));
  assert(Math.abs(st.pitch) <= PITCH_LIMIT, "the camera pitch stays in range");
}

function onSpeedEntry(): void {
  assert(input("speedNumber").isConnected, "the speed box exists");
  const entry = input("speedNumber");
  if (!entry.checkValidity()) {
    entry.value = st.speed.toFixed(2);
    $("speedEntryStatus").textContent = `Enter a speed from ${entry.min} to ${entry.max} m/s in steps of ${entry.step} m/s. The control kept the previous speed.`;
    return;
  }
  st.speed = entry.valueAsNumber;
  assert(st.speed >= Number(entry.min) && st.speed <= Number(entry.max), "the entered speed is in range");
  assert(st.speed > 0, "the speed is positive");
  input("speed").value = String(st.speed);
  showSpeed();
  if (st.phase === "idle") preview();
}

function wireSliders(): void {
  assert(input("speed").isConnected && input("skin").isConnected, "the sliders exist");
  input("speed").value = String(V_C);
  for (const attr of ["min", "max", "step"] as const) input("speedNumber")[attr] = input("speed")[attr];
  input("speedNumber").onchange = onSpeedEntry;
  input("speed").oninput = () => { st.speed = Number(input("speed").value); showSpeed(); };
  input("speed").onchange = () => { if (st.phase === "idle") preview(); };
  input("skin").value = String(T_SKIN * 1e3);
  input("skin").oninput = () => { st.tSkin = Number(input("skin").value) / 1e3; showSkin(); };
  input("skin").onchange = () => { if (st.phase === "idle") preview(); };
  assert(input("speedNumber").max === input("speed").max, "the number box has the slider range");
  assert(typeof input("skin").oninput === "function", "the skin slider has a handler");
}

function wireButtons(): void {
  assert(button("fire").isConnected && button("stop").isConnected, "the run buttons exist");
  button("fire").onclick = fire;
  button("stop").onclick = stopRun;
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || button("stop").disabled) return;
    e.preventDefault();
    assert(e.key === "Escape", "only Escape stops a run");
    stopRun();
  });
  button("wilbeck").onclick = wallCheck;
  button("gif").onclick = () => { void exportGif(); };
  button("mute").onclick = () => {
    st.muted = !st.muted;
    assert(typeof st.muted === "boolean", "the mute flag stays a boolean");
    try { localStorage.setItem(MUTED_KEY, st.muted ? "1" : "0"); } catch { showMute(); }
    showMute();
  };
  button("btnCostume").onclick = () => {
    st.costume = !st.costume;
    assert(typeof st.costume === "boolean", "the costume flag stays a boolean");
    setPressed($("btnCostume"), st.costume);
    legend();
  };
  button("btnPov").onclick = () => { drag = null; st.pov = !st.pov; setPressed($("btnPov"), st.pov); showZoom(); };
  button("btnReset").onclick = resetCamera;
  button("btnZoomIn").onclick = () => zoom(1 / 1.2);
  button("btnZoomOut").onclick = () => zoom(1.2);
  window.addEventListener("blur", () => { drag = null; });
  assert(typeof button("fire").onclick === "function", "the fire button has a handler");
  assert(typeof button("btnZoomOut").onclick === "function", "the zoom out button has a handler");
}

const TURNS: Record<string, Vec2> = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] };

function onCanvasKey(e: KeyboardEvent): void {
  assert(typeof e.key === "string", "a key event names its key");
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === "Home") {
    e.preventDefault();
    resetCamera();
    return;
  }
  if (st.pov) return;
  if (e.key === "+" || e.key === "-") {
    e.preventDefault();
    zoom(e.key === "+" ? 1 / 1.2 : 1.2);
    return;
  }
  const turn = TURNS[e.key];
  if (!turn) return;
  assert(turn.length === 2, "a turn has a yaw and a pitch step");
  e.preventDefault();
  rotate(turn[0], turn[1]);
}

function wireCanvas(): void {
  assert(canvas.isConnected, "the view canvas is in the page");
  assert(drag === null, "no drag is in progress at start");
  canvas.addEventListener("keydown", onCanvasKey);
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || drag || st.pov) return;
    drag = { pointerId: e.pointerId, x: e.clientX, y: e.clientY };
    assert(drag.pointerId === e.pointerId, "the drag follows the pointer");
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || drag.pointerId !== e.pointerId || st.pov) return;
    rotate((e.clientX - drag.x) * 0.008, (e.clientY - drag.y) * 0.008);
    assert(drag.pointerId === e.pointerId, "the drag follows the pointer");
    drag.x = e.clientX; drag.y = e.clientY;
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
    canvas.addEventListener(event, (e) => { if (drag?.pointerId === e.pointerId) drag = null; });
  }
  canvas.addEventListener("wheel", (e) => {
    if (st.pov || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    const unit = e.deltaMode === 1 ? WHEEL_LINE_PX : e.deltaMode === 2 ? canvas.clientHeight : 1;
    assert(unit > 0, "a wheel unit is positive");
    zoom(Math.exp(e.deltaY * unit * 0.001));
  }, { passive: false });
  assert(typeof canvas.setPointerCapture === "function", "the canvas can capture the pointer");
}

function main(): void {
  assert(PRESETS.length > 0, "there is a preset to start with");
  wireSliders();
  wireButtons();
  wireCanvas();
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
  assert(st.phase === "idle", "the page starts idle");
  assert(st.scene !== null, "the page starts with a scene");
  requestAnimationFrame(frame);
}

main();
