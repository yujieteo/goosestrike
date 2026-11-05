import createModule, { type SolverModule } from "./strike.mjs";
import type { GpuSolverModule } from "./strike-gpu.mjs";
import { assert } from "./checked.ts";

export const V_C = 350 * 1852 / 3600;
export const BIRD_MASS = 4 * 0.45359237;
const RHO_BIRD = 950.0;
const C0_BIRD = 1480.0;
const GAMMA_TAIT = 7.0;
export const T_SKIN = 0.0016;

export type SolverMode = 0 | 1;
export const PANEL: SolverMode = 0;
export const RIGID_WALL: SolverMode = 1;

const BYTES_PER_I64 = 8;
const SLICE_BYTES = 12;
const WGSL_URL = "strike.wgsl";

interface Engine {
  bytes: (fn: (lenPtr: number) => number) => ArrayBuffer;
  setSkin: (thickness: number) => void;
  time: () => number;
  dt: () => number;
  steps: () => number;
  tContact: () => number;
  np: () => number;
  nn: () => number;
  ne: () => number;
  particleMass: () => number;
  spacing: () => number;
  smoothing: () => number;
  birdD: () => number;
  birdL: () => number;
  birdV: () => number;
  contactForce: () => number;
  centerForce: () => number;
  sensorR: () => number;
  impulse: () => number;
  panelWork: () => number;
  hourglassWork: () => number;
  birdWork: () => number;
  failed: () => number;
  failedSkin: () => number;
  maxPlasticStrain: () => number;
  maxDeflection: () => number;
  failStrain: (material: number) => number;
  birdKE: () => number;
  panelKE: () => number;
  panelMass: () => number;
  fillRender: () => void;
  nodes: () => Float32Array;
  particles: () => Float32Array;
  pressure: () => Float32Array;
  strain: () => Float32Array;
  elements: () => Int32Array;
}

export interface CpuEngine extends Engine {
  gpu: false;
  setup: (mode: SolverMode, cells: number, spacing: number, speed: number) => number;
  advance: (steps: number, tEnd: number) => number;
  stepPhase: (tEnd: number) => number;
}

export interface GpuEngine extends Engine {
  gpu: true;
  errors: string[];
  setup: (mode: SolverMode, cells: number, spacing: number, speed: number) => number;
  run: (steps: number, tEnd: number) => number;
  poll: () => number;
  pending: () => number;
  running: () => boolean;
  history: () => Float32Array;
  gpuObject: GPU;
}

export type AnyEngine = CpuEngine | GpuEngine;

type Scalar<M> = { [K in keyof M]: M[K] extends () => number ? K : never }[keyof M];

function scalar<M>(m: M, name: Scalar<M>): () => number {
  assert(typeof m === "object" && m !== null, "the solver module is an object");
  assert(typeof m[name] === "function", `${String(name)} is a solver export`);
  return m[name] as () => number;
}

type Bytes = (fn: (lenPtr: number) => number) => ArrayBuffer;

function floats(bytes: Bytes, fn: (lenPtr: number) => number, n: number): Float32Array {
  assert(Number.isInteger(n) && n >= 0, "a buffer holds a whole count of floats");
  const out = new Float32Array(bytes(fn), 0, n);
  assert(out.length === n, "the float view has the requested length");
  assert(out.byteLength === 4 * n, "the float view covers its bytes");
  return out;
}

function ints(bytes: Bytes, fn: (lenPtr: number) => number, n: number): Int32Array {
  assert(Number.isInteger(n) && n >= 0, "a buffer holds a whole count of integers");
  const out = new Int32Array(bytes(fn), 0, n);
  assert(out.length === n, "the integer view has the requested length");
  assert(out.byteLength === 4 * n, "the integer view covers its bytes");
  return out;
}

function common(m: SolverModule): Engine {
  const lenPtr = m._malloc(BYTES_PER_I64);
  assert(lenPtr > 0, "the length slot was allocated");
  assert(m.HEAPU8.length > lenPtr, "the length slot lies inside the heap");

  function bytes(fn: (lenPtr: number) => number): ArrayBuffer {
    const ptr = fn(lenPtr);
    const len = Number(new BigInt64Array(m.HEAPU8.buffer, lenPtr, 1)[0]);
    assert(ptr >= 0 && len >= 0, "the buffer has a start and a length");
    assert(ptr + len <= m.HEAPU8.length, "the buffer lies inside the heap");
    return m.HEAPU8.buffer.slice(ptr, ptr + len) as ArrayBuffer;
  }

  const np = scalar(m, "_get_np"), nn = scalar(m, "_get_nn"), ne = scalar(m, "_get_ne");
  assert(typeof m._malloc === "function" && typeof m._free === "function", "the module has an allocator");
  assert(np !== nn && nn !== ne, "the counts are separate exports");
  return {
    bytes,
    setSkin: (t) => { assert(t > 0, "the skin has a thickness"); m._set_skin(t); },
    time: scalar(m, "_get_time"),
    dt: scalar(m, "_get_dt"),
    steps: scalar(m, "_get_steps"),
    tContact: scalar(m, "_get_t_contact"),
    np,
    nn,
    ne,
    particleMass: scalar(m, "_get_particle_mass"),
    spacing: scalar(m, "_get_spacing"),
    smoothing: scalar(m, "_get_smoothing"),
    birdD: scalar(m, "_get_bird_d"),
    birdL: scalar(m, "_get_bird_l"),
    birdV: scalar(m, "_get_bird_v"),
    contactForce: scalar(m, "_get_contact_force"),
    centerForce: scalar(m, "_get_center_force"),
    sensorR: scalar(m, "_get_sensor_r"),
    impulse: scalar(m, "_get_impulse"),
    panelWork: scalar(m, "_get_panel_work"),
    hourglassWork: scalar(m, "_get_hg_work"),
    birdWork: scalar(m, "_get_bird_work"),
    failed: scalar(m, "_get_failed"),
    failedSkin: scalar(m, "_get_failed_skin"),
    maxPlasticStrain: scalar(m, "_get_max_ep"),
    maxDeflection: scalar(m, "_get_max_defl"),
    failStrain: (mat) => { assert(mat === 0 || mat === 1, "the material is skin or stiffener"); return m._get_eps_fail(mat); },
    birdKE: scalar(m, "_get_bird_ke"),
    panelKE: scalar(m, "_get_panel_ke"),
    panelMass: scalar(m, "_get_panel_mass"),
    fillRender: () => { void m._fill_render(); },
    nodes: () => floats(bytes, m._buf_nodes, 3 * nn()),
    particles: () => floats(bytes, m._buf_particles, 3 * np()),
    pressure: () => floats(bytes, m._buf_pressure, np()),
    strain: () => floats(bytes, m._buf_strain, ne()),
    elements: () => ints(bytes, m._buf_elements, 4 * ne()),
  };
}

export async function loadEngine(): Promise<CpuEngine> {
  const m = await createModule();
  m._goose_init(0, 0);
  assert(m.HEAPU8.length > 0, "the module has a heap");
  assert(typeof m._set_skin === "function", "the module exports set_skin");
  assert(typeof m._advance === "function", "the module exports advance");
  assert(typeof m._setup === "function", "the module exports setup");
  assert(typeof m._step_phase === "function", "the module exports step_phase");
  return {
    ...common(m),
    gpu: false,
    setup: m._setup,
    advance: m._advance,
    stepPhase: m._step_phase,
  };
}

async function text(name: string): Promise<string> {
  assert(name.length > 0, "a file name is given");
  const url = new URL(name, import.meta.url);
  assert(url.pathname.endsWith(name), "the file is next to this module");
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return readFile(url, "utf8");
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  const body = await r.text();
  assert(body.length > 0, `${name} is not empty`);
  return body;
}

export async function loadGpuEngine(gpu: GPU | undefined = globalThis.navigator?.gpu): Promise<GpuEngine | null> {
  const adapter = await gpu?.requestAdapter();
  if (!gpu || !adapter) return null;
  const device = await adapter.requestDevice();
  const wgsl = new TextEncoder().encode(await text(WGSL_URL));
  assert(wgsl.length > 0, "the WGSL source is not empty");
  const { default: createGpuModule } = await import("./strike-gpu.mjs");
  const m: GpuSolverModule = await createGpuModule({ preinitializedWebGPUDevice: device });
  m._goose_init(0, 0);
  const src = m._malloc(wgsl.length), sl = m._malloc(SLICE_BYTES);
  assert(src > 0 && sl > 0, "the WGSL source and its slice were allocated");
  assert(src + wgsl.length <= m.HEAPU8.length, "the WGSL source fits in the heap");
  m.HEAPU8.set(wgsl, src);
  const dv = new DataView(m.HEAPU8.buffer, sl, SLICE_BYTES);
  dv.setUint32(0, src, true);
  dv.setBigInt64(4, BigInt(wgsl.length), true);
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  assert(wgsl.length < m.HEAPU8.length, "the WGSL source is smaller than the heap");
  assert(sl + SLICE_BYTES <= m.HEAPU8.length, "the slice lies inside the heap");
  const ok = m._gpu_init(sl) && m._gpu_setup(0, 56, 0.009, V_C);
  m._free(sl); m._free(src);
  const internal = await device.popErrorScope(), validation = await device.popErrorScope();
  const failure = internal || validation;
  assert(typeof ok === "number" || typeof ok === "boolean", "the set-up reports a result");
  assert(failure === null || typeof failure.message === "string", "a GPU error has a message");
  if (!ok || failure) throw new Error(`the WebGPU solver did not start${failure ? `: ${failure.message}` : ""}`);
  const base = common(m);
  assert(Array.isArray(errors), "the GPU errors are collected");
  assert(typeof m._gpu_run === "function", "the module exports gpu_run");
  return {
    ...base,
    gpu: true,
    errors,
    setup: m._gpu_setup,
    run: m._gpu_run,
    fillRender: () => { assert(Array.isArray(errors), "the GPU errors are collected"); },
    poll: m._gpu_poll,
    pending: m._gpu_pending,
    running: () => {
      const flag = m._gpu_running();
      assert(Number.isInteger(flag), "the running flag is a whole number");
      return flag !== 0;
    },
    history: () => {
      const rows = Math.min(m._get_steps(), m._gpu_hist_cap());
      assert(rows >= 0, "the history has a row count");
      return floats(base.bytes, m._gpu_buf_hist, 4 * rows);
    },
    birdKE: m._gpu_bird_ke,
    panelKE: m._gpu_panel_ke,
    gpuObject: gpu,
  };
}

export function hugoniot(v: number): [number, number] {
  assert(Number.isFinite(v) && v > 0, "the impact speed is positive");
  assert(v < C0_BIRD, "the impact speed is below the sound speed");
  const b = (RHO_BIRD * C0_BIRD * C0_BIRD) / GAMMA_TAIT;
  let lo = RHO_BIRD * 1.000001;
  let hi = RHO_BIRD * 2.0;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    const f = b * (Math.pow(mid / RHO_BIRD, GAMMA_TAIT) - 1) - (RHO_BIRD * v * v * mid) / (mid - RHO_BIRD);
    if (f > 0) hi = mid; else lo = mid;
  }
  assert(lo >= RHO_BIRD && hi <= 2 * RHO_BIRD && lo <= hi, "the bracket stays around the shocked density");
  const rho1 = 0.5 * (lo + hi);
  const us = (v * rho1) / (rho1 - RHO_BIRD);
  assert(rho1 > RHO_BIRD && Number.isFinite(us), "the shocked density is above the rest density");
  assert(us > v, "the shock runs faster than the bird");
  return [RHO_BIRD * us * v, us];
}

export function stagnation(v: number): number {
  assert(Number.isFinite(v), "the impact speed is finite");
  assert(RHO_BIRD > 0, "the bird has a density");
  return 0.5 * RHO_BIRD * v * v;
}
