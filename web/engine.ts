import createModule, { type SolverModule } from "./strike.mjs";
import type { GpuSolverModule } from "./strike-gpu.mjs";

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

function common(m: SolverModule): Engine {
  const lenPtr = m._malloc(BYTES_PER_I64);

  function bytes(fn: (lenPtr: number) => number): ArrayBuffer {
    const ptr = fn(lenPtr);
    const len = Number(new BigInt64Array(m.HEAPU8.buffer, lenPtr, 1)[0]);
    return m.HEAPU8.buffer.slice(ptr, ptr + len) as ArrayBuffer;
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
    elements: () => new Int32Array(bytes(m._buf_elements), 0, 4 * m._get_ne()),
  };
}

export async function loadEngine(): Promise<CpuEngine> {
  const m = await createModule();
  m._goose_init(0, 0);
  return {
    ...common(m),
    gpu: false,
    setup: (mode, cells, spacing, speed) => m._setup(mode, cells, spacing, speed),
    advance: (steps, tEnd) => m._advance(steps, tEnd),
    stepPhase: (tEnd) => m._step_phase(tEnd),
  };
}

async function text(name: string): Promise<string> {
  const url = new URL(name, import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return readFile(url, "utf8");
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  return r.text();
}

export async function loadGpuEngine(gpu: GPU | undefined = globalThis.navigator?.gpu): Promise<GpuEngine | null> {
  const adapter = await gpu?.requestAdapter();
  if (!gpu || !adapter) return null;
  const device = await adapter.requestDevice();
  const wgsl = new TextEncoder().encode(await text(WGSL_URL));
  const { default: createGpuModule } = await import("./strike-gpu.mjs");
  const m: GpuSolverModule = await createGpuModule({ preinitializedWebGPUDevice: device });
  m._goose_init(0, 0);
  const src = m._malloc(wgsl.length), sl = m._malloc(SLICE_BYTES);
  m.HEAPU8.set(wgsl, src);
  const dv = new DataView(m.HEAPU8.buffer, sl, SLICE_BYTES);
  dv.setUint32(0, src, true);
  dv.setBigInt64(4, BigInt(wgsl.length), true);
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (e) => errors.push((e as GPUUncapturedErrorEvent).error.message));
  device.pushErrorScope("validation");
  device.pushErrorScope("internal");
  const ok = m._gpu_init(sl) && m._gpu_setup(0, 56, 0.009, V_C);
  m._free(sl); m._free(src);
  const internal = await device.popErrorScope(), validation = await device.popErrorScope();
  const failure = internal || validation;
  if (!ok || failure) throw new Error(`the WebGPU solver did not start${failure ? `: ${failure.message}` : ""}`);
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
    gpuObject: gpu,
  };
}

export function hugoniot(v: number): [number, number] {
  const b = (RHO_BIRD * C0_BIRD * C0_BIRD) / GAMMA_TAIT;
  let lo = RHO_BIRD * 1.000001;
  let hi = RHO_BIRD * 2.0;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    const f = b * (Math.pow(mid / RHO_BIRD, GAMMA_TAIT) - 1) - (RHO_BIRD * v * v * mid) / (mid - RHO_BIRD);
    if (f > 0) hi = mid; else lo = mid;
  }
  const rho1 = 0.5 * (lo + hi);
  const us = (v * rho1) / (rho1 - RHO_BIRD);
  return [RHO_BIRD * us * v, us];
}

export function stagnation(v: number): number {
  return 0.5 * RHO_BIRD * v * v;
}
