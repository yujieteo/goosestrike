// The Goose solver compiled to WebAssembly, with typed views of its buffers.
// loadEngine: the CPU solver (sim/strike.goose). loadGpuEngine: the same
// solver on the GPU (sim/gpu.goose, with the kernels of web/strike.wgsl).
import createModule from "./strike.mjs";

export const V_C = 350 * 1852 / 3600;  // 350 KCAS at sea level, m/s
export const BIRD_MASS = 4 * 0.45359237; // 4 lb, kg
export const RHO_BIRD = 950.0;         // kg/m^3
export const C0_BIRD = 1480.0;         // m/s
export const GAMMA_TAIT = 7.0;
export const T_SKIN = 0.0016;          // default skin thickness, m

// The getters of strike.goose. Both modules have them: on the GPU,
// gpu_poll puts the results in the same globals.
function common(m) {
  const lenPtr = m._malloc(8);

  // A const u8[:] export returns a pointer and writes the byte length.
  // The bytes are copied, since Goose packs its globals without alignment.
  function bytes(fn) {
    const ptr = fn(lenPtr);
    const len = Number(new BigInt64Array(m.HEAPU8.buffer, lenPtr, 1)[0]);
    return m.HEAPU8.buffer.slice(ptr, ptr + len);
  }

  return {
    bytes,
    // the skin thickness of the next set-up, m
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
    // render data, valid after fillRender() (CPU) or poll() (GPU)
    fillRender: () => m._fill_render(),
    nodes: () => new Float32Array(bytes(m._buf_nodes), 0, 3 * m._get_nn()),
    particles: () => new Float32Array(bytes(m._buf_particles), 0, 3 * m._get_np()),
    pressure: () => new Float32Array(bytes(m._buf_pressure), 0, m._get_np()),
    strain: () => new Float32Array(bytes(m._buf_strain), 0, m._get_ne()),
    elements: () => new Int32Array(bytes(m._buf_elements), 0, 4 * m._get_ne()),
  };
}

export async function loadEngine() {
  const m = await createModule();
  m._goose_init(0, 0);
  return {
    ...common(m),
    gpu: false,
    // mode 0: fuselage panel, 1: rigid wall. cells: skin elements along the
    // panel (multiple of 8). spacing: bird particle lattice, m.
    setup: (mode, cells, spacing, speed) => m._setup(mode, cells, spacing, speed),
    advance: (steps, tEnd) => m._advance(steps, tEnd),
    // runs the next phase of the step (see run_phase in strike.goose) and
    // returns its number, or -1 at tEnd
    stepPhase: (tEnd) => m._step_phase(tEnd),
  };
}

async function text(name) {
  const url = new URL(name, import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return readFile(url, "utf8");
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  return r.text();
}

// The GPU solver, or null if this browser has no WebGPU adapter. gpu: the
// GPU object (navigator.gpu in a browser).
export async function loadGpuEngine(gpu = globalThis.navigator?.gpu) {
  const adapter = await gpu?.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const wgsl = new TextEncoder().encode(await text("strike.wgsl"));
  const { default: createGpuModule } = await import("./strike-gpu.mjs");
  const m = await createGpuModule({ preinitializedWebGPUDevice: device });
  m._goose_init(0, 0);
  // a const u8[:] argument is a pointer to { data, len }
  const src = m._malloc(wgsl.length), sl = m._malloc(12);
  m.HEAPU8.set(wgsl, src);
  const dv = new DataView(m.HEAPU8.buffer, sl, 12);
  dv.setUint32(0, src, true);
  dv.setBigInt64(4, BigInt(wgsl.length), true);
  const errors = [];
  device.addEventListener("uncapturederror", (e) => errors.push(e.error.message));
  const ok = m._gpu_init(sl);
  m._free(sl); m._free(src);
  if (!ok) throw new Error("the WebGPU solver did not start");
  const base = common(m);
  const info = adapter.info || {};
  return {
    ...base,
    gpu: true,
    adapter: [info.vendor, info.architecture, info.description].filter(Boolean).join(" "),
    // the WebGPU validation errors so far
    errors,
    // as setup of the CPU engine
    setup: (mode, cells, spacing, speed) => m._gpu_setup(mode, cells, spacing, speed),
    // sends `steps` steps (at most 512) and a read-back of their results,
    // and returns the steps, or 0 while two runs are not read back. The GPU
    // does no step at or after tEnd (s), the same for each run.
    run: (steps, tEnd) => m._gpu_run(steps, tEnd),
    // setup and poll fill the render data
    fillRender: () => {},
    // 1 when the results of the oldest run that is not read back are in,
    // 0 while the GPU works (or no run is pending), -1 if the read-back failed
    poll: () => m._gpu_poll(),
    // the runs that poll did not read back
    pending: () => m._gpu_pending(),
    // false when the GPU reached tEnd (the f32 time can end just below it)
    running: () => m._gpu_running() !== 0,
    // a Float32Array of 4 per step since the set-up: time, contact force,
    // sensor force, max deflection
    history: () => new Float32Array(base.bytes(m._gpu_buf_hist), 0, 4 * Math.min(m._get_steps(), m._gpu_hist_cap())),
    birdKE: () => m._gpu_bird_ke(),
    panelKE: () => m._gpu_panel_ke(),
    // Node's webgpu package stops the device when its GPU object is
    // collected, so the engine keeps it.
    gpuObject: gpu,
    destroy: () => device.destroy(),
  };
}

// Pressure behind the shock when the Tait bird hits a rigid wall at v:
// rho0 Us = rho1 (Us - v) and p1 = rho0 Us v = p(rho1). Returns [p, Us].
export function hugoniot(v) {
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

// Steady-flow stagnation pressure (Wilbeck), Pa.
export function stagnation(v) {
  return 0.5 * RHO_BIRD * v * v;
}
