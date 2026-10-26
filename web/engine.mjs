// The Goose solver compiled to WebAssembly, with typed views of its buffers.
import createModule from "./strike.mjs";

export const V_C = 350 * 1852 / 3600;  // 350 KCAS at sea level, m/s
export const BIRD_MASS = 4 * 0.45359237; // 4 lb, kg
export const RHO_BIRD = 950.0;         // kg/m^3
export const C0_BIRD = 1480.0;         // m/s
export const GAMMA_TAIT = 7.0;

export async function loadEngine() {
  const m = await createModule();
  m._goose_init(0, 0);
  const lenPtr = m._malloc(8);

  // A const u8[:] export returns a pointer and writes the byte length.
  // The bytes are copied, since Goose packs its globals without alignment.
  function bytes(fn) {
    const ptr = fn(lenPtr);
    const len = Number(new BigInt64Array(m.HEAPU8.buffer, lenPtr, 1)[0]);
    return m.HEAPU8.buffer.slice(ptr, ptr + len);
  }

  return {
    // mode 0: fuselage panel, 1: rigid wall. cells: skin elements along the
    // panel (multiple of 8). spacing: bird particle lattice, m.
    setup: (mode, cells, spacing, speed) => m._setup(mode, cells, spacing, speed),
    advance: (steps, tEnd) => m._advance(steps, tEnd),
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
    // render data, valid after fillRender()
    fillRender: () => m._fill_render(),
    nodes: () => new Float32Array(bytes(m._buf_nodes), 0, 3 * m._get_nn()),
    particles: () => new Float32Array(bytes(m._buf_particles), 0, 3 * m._get_np()),
    pressure: () => new Float32Array(bytes(m._buf_pressure), 0, m._get_np()),
    strain: () => new Float32Array(bytes(m._buf_strain), 0, m._get_ne()),
    elements: () => new Int32Array(bytes(m._buf_elements), 0, 4 * m._get_ne()),
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
