// Checks of the WebGPU solver (web/strike-gpu.wasm and web/strike.wgsl)
// against the CPU solver. Node gets WebGPU from the webgpu package (Dawn).
// With no GPU adapter the tests skip, but GOOSESTRIKE_GPU=1 (CI) makes
// that a failure.
import { test } from "node:test";
import assert from "node:assert/strict";
import { loadEngine, loadGpuEngine, V_C, BIRD_MASS } from "../web/engine.mjs";

let gpu = null, why = "";
try {
  const { create, globals } = await import("webgpu");
  Object.assign(globalThis, globals);
  gpu = await loadGpuEngine(create([]));
  if (!gpu) why = "no WebGPU adapter";
} catch (e) {
  why = `no WebGPU: ${e.message}`;
}
const skip = !gpu && !process.env.GOOSESTRIKE_GPU && why;

// Dawn does not keep Node alive while the GPU works.
async function runGpu(tEnd) {
  const keep = setInterval(() => {}, 1000);
  try {
    while (gpu.running()) {
      gpu.run(256, tEnd);
      let r;
      while (!(r = gpu.poll())) await new Promise((res) => setTimeout(res, 0));
      assert.equal(r, 1, "the read-back failed");
      assert.deepEqual(gpu.errors, []);
    }
  } finally {
    clearInterval(keep);
  }
}

test("the GPU solver matches the CPU solver on the panel", { skip }, async () => {
  assert.ok(gpu, why);
  const tEnd = 3.0e-3;   // the run of the page
  const cpu = await loadEngine();
  cpu.setup(0, 56, 0.009, V_C);
  const defl = [];
  let peakF = 0;
  while (cpu.time() < tEnd) {
    cpu.advance(1, tEnd);
    defl.push(cpu.maxDeflection());
    peakF = Math.max(peakF, cpu.contactForce());
  }
  gpu.setup(0, 56, 0.009, V_C);
  await runGpu(tEnd);
  const h = gpu.history();
  const n = h.length / 4;
  assert.ok(Math.abs(n - defl.length) <= 0.01 * defl.length, `steps: GPU ${n}, CPU ${defl.length}`);
  // the deflection history, step by step (f32 on the GPU, f64 on the CPU)
  let worst = 0, gPeak = 0;
  for (let i = 0; i < Math.min(n, defl.length); i++) {
    worst = Math.max(worst, Math.abs(h[4 * i + 3] - defl[i]));
    gPeak = Math.max(gPeak, h[4 * i + 1]);
  }
  assert.ok(worst < 1e-4, `deflection differs by ${worst * 1e3} mm`);
  // the values of the telemetry and of the verdict
  const rel = (a, b) => Math.abs(a - b) / Math.abs(b);
  assert.ok(rel(gPeak, peakF) < 0.01, `peak force: GPU ${gPeak}, CPU ${peakF}`);
  assert.ok(Math.abs(gpu.maxDeflection() - cpu.maxDeflection()) < 1e-4, "max deflection");
  assert.ok(rel(gpu.maxPlasticStrain(), cpu.maxPlasticStrain()) < 0.01, "max plastic strain");
  assert.ok(rel(gpu.panelWork(), cpu.panelWork()) < 0.01, "panel work");
  assert.ok(Math.abs(gpu.failed() - cpu.failed()) <= 2, `eroded: GPU ${gpu.failed()}, CPU ${cpu.failed()}`);
  assert.ok(Math.abs(gpu.failedSkin() - cpu.failedSkin()) <= 2, "eroded skin");
  // what the page draws: the strain of each element and the node positions
  cpu.fillRender();
  const gs = gpu.strain(), cs = cpu.strain(), gn = gpu.nodes(), cn = cpu.nodes();
  let ds = 0, dx = 0;
  for (let e = 0; e < cs.length; e++) if (gs[e] >= 0 && cs[e] >= 0) ds = Math.max(ds, Math.abs(gs[e] - cs[e]));
  for (let i = 0; i < cn.length; i++) dx = Math.max(dx, Math.abs(gn[i] - cn[i]));
  assert.ok(Math.max(...gs) > 0.5 * cpu.maxPlasticStrain(), "the GPU strain is drawn");
  assert.ok(ds < 0.01, `element strain differs by ${ds}`);
  assert.ok(dx < 1e-3, `node positions differ by ${dx * 1e3} mm`);
});

test("GPU rigid wall: the impulse is m v and the energy balance closes", { skip }, async () => {
  assert.ok(gpu, why);
  const ke0 = 0.5 * BIRD_MASS * V_C * V_C;
  gpu.setup(1, 8, 0.009, V_C);
  assert.ok(Math.abs(gpu.birdKE() - ke0) / ke0 < 1e-6, "the bird starts with all the kinetic energy");
  const tEnd = 1.5 * gpu.birdL() / V_C;
  await runGpu(tEnd);
  const imp = gpu.impulse() / (BIRD_MASS * V_C);
  assert.ok(Math.abs(imp - 1) < 0.02, `impulse / (m v) = ${imp}`);
  const bal = (gpu.birdKE() + gpu.birdWork()) / ke0;
  assert.ok(Math.abs(bal - 1) < 0.01, `energy balance = ${bal}`);
});
