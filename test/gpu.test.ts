import { test, expect } from "bun:test";
import { loadEngine, loadGpuEngine, V_C, BIRD_MASS, PANEL, RIGID_WALL, type GpuEngine } from "../web/engine.ts";
import { at } from "../web/checked.ts";

let gpu: GpuEngine | null = null;
let why = "";
try {
  const { create, globals } = await import("webgpu");
  Object.assign(globalThis, globals);
  gpu = await loadGpuEngine(create([]));
  if (!gpu) why = "no WebGPU adapter";
} catch (e) {
  why = `no WebGPU: ${e instanceof Error ? e.message : String(e)}`;
}
const mustRun = process.env["GOOSESTRIKE_GPU"] !== undefined;
const gpuTest = gpu || mustRun ? test : test.skip;

function needGpu(): GpuEngine {
  if (!gpu) throw new Error(why);
  return gpu;
}

async function runGpu(g: GpuEngine, tEnd: number): Promise<void> {
  const keep = setInterval(() => {}, 1000);
  try {
    while (g.running()) {
      g.run(256, tEnd);
      let r = g.poll();
      while (!r) { await new Promise((res) => setTimeout(res, 0)); r = g.poll(); }
      expect(r, "the read-back failed").toBe(1);
      expect(g.errors).toEqual([]);
    }
  } finally {
    clearInterval(keep);
  }
}

gpuTest("the GPU solver matches the CPU solver on the panel", async () => {
  const g = needGpu();
  const tEnd = 3.0e-3;
  const cpu = await loadEngine();
  cpu.setup(PANEL, 56, 0.009, V_C);
  const defl: number[] = [];
  let peakF = 0;
  while (cpu.time() < tEnd) {
    cpu.advance(1, tEnd);
    defl.push(cpu.maxDeflection());
    peakF = Math.max(peakF, cpu.contactForce());
  }
  g.setup(PANEL, 56, 0.009, V_C);
  await runGpu(g, tEnd);
  const h = g.history();
  const n = h.length / 4;
  expect(Math.abs(n - defl.length), `steps: GPU ${n}, CPU ${defl.length}`).toBeLessThanOrEqual(0.01 * defl.length);
  let worst = 0, gPeak = 0;
  for (let i = 0; i < Math.min(n, defl.length); i++) {
    worst = Math.max(worst, Math.abs(at(h, 4 * i + 3) - at(defl, i)));
    gPeak = Math.max(gPeak, at(h, 4 * i + 1));
  }
  expect(worst, `deflection differs by ${worst * 1e3} mm`).toBeLessThan(1e-4);
  const rel = (a: number, b: number): number => Math.abs(a - b) / Math.abs(b);
  expect(rel(gPeak, peakF), `peak force: GPU ${gPeak}, CPU ${peakF}`).toBeLessThan(0.01);
  expect(Math.abs(g.maxDeflection() - cpu.maxDeflection()), "max deflection").toBeLessThan(1e-4);
  expect(rel(g.maxPlasticStrain(), cpu.maxPlasticStrain()), "max plastic strain").toBeLessThan(0.01);
  expect(rel(g.panelWork(), cpu.panelWork()), "panel work").toBeLessThan(0.01);
  expect(Math.abs(g.failed() - cpu.failed()), `eroded: GPU ${g.failed()}, CPU ${cpu.failed()}`).toBeLessThanOrEqual(2);
  expect(Math.abs(g.failedSkin() - cpu.failedSkin()), "eroded skin").toBeLessThanOrEqual(2);
  cpu.fillRender();
  const gs = g.strain(), cs = cpu.strain(), gn = g.nodes(), cn = cpu.nodes();
  const el = cpu.elements(), eroded = new Uint8Array(cn.length / 3);
  let ds = 0, dx = 0;
  for (let e = 0; e < cs.length; e++) {
    if (at(gs, e) >= 0 && at(cs, e) >= 0) ds = Math.max(ds, Math.abs(at(gs, e) - at(cs, e)));
    else for (let k = 0; k < 4; k++) if (at(el, 4 * e + k) >= 0) eroded[at(el, 4 * e + k)] = 1;
  }
  for (let i = 0; i < cn.length; i++) if (!at(eroded, (i / 3) | 0)) dx = Math.max(dx, Math.abs(at(gn, i) - at(cn, i)));
  expect(Math.max(...gs), "the GPU strain is drawn").toBeGreaterThan(0.5 * cpu.maxPlasticStrain());
  expect(ds, `element strain differs by ${ds}`).toBeLessThan(0.01);
  expect(dx, `node positions differ by ${dx * 1e3} mm`).toBeLessThan(5e-3);
});

gpuTest("GPU rigid wall: the impulse is m v and the energy balance closes", async () => {
  const g = needGpu();
  const ke0 = 0.5 * BIRD_MASS * V_C * V_C;
  g.setup(RIGID_WALL, 8, 0.009, V_C);
  expect(Math.abs(g.birdKE() - ke0) / ke0, "the bird starts with all the kinetic energy").toBeLessThan(1e-6);
  const tEnd = 1.5 * g.birdL() / V_C;
  await runGpu(g, tEnd);
  const imp = g.impulse() / (BIRD_MASS * V_C);
  expect(Math.abs(imp - 1), `impulse / (m v) = ${imp}`).toBeLessThan(0.02);
  const bal = (g.birdKE() + g.birdWork()) / ke0;
  expect(Math.abs(bal - 1), `energy balance = ${bal}`).toBeLessThan(0.01);
});
