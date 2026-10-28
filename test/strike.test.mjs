// Checks of the committed WebAssembly solver (web/strike.wasm).
// Run: npm test (node --test)
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { loadEngine, hugoniot, stagnation, V_C, BIRD_MASS } from "../web/engine.mjs";

const eng = await loadEngine();
const ke0 = 0.5 * BIRD_MASS * V_C * V_C;

test("gen/ and the wasm files are built from the current sim/", () => {
  // build.sh writes the sums when it generates the C and builds the wasm
  const sums = readFileSync(new URL("../gen/sha256sums", import.meta.url), "utf8").trim().split("\n");
  assert.deepEqual(sums.map((l) => l.split(/\s+/)[1]), ["sim/strike.goose", "gen/strike.c", "gen/strike.h", "web/strike.wasm",
    "sim/gpu.goose", "sim/gpu.c", "gen/gpu.c", "gen/gpu.h", "web/strike-gpu.wasm"]);
  for (const l of sums) {
    const [sum, path] = l.split(/\s+/);
    assert.equal(createHash("sha256").update(readFileSync(new URL(`../${path}`, import.meta.url))).digest("hex"), sum, `${path}: run ./build.sh`);
  }
  // and copies the source for the code view of the page
  const src = readFileSync(new URL("../sim/strike.goose", import.meta.url), "utf8");
  assert.ok(src === readFileSync(new URL("../web/strike.goose", import.meta.url), "utf8"), "web/strike.goose: run ./build.sh");
});

test("the load case is 4 lb at 350 kt", () => {
  assert.equal(BIRD_MASS, 4 * 0.45359237);
  assert.ok(Math.abs(V_C - 350 * 1852 / 3600) < 1e-4);
});

test("the Wilbeck reference pressures", () => {
  const [ph, us] = hugoniot(V_C);
  assert.ok(Math.abs(ph / 1e6 - 314.43) < 0.01, `P_H = ${ph / 1e6} MPa`);
  assert.ok(Math.abs(us - 1838.2) < 0.1, `U_s = ${us} m/s`);
  assert.ok(Math.abs(stagnation(V_C) / 1e6 - 15.40) < 0.01);
});

test("rigid wall: the impulse is m v and the energy balance closes", () => {
  const np = eng.setup(1, 8, 0.009, V_C);
  assert.equal(eng.birdV(), V_C);
  assert.ok(Math.abs(np * eng.particleMass() - BIRD_MASS) < 1e-12, "the particles weigh 4 lb");
  assert.ok(Math.abs(eng.birdKE() - ke0) / ke0 < 1e-9, "the bird starts with all the kinetic energy");
  const tEnd = 1.5 * eng.birdL() / V_C;
  while (eng.time() < tEnd) eng.advance(100, tEnd);
  const imp = eng.impulse() / (BIRD_MASS * V_C);
  assert.ok(Math.abs(imp - 1) < 0.02, `impulse / (m v) = ${imp}`);
  const bal = (eng.birdKE() + eng.birdWork()) / ke0;
  assert.ok(Math.abs(bal - 1) < 0.01, `energy balance = ${bal}`);
});

test("panel: stable, energy balance within 1 %, and set-up resets the state", () => {
  const run = () => {
    eng.setup(0, 56, 0.009, V_C);
    let worst = 0;
    while (eng.time() < 5e-4) {
      eng.advance(1, 5e-4);
      const bal = (eng.birdKE() + eng.panelKE() + eng.panelWork() + eng.birdWork()) / ke0;
      worst = Math.max(worst, Math.abs(bal - 1));
    }
    return { worst, defl: eng.maxDeflection(), steps: eng.steps() };
  };
  const a = run();
  assert.ok(a.worst < 0.01, `worst energy balance error ${a.worst}`);
  assert.ok(a.defl > 0.01 && a.defl < 0.2, `deflection ${a.defl} m`);
  assert.ok(Math.abs(eng.panelMass() - 5.0526) < 1e-3, `panel mass ${eng.panelMass()} kg`);
  const b = run();
  assert.equal(b.defl, a.defl);
  assert.equal(b.steps, a.steps);
});

test("the phases one at a time give the same result as whole steps", () => {
  const run = (phased) => {
    eng.setup(0, 56, 0.009, V_C);
    if (phased) { while (eng.stepPhase(2e-4) >= 0); } else { while (eng.time() < 2e-4) eng.advance(10, 2e-4); }
    return [eng.steps(), eng.maxDeflection(), eng.panelWork(), eng.birdWork(), eng.contactForce()];
  };
  assert.deepEqual(run(true), run(false));
});
