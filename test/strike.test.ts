import { test, expect } from "bun:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { capture } from "./capture.ts";
import { loadEngine, hugoniot, stagnation, V_C, BIRD_MASS, PANEL, RIGID_WALL } from "../web/engine.ts";

const eng = await loadEngine();
const ke0 = 0.5 * BIRD_MASS * V_C * V_C;
const read = (path: string): Buffer => readFileSync(new URL(`../${path}`, import.meta.url));
const SUMMED_FILES = ["sim/strike.goose", "gen/strike.c", "gen/strike.h", "web/strike.wasm",
  "sim/gpu.goose", "sim/gpu.c", "gen/gpu.c", "gen/gpu.h", "web/strike-gpu.wasm"];

function generatedFilesAreCurrent(): void {
  const sums = read("gen/sha256sums").toString("utf8").trim().split("\n").map((l) => l.split(/\s+/));
  expect(sums.map((l) => l[1])).toEqual(SUMMED_FILES);
  for (const [sum, path] of sums) {
    if (sum === undefined || path === undefined) throw new Error("a line of gen/sha256sums has no sum and path");
    expect(createHash("sha256").update(read(path)).digest("hex"), `${path}: run ./build.sh`).toBe(sum);
  }
  expect(read("sim/strike.goose").equals(read("web/strike.goose")), "web/strike.goose: run ./build.sh").toBe(true);
}

test("gen/ and the wasm files are built from the current sim/", generatedFilesAreCurrent);

test("the load case is 4 lb at 350 kt", () => {
  expect(BIRD_MASS).toBe(4 * 0.45359237);
  expect(Math.abs(V_C - 350 * 1852 / 3600)).toBeLessThan(1e-4);
});

test("the Wilbeck reference pressures", () => {
  const [ph, us] = hugoniot(V_C);
  expect(Math.abs(ph / 1e6 - 314.43), `P_H = ${ph / 1e6} MPa`).toBeLessThan(0.01);
  expect(Math.abs(us - 1838.2), `U_s = ${us} m/s`).toBeLessThan(0.1);
  expect(Math.abs(stagnation(V_C) / 1e6 - 15.40)).toBeLessThan(0.01);
});

function rigidWallBalances(): void {
  const np = eng.setup(RIGID_WALL, 8, 0.009, V_C);
  expect(eng.birdV()).toBe(V_C);
  expect(Math.abs(np * eng.particleMass() - BIRD_MASS), "the particles weigh 4 lb").toBeLessThan(1e-12);
  expect(Math.abs(eng.birdKE() - ke0) / ke0, "the bird starts with all the kinetic energy").toBeLessThan(1e-9);
  const tEnd = 1.5 * eng.birdL() / V_C;
  capture({ kind: "solver", mode: RIGID_WALL, cells: 8, spacing: 0.009, speed: V_C, tEnd, stepping: "advance", batch: 100 });
  while (eng.time() < tEnd) eng.advance(100, tEnd);
  const imp = eng.impulse() / (BIRD_MASS * V_C);
  expect(Math.abs(imp - 1), `impulse / (m v) = ${imp}`).toBeLessThan(0.02);
  const bal = (eng.birdKE() + eng.birdWork()) / ke0;
  expect(Math.abs(bal - 1), `energy balance = ${bal}`).toBeLessThan(0.01);
}

test("rigid wall: the impulse is m v and the energy balance closes", rigidWallBalances);

function panelIsStable(): void {
  const run = () => {
    capture({ kind: "solver", mode: PANEL, cells: 56, spacing: 0.009, speed: V_C, tEnd: 5e-4, stepping: "advance", batch: 1 });
    eng.setup(PANEL, 56, 0.009, V_C);
    let worst = 0;
    while (eng.time() < 5e-4) {
      eng.advance(1, 5e-4);
      const bal = (eng.birdKE() + eng.panelKE() + eng.panelWork() + eng.birdWork()) / ke0;
      worst = Math.max(worst, Math.abs(bal - 1));
    }
    return { worst, defl: eng.maxDeflection(), steps: eng.steps() };
  };
  const a = run();
  expect(a.worst, `worst energy balance error ${a.worst}`).toBeLessThan(0.01);
  expect(a.defl > 0.01 && a.defl < 0.2, `deflection ${a.defl} m`).toBe(true);
  expect(Math.abs(eng.panelMass() - 5.0526), `panel mass ${eng.panelMass()} kg`).toBeLessThan(1e-3);
  const b = run();
  expect(b.defl).toBe(a.defl);
  expect(b.steps).toBe(a.steps);
}

test("panel: stable, energy balance within 1 %, and set-up resets the state", panelIsStable);

test("the phases one at a time give the same result as whole steps", () => {
  const run = (phased: boolean) => {
    capture({ kind: "solver", mode: PANEL, cells: 56, spacing: 0.009, speed: V_C, tEnd: 2e-4, stepping: phased ? "phase" : "advance", batch: 10 });
    eng.setup(PANEL, 56, 0.009, V_C);
    if (phased) { while (eng.stepPhase(2e-4) >= 0); } else { while (eng.time() < 2e-4) eng.advance(10, 2e-4); }
    return [eng.steps(), eng.maxDeflection(), eng.panelWork(), eng.birdWork(), eng.contactForce()];
  };
  expect(run(true)).toEqual(run(false));
});
