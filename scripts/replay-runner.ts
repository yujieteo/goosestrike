import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import type { GifInput, ReplayInput, SolverInput } from "./replay-inputs.ts";

interface Solver {
  setup(mode: number, cells: number, spacing: number, speed: number): number;
  advance(steps: number, tEnd: number): number;
  stepPhase(tEnd: number): number;
  fillRender(): void;
  [getter: string]: unknown;
}
interface EngineModule { loadEngine(): Promise<Solver> }
interface Recorder { add(image: { data: Uint8Array }, delay: number): void; encode(): Uint8Array }
interface GifModule { GifRecorder: new (w: number, h: number) => Recorder }

const GETTERS = ["time", "dt", "steps", "tContact", "np", "nn", "ne", "contactForce", "centerForce", "impulse", "panelWork",
  "hourglassWork", "birdWork", "failed", "failedSkin", "maxPlasticStrain", "maxDeflection", "birdKE", "panelKE", "panelMass"];
const BUFFERS = ["nodes", "particles", "pressure", "strain", "elements"];

const sha = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

async function load<T>(root: string, stem: string): Promise<T> {
  for (const ext of ["ts", "mjs"]) {
    const path = `${root}/web/${stem}.${ext}`;
    if (await Bun.file(path).exists()) return await import(pathToFileURL(path).href) as T;
  }
  throw new Error(`${root} has no web/${stem}.ts or web/${stem}.mjs`);
}

function call(solver: Solver, name: string): unknown {
  const fn = solver[name];
  if (typeof fn !== "function") throw new Error(`the solver has no ${name}`);
  return fn.call(solver);
}

async function runSolver(root: string, input: SolverInput): Promise<unknown> {
  const { loadEngine } = await load<EngineModule>(root, "engine");
  const eng = await loadEngine();
  eng.setup(input.mode, input.cells, input.spacing, input.speed);
  if (input.stepping === "phase") { while (eng.stepPhase(input.tEnd) >= 0); }
  else { while (call(eng, "time") as number < input.tEnd) eng.advance(input.batch, input.tEnd); }
  eng.fillRender();
  const out: Record<string, unknown> = {};
  for (const g of GETTERS) out[g] = call(eng, g);
  for (const b of BUFFERS) {
    const view = call(eng, b) as { buffer: ArrayBufferLike; byteOffset: number; byteLength: number };
    out[b] = sha(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  }
  return out;
}

async function runGif(root: string, input: GifInput): Promise<unknown> {
  const { GifRecorder } = await load<GifModule>(root, "gif");
  const rec = new GifRecorder(input.width, input.height);
  for (const f of input.frames) rec.add({ data: new Uint8Array(Buffer.from(f.rgba, "base64")) }, f.delay);
  const gif = rec.encode();
  return { bytes: gif.length, sha256: sha(gif) };
}

async function main(): Promise<void> {
  const [root, inputsDir] = process.argv.slice(2);
  if (root === undefined || inputsDir === undefined) throw new Error("usage: replay-runner <tree> <inputs>");
  for (const name of readdirSync(inputsDir).sort()) {
    const input = JSON.parse(readFileSync(`${inputsDir}/${name}`, "utf8")) as ReplayInput;
    let result: unknown;
    try {
      result = input.kind === "gif" ? await runGif(root, input) : await runSolver(root, input);
    } catch (e) {
      result = { error: e instanceof Error ? e.message : String(e) };
    }
    process.stdout.write(`${name}\t${JSON.stringify(result)}\n`);
  }
}

await main();
