import { mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { BUNDLE, NONDETERMINISTIC_BANNED_IN, NONDETERMINISTIC_CALLS } from "./config.ts";
import { checkComments, checkNondeterminism } from "./comments.ts";
import { buildBundle } from "./bundle.ts";
import { checkGoose, checkLint } from "./lint.ts";
import { checkPower10 } from "./power10.ts";
import { checkBudget } from "./budget.ts";
import { cacheDir, replay } from "./replay.ts";
import { checkShell } from "./shell.ts";
import { checkAnalyzer, checkGeneratedWarnings } from "./cstatic.ts";

const root = path.dirname(import.meta.dir);
const accept = process.argv.includes("--accept");
const only = process.argv.slice(2).filter((a) => !a.startsWith("--"));

function run(cmd: string[], env: Record<string, string> = {}): string[] {
  const r = Bun.spawnSync(cmd, { cwd: root, env: { ...process.env, ...env }, stdout: "pipe", stderr: "pipe" });
  return r.exitCode === 0 ? [] : [`${cmd.join(" ")} exited ${r.exitCode}`, r.stdout.toString(), r.stderr.toString()].filter(Boolean);
}

function bundleIsFresh(): string[] {
  const pinned = (JSON.parse(readFileSync(`${root}/package.json`, "utf8")) as { packageManager: string }).packageManager;
  if (pinned !== `bun@${Bun.version}`) return [`the bundle needs ${pinned}, not bun@${Bun.version}`];
  const out = `${tmpdir()}/goosestrike-bundle.js`;
  const problems = buildBundle(root, out);
  if (problems.length) return problems;
  const same = readFileSync(out).equals(readFileSync(`${root}/${BUNDLE.outfile}`));
  return same ? [] : [`${BUNDLE.outfile} is not the bundle of ${BUNDLE.entry}: run bun run build:web`];
}

function testsWithCapture(): string[] {
  const inputs = `${cacheDir()}/inputs`;
  rmSync(inputs, { recursive: true, force: true });
  mkdirSync(inputs, { recursive: true });
  return run(["bun", "run", "test"], { GOOSESTRIKE_CAPTURE: inputs });
}

function sanitizerRun(): string[] {
  const out = `${tmpdir()}/goosestrike-sanitize`;
  const strict = run(["clang", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-fsyntax-only", "-Igen", "test/sanitize/main.c"]);
  if (strict.length) return strict;
  const build = run(["clang", "-O1", "-g", "-fsanitize=undefined,address", "-fno-sanitize-recover=all", "-Igen",
    "test/sanitize/main.c", "gen/strike.c", "-o", out, "-lm", "-lpthread"]);
  return build.length ? build : run([out]);
}

const steps: [string, () => string[] | Promise<string[]>][] = [
  ["typecheck", () => run(["bunx", "tsc", "--noEmit"])],
  ["comments", () => checkComments(root)],
  ["nondeterminism", () => checkNondeterminism(root, NONDETERMINISTIC_BANNED_IN, NONDETERMINISTIC_CALLS)],
  ["goose", () => checkGoose(root)],
  ["lint", () => checkLint(root)],
  ["power10", () => checkPower10()],
  ["dead-code", () => run(["bunx", "knip"])],
  ["bundle", bundleIsFresh],
  ["budget", () => {
    const { problems, report } = checkBudget(root);
    for (const line of report) console.log(`  ${line}`);
    return problems;
  }],
  ["sanitizer", sanitizerRun],
  ["c-warnings", () => checkGeneratedWarnings(root)],
  ["analyzer", () => checkAnalyzer(root)],
  ["shellcheck", () => checkShell(root)],
  ["tests", testsWithCapture],
  ["replay", () => replay(root, accept)],
];

let failed = false;
for (const [name, step] of steps) {
  if (only.length && !only.includes(name)) continue;
  if (name === "replay" && process.env["CI"] !== undefined && only.length === 0) continue;
  const started = performance.now();
  const problems = await step();
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  console.log(`${problems.length ? "FAIL" : "ok  "} ${name} (${seconds} s)`);
  for (const p of problems) console.log(`  ${p}`);
  failed ||= problems.length > 0;
}
process.exit(failed ? 1 : 0);
