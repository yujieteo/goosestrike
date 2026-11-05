import { readdirSync } from "node:fs";
import { ANALYZER_KNOWN, GENERATED_C_OFF } from "./config.ts";

function generatedC(root: string): string[] {
  return readdirSync(`${root}/gen`).filter((f) => f.endsWith(".c")).sort().map((f) => `gen/${f}`);
}

function spawn(root: string, cmd: string[]): { exitCode: number; stderr: string } {
  const r = Bun.spawnSync(cmd, { cwd: root, stdout: "pipe", stderr: "pipe" });
  return { exitCode: r.exitCode, stderr: r.stderr.toString() };
}

export function checkGeneratedWarnings(root: string): string[] {
  const problems: string[] = [];
  for (const file of generatedC(root)) {
    const r = spawn(root, ["clang", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-fsyntax-only", "-Igen", ...GENERATED_C_OFF, file]);
    if (r.exitCode !== 0) problems.push(`${file} has warnings:`, r.stderr);
  }
  return problems;
}

export function checkAnalyzer(root: string): string[] {
  const problems: string[] = [];
  for (const file of generatedC(root)) {
    const r = spawn(root, ["clang", "--analyze", "-Xanalyzer", "-analyzer-disable-checker=deadcode.DeadStores", "-Igen", file, "-o", "/dev/null"]);
    const found: string[] = [];
    for (const line of r.stderr.split("\n")) {
      const m = /^[^:]+:\d+:\d+: warning: (.*) \[([^\]]+)\]$/.exec(line);
      if (m) found.push(`[${m[2]}] ${m[1]}`);
    }
    const known = [...ANALYZER_KNOWN];
    for (const f of found) {
      const i = known.indexOf(f);
      if (i < 0) problems.push(`${file}: new analyser finding: ${f}`);
      else known.splice(i, 1);
    }
    for (const k of known) problems.push(`${file}: analyser finding no longer reported, remove it from ANALYZER_KNOWN in scripts/config.ts: ${k}`);
  }
  return problems;
}
