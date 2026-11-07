import path from "node:path";
import { BUNDLE } from "./config.ts";

export function buildBundle(root: string, outfile: string): string[] {
  const r = Bun.spawnSync(["bun", "build", BUNDLE.entry, "--outfile", outfile, ...BUNDLE.args], { cwd: root, stdout: "pipe", stderr: "pipe" });
  return r.exitCode === 0 ? [] : [r.stderr.toString()];
}

if (import.meta.main) {
  const problems = buildBundle(path.dirname(import.meta.dir), BUNDLE.outfile);
  if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
}
