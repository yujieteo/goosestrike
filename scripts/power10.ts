import { homedir } from "node:os";
import { POWER10 } from "./config.ts";

function sourceUrl(): string {
  const configured = process.env[POWER10.sourceEnv];
  const source = configured !== undefined && configured !== "" ? configured : `${homedir()}/${POWER10.defaultSource}`;
  return /^[a-z+]+:\/\//.test(source) ? source : `git+file://${source}`;
}

export function checkPower10(root: string): string[] {
  const from = `${sourceUrl()}@${POWER10.commit}`;
  const r = Bun.spawnSync(["uvx", "--from", from, "power10-check", "--require-version", POWER10.version], { cwd: root, stdout: "pipe", stderr: "pipe" });
  if (r.exitCode === 0) return [];
  return [...r.stdout.toString().split("\n"), ...r.stderr.toString().split("\n")].filter((l) => l !== "");
}
