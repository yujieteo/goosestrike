import { POWER10 } from "./config.ts";

export function checkPower10(root: string): string[] {
  const from = `${POWER10.source}@${POWER10.commit}`;
  const r = Bun.spawnSync(["uvx", "--from", from, "power10-check", "--require-version", POWER10.version], { cwd: root, stdout: "pipe", stderr: "pipe" });
  if (r.exitCode === 0) return [];
  return [...r.stdout.toString().split("\n"), ...r.stderr.toString().split("\n")].filter((l) => l !== "");
}
