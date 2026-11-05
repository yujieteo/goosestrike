export function checkPower10(): string[] {
  const r = Bun.spawnSync([`${import.meta.dir}/power10-check`], { cwd: import.meta.dir, stdout: "pipe", stderr: "pipe" });
  if (r.exitCode === 0) return [];
  return [...r.stdout.toString().split("\n"), ...r.stderr.toString().split("\n")].filter((l) => l !== "");
}
