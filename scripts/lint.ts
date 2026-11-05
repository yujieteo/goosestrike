import { Glob } from "bun";

function spawn(root: string, cmd: string[], quiet: boolean): string[] {
  const r = Bun.spawnSync(cmd, { cwd: root, stdout: "pipe", stderr: "pipe" });
  const said = [r.stdout.toString(), r.stderr.toString()].filter((t) => t.trim() !== "");
  return r.exitCode === 0 && (!quiet || said.length === 0) ? [] : [`${cmd.join(" ")} exited ${r.exitCode}`, ...said];
}

export function checkGoose(root: string): string[] {
  if (process.env["CI"] !== undefined && Bun.which("goose") === null) {
    console.log("  skipped: CI has no Goose compiler, run it on your computer");
    return [];
  }
  const files = [...new Glob("sim/*.goose").scanSync({ cwd: root })].sort();
  const checks = files.flatMap((f) => spawn(root, ["goose", "check", "-q", f], true));
  return [...checks, ...spawn(root, ["goose", "format", "--check", "sim"], true)];
}

export function checkLint(root: string): string[] {
  return spawn(root, ["bunx", "oxlint", "--deny-warnings", "web", "test", "scripts", "docs/figures/src"], false);
}
