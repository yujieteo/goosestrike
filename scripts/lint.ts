import { Glob } from "bun";

function spawn(root: string, cmd: string[]): string[] {
  const r = Bun.spawnSync(cmd, { cwd: root, stdout: "pipe", stderr: "pipe" });
  const said = [r.stdout.toString(), r.stderr.toString()].filter((t) => t.trim() !== "");
  return r.exitCode === 0 && said.length === 0 ? [] : [`${cmd.join(" ")} exited ${r.exitCode}`, ...said];
}

export function checkGoose(root: string): string[] {
  const files = [...new Glob("sim/*.goose").scanSync({ cwd: root })].sort();
  const checks = files.flatMap((f) => spawn(root, ["goose", "check", "-q", f]));
  return [...checks, ...spawn(root, ["goose", "format", "--check", "sim"])];
}

export function checkLint(root: string): string[] {
  return spawn(root, ["bunx", "oxlint", "--deny-warnings", "web", "test", "scripts", "docs/figures/src"]);
}
