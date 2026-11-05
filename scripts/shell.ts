import { readFileSync } from "node:fs";
import { SHELL_EXCLUDED } from "./config.ts";

const SHELLCHECK = ["uvx", "--from", "shellcheck-py==0.11.0.1", "shellcheck", "-o", "all"];
const SHELL_FIRST_LINE = /^#!\s*(?:\/\S*\/)?(?:env\s+(?:-\S+\s+)*)?(?:sh|bash|ksh)(?:\s|$)/;

function committedFiles(root: string): string[] {
  const r = Bun.spawnSync(["git", "ls-files", "-z"], { cwd: root, stdout: "pipe", stderr: "pipe" });
  return r.stdout.toString().split("\0").filter(Boolean);
}

function isShell(root: string, file: string): boolean {
  if (file.endsWith(".sh")) return true;
  const head = readFileSync(`${root}/${file}`).subarray(0, 256).toString("latin1").split("\n")[0] ?? "";
  return SHELL_FIRST_LINE.test(head);
}

export function checkShell(root: string): string[] {
  const files = committedFiles(root).filter((f) => !(f in SHELL_EXCLUDED) && isShell(root, f));
  if (files.length === 0) return [];
  const r = Bun.spawnSync([...SHELLCHECK, ...files], { cwd: root, stdout: "pipe", stderr: "pipe" });
  return r.exitCode === 0 ? [] : [`shellcheck exited ${r.exitCode}`, r.stdout.toString(), r.stderr.toString()].filter(Boolean);
}
