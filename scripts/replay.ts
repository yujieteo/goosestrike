import { existsSync, mkdtempSync, mkdirSync, rmSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { REPLAY } from "./config.ts";

interface Line { name: string; output: string }

function sh(cwd: string, cmd: string[]): string {
  const r = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
  if (r.exitCode !== 0) throw new Error(`${cmd.join(" ")}: ${r.stderr.toString()}`);
  return r.stdout.toString().trim();
}

export function cacheDir(): string {
  return `${tmpdir()}/${REPLAY.cacheDir}`;
}

function extract(root: string, rev: string, into: string): void {
  mkdirSync(into, { recursive: true });
  const r = Bun.spawnSync(["sh", "-c", `git -C '${root}' archive '${rev}' | tar -x -C '${into}'`], { stderr: "pipe" });
  if (r.exitCode !== 0) throw new Error(`git archive ${rev}: ${r.stderr.toString()}`);
}

async function* lines(stream: ReadableStream<Uint8Array>): AsyncGenerator<Line> {
  let pending = "";
  const decoder = new TextDecoder();
  const reader = stream.getReader();
  for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) {
    pending += decoder.decode(chunk.value, { stream: true });
    let nl: number;
    while ((nl = pending.indexOf("\n")) >= 0) {
      const text = pending.slice(0, nl);
      pending = pending.slice(nl + 1);
      const tab = text.indexOf("\t");
      yield { name: text.slice(0, tab), output: text.slice(tab + 1) };
    }
  }
}

export async function replay(root: string, accept: boolean): Promise<string[]> {
  const inputs = cacheDir() + "/inputs";
  const total = existsSync(inputs) ? readdirSync(inputs).length : 0;
  if (total === 0) return [`${inputs} has no captured input: run the tests with GOOSESTRIKE_CAPTURE=${inputs}`];
  const base = sh(root, ["git", "rev-parse", "--verify", REPLAY.baseRef]);
  const work = mkdtempSync(`${tmpdir()}/goosestrike-replay-run-`);
  try {
    const baseTree = `${work}/base`, branchTree = `${work}/branch`;
    extract(root, base, baseTree);
    const dirty = sh(root, ["git", "status", "--porcelain"]);
    if (dirty) {
      const r = Bun.spawnSync(["sh", "-c", `rsync -a --exclude .git --exclude node_modules '${root}/' '${branchTree}/'`]);
      if (r.exitCode !== 0) throw new Error("rsync of the branch failed");
    } else {
      extract(root, "HEAD", branchTree);
    }
    const runner = `${root}/scripts/replay-runner.ts`;
    const start = (tree: string) => Bun.spawn(["bun", "run", runner, tree, inputs], { stdout: "pipe", stderr: "inherit" });
    const a = start(baseTree), b = start(branchTree);
    const baseLines: Line[] = [], branchLines: Line[] = [];
    const diffs: string[] = [];
    const errors: string[] = [];
    let stopped = false;
    let compared = 0;
    const compare = (): void => {
      while (compared < Math.min(baseLines.length, branchLines.length)) {
        const x = baseLines[compared], y = branchLines[compared];
        compared++;
        if (x === undefined || y === undefined) continue;
        if ("error" in (JSON.parse(y.output) as object)) errors.push(`${y.name} failed on branch: ${y.output}`);
        if (x.output !== y.output) diffs.push(`${x.name}\n  base:   ${x.output}\n  branch: ${y.output}`);
        if (diffs.length > REPLAY.largeChangeFraction * total && !stopped) { stopped = true; a.kill(); b.kill(); }
      }
    };
    const collect = async (proc: typeof a, into: Line[]): Promise<void> => {
      for await (const l of lines(proc.stdout)) { into.push(l); compare(); }
    };
    await Promise.all([collect(a, baseLines), collect(b, branchLines)]);
    await Promise.all([a.exited, b.exited]);
    compare();
    if (stopped) return [`large change: more than ${REPLAY.largeChangeFraction * 100} % of ${total} outputs differ`, ...diffs.slice(0, 3)];
    if (baseLines.length !== total || branchLines.length !== total) {
      return [`replay produced ${baseLines.length} base and ${branchLines.length} branch outputs for ${total} inputs`];
    }
    if (errors.length) return errors;
    if (diffs.length === 0) return [];
    if (accept) {
      console.log(diffs.join("\n"));
      console.log(`${diffs.length} of ${total} outputs differ: accepted`);
      return [];
    }
    return [`${diffs.length} of ${total} outputs differ from ${REPLAY.baseRef}; rerun with --accept to accept`, ...diffs];
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}
