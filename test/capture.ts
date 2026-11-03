import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import type { ReplayInput } from "../scripts/replay-inputs.ts";

export function capture(input: ReplayInput): void {
  const dir = process.env["GOOSESTRIKE_CAPTURE"];
  if (dir === undefined) return;
  mkdirSync(dir, { recursive: true });
  const body = JSON.stringify(input);
  const name = `${input.kind}-${createHash("sha256").update(body).digest("hex").slice(0, 16)}.json`;
  writeFileSync(`${dir}/${name}`, body);
}
