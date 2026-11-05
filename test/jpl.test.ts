import { expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { checkJpl } from "../scripts/jpl.ts";

function project(files: Record<string, string>): string {
  const root = mkdtempSync(`${tmpdir()}/jpl-`);
  for (const [name, body] of Object.entries(files)) {
    mkdirSync(`${root}/${name.split("/")[0]}`, { recursive: true });
    writeFileSync(`${root}/${name}`, body);
  }
  return root;
}

const body = (n: number): string => Array.from({ length: n }, (_, i) => `  x = ${i};`).join("\n");

test("SCR-187: recursion, direct or through another function, fails in Goose and TypeScript", () => {
  const root = project({
    "sim/a.goose": "fn f(n: i64) -> i64 { g(n) }\nfn g(n: i64) -> i64 { f(n) }\nfn ok() { f2() }\nfn f2() {}\n",
    "web/a.ts": "function h(n: number): number { return n ? h(n - 1) : 0; }\nfunction fine(): number { return 1; }\n",
  });
  const problems = checkJpl(root, {});
  expect(problems.filter((p) => p.includes("rule 1")).map((p) => p.split(" ")[1]).sort()).toEqual(["f", "g", "h"]);
});

test("SCR-187: a function of 61 lines fails and one of 60 passes, in Goose and TypeScript", () => {
  const root = project({
    "sim/a.goose": `fn long() {\n${body(59)}\n}\nfn exact() {\n${body(58)}\n}\n`,
    "web/a.ts": `function long(): void {\n${body(59)}\n}\nfunction exact(): void {\n${body(58)}\n}\n`,
  });
  const problems = checkJpl(root, {});
  expect(problems.map((p) => p.split(" ").slice(0, 2).join(" "))).toEqual(["sim/a.goose:1: long", "web/a.ts:1: long"]);
});
