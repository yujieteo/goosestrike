import { gzipSync } from "node:zlib";
import { readFileSync, readdirSync } from "node:fs";
import { BUDGET_KB, FIRST_LOAD_FILES } from "./config.ts";

function gzippedBytes(root: string, file: string): number {
  return gzipSync(readFileSync(`${root}/${file}`)).length;
}

export function checkBudget(root: string): { problems: string[]; report: string[] } {
  const problems: string[] = [];
  const report: string[] = [];
  const pages = readdirSync(`${root}/web`).filter((f) => f.endsWith(".html"));
  if (pages.join(",") !== "index.html") problems.push(`web/ has the pages ${pages.join(", ")}: list the first-load files of each page in scripts/config.ts`);
  for (const [route, files] of Object.entries(FIRST_LOAD_FILES)) {
    const total = files.reduce((sum, f) => sum + gzippedBytes(root, f), 0);
    const budget = BUDGET_KB.default * 1024;
    report.push(`index.html with the ${route} solver: ${(total / 1024).toFixed(1)} KB gzipped of ${BUDGET_KB.default} KB`);
    if (total > budget) problems.push(`index.html with the ${route} solver: ${total} bytes gzipped is over ${budget}`);
  }
  return { problems, report };
}
