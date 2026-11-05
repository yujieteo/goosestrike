import ts from "ts-parser";
import { Glob } from "bun";
import { readFileSync } from "node:fs";
import { COMMENT_EXCLUDED, SOURCE_GLOBS } from "./config.ts";

const OUTSIDE_SOURCE = /https?:\/\//;
const DIRECTIVE = /^\/\/\/?\s*(@ts-|<reference)/;

function sourceFiles(root: string): string[] {
  const found = new Set<string>();
  for (const pattern of SOURCE_GLOBS) {
    for (const f of new Glob(pattern).scanSync({ cwd: root })) found.add(f);
  }
  return [...found].filter((f) => !(f in COMMENT_EXCLUDED)).sort();
}

function commentsIn(text: string, fileName: string): { line: number; text: string }[] {
  const source = ts.createSourceFile(fileName, text, ts.ScriptTarget.Latest, true);
  const found = new Map<number, string>();
  const stack: ts.Node[] = [source];
  for (let node = stack.pop(); node !== undefined; node = stack.pop()) {
    const children = [...node.getChildren(source)];
    if (children.length === 0) {
      for (const range of ts.getLeadingCommentRanges(text, node.getFullStart()) ?? []) {
        found.set(range.pos, text.slice(range.pos, range.end));
      }
    }
    stack.push(...children.reverse());
  }
  return [...found].map(([pos, body]) => ({ line: source.getLineAndCharacterOfPosition(pos).line + 1, text: body }))
    .filter((c) => !OUTSIDE_SOURCE.test(c.text) && !DIRECTIVE.test(c.text));
}

export function checkComments(root: string): string[] {
  const problems: string[] = [];
  for (const file of sourceFiles(root)) {
    for (const c of commentsIn(readFileSync(`${root}/${file}`, "utf8"), file)) {
      problems.push(`${file}:${c.line}: comment: ${c.text.split("\n")[0]}`);
    }
  }
  return problems;
}

export function checkNondeterminism(root: string, files: readonly string[], calls: readonly string[]): string[] {
  const problems: string[] = [];
  for (const file of files) {
    const text = readFileSync(`${root}/${file}`, "utf8");
    const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
    const stack: ts.Node[] = [source];
    for (let node = stack.pop(); node !== undefined; node = stack.pop()) {
      if (ts.isPropertyAccessExpression(node) || ts.isNewExpression(node)) {
        const shown = ts.isNewExpression(node) ? `new ${node.expression.getText(source)}` : node.getText(source);
        if (calls.includes(shown)) problems.push(`${file}:${source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1}: ${shown}`);
      }
      const children: ts.Node[] = [];
      ts.forEachChild(node, (child) => { children.push(child); });
      stack.push(...children.reverse());
    }
  }
  return problems;
}
