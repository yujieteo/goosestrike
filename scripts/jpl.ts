import ts from "ts-parser";
import { Glob } from "bun";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { COMMENT_EXCLUDED, JPL_GOOSE_GLOBS, JPL_LONG_FUNCTIONS, JPL_MAX_LINES, SOURCE_GLOBS } from "./config.ts";

type Site = ts.CallExpression | ts.NewExpression;
type Fn = {
  file: string; name: string; start: number; lines: number;
  calls: string[]; sites: Site[]; refs: ts.Expression[]; inline: ts.Node[]; bases: ts.Expression[];
  node: ts.Node | null; edges: Fn[];
};

function files(root: string, patterns: readonly string[]): string[] {
  const found = new Set<string>();
  for (const pattern of patterns) for (const f of new Glob(pattern).scanSync({ cwd: root })) found.add(f);
  return [...found].filter((f) => !(f in COMMENT_EXCLUDED)).sort();
}

function blankCommentsAndStrings(text: string): string {
  return text.replace(/\/\/[^\n]*|"(?:[^"\\\n]|\\.)*"/g, (m) => " ".repeat(m.length));
}

const blank = (text: string): string => text.replace(/[^\n]/g, " ");

function closing(code: string, open: number, up: string, down: string): number {
  let depth = 0;
  for (let i = open; i < code.length; i++) {
    if (code[i] === up) depth++;
    else if (code[i] === down && --depth === 0) return i;
  }
  return code.length;
}

type GooseHeader = { name: string; at: number; open: number };

function gooseHeaders(code: string): GooseHeader[] {
  const found: GooseHeader[] = [];
  const header = /^[ \t]*(?:export\s+)?(?:extern\s+)?(?:recursive\s+)?(?:thread_fn|fn)\s+(\w+)\s*\(/gm;
  for (let m = header.exec(code); m !== null; m = header.exec(code)) {
    const params = closing(code, header.lastIndex - 1, "(", ")");
    const tail = /[{;]/.exec(code.slice(params));
    if (tail === null || tail[0] === ";") continue;
    found.push({ name: m[1] as string, at: m.index + m[0].search(/\S/), open: params + tail.index });
  }
  return found;
}

function gooseFunctions(file: string, text: string): Fn[] {
  const code = blankCommentsAndStrings(text);
  const headers = gooseHeaders(code);
  const ends = headers.map((h) => closing(code, h.open, "{", "}"));
  const line = (at: number): number => code.slice(0, at).split("\n").length;
  return headers.map((h, i) => {
    const end = ends[i] as number;
    let body = code.slice(h.open, end);
    let nested = 0;
    headers.forEach((n, j) => {
      const nestedEnd = ends[j] as number;
      if (n.at > h.open && nestedEnd <= end) {
        body = body.slice(0, n.at - h.open) + blank(body.slice(n.at - h.open, nestedEnd - h.open + 1)) + body.slice(nestedEnd - h.open + 1);
        nested += line(nestedEnd) - line(n.at) + 1;
      }
    });
    const calls = [...body.matchAll(/\b(\w+)\s*\(/g)].map((c) => c[1] as string);
    const lines = line(end) - line(h.at) + 1 - nested;
    return { file, name: h.name, start: line(h.at), lines, calls, sites: [], refs: [], inline: [], bases: [], node: null, edges: [] };
  });
}

const FUNCTION_KINDS = [
  ts.SyntaxKind.FunctionDeclaration,
  ts.SyntaxKind.FunctionExpression,
  ts.SyntaxKind.ArrowFunction,
  ts.SyntaxKind.MethodDeclaration,
  ts.SyntaxKind.Constructor,
  ts.SyntaxKind.GetAccessor,
  ts.SyntaxKind.SetAccessor,
];

function isInstanceField(node: ts.Node): node is ts.PropertyDeclaration {
  return ts.isPropertyDeclaration(node) && !node.modifiers?.some((m) => m.kind === ts.SyntaxKind.StaticKeyword);
}

function isCallback(node: ts.Node): boolean {
  return (ts.isFunctionExpression(node) || ts.isArrowFunction(node)) && (ts.isCallExpression(node.parent) || ts.isNewExpression(node.parent)) && node.parent.arguments?.includes(node as ts.Expression) === true;
}

function isReference(node: ts.Node): node is ts.Identifier | ts.PropertyAccessExpression {
  if (!ts.isIdentifier(node) && !ts.isPropertyAccessExpression(node)) return false;
  const parent = node.parent;
  if (ts.isPropertyAccessExpression(parent) && (parent.expression === node || parent.name === node)) return false;
  if (ts.isCallExpression(parent) || ts.isNewExpression(parent)) { if (parent.expression === node) return false; }
  return !(ts.isIdentifier(node) && "name" in parent && (parent as { name?: ts.Node }).name === node);
}

const DEFERRING_CALLS = ["requestAnimationFrame", "setTimeout", "setInterval", "queueMicrotask", "addEventListener"];

function isDeferred(node: ts.Node): boolean {
  const parent = node.parent;
  if (ts.isCallExpression(parent) && parent.arguments.includes(node as ts.Expression)) {
    const callee = ts.isPropertyAccessExpression(parent.expression) ? parent.expression.name.text : ts.isIdentifier(parent.expression) ? parent.expression.text : "";
    return DEFERRING_CALLS.includes(callee);
  }
  return ts.isBinaryExpression(parent) && parent.right === node && ts.isPropertyAccessExpression(parent.left) && parent.left.name.text.startsWith("on");
}

type Uses = { sites: Site[]; refs: ts.Expression[]; inline: ts.Node[] };

function usesIn(function_: ts.Node): Uses {
  const uses: Uses = { sites: [], refs: [], inline: [] };
  const stack = [...function_.getChildren()];
  for (let c = stack.pop(); c !== undefined; c = stack.pop()) {
    if (isCallback(c) && !isDeferred(c)) uses.inline.push(c);
    if (FUNCTION_KINDS.includes(c.kind) || isInstanceField(c)) continue;
    if (ts.isCallExpression(c) || ts.isNewExpression(c)) uses.sites.push(c);
    else if (isReference(c) && !isDeferred(c)) uses.refs.push(c);
    stack.push(...c.getChildren());
  }
  return uses;
}

function ownName(node: ts.Node, source: ts.SourceFile): string {
  const parent = node.parent;
  if ((ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node) || ts.isFunctionExpression(node)) && node.name) return node.name.getText(source);
  if (ts.isGetAccessor(node) || ts.isSetAccessor(node)) return `${ts.isGetAccessor(node) ? "get" : "set"} ${node.name.getText(source)}`;
  if (ts.isVariableDeclaration(parent) || ts.isPropertyAssignment(parent) || ts.isPropertyDeclaration(parent)) return parent.name.getText(source);
  return ts.isConstructorDeclaration(node) ? "constructor" : "";
}

function baseOf(cls: ts.ClassLikeDeclaration): ts.Expression[] {
  const base = cls.heritageClauses?.find((h) => h.token === ts.SyntaxKind.ExtendsKeyword)?.types[0];
  return base === undefined ? [] : [base.expression];
}

function initializerUses(cls: ts.ClassLikeDeclaration): Uses {
  const all = cls.members.filter(isInstanceField).map((field) => usesIn(field));
  return { sites: all.flatMap((u) => u.sites), refs: all.flatMap((u) => u.refs), inline: all.flatMap((u) => u.inline) };
}

const fn = (file: string, name: string, start: number, lines: number, uses: Uses, bases: ts.Expression[], node: ts.Node): Fn =>
  ({ file, name, start, lines, calls: [], sites: uses.sites, refs: uses.refs, inline: uses.inline, bases, node, edges: [] });

function tsFile(file: string, source: ts.SourceFile): Fn[] {
  const found: Fn[] = [];
  const seen = new Map<string, number>();
  const pending: [ts.Node, string][] = [[source, "file"]];
  for (let item = pending.shift(); item !== undefined; item = pending.shift()) {
    const [node, outer] = item;
    let scope = outer;
    if (ts.isClassLike(node) && !node.members.some(ts.isConstructorDeclaration)) {
      const uses = initializerUses(node);
      const line = source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
      const bases = baseOf(node);
      if (uses.sites.length + uses.refs.length + uses.inline.length > 0 || bases.length > 0) found.push(fn(file, `${node.name?.getText(source) ?? "class"}>init`, line, 1, uses, bases, node));
    }
    if (FUNCTION_KINDS.includes(node.kind)) {
      const own = ownName(node, source);
      const base = own === "" ? `${outer}>anonymous` : own;
      const n = (seen.get(base) ?? 0) + 1;
      seen.set(base, n);
      const name = n === 1 ? base : `${base}#${n}`;
      const first = source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
      const last = source.getLineAndCharacterOfPosition(node.getEnd()).line + 1;
      const ctor = ts.isConstructorDeclaration(node);
      const own_ = usesIn(node);
      const init = ctor ? initializerUses(node.parent) : { sites: [], refs: [], inline: [] };
      const uses = { sites: [...own_.sites, ...init.sites], refs: [...own_.refs, ...init.refs], inline: [...own_.inline, ...init.inline] };
      found.push(fn(file, name, first, last - first + 1, uses, ctor ? baseOf(node.parent) : [], node));
      scope = name;
    }
    for (const child of node.getChildren(source)) pending.push([child, scope]);
  }
  return found;
}

function classNodes(declaration: ts.ClassLikeDeclaration): ts.Node[] {
  const constructors = declaration.members.filter(ts.isConstructorDeclaration);
  return constructors.length > 0 ? constructors : [declaration];
}

function functionNodes(declaration: ts.Node): ts.Node[] {
  if (ts.isClassLike(declaration)) return classNodes(declaration);
  if (ts.isVariableDeclaration(declaration) || ts.isPropertyAssignment(declaration) || ts.isPropertyDeclaration(declaration)) {
    let init = declaration.initializer;
    while (init !== undefined && ts.isParenthesizedExpression(init)) init = init.expression;
    if (init !== undefined && ts.isClassExpression(init)) return classNodes(init);
    return init !== undefined && FUNCTION_KINDS.includes(init.kind) ? [init] : [];
  }
  return FUNCTION_KINDS.includes(declaration.kind) ? [declaration] : [];
}

function nodesOf(checker: ts.TypeChecker, expression: ts.Expression, construct: boolean): ts.Node[] {
  let symbol = checker.getSymbolAtLocation(ts.isPropertyAccessExpression(expression) ? expression.name : expression);
  if (symbol !== undefined && symbol.flags & ts.SymbolFlags.Alias) symbol = checker.getAliasedSymbol(symbol);
  return (symbol?.declarations ?? []).flatMap(functionNodes).filter((n) => construct || !ts.isClassLike(n) && !ts.isConstructorDeclaration(n));
}

const folder = (file: string): string => file.slice(0, file.lastIndexOf("/") + 1);

function link(fns: Fn[], checker: ts.TypeChecker): void {
  const byNode = new Map<ts.Node, Fn>(fns.flatMap((f) => (f.node === null ? [] : [[f.node, f] as [ts.Node, Fn]])));
  const byName = new Map<string, Fn[]>();
  for (const f of fns) if (f.node === null) byName.set(`${folder(f.file)}\0${f.name}`, [...(byName.get(`${folder(f.file)}\0${f.name}`) ?? []), f]);
  for (const f of fns) {
    const named = f.calls.flatMap((c) => byName.get(`${folder(f.file)}\0${c}`) ?? []);
    const reached = (e: ts.Expression, construct: boolean): Fn[] => nodesOf(checker, e, construct).flatMap((n) => byNode.get(n) ?? []);
    const typed = [...f.sites.map((site) => site.expression), ...f.bases].flatMap((e) => reached(e, true)).concat(f.refs.flatMap((e) => reached(e, false)));
    const inline = f.inline.flatMap((n) => byNode.get(n) ?? []);
    f.edges = [...named, ...typed, ...inline];
  }
}

function recursion(fns: Fn[]): string[] {
  const problems: string[] = [];
  for (const f of fns) {
    const visited = new Set<Fn>();
    const stack = [...f.edges];
    for (let g = stack.pop(); g !== undefined; g = stack.pop()) {
      if (g === f) { problems.push(`${f.file}:${f.start}: ${f.name} can call itself (JPL rule 1: no recursion)`); break; }
      if (!visited.has(g)) { visited.add(g); stack.push(...g.edges); }
    }
  }
  return problems;
}

function lengths(fns: Fn[], allowedLong: Record<string, number>): string[] {
  const problems: string[] = [];
  const listed = new Set<string>();
  for (const f of fns) {
    const key = `${f.file}:${f.name}`;
    const allowed = allowedLong[key];
    if (allowed !== undefined) listed.add(key);
    if (f.lines > (allowed ?? JPL_MAX_LINES)) {
      problems.push(`${f.file}:${f.start}: ${f.name} has ${f.lines} lines (JPL rule 4: at most ${allowed ?? JPL_MAX_LINES})`);
    } else if (allowed !== undefined && f.lines < allowed) {
      problems.push(`${f.file}:${f.start}: ${f.name} has ${f.lines} lines: lower its limit in JPL_LONG_FUNCTIONS to ${f.lines > JPL_MAX_LINES ? f.lines : "nothing (remove it)"}`);
    }
  }
  for (const key of Object.keys(allowedLong)) if (!listed.has(key)) problems.push(`${key} is in JPL_LONG_FUNCTIONS but no such function exists: remove it`);
  return problems;
}

const COMPILER_OPTIONS: ts.CompilerOptions = {
  noEmit: true,
  noLib: true,
  types: [],
  allowImportingTsExtensions: true,
  target: ts.ScriptTarget.ESNext,
  module: ts.ModuleKind.ESNext,
  moduleResolution: ts.ModuleResolutionKind.Bundler,
};

export function checkJpl(root: string, allowedLong: Record<string, number> = JPL_LONG_FUNCTIONS): string[] {
  const tsFiles = files(root, SOURCE_GLOBS);
  const program = ts.createProgram(tsFiles.map((f) => `${root}/${f}`), COMPILER_OPTIONS, ts.createCompilerHost(COMPILER_OPTIONS, true));
  const fns = [
    ...files(root, JPL_GOOSE_GLOBS).flatMap((f) => gooseFunctions(f, readFileSync(`${root}/${f}`, "utf8"))),
    ...tsFiles.flatMap((f) => {
      const source = program.getSourceFile(`${root}/${f}`);
      return source === undefined ? [] : tsFile(f, source);
    }),
  ];
  link(fns, program.getTypeChecker());
  return [...recursion(fns), ...lengths(fns, allowedLong)];
}

if (import.meta.main) {
  const root = fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, "");
  const list = process.argv.includes("--list");
  const problems = checkJpl(root);
  if (list) {
    for (const p of problems) { const m = /^(.+?):\d+: (\S+) has (\d+) lines \(/.exec(p); if (m) console.log(`  "${m[1]}:${m[2]}": ${m[3]},`); }
  } else for (const p of problems) console.log(p);
}
