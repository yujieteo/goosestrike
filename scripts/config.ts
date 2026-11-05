export const COMMENT_EXCLUDED: Record<string, string> = {
  "web/strike.mjs": "written by Emscripten from gen/strike.c",
  "web/strike-gpu.mjs": "written by Emscripten from gen/gpu.c",
  "web/app.js": "written by bun build from web/app.ts",
};

export const SOURCE_GLOBS = ["web/**/*.ts", "web/**/*.mts", "test/**/*.ts", "scripts/**/*.ts"];

export const BUNDLE = {
  entry: "web/app.ts",
  outfile: "web/app.js",
  args: ["--target", "browser", "--format", "esm", "--external", "./strike*.mjs", "--external", "node:*"],
} as const;

export const NONDETERMINISTIC_BANNED_IN = ["web/engine.ts", "web/gif.ts", "web/checked.ts"];
export const NONDETERMINISTIC_CALLS = ["Math.random", "Date.now", "performance.now", "new Date"];

export const FIRST_LOAD_FILES = {
  cpu: ["web/index.html", "web/style.css", "web/app.js", "web/strike.mjs", "web/strike.wasm", "web/strike.goose"],
  gpu: ["web/index.html", "web/style.css", "web/app.js", "web/strike-gpu.mjs", "web/strike-gpu.wasm", "web/strike.wgsl", "web/strike.goose"],
} as const;

export const BUDGET_KB = { default: 120 } as const;

export const REPLAY = {
  cacheDir: "goosestrike-replay",
  baseRef: "origin/main",
  largeChangeFraction: 0.2,
} as const;

export const POWER10 = {
  source: "git+https://github.com/yujieteo/power10-check",
  commit: "502925100c8d3b586662d0d87caa6531dbe8676c",
  version: "0.1.0",
} as const;

export const GENERATED_C_OFF = [
  "-Wno-unused-const-variable",
  "-Wno-unused-function",
  "-Wno-unused-parameter",
  "-Wno-unused-variable",
  "-Wno-unterminated-string-initialization",
] as const;

export const ANALYZER_KNOWN: string[] = [
  "[core.NullDereference] Access to field 'n' results in a dereference of a null pointer (loaded from variable 'lf')",
  "[core.UndefinedBinaryOperatorResult] The left operand of '!=' is a garbage value due to array index out of bounds",
];

export const ANALYZER_VERSION_DEPENDENT: string[] = [
  "[unix.Stream] File position of the stream might be 'indeterminate' after a failed operation. Can cause undefined behavior",
  "[unix.Stream] Read function called when stream is in EOF state. Function has no effect",
];
