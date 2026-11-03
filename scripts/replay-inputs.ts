type Stepping = "phase" | "advance";

export interface SolverInput {
  kind: "solver";
  mode: 0 | 1;
  cells: number;
  spacing: number;
  speed: number;
  tEnd: number;
  stepping: Stepping;
  batch: number;
}

export interface GifInput {
  kind: "gif";
  width: number;
  height: number;
  frames: { delay: number; rgba: string }[];
}

export type ReplayInput = SolverInput | GifInput;
