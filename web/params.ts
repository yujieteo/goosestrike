import { assert } from "./checked.ts";

export type Vec3 = [number, number, number];

export const KNOT_MS = 1852 / 3600;
export const T_END = 3.0e-3;
export const BUDGET_MS = 12;
export const APPROACH_S = 1.6;
export const APPROACH_M = 1.2;
export const PRESSURE_LOW_PA = 1e5, PRESSURE_HIGH_PA = 4e8;
export const EP_MAX = 0.18;
export const GIF_W = 480;
export const GIF_SIM_US = 25;
export const GIF_CAPTION_PX = 28;
export const GIF_APPROACH_PERIOD_MS = 100;
export const GIF_APPROACH_DELAY_CS = 10;
export const GIF_SIM_DELAY_CS = 4;
export const GIF_LAST_DELAY_CS = 300;
export const GPU_BATCH_MAX = 512;
export const GPU_BATCH_MIN = 8;
export const STRUCTURE_FAIL_PHASE = 4;
export const MIN_DEPTH = 0.01;
export const SKIN_COLUMN_FRACTION = 0.760 / 1.0668;
export const SKIN_ROW_MULTIPLE = 8;
export const WALL_RUN_LENGTHS = 1.5;
export const MUTED_KEY = "goosestrike-muted";
export const SOURCE_URL = "strike.goose";
export const DEFAULT_YAW = 1.0, DEFAULT_PITCH = 0.25, DEFAULT_DIST = 1.3;
export const DIST_MIN = 0.4, DIST_MAX = 5;
export const PITCH_LIMIT = 1.45;
export const DEPTH_BUCKETS = 4096;
export const SHADES = 64;
export const GRID = "rgba(43, 41, 37, 0.35)";
export const WHEEL_LINE_PX = 16;
export const GOOSE = {
  belly: [252, 250, 244] as Vec3, back: [234, 228, 216] as Vec3,
  orange: "#e8913a", ink: "#2b2925", blush: "rgba(226, 120, 110, 0.5)", star: "#e9b640",
};

assert(T_END > 0 && APPROACH_S > 0 && APPROACH_M > 0, "the run and approach lengths are positive");
assert(PRESSURE_LOW_PA < PRESSURE_HIGH_PA && EP_MAX > 0, "the colour scales have a range");
assert(GPU_BATCH_MIN <= GPU_BATCH_MAX && DIST_MIN < DIST_MAX, "the limits are ordered");
