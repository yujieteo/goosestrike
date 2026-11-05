import { assert } from "./checked.ts";

export interface Preset {
  name: string; spacing: number; cells: number; peak: number; plateau: number; defl: number; skin: number; stiff: number;
  run: string | null; gpuRun: string; gpuOnly: boolean;
}

export const ALL_PRESETS: Preset[] = [
  { name: "Potato goose", spacing: 0.009, cells: 56, peak: -42.6, plateau: 25.8, defl: 150.8, skin: 0, stiff: 36, run: "10 s", gpuRun: "1 to 2 s", gpuOnly: false },
  { name: "Good goose", spacing: 0.006, cells: 88, peak: 0.9, plateau: 27.7, defl: 152.1, skin: 8, stiff: 62, run: "35 s", gpuRun: "4 to 6 s", gpuOnly: false },
  { name: "Very good goose", spacing: 0.0045, cells: 112, peak: 2.7, plateau: 13.1, defl: 152.9, skin: 25, stiff: 152, run: "2 min", gpuRun: "10 to 12 s", gpuOnly: false },
  { name: "Excellent goose", spacing: 0.0035, cells: 144, peak: 11.3, plateau: 3.2, defl: 228.1, skin: 249, stiff: 191, run: null, gpuRun: "20 to 60 s", gpuOnly: true },
];

export const PHASES = [
  { name: "time step", fns: ["stable_dt", "shell_dt"] },
  { name: "SPH bird", fns: ["sph_forces", "bin_particles"] },
  { name: "shells, plasticity, erosion", fns: ["shell_forces", "shell_element", "plane_stress_return"] },
  { name: "contact", fns: ["contact", "bin_elements", "closest_tri"] },
  { name: "integration", fns: ["integrate", "tait"] },
];

assert(ALL_PRESETS[0]?.gpuOnly === false, "the first preset runs without the GPU");
assert(ALL_PRESETS.length === 4, "there are four presets");
assert(PHASES.length === 5, "a time step has five phases");
