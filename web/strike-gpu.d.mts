import type { SolverModule } from "./strike.mjs";

export interface GpuSolverModule extends SolverModule {
  _gpu_init(wgslSlice: number): number;
  _gpu_setup(mode: number, cells: number, spacing: number, speed: number): number;
  _gpu_run(steps: number, tEnd: number): number;
  _gpu_poll(): number;
  _gpu_running(): number;
  _gpu_pending(): number;
  _gpu_bird_ke(): number;
  _gpu_panel_ke(): number;
  _gpu_hist_cap(): number;
  _gpu_buf_hist(lenPtr: number): number;
}
declare const createGpuModule: (options?: Record<string, unknown>) => Promise<GpuSolverModule>;
export default createGpuModule;
