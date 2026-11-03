// goosestrike README figure 2: one time step, the five phases of run_phase on the CPU
// and the 10 kernels of web/strike.wgsl on the GPU. Sources: sim/strike.goose run_phase,
// sim/gpu.goose step_kernels. Style: schematic-emil.
// Build: put style.js of /schematic-emil beside this file as emil.js, with its fonts/, then run the upstream
// lgwrightlab/schematic-figure tools: node tools/render.mjs --root <dir> --fig step.js. The SVG in docs/figures/
// also embeds a subset of Inter, so that GitHub, which shows it in an <img>, uses Inter.
import * as E from './emil.js';
export const figure = { width: 120, height: 70, name: 'step', format: 'journal' };

const PHASES = [
  ['time step', []],
  ['SPH bird', ['clear', 'bin_particles', 'sph']],
  ['shells', ['shell']],
  ['contact', ['bin_elements', 'contact']],
  ['integration', ['integrate_particles', 'integrate_nodes', 'elem_geom']],
  [null, ['reduce']],
];

export default function draw(F, L) {
  const { r, line, text } = E.useEmil(F, L, 'web');
  const g = E.gray, LOOP = E.hue.blue[11];
  const mm = L.mm, root = F.root;
  const cpuX = mm(8), cpuW = mm(34), gpuX = mm(47);
  const top = mm(13), rowH = mm(9), cardH = mm(7);
  const rowY = (i) => top + i * rowH;
  const head = (x, s, sub) => {
    L.text(root, x, mm(5.5), s, { size: text.label, weight: E.weight.title, valign: 'middle' });
    L.text(root, x, mm(9.5), sub, { size: text.small, color: g[1100], valign: 'middle' });
  };
  head(cpuX, 'CPU', 'strike.wasm, f64');
  head(gpuX, 'GPU', 'strike.wgsl, f32');

  const chips = {};
  PHASES.forEach(([name, kernels], i) => {
    const y = rowY(i);
    if (name) {
      L.rect(root, cpuX, y, cpuW, cardH, E.card(r.chip));
      L.text(root, cpuX + mm(2.5), y + cardH / 2, String(i), { size: text.small, color: g[1100], valign: 'middle' });
      L.text(root, cpuX + mm(6), y + cardH / 2, name, { size: text.label, weight: E.weight.name, valign: 'middle' });
    }
    let x = gpuX;
    for (const k of kernels) {
      const t = L.text(root, x + mm(2), y + cardH / 2, k.replace(/_/g, '\\_'), { size: text.small, valign: 'middle' });
      const w = t.getBBox().width + mm(4);
      const chip = L.rect(root, x, y, w, cardH, E.card(r.chip));
      t.parentNode.insertBefore(chip, t);
      chips[k] = { x, y, w };
      x += w + mm(1.5);
    }
  });

  // the time loops: CPU phase 4 back to phase 0, GPU reduce back to phase 0 (the next time step)
  const arrow = { stroke: LOOP, width: line.normal, head: 3, radius: 6 };
  const lx = cpuX - mm(3);
  L.arrow(root, [cpuX, rowY(4) + cardH / 2], [cpuX, rowY(0) + cardH / 2], { ...arrow, via: [[lx, rowY(4) + cardH / 2], [lx, rowY(0) + cardH / 2]] });
  const red = chips.reduce, rx = Math.max(...Object.values(chips).map((c) => c.x + c.w)) + mm(3);
  const t0 = L.text(root, gpuX, rowY(0) + cardH / 2, 'next time step', { size: text.small, color: LOOP, valign: 'middle' });
  const tw = t0.getBBox().width;
  L.arrow(root, [red.x + red.w, red.y + cardH / 2], [gpuX + tw + mm(1.5), rowY(0) + cardH / 2], { ...arrow, via: [[rx, red.y + cardH / 2], [rx, rowY(0) + cardH / 2]] });

  E.emilTheme(F);
}
