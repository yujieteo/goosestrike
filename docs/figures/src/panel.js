// goosestrike README figure 1: the load case and the stiffened panel, at true scale.
// Geometry from sim/strike.goose (build_panel, panel_point, build_bird). Style: schematic-emil.
// Build: put style.js of /schematic-emil beside this file as emil.js, with its fonts/, then run the upstream
// lgwrightlab/schematic-figure tools: node tools/render.mjs --root <dir> --fig panel.js. The SVG in docs/figures/
// also embeds a subset of Inter, so that GitHub, which shows it in an <img>, uses Inter.
import * as E from './emil.js';
export const figure = { width: 120, height: 90, name: 'panel', format: 'journal' };

// geometry, m
const LX = 1.0668, LS = 0.760, R = 1.975;
const STR_H = 0.0254, STR_F = 0.015, FRM_H = 0.0762, FRM_F = 0.025;
const BD = 0.1131, BL = 0.2262, GAP = 0.0065;
const STR_S = [-3, -1, 1, 3].map((k) => (k * LS) / 8), FRM_X = [-LX / 4, LX / 4];

export default function draw(F, L) {
  const { r, line, text } = E.useEmil(F, L, 'web');
  const g = E.gray, h = E.hue;
  const SKIN = g[1200], STR = h.blue[11], FRM = h.violet[11];
  const BIRD = { fill: h.amber[9], stroke: h.amber[11], width: line.thin };
  const lab = (p, x, y, s, o = {}) => L.text(p, x, y, s, { size: text.label, weight: E.weight.name, ...o });
  const sec = (p, x, y, s, o = {}) => L.text(p, x, y, s, { size: text.small, color: g[1100], ...o });
  const names = { a: 'Plan', b: 'Section A–A', c: 'Section B–B' };
  const panel = (k, box) => F.panel(k, { ...box, labelWeight: E.weight.title, labelFormat: (s) => names[s] });
  const k = L.mm(1000 / 12);            // section scale 1:12, pt per m
  const ticks = (p, a, b, n, o) => {     // clamped edge: ticks on the outer side of the line a-b, o = outward normal
    for (let i = 0; i <= n; i++) {
      const x = a[0] + ((b[0] - a[0]) * i) / n, y = a[1] + ((b[1] - a[1]) * i) / n;
      L.line(p, [x, y], [x + o[0] * 3 - o[1] * 2, y + o[1] * 3 + o[0] * 2], { stroke: g[900], width: line.thin });
    }
  };
  const capsule = (p, cx, yNose, d, l) => {   // bird axis vertical, nose down
    const rr = d / 2;
    L.rect(p, cx - rr, yNose - l, d, l, { r: rr, ...BIRD });
  };

  // ---- a: plan, seen from outside; the stiffeners are on the inner side (dashed) ----
  const a = panel('a', { x: 2, y: 2, w: 44, h: 46 });
  const ka = L.mm(30) / LX, ax = 8, ay = 24;
  const aw = LX * ka, ah = LS * ka;
  const PX = (x) => ax + (x + LX / 2) * ka, PY = (s) => ay + (LS / 2 - s) * ka;
  L.rect(a, ax, ay, aw, ah, { fill: g[200], stroke: SKIN, width: line.bold });
  ticks(a, [ax, ay], [ax + aw, ay], 12, [0, -1]);
  ticks(a, [ax, ay + ah], [ax + aw, ay + ah], 12, [0, 1]);
  ticks(a, [ax, ay], [ax, ay + ah], 8, [-1, 0]);
  ticks(a, [ax + aw, ay], [ax + aw, ay + ah], 8, [1, 0]);
  for (const s of STR_S) L.line(a, [ax, PY(s)], [ax + aw, PY(s)], { stroke: STR, width: line.normal, dash: '3 1.6' });
  for (const x of FRM_X) L.line(a, [PX(x), ay], [PX(x), ay + ah], { stroke: FRM, width: line.normal, dash: '3 1.6' });
  L.circle(a, PX(0), PY(0), (BD / 2) * ka, BIRD);
  // cut lines
  const cut = { stroke: g[1100], width: line.thin, dash: '5 1.5 1 1.5' };
  L.line(a, [PX(0), ay - 7], [PX(0), ay + ah + 7], cut);
  L.line(a, [ax - 6, PY(0)], [ax + aw + 6, PY(0)], cut);
  const cutTag = (x, y, dir, s) => {
    L.arrow(a, [x, y], [x + dir[0] * 5, y + dir[1] * 5], { stroke: g[1100], width: line.thin, head: 2.6 });
    sec(a, x + dir[0] * 7.5, y + dir[1] * 7.5, s, { anchor: 'middle', valign: 'middle', weight: E.weight.name, color: g[1200] });
  };
  cutTag(PX(0), ay - 7, [-1, 0], 'A'); cutTag(PX(0), ay + ah + 7, [-1, 0], 'A');
  cutTag(ax - 6, PY(0), [0, -1], 'B'); cutTag(ax + aw + 6, PY(0), [0, -1], 'B');
  sec(a, ax + aw / 2, ay + ah + 12, '1067 mm × 760 mm', { anchor: 'middle', valign: 'top' });
  lab(a, ax + aw + 4, PY(STR_S[3]), 'stringer', { valign: 'middle', color: STR });
  lab(a, PX(FRM_X[1]), ay - 9, 'frame', { anchor: 'middle', valign: 'bottom', color: FRM });

  // ---- b: section A–A, around the fuselage (s right, z up) ----
  const b = panel('b', { x: 50, y: 2, w: 68, h: 46 });
  const bx = b.w / 2, by = 92;           // the skin top at s = 0
  const sp = (s, d) => {                    // panel_point: s along the arc, d inward
    const rr = R - d, t = s / R;
    return [bx + rr * Math.sin(t) * k, by - (rr * Math.cos(t) - R) * k];
  };
  const arcPts = (s0, s1, d) => Array.from({ length: 41 }, (_, i) => sp(s0 + ((s1 - s0) * i) / 40, d));
  L.polyline(b, arcPts(-LS / 2, LS / 2, 0), { stroke: SKIN, width: line.bold, fill: 'none' });
  for (const s of STR_S) L.polyline(b, [sp(s, 0), sp(s, STR_H), ...arcPts(s, s + STR_F, STR_H)], { stroke: STR, width: line.bold, fill: 'none', join: 'round' });
  for (const s of [-LS / 2, LS / 2]) {
    const p0 = sp(s, -0.002), p1 = sp(s, 0.04), t = [Math.cos(s / R), Math.sin(s / R)];
    ticks(b, p0, p1, 4, s < 0 ? [-t[0], -t[1]] : t);
  }
  capsule(b, bx, by - GAP * k, BD * k, BL * k);
  L.callout(b, sp(STR_S[2] + STR_F / 2, STR_H), [sp(STR_S[2], 0)[0] + 14, by + 22], 'stringer', { anchor: 'start', size: text.label, color: STR });
  L.callout(b, sp(-0.24, 0), [sp(-0.24, 0)[0] - 4, by - 26], 'skin', { anchor: 'end', size: text.label });
  L.arrow(b, [bx, by + 19], [bx, by + 2], { stroke: g[900], width: line.thin, head: 2.4 });
  sec(b, bx, by + 21, 'R 1975 mm', { anchor: 'middle', valign: 'top' });

  // ---- c: section B–B, along the fuselage (x right, z up) ----
  const c = panel('c', { x: 2, y: 50, w: 116, h: 38 });
  const cx = c.w / 2, cy = 76;
  const cp = (x, d) => [cx + x * k, cy + d * k];
  L.line(c, cp(-LX / 2, 0), cp(LX / 2, 0), { stroke: SKIN, width: line.bold });
  for (const x of FRM_X) L.polyline(c, [cp(x, 0), cp(x, FRM_H), cp(x + FRM_F, FRM_H)], { stroke: FRM, width: line.bold, fill: 'none', join: 'round' });
  ticks(c, cp(-LX / 2, -0.002), cp(-LX / 2, 0.04), 4, [-1, 0]);
  ticks(c, cp(LX / 2, -0.002), cp(LX / 2, 0.04), 4, [1, 0]);
  capsule(c, cx, cy - GAP * k, BD * k, BL * k);
  L.arrow(c, [cx + BD * k * 0.5 + 9, cy - BL * k - 1], [cx + BD * k * 0.5 + 9, cy - BL * k * 0.45], { stroke: g[1200], width: line.bold, head: 3.4 });
  lab(c, cx + BD * k * 0.5 + 13, cy - BL * k * 0.75, '180 m/s', { valign: 'middle' });
  lab(c, cx - BD * k * 0.5 - 4, cy - BL * k * 0.6, '4 lb bird', { anchor: 'end', valign: 'middle' });
  L.callout(c, cp(FRM_X[1], FRM_H * 0.5), [cp(FRM_X[1], 0)[0] + 18, cy + 18], 'frame', { anchor: 'start', size: text.label, color: FRM });
  L.callout(c, cp(-LX / 2 - 0.01, 0.02), [cp(-LX / 2, 0)[0] + 12, cy + 18], 'clamped edge', { anchor: 'start', size: text.small, color: g[1100] });

  E.emilTheme(F);
}
