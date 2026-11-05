// goosestrike README figure 1: the load case and the stiffened panel, at true scale.
// Geometry from sim/strike.goose (build_panel, panel_point, build_bird). Style: schematic-emil.
// Build: put style.js of /schematic-emil beside this file as emil.js, with its fonts/, then run the upstream
// lgwrightlab/schematic-figure tools: node tools/render.mjs --root <dir> --fig panel.js. The SVG in docs/figures/
// also embeds a subset of Inter, so that GitHub, which shows it in an <img>, uses Inter.
import * as E from './emil.js';
import { assert } from './assert.js';
export const figure = { width: 120, height: 90, name: 'panel', format: 'journal' };

// geometry, m
const LX = 1.0668, LS = 0.760, R = 1.975;
const STR_H = 0.0254, STR_F = 0.015, FRM_H = 0.0762, FRM_F = 0.025;
const BD = 0.1131, BL = 0.2262, GAP = 0.0065;
const STR_S = [-3, -1, 1, 3].map((k) => (k * LS) / 8), FRM_X = [-LX / 4, LX / 4];
assert(STR_S.length === 4 && FRM_X.length === 2, 'there are four stringers and two frames');
assert(LS < LX && BD < BL, 'the panel is longer than it is wide and the bird is a capsule');
assert(GAP > 0 && R > LX, 'the bird sits above a curved skin');

function kit(F, L) {
  assert(typeof F.panel === 'function' && typeof L.text === 'function', 'the figure has a panel and a text layer');
  assert(typeof L.mm === 'function', 'the figure measures in millimetres');
  const { line, text } = E.useEmil(F, L, 'web');
  const g = E.gray, h = E.hue;
  const BIRD = { fill: h.amber[9], stroke: h.amber[11], width: line.thin };
  const lab = (p, x, y, s, o = {}) => {
    assert(typeof s === 'string' && Number.isFinite(x), 'a label has text and an x position');
    assert(Number.isFinite(y), 'a label has a y position');
    return L.text(p, x, y, s, { size: text.label, weight: E.weight.name, ...o });
  };
  const sec = (p, x, y, s, o = {}) => {
    assert(typeof s === 'string' && Number.isFinite(x), 'a note has text and an x position');
    assert(Number.isFinite(y), 'a note has a y position');
    return L.text(p, x, y, s, { size: text.small, color: g[1100], ...o });
  };
  const names = { a: 'Plan', b: 'Section A–A', c: 'Section B–B' };
  const panel = (key, box) => {
    assert(key in names, 'the panel is one of a, b and c');
    assert(box.w > 0 && box.h > 0, 'the panel has a size');
    return F.panel(key, { ...box, labelWeight: E.weight.title, labelFormat: (s) => {
      assert(s in names, 'the label belongs to a known panel');
      assert(names[s].length > 0, 'the label has text');
      return names[s];
    } });
  };
  const k = L.mm(1000 / 12);            // section scale 1:12, pt per m
  assert(k > 0, 'the section scale is positive');
  const ticks = (p, a, b, n, o) => {     // clamped edge: ticks on the outer side of the line a-b, o = outward normal
    assert(n > 0 && o.length === 2, 'a tick row has a count and a normal');
    assert(a.length === 2 && b.length === 2, 'a tick row runs between two points');
    for (let i = 0; i <= n; i++) {
      const x = a[0] + ((b[0] - a[0]) * i) / n, y = a[1] + ((b[1] - a[1]) * i) / n;
      L.line(p, [x, y], [x + o[0] * 3 - o[1] * 2, y + o[1] * 3 + o[0] * 2], { stroke: g[900], width: line.thin });
    }
  };
  const capsule = (p, cx, yNose, d, l) => {   // bird axis vertical, nose down
    assert(d > 0 && l >= d, 'the capsule is longer than it is wide');
    assert(Number.isFinite(cx) && Number.isFinite(yNose), 'the capsule has a position');
    const rr = d / 2;
    L.rect(p, cx - rr, yNose - l, d, l, { r: rr, ...BIRD });
  };
  return { L, g, h, line, text, lab, sec, panel, k, ticks, capsule, BIRD };
}

function drawPlan(K) {
  const { L, g, line, lab, sec, panel, ticks, BIRD } = K;
  assert(typeof panel === 'function' && typeof L.circle === 'function', 'the plan has a panel and a circle primitive');
  assert(STR_S.length > 0, 'the plan has stringers');
  const SKIN = g[1200], STR = K.h.blue[11], FRM = K.h.violet[11];
  // ---- a: plan, seen from outside; the stiffeners are on the inner side (dashed) ----
  const a = panel('a', { x: 2, y: 2, w: 44, h: 46 });
  const ka = L.mm(30) / LX, ax = 8, ay = 24;
  const aw = LX * ka, ah = LS * ka;
  const PX = (x) => {
    assert(Math.abs(x) <= LX / 2, 'a plan x position lies on the panel');
    assert(ka > 0, 'the plan scale is positive');
    return ax + (x + LX / 2) * ka;
  };
  const PY = (s) => {
    assert(Math.abs(s) <= LS / 2, 'a plan s position lies on the panel');
    assert(ka > 0, 'the plan scale is positive');
    return ay + (LS / 2 - s) * ka;
  };
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
    assert(dir.length === 2 && typeof s === 'string', 'a cut tag has a direction and a letter');
    assert(Number.isFinite(x) && Number.isFinite(y), 'a cut tag has a position');
    L.arrow(a, [x, y], [x + dir[0] * 5, y + dir[1] * 5], { stroke: g[1100], width: line.thin, head: 2.6 });
    sec(a, x + dir[0] * 7.5, y + dir[1] * 7.5, s, { anchor: 'middle', valign: 'middle', weight: E.weight.name, color: g[1200] });
  };
  cutTag(PX(0), ay - 7, [-1, 0], 'A'); cutTag(PX(0), ay + ah + 7, [-1, 0], 'A');
  cutTag(ax - 6, PY(0), [0, -1], 'B'); cutTag(ax + aw + 6, PY(0), [0, -1], 'B');
  sec(a, ax + aw / 2, ay + ah + 12, '1067 mm × 760 mm', { anchor: 'middle', valign: 'top' });
  lab(a, ax + aw + 4, PY(STR_S[3]), 'stringer', { valign: 'middle', color: STR });
  lab(a, PX(FRM_X[1]), ay - 9, 'frame', { anchor: 'middle', valign: 'bottom', color: FRM });
}

function drawAround(K) {
  const { L, g, line, text, sec, panel, k, ticks, capsule } = K;
  assert(k > 0 && typeof panel === 'function', 'section b has a scale and a panel');
  assert(typeof L.callout === 'function', 'section b has callouts');
  const SKIN = g[1200], STR = K.h.blue[11];
  // ---- b: section A–A, around the fuselage (s right, z up) ----
  const b = panel('b', { x: 50, y: 2, w: 68, h: 46 });
  const bx = b.w / 2, by = 92;           // the skin top at s = 0
  const sp = (s, d) => {                    // panel_point: s along the arc, d inward
    assert(Number.isFinite(s) && Number.isFinite(d), 'a panel point has an arc position and a depth');
    assert(d < R, 'a panel point lies outside the axis');
    const rr = R - d, t = s / R;
    return [bx + rr * Math.sin(t) * k, by - (rr * Math.cos(t) - R) * k];
  };
  const arcPts = (s0, s1, d) => {
    assert(s0 <= s1, 'an arc runs forwards');
    assert(d >= 0, 'an arc lies on or below the skin');
    return Array.from({ length: 41 }, (_, i) => {
      assert(i >= 0 && i <= 40, 'an arc has 41 points');
      assert(Number.isFinite(s0 + s1), 'an arc has finite ends');
      return sp(s0 + ((s1 - s0) * i) / 40, d);
    });
  };
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
}

function drawAlong(K) {
  const { L, g, line, text, lab, panel, k, ticks, capsule } = K;
  assert(k > 0 && typeof panel === 'function', 'section c has a scale and a panel');
  assert(typeof L.callout === 'function', 'section c has callouts');
  const SKIN = g[1200], FRM = K.h.violet[11];
  // ---- c: section B–B, along the fuselage (x right, z up) ----
  const c = panel('c', { x: 2, y: 50, w: 116, h: 38 });
  const cx = c.w / 2, cy = 76;
  const cp = (x, d) => {
    assert(Math.abs(x) <= LX, 'a point along the panel lies within its length');
    assert(Number.isFinite(d), 'a point along the panel has a depth');
    return [cx + x * k, cy + d * k];
  };
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
}

export default function draw(F, L) {
  assert(F !== undefined && L !== undefined, 'the figure and its layer are given');
  assert(figure.name === 'panel', 'this is the panel figure');
  const K = kit(F, L);
  drawPlan(K);
  drawAround(K);
  drawAlong(K);
  E.emilTheme(F);
}
