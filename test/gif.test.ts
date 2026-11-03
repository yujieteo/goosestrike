import { test, expect } from "bun:test";
import { GifRecorder, type Rgb } from "../web/gif.ts";
import { capture } from "./capture.ts";
import { at } from "../web/checked.ts";

const BACKGROUND: Rgb = [232, 224, 208];
const HEADER_BYTES = 13;
const PALETTE_COLOURS = 256;
const MAX_STEP_ERROR = 12;

test("the GIF palette keeps a colour ramp next to a large light background", () => {
  const w = 576, h = 64, img = { data: new Uint8ClampedArray(4 * w * h) };
  const ramp: Rgb[] = [];
  for (let r = 4; r < 28; r++) for (let g = 4; g < 28; g++) ramp.push([8 * r, 8 * g, 48]);
  for (let i = 0; i < w * h; i++) img.data.set([...(i < w ? at(ramp, i) : BACKGROUND), 255], 4 * i);
  capture({ kind: "gif", width: w, height: h, frames: [{ delay: 4, rgba: Buffer.from(img.data).toString("base64") }] });
  const rec = new GifRecorder(w, h);
  rec.add(img, 4);
  const gif = rec.encode();
  const pal = Array.from({ length: PALETTE_COLOURS }, (_, i) => [...gif.subarray(HEADER_BYTES + 3 * i, HEADER_BYTES + 3 * i + 3)]);
  const err = (c: Rgb): number => Math.min(...pal.map((p) => Math.hypot(...p.map((v, k) => v - at(c, k) * 255 / 248))));
  expect(err(BACKGROUND), "the background has its own palette colour").toBeLessThan(1);
  for (const c of ramp) expect(err(c), `${c}: the nearest palette colour`).toBeLessThanOrEqual(MAX_STEP_ERROR);
});
