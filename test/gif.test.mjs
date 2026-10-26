// Checks of the GIF export (web/gif.mjs).
import { test } from "node:test";
import assert from "node:assert/strict";
import { GifRecorder } from "../web/gif.mjs";

test("the GIF palette keeps a colour ramp next to a large light background", () => {
  // like the page: most pixels are the light background, and the panel shows
  // more shaded ramp colours than the palette holds (multiples of 8, so that
  // each is one 15-bit colour)
  const w = 576, h = 64, img = { data: new Uint8ClampedArray(4 * w * h) };
  const ramp = [];
  for (let r = 4; r < 28; r++) for (let g = 4; g < 28; g++) ramp.push([8 * r, 8 * g, 48]);
  for (let i = 0; i < w * h; i++) img.data.set([...(i < w ? ramp[i] : [232, 224, 208]), 255], 4 * i);
  const rec = new GifRecorder(w, h);
  rec.add(img, 4);
  const gif = rec.encode();
  // the global colour table follows the 13-byte header
  const pal = Array.from({ length: 256 }, (_, i) => [...gif.subarray(13 + 3 * i, 16 + 3 * i)]);
  const err = (c) => Math.min(...pal.map((p) => Math.hypot(...p.map((v, k) => v - c[k] * 255 / 248))));
  assert.ok(err([232, 224, 208]) < 1, "the background has its own palette colour");
  // 256 colours for 576: each ramp colour is at most one 15-bit step from the palette
  for (const c of ramp) assert.ok(err(c) <= 12, `${c}: the nearest palette colour is ${err(c).toFixed(1)} away`);
});
