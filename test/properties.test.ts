import { test, expect } from "bun:test";
import fc from "fast-check";
import { hugoniot, stagnation } from "../web/engine.ts";
import { GifRecorder } from "../web/gif.ts";
import { capture } from "./capture.ts";

const SPEED_RANGE = { min: 20, max: 400, noNaN: true };
const GIF_HEADER = "GIF89a";
const GIF_TRAILER = 0x3b;
const PROPERTY_SEED = 20260101;

test("the Hugoniot pressure grows with speed and stays above the stagnation pressure", () => {
  fc.assert(fc.property(fc.double(SPEED_RANGE), fc.double(SPEED_RANGE), (a, b) => {
    const [lo, hi] = a < b ? [a, b] : [b, a];
    const [pLo] = hugoniot(lo), [pHi, us] = hugoniot(hi);
    expect(pHi).toBeGreaterThanOrEqual(pLo);
    expect(pHi).toBeGreaterThan(stagnation(hi));
    expect(us).toBeGreaterThan(hi);
  }));
});

test("a GIF of any small image has the header, the trailer and one frame per add", () => {
  const image = fc.record({
    width: fc.integer({ min: 1, max: 12 }),
    height: fc.integer({ min: 1, max: 12 }),
    seed: fc.integer({ min: 0, max: 255 }),
    frames: fc.integer({ min: 1, max: 3 }),
  });
  fc.assert(fc.property(image, ({ width, height, seed, frames }) => {
    const rec = new GifRecorder(width, height);
    const captured: { delay: number; rgba: string }[] = [];
    for (let f = 0; f < frames; f++) {
      const data = new Uint8Array(4 * width * height).map((_, i) => (i * (seed + 1) + f * 31) & 255);
      rec.add({ data }, 2);
      captured.push({ delay: 2, rgba: Buffer.from(data).toString("base64") });
    }
    if (seed === 0) capture({ kind: "gif", width, height, frames: captured });
    const gif = rec.encode();
    expect(String.fromCharCode(...gif.subarray(0, 6))).toBe(GIF_HEADER);
    expect(gif[gif.length - 1]).toBe(GIF_TRAILER);
    expect(rec.frames.length).toBe(frames);
  }), { seed: PROPERTY_SEED });
});
