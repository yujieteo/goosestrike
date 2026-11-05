import { assert, at } from "./checked.ts";

export type Rgb = [number, number, number];
interface Frame { px: Uint16Array; delay: number }
export interface RgbaImage { data: ArrayLike<number> }

const COLOURS_15_BIT = 32768;
const PALETTE_SIZE = 256;
const LZW_CLEAR = 256;
const LZW_END = 257;
const LZW_FIRST_FREE = 258;
const LZW_MAX_CODES = 4096;
const LZW_MIN_SIZE = 9;
const LZW_MAX_SIZE = 12;
const SUB_BLOCK = 255;

export class GifRecorder {
  readonly w: number;
  readonly h: number;
  readonly frames: Frame[] = [];
  readonly hist = new Uint32Array(COLOURS_15_BIT);

  constructor(width: number, height: number) {
    assert(Number.isInteger(width) && width > 0, "the GIF has a whole width");
    assert(Number.isInteger(height) && height > 0, "the GIF has a whole height");
    this.w = width;
    this.h = height;
  }

  add(image: RgbaImage, delay: number): void {
    const d = image.data, n = this.w * this.h, px = new Uint16Array(n);
    assert(d.length === 4 * n, "the image has four bytes per pixel");
    assert(Number.isInteger(delay) && delay > 0, "the frame delay is a whole number of centiseconds");
    for (let i = 0; i < n; i++) {
      const c = ((at(d, 4 * i) >> 3) << 10) | ((at(d, 4 * i + 1) >> 3) << 5) | (at(d, 4 * i + 2) >> 3);
      px[i] = c;
      this.hist[c] = at(this.hist, c) + 1;
    }
    assert(this.hist.length === COLOURS_15_BIT, "the histogram is complete");
    this.frames.push({ px, delay });
    assert(this.frames.length > 0, "the frame was kept");
  }

  encode(): Uint8Array {
    assert(this.frames.length > 0, "there is a frame to encode");
    const pal = medianCut(this.hist, PALETTE_SIZE);
    assert(pal.length <= PALETTE_SIZE, "the palette fits in 256 colours");
    const map = new Int16Array(COLOURS_15_BIT).fill(-1);
    const out = new ByteWriter();
    out.str("GIF89a");
    out.u16(this.w); out.u16(this.h);
    out.byte(0xf7); out.byte(0); out.byte(0);
    for (let i = 0; i < PALETTE_SIZE; i++) {
      const c = pal[i] ?? [0, 0, 0];
      out.byte(c[0]); out.byte(c[1]); out.byte(c[2]);
    }
    out.bytes([0x21, 0xff, 11]); out.str("NETSCAPE2.0"); out.bytes([3, 1, 0, 0, 0]);
    const idx = new Uint8Array(this.w * this.h);
    for (const f of this.frames) {
      assert(f.px.length === idx.length, "every frame has the size of the image");
      for (let i = 0; i < idx.length; i++) {
        const c = at(f.px, i);
        let k = at(map, c);
        if (k < 0) { k = nearest(pal, c); map[c] = k; }
        idx[i] = k;
      }
      out.bytes([0x21, 0xf9, 4, 0]); out.u16(f.delay); out.bytes([0, 0]);
      out.byte(0x2c); out.u16(0); out.u16(0); out.u16(this.w); out.u16(this.h); out.byte(0);
      lzw(idx, out);
    }
    out.byte(0x3b);
    assert(out.result().length > 0, "the GIF has bytes");
    assert(out.result()[out.result().length - 1] === 0x3b, "the GIF ends with its trailer");
    return out.result();
  }
}

const channel = (c: number, k: number): number => (c >> (10 - 5 * k)) & 31;

function medianCut(hist: Uint32Array, n: number): Rgb[] {
  assert(hist.length === COLOURS_15_BIT, "the histogram has one bin per 15-bit colour");
  assert(n >= 1 && n <= PALETTE_SIZE, "the palette size is 1 to 256");
  const cols: number[] = [];
  for (let c = 0; c < COLOURS_15_BIT; c++) if (at(hist, c)) cols.push(c);
  assert(cols.length > 0, "the image has at least one colour");
  const boxes: number[][] = [cols];
  while (boxes.length < n) {
    let bi = -1, best = 0, axis = 0;
    boxes.forEach((b, i) => {
      assert(b.length > 0, "no box is empty");
      if (b.length < 2) return;
      for (let k = 0; k < 3; k++) {
        let lo = 31, hi = 0, w = 0;
        for (const c of b) { const v = channel(c, k); if (v < lo) lo = v; if (v > hi) hi = v; w += at(hist, c); }
        const score = (hi - lo) * Math.sqrt(w);
        if (score > best) { best = score; bi = i; axis = k; }
      }
    });
    if (bi < 0) break;
    const b = at(boxes, bi).sort((p, q) => channel(p, axis) - channel(q, axis));
    let total = 0, half = 0, cut = b.length - 1;
    for (const c of b) total += at(hist, c);
    for (let i = 0; i < b.length - 1; i++) { half += at(hist, at(b, i)); if (half >= total / 2) { cut = i + 1; break; } }
    assert(cut > 0 && cut < b.length, "a cut leaves both halves non-empty");
    boxes.splice(bi, 1, b.slice(0, cut), b.slice(cut));
  }
  assert(boxes.length <= n, "no more boxes than palette entries");
  return boxes.map((b): Rgb => {
    let r = 0, g = 0, bl = 0, w = 0;
    assert(b.length > 0, "a box has colours to average");
    for (const c of b) { const k = at(hist, c); r += channel(c, 0) * k; g += channel(c, 1) * k; bl += channel(c, 2) * k; w += k; }
    assert(w > 0, "a box has weight");
    const mean = (v: number): number => Math.round((v / w) * 255 / 31);
    return [mean(r), mean(g), mean(bl)];
  });
}

function nearest(pal: Rgb[], c: number): number {
  assert(pal.length > 0, "the palette has colours");
  assert(c >= 0 && c < COLOURS_15_BIT, "the colour is a 15-bit value");
  const r = ((c >> 10) & 31) * 255 / 31, g = ((c >> 5) & 31) * 255 / 31, b = (c & 31) * 255 / 31;
  let best = 0, bd = Infinity;
  for (let i = 0; i < pal.length; i++) {
    const p = at(pal, i), d = 2 * (p[0] - r) ** 2 + 4 * (p[1] - g) ** 2 + 3 * (p[2] - b) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  assert(best >= 0 && best < pal.length, "the nearest entry is in the palette");
  return best;
}

function lzw(idx: Uint8Array, out: ByteWriter): void {
  assert(idx.length > 0, "the frame has pixels");
  assert(idx.length <= 1 << 24, "a frame has a bounded number of pixels");
  out.byte(8);
  let size = LZW_MIN_SIZE, next = LZW_FIRST_FREE, acc = 0, nbits = 0;
  const block: number[] = [];
  const emit = (code: number): void => {
    acc |= code << nbits; nbits += size;
    while (nbits >= 8) {
      block.push(acc & 255); acc >>>= 8; nbits -= 8;
      if (block.length === SUB_BLOCK) { out.byte(SUB_BLOCK); out.bytes(block); block.length = 0; }
    }
  };
  const dict = new Map<number, number>();
  emit(LZW_CLEAR);
  assert(dict.size === 0 && next === LZW_FIRST_FREE, "the dictionary starts empty");
  let prefix = at(idx, 0);
  assert(prefix < PALETTE_SIZE, "the first pixel indexes the palette");
  for (let i = 1; i < idx.length; i++) {
    const k = at(idx, i), key = prefix * 256 + k, code = dict.get(key);
    if (code !== undefined) { prefix = code; continue; }
    emit(prefix);
    if (next < LZW_MAX_CODES) {
      dict.set(key, next++);
      if (next > (1 << size) && size < LZW_MAX_SIZE) size++;
    } else {
      emit(LZW_CLEAR); dict.clear(); size = LZW_MIN_SIZE; next = LZW_FIRST_FREE;
    }
    prefix = k;
  }
  emit(prefix);
  emit(LZW_END);
  assert(size <= LZW_MAX_SIZE, "the code size stays within 12 bits");
  assert(next <= LZW_MAX_CODES, "the dictionary stays within 4096 codes");
  if (nbits > 0) block.push(acc & 255);
  if (block.length) { out.byte(block.length); out.bytes(block); }
  out.byte(0);
  assert(block.length <= SUB_BLOCK, "the last block fits in a sub-block");
  assert(nbits >= 0 && nbits < 8, "fewer than eight bits are left over");
}

class ByteWriter {
  private buf = new Uint8Array(1 << 20);
  private n = 0;
  byte(b: number): void {
    if (this.n === this.buf.length) { const nb = new Uint8Array(2 * this.n); nb.set(this.buf); this.buf = nb; }
    this.buf[this.n++] = b;
  }
  bytes(a: Iterable<number>): void {
    assert(a !== null && typeof a === "object", "the bytes come as a list");
    assert(this.n <= this.buf.length, "the writer is within its buffer");
    for (const b of a) this.byte(b);
  }
  u16(v: number): void {
    assert(Number.isInteger(v) && v >= 0, "a 16-bit value is a whole number");
    assert(v < 65536, "a 16-bit value fits in two bytes");
    this.byte(v & 255); this.byte((v >> 8) & 255);
  }
  str(s: string): void {
    assert(s.length > 0, "a string has characters");
    assert(/^[\u0020-\u007E]*$/.test(s), "a string is printable ASCII");
    for (let i = 0; i < s.length; i++) this.byte(s.charCodeAt(i));
  }
  result(): Uint8Array {
    assert(this.n <= this.buf.length, "the writer is within its buffer");
    assert(this.n >= 0, "the writer has a length");
    return this.buf.subarray(0, this.n);
  }
}
