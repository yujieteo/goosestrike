// An animated GIF recorder with no dependency: frames are kept as 15-bit
// colour (5 bits per channel), the palette is a median cut of all the frames,
// and the encoder is GIF89a with LZW compression.

export class GifRecorder {
  constructor(width, height) {
    this.w = width;
    this.h = height;
    this.frames = [];               // { px: Uint16Array, delay: centiseconds }
    this.hist = new Uint32Array(32768);
  }

  // Adds the RGBA pixels of one frame (an ImageData) that shows for `delay` cs.
  add(image, delay) {
    const d = image.data, n = this.w * this.h, px = new Uint16Array(n);
    for (let i = 0; i < n; i++) {
      const c = ((d[4 * i] >> 3) << 10) | ((d[4 * i + 1] >> 3) << 5) | (d[4 * i + 2] >> 3);
      px[i] = c;
      this.hist[c]++;
    }
    this.frames.push({ px, delay });
  }

  // Makes the GIF file as a Uint8Array.
  encode() {
    const pal = medianCut(this.hist, 256);
    const map = new Int16Array(32768).fill(-1);
    const out = new ByteWriter();
    out.str("GIF89a");
    out.u16(this.w); out.u16(this.h);
    out.byte(0xf7); out.byte(0); out.byte(0);          // global table of 256 colours
    for (let i = 0; i < 256; i++) {
      const c = pal[i] ?? [0, 0, 0];
      out.byte(c[0]); out.byte(c[1]); out.byte(c[2]);
    }
    out.bytes([0x21, 0xff, 11]); out.str("NETSCAPE2.0"); out.bytes([3, 1, 0, 0, 0]);  // loop forever
    const idx = new Uint8Array(this.w * this.h);
    for (const f of this.frames) {
      for (let i = 0; i < idx.length; i++) {
        const c = f.px[i];
        let k = map[c];
        if (k < 0) k = map[c] = nearest(pal, c);
        idx[i] = k;
      }
      out.bytes([0x21, 0xf9, 4, 0]); out.u16(f.delay); out.bytes([0, 0]);
      out.byte(0x2c); out.u16(0); out.u16(0); out.u16(this.w); out.u16(this.h); out.byte(0);
      lzw(idx, out);
    }
    out.byte(0x3b);
    return out.result();
  }
}

// Splits the colour histogram into `n` boxes, each time the box with the
// most pixels along its longest side, and returns the mean colour of each box.
function medianCut(hist, n) {
  const cols = [];
  for (let c = 0; c < 32768; c++) if (hist[c]) cols.push(c);
  const ch = (c, k) => (c >> (10 - 5 * k)) & 31;
  const boxes = [cols];
  while (boxes.length < n) {
    let bi = -1, best = 0, axis = 0;
    boxes.forEach((b, i) => {
      if (b.length < 2) return;
      for (let k = 0; k < 3; k++) {
        let lo = 31, hi = 0, w = 0;
        for (const c of b) { const v = ch(c, k); if (v < lo) lo = v; if (v > hi) hi = v; w += hist[c]; }
        const score = (hi - lo) * Math.sqrt(w);
        if (score > best) { best = score; bi = i; axis = k; }
      }
    });
    if (bi < 0) break;
    const b = boxes[bi].sort((p, q) => ch(p, axis) - ch(q, axis));
    let total = 0, half = 0, cut = 1;
    for (const c of b) total += hist[c];
    for (let i = 0; i < b.length - 1; i++) { half += hist[b[i]]; if (half >= total / 2) { cut = i + 1; break; } }
    boxes.splice(bi, 1, b.slice(0, cut), b.slice(cut));
  }
  return boxes.map((b) => {
    let r = 0, g = 0, bl = 0, w = 0;
    for (const c of b) { const k = hist[c]; r += ch(c, 0) * k; g += ch(c, 1) * k; bl += ch(c, 2) * k; w += k; }
    return [r, g, bl].map((v) => Math.round((v / w) * 255 / 31));
  });
}

function nearest(pal, c) {
  const r = ((c >> 10) & 31) * 255 / 31, g = ((c >> 5) & 31) * 255 / 31, b = (c & 31) * 255 / 31;
  let best = 0, bd = Infinity;
  for (let i = 0; i < pal.length; i++) {
    const p = pal[i], d = 2 * (p[0] - r) ** 2 + 4 * (p[1] - g) ** 2 + 3 * (p[2] - b) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

// Image data: LZW codes of 8-bit indices, packed LSB first, in sub-blocks.
function lzw(idx, out) {
  const clear = 256, eoi = 257;
  out.byte(8);
  let size = 9, next = 258, acc = 0, nbits = 0;
  const block = [];
  const emit = (code) => {
    acc |= code << nbits; nbits += size;
    while (nbits >= 8) {
      block.push(acc & 255); acc >>>= 8; nbits -= 8;
      if (block.length === 255) { out.byte(255); out.bytes(block); block.length = 0; }
    }
  };
  const dict = new Map();
  emit(clear);
  let prefix = idx[0];
  for (let i = 1; i < idx.length; i++) {
    const k = idx[i], key = prefix * 256 + k, code = dict.get(key);
    if (code !== undefined) { prefix = code; continue; }
    emit(prefix);
    if (next < 4096) {
      dict.set(key, next++);
      if (next > (1 << size) && size < 12) size++;
    } else {
      emit(clear); dict.clear(); size = 9; next = 258;
    }
    prefix = k;
  }
  emit(prefix);
  emit(eoi);
  if (nbits > 0) block.push(acc & 255);
  if (block.length) { out.byte(block.length); out.bytes(block); }
  out.byte(0);
}

class ByteWriter {
  constructor() { this.buf = new Uint8Array(1 << 20); this.n = 0; }
  byte(b) {
    if (this.n === this.buf.length) { const nb = new Uint8Array(2 * this.n); nb.set(this.buf); this.buf = nb; }
    this.buf[this.n++] = b;
  }
  bytes(a) { for (const b of a) this.byte(b); }
  u16(v) { this.byte(v & 255); this.byte((v >> 8) & 255); }
  str(s) { for (let i = 0; i < s.length; i++) this.byte(s.charCodeAt(i)); }
  result() { return this.buf.subarray(0, this.n); }
}
