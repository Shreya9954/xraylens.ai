// Image handling for XRayLens in the browser.
//
// Everything that feeds the model is a line-by-line port of the Pillow code the
// model card's inference.py uses (Resample.c fixed-point kernels, ImageOps.pad,
// Geometry.c affine bilinear, ImagingBlend), so browser inputs are bit-identical
// to PyTorch's. tests/test_web_parity.py checks this in a real browser.

import { INFERNO } from "./inferno.js";

// An image is { w, h, c, data: Uint8Array } with c = 1 (grey) or 3 (RGB), row-major, interleaved.
export const makeImage = (w, h, c, data) => ({ w, h, c, data: data || new Uint8Array(w * h * c) });

// Python's round(): ties go to the even neighbour.
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ── decoding ──────────────────────────────────────────────────────────────────
export async function decodeBlob(blob) {
  // No colour management and no premultiplication: PIL applies neither.
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none",
                                              imageOrientation: "from-image" });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
  bmp.close();
  const out = makeImage(canvas.width, canvas.height, 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    out.data[j] = rgba[i]; out.data[j + 1] = rgba[i + 1]; out.data[j + 2] = rgba[i + 2];
  }
  return out;
}

export function toRGB(img) {
  if (img.c === 3) return img;
  const out = makeImage(img.w, img.h, 3);
  for (let i = 0; i < img.w * img.h; i++) out.data[3 * i] = out.data[3 * i + 1] = out.data[3 * i + 2] = img.data[i];
  return out;
}

// PIL "L" conversion: ITU-R 601-2 luma in 16.16 fixed point.
export function toGray(img) {
  if (img.c === 1) return img;
  const out = makeImage(img.w, img.h, 1);
  const d = img.data;
  for (let i = 0; i < img.w * img.h; i++) {
    out.data[i] = (d[3 * i] * 19595 + d[3 * i + 1] * 38470 + d[3 * i + 2] * 7471 + 0x8000) >>> 16;
  }
  return out;
}

// ── Pillow Resample.c ─────────────────────────────────────────────────────────
const PRECISION_BITS = 32 - 8 - 2;
const FILTERS = {
  bilinear: { support: 1.0, fn: (x) => { x = Math.abs(x); return x < 1 ? 1 - x : 0; } },
  bicubic: { support: 2.0, fn: (x) => {
    const a = -0.5; x = Math.abs(x);
    if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
    return 0;
  } },
};

function coeffs(inSize, outSize, filter) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1.0);
  const support = filter.support * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const kk = new Int32Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  const ss = 1.0 / filterscale;
  const pre = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) { const w = filter.fn((x + xmin - center + 0.5) * ss); pre[x] = w; ww += w; }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? pre[x] / ww : pre[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * (1 << PRECISION_BITS)) : Math.trunc(0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[2 * xx] = xmin; bounds[2 * xx + 1] = xmax;
  }
  return { kk, bounds, ksize };
}

const clip8 = (ss) => { const v = Math.floor(ss / (1 << PRECISION_BITS)); return v < 0 ? 0 : v > 255 ? 255 : v; };

function resampleH(img, outW, filter) {
  const { kk, bounds, ksize } = coeffs(img.w, outW, filter);
  const out = makeImage(outW, img.h, img.c);
  const c = img.c;
  for (let y = 0; y < img.h; y++) {
    const row = y * img.w * c;
    for (let xx = 0; xx < outW; xx++) {
      const xmin = bounds[2 * xx], xmax = bounds[2 * xx + 1], k = xx * ksize;
      for (let ch = 0; ch < c; ch++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < xmax; x++) ss += img.data[row + (x + xmin) * c + ch] * kk[k + x];
        out.data[(y * outW + xx) * c + ch] = clip8(ss);
      }
    }
  }
  return out;
}

function resampleV(img, outH, filter) {
  const { kk, bounds, ksize } = coeffs(img.h, outH, filter);
  const out = makeImage(img.w, outH, img.c);
  const stride = img.w * img.c;
  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[2 * yy], ymax = bounds[2 * yy + 1], k = yy * ksize;
    for (let i = 0; i < stride; i++) {
      let ss = 1 << (PRECISION_BITS - 1);
      for (let y = 0; y < ymax; y++) ss += img.data[(y + ymin) * stride + i] * kk[k + y];
      out.data[yy * stride + i] = clip8(ss);
    }
  }
  return out;
}

export function resize(img, w, h, method = "bilinear") {
  const f = FILTERS[method];
  let out = img;
  if (w !== img.w) out = resampleH(out, w, f);
  if (h !== img.h) out = resampleV(out, h, f);
  return out === img ? makeImage(img.w, img.h, img.c, img.data.slice()) : out;
}

// ImageOps.pad(image, (size, size), BILINEAR, color=black, centering=(0.5, 0.5)).
export function letterbox(img, size) {
  const rgb = toRGB(img);
  let w = size, h = size;
  const ratio = rgb.w / rgb.h;
  if (ratio > 1) h = roundHalfEven(rgb.h / rgb.w * size);
  else if (ratio < 1) w = roundHalfEven(rgb.w / rgb.h * size);
  const resized = resize(rgb, w, h, "bilinear");
  const out = makeImage(size, size, 3);
  const x0 = w !== size ? roundHalfEven((size - w) * 0.5) : 0;
  const y0 = h !== size ? roundHalfEven((size - h) * 0.5) : 0;
  for (let y = 0; y < h; y++) out.data.set(resized.data.subarray(y * w * 3, (y + 1) * w * 3), ((y + y0) * size + x0) * 3);
  return out;
}

// torchvision ToTensor + Normalize, written into a (n, 3, H, W) batch at index b.
export function writeTensor(boxed, mean, std, target, b = 0) {
  const n = boxed.w * boxed.h, off = b * 3 * n;
  for (let i = 0; i < n; i++) {
    for (let ch = 0; ch < 3; ch++) target[off + ch * n + i] = (boxed.data[3 * i + ch] / 255 - mean[ch]) / std[ch];
  }
}

// ── explanation helpers (mirror xraylens/explainability.py) ──────────────────
// torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)
export function upsampleCam(cam, inH, inW, outH, outW) {
  const out = new Float32Array(outH * outW);
  const sy = inH / outH, sx = inW / outW;
  for (let y = 0; y < outH; y++) {
    let fy = (y + 0.5) * sy - 0.5; if (fy < 0) fy = 0;
    const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, inH - 1), ly = fy - y0;
    for (let x = 0; x < outW; x++) {
      let fx = (x + 0.5) * sx - 0.5; if (fx < 0) fx = 0;
      const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, inW - 1), lx = fx - x0;
      const top = cam[y0 * inW + x0] * (1 - lx) + cam[y0 * inW + x1] * lx;
      const bot = cam[y1 * inW + x0] * (1 - lx) + cam[y1 * inW + x1] * lx;
      out[y * outW + x] = top * (1 - ly) + bot * ly;
    }
  }
  let peak = 0;
  for (const v of out) if (v > peak) peak = v;
  if (peak > 1e-12) for (let i = 0; i < out.length; i++) out[i] /= peak; else out.fill(0);
  return out;
}

export function contentMask(boxed) {
  const g = toGray(boxed), { w, h } = g;
  const rows = new Array(h).fill(false), cols = new Array(w).fill(false);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (g.data[y * w + x] > 2) { rows[y] = true; cols[x] = true; }
  const mask = new Uint8Array(w * h);
  const r0 = rows.indexOf(true), c0 = cols.indexOf(true);
  if (r0 < 0 || c0 < 0) return { mask, box: null };
  const r1 = h - rows.slice().reverse().indexOf(true), c1 = w - cols.slice().reverse().indexOf(true);
  for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) mask[y * w + x] = 1;
  return { mask, box: [r0, r1, c0, c1] };
}

export function attributionStats(cam, boxed, lung) {
  const n = cam.length;
  let total = 0; for (const v of cam) total += v; total = total || 1;
  const { mask, box } = contentMask(boxed);
  let pad = 0, padArea = 0;
  for (let i = 0; i < n; i++) if (!mask[i]) { pad += cam[i]; padArea++; }
  const s = { padding: pad / total, padding_area: padArea / n, border: 0, border_area: 0 };
  if (box) {
    const [r0, r1, c0, c1] = box, mh = Math.trunc((r1 - r0) * 0.12), mw = Math.trunc((c1 - c0) * 0.12), w = boxed.w;
    let b = 0, bArea = 0;
    for (let y = r0; y < r1; y++) for (let x = c0; x < c1; x++) {
      const inner = y >= r0 + mh && y < r1 - mh && x >= c0 + mw && x < c1 - mw;
      if (!inner) { b += cam[y * w + x]; bArea++; }
    }
    s.border = b / total; s.border_area = bArea / n;
  }
  if (lung) {
    let l = 0, la = 0;
    for (let i = 0; i < n; i++) if (lung[i]) { l += cam[i]; la++; }
    s.lungs = l / total; s.lungs_area = la / n;
  }
  return s;
}

// matplotlib lookup: index = min(floor(t * N), N - 1)
const inferno = (t) => INFERNO[Math.min(Math.floor(Math.min(Math.max(t, 0), 1) * 256), 255)];

// Same blending as explainability.overlay: weak activations stay transparent.
export function overlayToCanvas(canvas, boxed, cam, alpha, lung) {
  const { w, h } = boxed;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(w, h);
  let outline = null;
  if (lung) outline = lungOutline(lung, w, h);
  for (let i = 0; i < w * h; i++) {
    const a = cam ? alpha * Math.min(Math.max(cam[i] * 1.4, 0), 1) : 0;
    const heat = cam ? inferno(cam[i]) : [0, 0, 0];
    for (let ch = 0; ch < 3; ch++) out.data[4 * i + ch] = boxed.data[3 * i + ch] * (1 - a) + heat[ch] * a;
    out.data[4 * i + 3] = 255;
    if (outline && outline[i]) { out.data[4 * i] = 56; out.data[4 * i + 1] = 165; out.data[4 * i + 2] = 150; }
  }
  ctx.putImageData(out, 0, 0);
}

function lungOutline(mask, w, h) {
  // mask & ~erode(mask, 5x5), as in app_state.lung_outline
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue;
    let edge = false;
    for (let dy = -2; dy <= 2 && !edge; dy++) for (let dx = -2; dx <= 2; dx++) {
      const yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w || !mask[yy * w + xx]) { edge = true; break; }
    }
    out[y * w + x] = edge ? 1 : 0;
  }
  return out;
}

export function imageToCanvas(canvas, img) {
  const rgb = toRGB(img);
  canvas.width = rgb.w; canvas.height = rgb.h;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(rgb.w, rgb.h);
  for (let i = 0; i < rgb.w * rgb.h; i++) {
    out.data[4 * i] = rgb.data[3 * i]; out.data[4 * i + 1] = rgb.data[3 * i + 1]; out.data[4 * i + 2] = rgb.data[3 * i + 2]; out.data[4 * i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

// ── perturbations (mirror xraylens/perturbations.py) ─────────────────────────
const gray = (img) => toGray(img);

export function hflip(img) {
  const g = gray(img), out = makeImage(g.w, g.h, 1);
  for (let y = 0; y < g.h; y++) for (let x = 0; x < g.w; x++) out.data[y * g.w + x] = g.data[y * g.w + (g.w - 1 - x)];
  return out;
}

export function invert(img) {
  const g = gray(img), out = makeImage(g.w, g.h, 1);
  for (let i = 0; i < g.data.length; i++) out.data[i] = 255 - g.data[i];
  return out;
}

// Image.rotate(10, BILINEAR, fillcolor=0): Pillow's affine matrix + Geometry.c bilinear_filter8.
export function rotate10(img) {
  const g = gray(img), { w, h } = g, out = makeImage(w, h, 1);
  const ang = -10 * Math.PI / 180;
  const r15 = (v) => Math.round(v * 1e15) / 1e15;
  const m = [r15(Math.cos(ang)), r15(Math.sin(ang)), 0, r15(-Math.sin(ang)), r15(Math.cos(ang)), 0];
  const cx = w / 2, cy = h / 2;
  m[2] = m[0] * -cx + m[1] * -cy + cx;
  m[5] = m[3] * -cx + m[4] * -cy + cy;
  const clipX = (x) => (x < 0 ? 0 : x >= w ? w - 1 : x), clipY = (y) => (y < 0 ? 0 : y >= h ? h - 1 : y);
  for (let y = 0; y < h; y++) {
    let xin = m[0] * 0.5 + m[1] * (y + 0.5) + m[2];
    let yin = m[3] * 0.5 + m[4] * (y + 0.5) + m[5];
    for (let x = 0; x < w; x++, xin += m[0], yin += m[3]) {
      if (xin < 0 || xin >= w || yin < 0 || yin >= h) { out.data[y * w + x] = 0; continue; }
      const xs = xin - 0.5, ys = yin - 0.5, x0 = Math.floor(xs), y0 = Math.floor(ys), dx = xs - x0, dy = ys - y0;
      const r = clipY(y0) * w, a0 = clipX(x0), a1 = clipX(x0 + 1);
      let v1 = g.data[r + a0] + (g.data[r + a1] - g.data[r + a0]) * dx;
      let v2 = v1;
      if (y0 + 1 >= 0 && y0 + 1 < h) { const r2 = (y0 + 1) * w; v2 = g.data[r2 + a0] + (g.data[r2 + a1] - g.data[r2 + a0]) * dx; }
      out.data[y * w + x] = Math.trunc(v1 + (v2 - v1) * dy);
    }
  }
  return out;
}

// ImageEnhance.Contrast(0.55): blend toward the rounded mean grey with Pillow's truncation.
export function lowContrast(img) {
  const g = gray(img);
  let sum = 0; for (const v of g.data) sum += v;
  const mean = Math.trunc(sum / g.data.length + 0.5), out = makeImage(g.w, g.h, 1);
  // ImagingBlend takes alpha as a C float and evaluates in single precision.
  const f = Math.fround, alpha = f(0.55);
  for (let i = 0; i < g.data.length; i++) out.data[i] = Math.trunc(f(mean + f(alpha * (g.data[i] - mean))));
  return out;
}

export function cropCenter(img) {
  const g = gray(img);
  const x0 = Math.trunc(g.w * 0.15), y0 = Math.trunc(g.h * 0.15), x1 = Math.trunc(g.w * 0.85), y1 = Math.trunc(g.h * 0.85);
  const out = makeImage(x1 - x0, y1 - y0, 1);
  for (let y = y0; y < y1; y++) out.data.set(g.data.subarray(y * g.w + x0, y * g.w + x1), (y - y0) * out.w);
  return out;
}

// Pillow approximates a Gaussian with three box blurs; we use a true Gaussian of the same radius.
export function blur(img) {
  const g = gray(img), sigma = Math.max(g.w, g.h) / 300;
  const r = Math.ceil(sigma * 3), k = [];
  let ks = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); ks += v; }
  const pass = (src, w, h, horiz) => {
    const dst = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -r; i <= r; i++) {
        const xx = horiz ? Math.min(Math.max(x + i, 0), w - 1) : x, yy = horiz ? y : Math.min(Math.max(y + i, 0), h - 1);
        s += src[yy * w + xx] * k[i + r];
      }
      dst[y * w + x] = s / ks;
    }
    return dst;
  };
  const t = pass(pass(g.data, g.w, g.h, true), g.w, g.h, false);
  const out = makeImage(g.w, g.h, 1);
  for (let i = 0; i < t.length; i++) out.data[i] = Math.min(255, Math.max(0, Math.round(t[i])));
  return out;
}

// Seeded Gaussian noise (σ = 18). The stream differs from NumPy's, so per-pixel values differ.
export function noise(img) {
  const g = gray(img), out = makeImage(g.w, g.h, 1);
  let s = 0x9e3779b9;
  const rnd = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return ((s >>> 0) + 0.5) / 4294967296; };
  for (let i = 0; i < g.data.length; i++) {
    const n = Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd());
    out.data[i] = Math.min(255, Math.max(0, Math.trunc(g.data[i] + 18 * n)));
  }
  return out;
}

export function marker(img) {
  const g = gray(img), size = Math.max(12, Math.trunc(g.w / 14));
  const canvas = new OffscreenCanvas(g.w, g.h), ctx = canvas.getContext("2d");
  const id = ctx.createImageData(g.w, g.h);
  for (let i = 0; i < g.data.length; i++) { id.data[4 * i] = id.data[4 * i + 1] = id.data[4 * i + 2] = g.data[i]; id.data[4 * i + 3] = 255; }
  ctx.putImageData(id, 0, 0);
  ctx.textBaseline = "top";
  ctx.font = `${size}px sans-serif`;
  ctx.fillStyle = "rgb(255,255,255)"; ctx.fillText("L", Math.trunc(g.w * 0.78), Math.trunc(g.h * 0.04));
  ctx.fillStyle = "rgb(235,235,235)"; ctx.fillText("PORTABLE", Math.trunc(g.w * 0.04), Math.trunc(g.h * 0.04));
  const px = ctx.getImageData(0, 0, g.w, g.h).data, out = makeImage(g.w, g.h, 1);
  for (let i = 0; i < g.data.length; i++) out.data[i] = px[4 * i];
  return out;
}

export const PERTURBATIONS = [
  ["hflip", "Mirror left–right", hflip, true],
  ["rotate10", "Rotate 10°", rotate10, true],
  ["crop", "Tighter collimation (70% crop)", cropCenter, true],
  ["contrast", "Low contrast (×0.55)", lowContrast, true],
  ["blur", "Mild blur", blur, false],
  ["noise", "Sensor noise (σ = 18)", noise, false],
  ["marker", "Burned-in text markers", marker, false],
  ["invert", "Inverted greyscale", invert, true],
]; // 4th field: bit-exact with the Python version

// ── input checks (mirror xraylens/inputs.py) ─────────────────────────────────
export class InputError extends Error {}

export function assess(img) {
  const { w, h } = img;
  if (Math.min(w, h) < 64) throw new InputError(`The image is ${w}×${h} px. It needs to be at least 64 px on each side.`);
  const small = resize(toRGB(img), 128, 128, "bicubic");
  let mean = 0; for (const v of small.data) mean += v; mean /= small.data.length;
  let vs = 0; for (const v of small.data) vs += (v - mean) ** 2;
  if (Math.sqrt(vs / small.data.length) < 2) throw new InputError("The image is blank (a single flat colour).");
  const warnings = [];
  let chroma = 0;
  for (let i = 0; i < 128 * 128; i++) {
    const r = small.data[3 * i], g = small.data[3 * i + 1], b = small.data[3 * i + 2];
    chroma += Math.max(r, g, b) - Math.min(r, g, b);
  }
  if (chroma / (128 * 128) > 12) warnings.push("This image contains colour. Radiographs are greyscale, so it may not be an X-ray. The model will still produce a confident-looking score.");
  const ratio = Math.max(w, h) / Math.min(w, h);
  if (ratio > 1.8) warnings.push(`Unusual aspect ratio (${ratio.toFixed(1)}:1). The image is letterboxed with black bars, and the model saw few images shaped like this.`);
  if (Math.min(w, h) < 256) warnings.push("Low resolution. The model works at 320 px, so fine detail is lost.");
  if (looksInverted(img)) warnings.push("The image looks inverted (bone dark, air bright). The model was trained on bone-is-white radiographs.");
  return warnings;
}

function looksInverted(img) {
  const g = resize(toGray(img), 128, 128, "bicubic").data;
  const at = (y, x) => g[y * 128 + x];
  const meanRect = (y0, y1, x0, x1) => { let s = 0, n = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { s += at(y, x); n++; } return s / n; };
  let bs = 0, bn = 0;
  for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) if (y < 8 || y >= 120 || x < 8 || x >= 120) {
    // inputs.py concatenates the four 8-px strips, so corner pixels are counted twice
    const times = (y < 8 || y >= 120 ? 1 : 0) + (x < 8 || x >= 120 ? 1 : 0);
    bs += at(y, x) * times; bn += times;
  }
  const darkCentre = meanRect(40, 88, 40, 88) < bs / bn - 40;
  const lungs = (meanRect(32, 77, 26, 45) + meanRect(32, 77, 83, 102)) / 2;
  const darkMidline = meanRect(26, 100, 58, 70) < lungs - 5;
  return darkCentre || darkMidline;
}

// ── DICOM (uncompressed transfer syntaxes only) ──────────────────────────────
export function decodeDicom(buffer, dicomParser) {
  let ds;
  try { ds = dicomParser.parseDicom(new Uint8Array(buffer)); }
  catch { throw new InputError("This DICOM file could not be parsed."); }
  const ts = ds.string("x00020010") || "";
  if (!["1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"].includes(ts.trim()))
    throw new InputError("This DICOM uses a compressed transfer syntax, which the browser version can't decode. Export it as PNG.");
  const rows = ds.uint16("x00280010"), cols = ds.uint16("x00280011"), bits = ds.uint16("x00280100");
  const spp = ds.uint16("x00280002") || 1, frames = parseInt(ds.string("x00280008") || "1", 10);
  const signed = ds.uint16("x00280103") === 1, photo = (ds.string("x00280004") || "").trim();
  const el = ds.elements.x7fe00010;
  if (!el || !rows || !cols) throw new InputError("This DICOM has no readable pixel data.");
  if (spp !== 1) throw new InputError("Colour DICOM images are not supported.");
  const n = rows * cols, frame = Math.floor(frames / 2), start = el.dataOffset + frame * n * (bits / 8);
  const view = new DataView(ds.byteArray.buffer, ds.byteArray.byteOffset);
  const arr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    arr[i] = bits === 8 ? ds.byteArray[start + i]
      : signed ? view.getInt16(start + 2 * i, true) : view.getUint16(start + 2 * i, true);
  }
  let max = -Infinity; for (const v of arr) if (v > max) max = v;
  if (photo === "MONOCHROME1") for (let i = 0; i < n; i++) arr[i] = max - arr[i];
  const sorted = Float32Array.from(arr).sort();
  const pct = (p) => { const k = (sorted.length - 1) * p, f = Math.floor(k), c = Math.min(f + 1, sorted.length - 1); return sorted[f] + (sorted[c] - sorted[f]) * (k - f); };
  const lo = pct(0.005), hi = pct(0.995), span = Math.max(hi - lo, 1e-6);
  const out = makeImage(cols, rows, 1);
  for (let i = 0; i < n; i++) out.data[i] = Math.trunc(Math.min(Math.max((arr[i] - lo) / span, 0), 1) * 255);
  const meta = {};
  for (const [tag, name] of [["x00080060", "Modality"], ["x00185101", "ViewPosition"], ["x00180015", "BodyPartExamined"]]) {
    const v = ds.string(tag); if (v) meta[name] = v.trim();
  }
  return { image: out, meta };
}
