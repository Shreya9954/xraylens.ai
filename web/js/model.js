// In-browser inference with ONNX Runtime Web.
//
// The graph (model/xraylens.onnx, ~66 KB) references every weight by byte range
// inside the model author's original model.safetensors, which we download straight
// from Hugging Face at a pinned commit. We never re-host the weights.
import * as ort from "../vendor/ort/ort.wasm.min.mjs";
import { letterbox, writeTensor } from "./imaging.js";

export const REPO = "Purino/resnet50-chest-xray-pneumonia";
export const REVISION = "da2a7724bfa53ca6693fd4e1783b211963ed2a55";
export const WEIGHTS_URL = `https://huggingface.co/${REPO}/resolve/${REVISION}/model.safetensors`;
export const WEIGHTS_BYTES = 94289936;
export const CONFIG = { classes: ["NORMAL", "PNEUMONIA"], imageSize: 320, threshold: 0.5,
                        mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] };
const CACHE = "xraylens-weights-v1";

ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
// Threads need cross-origin isolation (COOP/COEP headers, see vercel.json). Without it, one thread.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1;

async function fetchWeights(onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* private mode: no Cache Storage */ }
  if (cache) {
    const hit = await cache.match(WEIGHTS_URL);
    if (hit) { onProgress?.(1, "cached"); return new Uint8Array(await hit.arrayBuffer()); }
  }
  const res = await fetch(WEIGHTS_URL, { mode: "cors" });
  if (!res.ok || !res.body) throw new Error(`Hugging Face returned ${res.status} for the model weights.`);
  const total = Number(res.headers.get("content-length")) || WEIGHTS_BYTES;
  const buf = new Uint8Array(total);
  const reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    onProgress?.(got / total, "download");
  }
  if (got !== WEIGHTS_BYTES) throw new Error(`Expected ${WEIGHTS_BYTES} bytes of weights, got ${got}.`);
  if (cache) {
    try { await cache.put(WEIGHTS_URL, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); }
    catch { /* quota exceeded: fine, we just re-download next time */ }
  }
  return buf;
}

let sessionPromise = null;

export function loadModel(onProgress) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const [graph, weights] = await Promise.all([
        fetch(new URL("../model/xraylens.onnx", import.meta.url)).then((r) => r.arrayBuffer()),
        fetchWeights(onProgress),
      ]);
      onProgress?.(1, "compile");
      const session = await ort.InferenceSession.create(graph, {
        executionProviders: ["wasm"],
        externalData: [{ path: "model.safetensors", data: weights }],
        graphOptimizationLevel: "all",
      });
      onProgress?.(1, "ready");
      return session;
    })().catch((e) => { sessionPromise = null; throw e; });
  }
  return sessionPromise;
}

const softmax = (a, b) => { const m = Math.max(a, b), ea = Math.exp(a - m), eb = Math.exp(b - m); return [ea / (ea + eb), eb / (ea + eb)]; };

// Run a batch of decoded images. Returns per image: boxed input, scores, logits, raw 10x10 CAMs.
export async function infer(images) {
  const session = await loadModel();
  const S = CONFIG.imageSize, n = images.length, plane = 3 * S * S;
  const data = new Float32Array(n * plane);
  const boxed = images.map((im, i) => { const b = letterbox(im, S); writeTensor(b, CONFIG.mean, CONFIG.std, data, i); return b; });
  const out = await session.run({ input: new ort.Tensor("float32", data, [n, 3, S, S]) });
  const L = out.logits.data, C = out.cams.data, [, , ch, cw] = out.cams.dims;
  return boxed.map((b, i) => {
    const z = [L[2 * i], L[2 * i + 1]];
    const [pn, pp] = softmax(z[0], z[1]);
    const cams = [0, 1].map((c) => C.slice((i * 2 + c) * ch * cw, (i * 2 + c + 1) * ch * cw));
    return { boxed: b, logits: z, scores: { NORMAL: pn, PNEUMONIA: pp }, pneumonia: pp,
             label: pp >= CONFIG.threshold ? "PNEUMONIA" : "NORMAL", cams, camDims: [ch, cw] };
  });
}

export const backendInfo = () => ({ threads: ort.env.wasm.numThreads, isolated: self.crossOriginIsolated });
