// Examine page: steps 01–04, running the classifier in the browser.
import { $, esc, pct, note, stats, getJSON, getScores, chart, COLORS } from "./common.js";
import { decodeBlob, letterbox, toGray, assess, InputError, upsampleCam, attributionStats, overlayToCanvas,
         imageToCanvas, PERTURBATIONS, decodeDicom } from "./imaging.js";
import { loadModel, infer, CONFIG, WEIGHTS_BYTES, backendInfo } from "./model.js";

const [gallery, metrics, scores, randPop] = await Promise.all([
  getJSON("gallery/gallery.json"), getJSON("data/metrics.json"), getScores(), getJSON("data/randomisation.json"),
]);
const rsna = scores.filter((r) => r.cohort === "rsna");
const state = { token: 0 };
let modelReady = false;

// ── live model status ─────────────────────────────────────────────────────────
const statusEl = $("#model-status");
function setStatus(html, frac) {
  statusEl.innerHTML = `${html}<span class="bar"><i style="width:${Math.round((frac ?? 0) * 100)}%"></i></span>`;
}
function startModel() {
  return loadModel((f, phase) => {
    if (phase === "download") setStatus(`Live model · downloading weights from Hugging Face <b>${pct(f, 0)}</b> of ${(WEIGHTS_BYTES / 1e6).toFixed(0)} MB`, f);
    else if (phase === "cached") setStatus("Live model · weights cached on this device", 1);
    else if (phase === "compile") setStatus("Live model · <b>compiling</b> ResNet-50…", 1);
  }).then(() => {
    const b = backendInfo();
    modelReady = true;
    setStatus(`Live model · <b>ready</b> · ONNX Runtime Web, ${b.threads} thread${b.threads > 1 ? "s" : ""}, on this device`, 1);
    if (state.kind === "gallery") liveCheck(state.token);
  }).catch((e) => {
    setStatus(`Live model · <b>failed to load</b>: ${esc(e.message)}. Gallery results still work.`, 0);
    throw e;
  });
}
const saveData = navigator.connection && (navigator.connection.saveData || /2g/.test(navigator.connection.effectiveType || ""));
let modelPromise = null;
const ensureModel = () => (modelPromise ||= startModel());
if (!saveData) ensureModel();
else setStatus('Live model · not loaded (data saver on) · <a href="#" id="load-model">load 94 MB</a>', 0);
document.addEventListener("click", (e) => { if (e.target.id === "load-model") { e.preventDefault(); ensureModel(); } });

// ── 01 contact sheet and upload ──────────────────────────────────────────────
function renderSheet() {
  $("#sheet").innerHTML = gallery.map((g) => `
    <div class="card${state.id === g.id ? " active" : ""}">
      <img src="gallery/${g.thumb}" alt="${esc(g.title)}" loading="lazy" width="240" height="300">
      <div class="cap"><b>${esc(g.title)}</b>${esc(g.subtitle)}</div>
      <button class="btn${state.id === g.id ? " primary" : ""}" data-id="${g.id}">${state.id === g.id ? "● Loaded" : "Load"}</button>
    </div>`).join("");
}
$("#sheet").addEventListener("click", (e) => { const id = e.target.closest("button")?.dataset.id; if (id) selectGallery(id); });

const drop = $("#drop");
$("#file").addEventListener("change", (e) => e.target.files[0] && handleUpload(e.target.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0]); });

const softmax = ([a, b]) => { const m = Math.max(a, b), ea = Math.exp(a - m), eb = Math.exp(b - m); return [ea / (ea + eb), eb / (ea + eb)]; };

async function maskFromUrl(url) {
  const m = letterbox(await decodeBlob(await (await fetch(url)).blob()), CONFIG.imageSize);
  const g = toGray(m).data, out = new Uint8Array(g.length);
  for (let i = 0; i < g.length; i++) out[i] = g[i] > 127 ? 1 : 0;
  return out;
}

async function selectGallery(id) {
  const token = ++state.token;
  const item = gallery.find((g) => g.id === id);
  Object.assign(state, { kind: "gallery", id, item, meta: {}, stressLive: null });
  renderSheet();
  $("#error").innerHTML = "";
  const img = await decodeBlob(await (await fetch(`gallery/${item.file}`)).blob());
  const lung = item.mask ? await maskFromUrl(`gallery/${item.mask}`) : null;
  if (token !== state.token) return;
  const pre = item.precomputed;
  const [pn, pp] = softmax(pre.logits);
  Object.assign(state, { img, lung, warnings: assess(img), boxed: letterbox(img, CONFIG.imageSize),
    result: { logits: pre.logits, scores: { NORMAL: pn, PNEUMONIA: pp }, pneumonia: pp, label: pp >= CONFIG.threshold ? "PNEUMONIA" : "NORMAL",
              cams: pre.cams.map((c) => Float32Array.from(c)), camDims: [10, 10], source: "pytorch" } });
  render();
  if (modelReady) liveCheck(token);
}

// Re-run a gallery case through the in-browser model and compare with PyTorch.
async function liveCheck(token) {
  if (state.kind !== "gallery" || !state.img) return;
  const [r] = await infer([state.img]);
  if (token !== state.token) return;
  state.live = { pneumonia: r.pneumonia, delta: Math.abs(r.pneumonia - state.result.pneumonia) };
  renderLive();
}

async function handleUpload(file) {
  const token = ++state.token;
  Object.assign(state, { kind: "upload", id: null, item: null, lung: null, live: null, stressLive: null });
  renderSheet();
  $("#error").innerHTML = "";
  try {
    if (file.size > 20 * 1024 * 1024) throw new InputError(`The file is ${(file.size / 1e6).toFixed(1)} MB. The limit is 20 MB.`);
    if (!file.size) throw new InputError("The file is empty.");
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const buf = await file.arrayBuffer();
    const isDicom = ["dcm", "dicom"].includes(ext) || (buf.byteLength > 132 && new TextDecoder().decode(new Uint8Array(buf, 128, 4)) === "DICM");
    let img, meta = {};
    if (isDicom) {
      await loadScript("vendor/dicom-parser/dicomParser.min.js");
      ({ image: img, meta } = decodeDicom(buf, window.dicomParser));
    } else {
      if (ext && !["png", "jpg", "jpeg", "bmp", "webp"].includes(ext)) throw new InputError(`'.${ext}' files are not supported. Use PNG, JPEG or DICOM.`);
      try { img = await decodeBlob(new Blob([buf], { type: file.type })); }
      catch { throw new InputError("This file is not a readable image, or it is truncated."); }
      if (img.w * img.h > 50_000_000) throw new InputError("The image has too many pixels to process safely.");
    }
    const warnings = assess(img);
    if (token !== state.token) return;
    Object.assign(state, { img, meta, warnings, source: file.name, boxed: letterbox(img, CONFIG.imageSize), result: null });
    renderPending();
    await ensureModel();
    const [r] = await infer([img]);
    if (token !== state.token) return;
    state.result = { ...r, source: "browser" };
    render();
    runLiveStress(token);
  } catch (e) {
    if (token !== state.token) return;
    $("#error").innerHTML = note(e instanceof InputError ? `<b>Can’t analyse this file.</b> ${esc(e.message)}`
      : `<b>Something went wrong.</b> ${esc(e.message)}`, "warn");
  }
}

function loadScript(src) {
  if (document.querySelector(`script[src="${src}"]`)) return Promise.resolve();
  return new Promise((ok, fail) => { const s = document.createElement("script"); s.src = src; s.onload = ok; s.onerror = fail; document.head.append(s); });
}

// ── 02 output ─────────────────────────────────────────────────────────────────
function renderPending() {
  imageToCanvas($("#input-canvas"), state.boxed);
  $("#readout").innerHTML = `<div class="readout"><div class="lab">Model output</div><div class="call"><span class="spin"></span></div>
    <p class="fig">Waiting for the in-browser model. First visit downloads 94 MB of weights from Hugging Face; later visits use the cached copy.</p></div>`;
  for (const id of ["#context", "#story", "#attr", "#attr-note", "#stress-head", "#stress-note", "#thumbs", "#stress-chart", "#cam-cap", "#rand-body"]) $(id).innerHTML = "";
}

function render() {
  const { result: r, boxed, img } = state;
  imageToCanvas($("#input-canvas"), boxed);
  $("#input-cap").innerHTML = `<b>MODEL INPUT</b> · ${img.w}×${img.h} source → letterboxed to 320×320, black bars added · ${esc(state.item ? state.item.title : state.source)}`;
  const p = r.pneumonia, sat = p < 1e-3 || p > 1 - 1e-3;
  $("#readout").innerHTML = `
    <div class="readout">
      <div class="lab">Model output · class with score ≥ threshold</div>
      <div class="call ${r.label.toLowerCase()}">${r.label === "PNEUMONIA" ? "Pneumonia" : "Normal"}</div>
      <div class="track" role="img" aria-label="Pneumonia score ${p.toFixed(4)} against threshold 0.5">
        <div class="bar"></div><div class="thr" style="left:50%"><span>threshold 0.5</span></div><div class="pin" style="left:${(p * 100).toFixed(2)}%"></div>
      </div>
      <div class="axis"><span>0 · NORMAL</span><span>PNEUMONIA · 1</span></div>
      <dl class="kv">
        <dt>PNEUMONIA score</dt><dd>${p.toFixed(6)}</dd>
        <dt>NORMAL score</dt><dd>${r.scores.NORMAL.toFixed(6)}</dd>
        <dt>Logit margin (z₁ − z₀)</dt><dd>${(r.logits[1] - r.logits[0] >= 0 ? "+" : "") + (r.logits[1] - r.logits[0]).toFixed(2)}</dd>
        <dt>Saturated (&lt;0.001 or &gt;0.999)</dt><dd>${sat ? "yes" : "no"}</dd>
        <dt>Computed by</dt><dd id="live-line">${r.source === "browser" ? "your browser (ONNX Runtime Web)" : "PyTorch (precomputed) · live check pending"}</dd>
      </dl>
    </div>`;
  const shareN = rsna.filter((x) => x.finding === "Normal" && x.score >= p).length / 200;
  const shareO = rsna.filter((x) => x.finding === "Lung Opacity" && x.score >= p).length / 200;
  $("#context").innerHTML = note(`<b>What the score is, and what it isn’t.</b> ${p.toFixed(4)} is the network’s output for this image, not the probability that a patient has pneumonia. For context, from our own external test: <b>${pct(shareN, 0)}</b> of 200 adult radiographs read as <i>normal</i> (RSNA) scored at least this high, as did <b>${pct(shareO, 0)}</b> of 200 with lung opacity.`);
  let story = "";
  if (state.item) {
    story += `<div style="margin-top:.6rem">${state.item.tags.map(([t, c]) => `<span class="tag" style="color:${c}">${esc(t)}</span>`).join("")}</div>`;
    story += note(state.item.story, state.item.seen_in_training ? "warn" : "cool");
  }
  for (const w of state.warnings) story += note(`<b>Input check.</b> ${esc(w)}`, "warn");
  if (state.meta && Object.keys(state.meta).length) story += `<div class="fig">DICOM · ${Object.entries(state.meta).map(([k, v]) => `${k} ${esc(v)}`).join(" · ")}</div>`;
  $("#story").innerHTML = story;
  renderLive();
  renderExplain();
  renderStress();
}

function renderLive() {
  const el = $("#live-line");
  if (!el || state.kind !== "gallery") return;
  if (state.live) el.innerHTML = `PyTorch ${state.result.pneumonia.toFixed(6)} · your browser ${state.live.pneumonia.toFixed(6)} · |Δ| ${state.live.delta.toExponential(1)}`;
  else el.textContent = modelReady ? "PyTorch (precomputed) · checking live…" : "PyTorch (precomputed) · live check when the model loads";
}

// ── 03 explanation ───────────────────────────────────────────────────────────
let cls = "pred";
$("#alpha").addEventListener("input", (e) => { $("#alpha-val").textContent = Number(e.target.value).toFixed(2); renderExplain(); });
$("#cls").addEventListener("change", (e) => { cls = e.target.value; renderExplain(); });

function renderExplain() {
  const r = state.result;
  if (!r) return;
  $("#cls").innerHTML = [["pred", `Predicted: ${r.label}`], ["NORMAL", "NORMAL"], ["PNEUMONIA", "PNEUMONIA"]]
    .map(([v, t]) => `<label><input type="radio" name="cls" value="${v}"${cls === v ? " checked" : ""}> ${t}</label>`).join("");
  const name = cls === "pred" ? r.label : cls, idx = name === "NORMAL" ? 0 : 1;
  const cam = upsampleCam(r.cams[idx], r.camDims[0], r.camDims[1], 320, 320);
  const alpha = Number($("#alpha").value);
  imageToCanvas($("#cam-a"), state.boxed);
  overlayToCanvas($("#cam-b"), state.boxed, cam, alpha, state.lung);
  $("#cam-cap").innerHTML = `<b>B</b> · Grad-CAM for ${name}, layer4 (10×10 upsampled)${state.lung ? " · teal outline = radiologist lung mask" : ""}`;
  const s = attributionStats(cam, state.boxed, state.lung);
  const cell = (label, mass, area, flag) => [label, pct(mass, 0), `of heat · ${pct(area, 0)} of pixels · ×${(mass / area).toFixed(2)}`, flag && mass / area > 1.3];
  const rows = [cell("Black padding", s.padding, s.padding_area, true), cell("Outer 12% frame", s.border, s.border_area, true)];
  if (s.lungs !== undefined) rows.push(cell("Inside lungs", s.lungs, s.lungs_area, false));
  $("#attr").innerHTML = stats(rows, "one");
  $("#attr-note").innerHTML = s.padding + s.border > 0.5 ? note("<b>Most of this map sits outside the anatomy</b>, on the black bars or the image frame. Either there is little evidence for this class, or the model is using acquisition artefacts. Neither supports a clinical reading.", "warn") : "";
  renderRandomisation(name);
}

function renderRandomisation(name) {
  const pop = `Across ${randPop.n} evaluation images (60 RSNA, 60 Montgomery), <b>${pct(randPop.pass_rate, 0)}</b> pass this test (|ρ| &lt; ${randPop.pass_threshold} at every stage). Median |ρ|: fc ${randPop.median_abs.rho_fc.toFixed(2)}, layer4 ${randPop.median_abs.rho_layer4.toFixed(2)}, layer3 ${randPop.median_abs.rho_layer3.toFixed(2)}.`;
  const intro = `<p>Adebayo et al. (2018) showed that some saliency maps look the same even when a network’s weights are randomised. Those maps act as edge detectors, not explanations. This test destroys the learned weights from the top down (fc → layer4 → layer3) and recomputes Grad-CAM. A faithful map should <b>lose its resemblance</b> (rank correlation near 0).</p>`;
  if (state.kind !== "gallery") {
    $("#rand-body").innerHTML = intro + note(`Re-initialising weights needs the full PyTorch model, so this check is precomputed for gallery cases rather than run on uploads. ${pop}`);
    return;
  }
  const stages = state.item.precomputed.randomisation[name];
  const base = upsampleCam(state.result.cams[name === "NORMAL" ? 0 : 1], 10, 10, 320, 320);
  const worst = Math.max(...stages.map((s) => Math.abs(s.spearman)));
  $("#rand-body").innerHTML = intro + `<div class="cols" style="grid-template-columns:repeat(4,minmax(0,1fr));gap:10px">
      <div><canvas class="frame" id="rand-0"></canvas><div class="fig">Trained weights</div></div>
      ${stages.map((s, i) => `<div><canvas class="frame" id="rand-${i + 1}"></canvas><div class="fig">+ random ${s.randomised} · ρ = ${s.spearman >= 0 ? "+" : ""}${s.spearman.toFixed(2)}</div></div>`).join("")}
    </div>` +
    (worst < 0.4 ? note(`<b>Passes.</b> Every randomised map is weakly correlated with the original (|ρ| ≤ ${worst.toFixed(2)}), so this heatmap depends on the learned weights. That is necessary but not sufficient: it can still faithfully show a shortcut.`, "cool")
                 : note(`<b>Caution.</b> A randomised model reproduces this map (|ρ| up to ${worst.toFixed(2)}). Some of what you see is image structure, not model reasoning.`, "warn")) +
    `<p class="fig">Computed in PyTorch (seed 0) and shipped with the page. ${pop}</p>`;
  overlayToCanvas($("#rand-0"), state.boxed, base, 0.6, null);
  stages.forEach((s, i) => overlayToCanvas($(`#rand-${i + 1}`), state.boxed, upsampleCam(Float32Array.from(s.cam), 10, 10, 320, 320), 0.6, null));
}

// ── 04 stress test ───────────────────────────────────────────────────────────
async function runLiveStress(token) {
  $("#stress-head").innerHTML = `<p class="fig"><span class="spin"></span> Scoring 9 images in your browser…</p>`;
  await ensureModel();
  const ims = [state.img, ...PERTURBATIONS.map(([, , fn]) => fn(state.img))];
  const res = await infer(ims);
  if (token !== state.token) return;
  state.stressLive = Object.fromEntries([["original", res[0].pneumonia], ...PERTURBATIONS.map(([k], i) => [k, res[i + 1].pneumonia])]);
  renderStress();
}

function renderStress() {
  const live = state.stressLive, pre = state.item?.precomputed.stress;
  const sc = live || pre;
  const thumbs = [["original", "Original", (x) => x], ...PERTURBATIONS.map(([k, l, fn]) => [k, l, fn])];
  if (!sc) { $("#stress-head").innerHTML = ""; return; }
  const base = sc.original, thr = CONFIG.threshold;
  const src = live ? "Scored live in your browser. Blur, noise and marker are browser re-implementations of the Python edits; the other five are pixel-identical."
                   : "Scored with PyTorch using the exact edits from the evaluation.";
  $("#stress-head").innerHTML = `<p class="fig">${src}${!live && state.kind === "gallery" ? ' <button class="btn inline" id="rerun" style="margin-left:.6rem">Re-run live in browser</button>' : ""}</p>`;
  $("#rerun")?.addEventListener("click", (e) => { e.target.disabled = true; runLiveStress(state.token); });
  const rows = PERTURBATIONS.map(([k, label]) => ({ edit: label, score: sc[k], baseline: base,
    label: sc[k] >= thr ? "PNEUMONIA" : "NORMAL", flipped: (sc[k] >= thr) !== (base >= thr) }));
  const order = rows.map((r) => r.edit);
  chart($("#stress-chart"), {
    height: 290, data: { values: rows },
    layer: [
      { data: { values: [{ t: thr }] }, mark: { type: "rule", strokeDash: [4, 4], color: COLORS.BONE, opacity: 0.5 }, encoding: { x: { field: "t", type: "quantitative" } } },
      { mark: { type: "rule", color: COLORS.MUTED, strokeWidth: 2 }, encoding: { y: { field: "edit", type: "nominal", sort: order, title: null, axis: { labelLimit: 260 } }, x: { field: "baseline", type: "quantitative" }, x2: { field: "score" } } },
      { mark: { type: "tick", color: COLORS.BONE, thickness: 2, size: 16 }, encoding: { y: { field: "edit", type: "nominal", sort: order }, x: { field: "baseline", type: "quantitative" } } },
      { mark: { type: "circle", size: 140, stroke: COLORS.INK, strokeWidth: 2, opacity: 1 },
        encoding: { y: { field: "edit", type: "nominal", sort: order },
          x: { field: "score", type: "quantitative", scale: { domain: [0, 1] }, title: "PNEUMONIA score  (│ = original image)" },
          color: { field: "label", type: "nominal", scale: { domain: ["NORMAL", "PNEUMONIA"], range: [COLORS.NORMAL, COLORS.PNEUMONIA] }, legend: { title: "Label after edit" } },
          tooltip: [{ field: "edit" }, { field: "score", format: ".5f" }, { field: "baseline", format: ".5f", title: "original" }, { field: "flipped" }] } },
    ],
  });
  $("#thumbs").innerHTML = thumbs.map(([k, l], i) => `<div><canvas id="th-${i}"></canvas><div class="fig${k !== "original" && (sc[k] >= thr) !== (base >= thr) ? " flip" : ""}"><b>${esc(l)}</b><br>${sc[k].toFixed(4)}${k !== "original" && (sc[k] >= thr) !== (base >= thr) ? " ⟲ flipped" : ""}</div></div>`).join("");
  thumbs.forEach(([, , fn], i) => imageToCanvas($(`#th-${i}`), fn(state.img)));
  const flips = rows.filter((r) => r.flipped).length, mk = metrics.robustness.marker, inv = metrics.robustness.invert;
  $("#stress-note").innerHTML = note(`<b>${flips ? `${flips} of 8 edits flip this image’s label.` : "No edit flips this image’s label."}</b> Across ${mk.n} external adult radiographs, burned-in text markers flipped ${pct(mk.flip_rate)} of labels, ${mk.to_pneumonia} toward PNEUMONIA and ${mk.to_normal} toward NORMAL. Inverting greyscale flipped ${pct(inv.flip_rate)}. Edits that only add artefacts push the model one way, toward PNEUMONIA.`);
}

renderSheet();
selectGallery(gallery[0].id);
