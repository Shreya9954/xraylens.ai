// Reliability dashboard. Every metric is recomputed from data/scores.json in the browser.
import { $, esc, pct, ci, note, stats, getJSON, getScores, binaryMetrics, wilson, chart, COLORS } from "./common.js";

const [M, S, R, P] = await Promise.all([getJSON("data/metrics.json"), getScores(), getJSON("data/randomisation.json"),
  getJSON("data/web_parity.json").catch(() => null)]);
const NAMES = { kermany_test: "Kermany test", rsna: "RSNA", montgomery: "Montgomery", shenzhen: "Shenzhen" };
const k = M.cohorts.kermany_test, r = M.cohorts.rsna_normal_vs_opacity;

$("#kick").textContent = `Reliability report · ${S.length.toLocaleString()} images · 4 cohorts · every number reproducible`;
$("#hero").innerHTML = `
  <div><div class="kicker" style="color:var(--muted)">Kermany “test” folder · n = ${k.n}</div>
    <div class="n"><span class="strike">${k.auc.toFixed(3)}</span> <small>AUC</small></div>
    <div class="cap">Sensitivity ${pct(k.sensitivity)} · specificity ${pct(k.specificity)}. Not a valid test: the published weights were refit on these exact images.</div></div>
  <div><div class="kicker">RSNA adults · external · n = ${r.n}</div>
    <div class="n" style="color:var(--pneumonia)">${pct(r.specificity)}<small> specificity</small></div>
    <div class="cap">${ci(r.specificity_ci)}. Normal adult radiographs that the model correctly calls NORMAL at the shipped 0.5 threshold. AUC ${r.auc.toFixed(3)} (${r.auc_ci[0].toFixed(3)}–${r.auc_ci[1].toFixed(3)}).</div></div>`;
$("#sat").innerHTML = stats([
  ["Kermany saturated", pct(M.saturation.kermany_test, 0), "score < 0.001 or > 0.999"],
  ["RSNA saturated", pct(M.saturation.rsna, 0), "same definition"],
  ["RSNA calibration error", M.calibration.rsna_normal_vs_opacity.ece.toFixed(2), "ECE, 10 bins"],
]);

// ── 02 threshold explorer ────────────────────────────────────────────────────
const ops = M.operating_points || {};
const choices = [["0.5 (shipped)", 0.5], ["0.9", 0.9], ["0.99", 0.99], ["0.999", 0.999], ["0.9999", 0.9999],
  ...Object.entries(ops).map(([n, o]) => [`spec ≥ ${n.split("_").pop()}% on RSNA (${o.threshold.toFixed(7)})`, o.threshold])];
const thrEl = $("#thr");
thrEl.max = choices.length - 1;
const rsMain = S.filter((x) => x.cohort === "rsna" && (x.finding === "Normal" || x.finding === "Lung Opacity"));
const y = rsMain.map((x) => x.label), sc = rsMain.map((x) => x.score);

const groups = [...new Set(S.filter((x) => !["Pneumonia", "Lung Opacity"].includes(x.finding)).map((x) => `${x.cohort}|${x.finding}`))];
const groupLabel = (g) => { const [c, f] = g.split("|"); return `${NAMES[c]} · ${f === "No Lung Opacity / Not Normal" ? "other abnormality" : f}`; };

const rocView = chart($("#roc"), {
  height: 280, title: `ROC, RSNA · AUC ${r.auc.toFixed(3)}`,
  layer: [
    { data: { values: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }, mark: { type: "line", strokeDash: [4, 4], color: COLORS.MUTED, strokeWidth: 1 },
      encoding: { x: { field: "x", type: "quantitative", title: "1 − specificity" }, y: { field: "y", type: "quantitative", title: "sensitivity" } } },
    { data: { values: M.roc.rsna_normal_vs_opacity.fpr.map((f, i) => ({ fpr: f, tpr: M.roc.rsna_normal_vs_opacity.tpr[i] })) },
      mark: { type: "line", color: COLORS.BONE, strokeWidth: 2, interpolate: "step-after" },
      encoding: { x: { field: "fpr", type: "quantitative" }, y: { field: "tpr", type: "quantitative" } } },
    { data: { name: "op" }, mark: { type: "point", size: 180, filled: true, color: COLORS.AMBER, stroke: COLORS.INK, strokeWidth: 2 },
      encoding: { x: { field: "fpr", type: "quantitative" }, y: { field: "tpr", type: "quantitative" },
        tooltip: [{ field: "label" }, { field: "fpr", format: ".3f" }, { field: "tpr", format: ".3f" }] } },
  ],
});

$("#roc-note").innerHTML = note(`<b>Good ranking, bad threshold.</b> An AUC of ${r.auc.toFixed(3)} means the model usually ranks an opacity image above a normal one. The shipped 0.5 cut-off just sits in the wrong place for adults. ` +
  Object.values(ops).map((o) => `For ${(o.specificity * 100).toFixed(1)}% specificity you need a threshold of ${o.threshold.toFixed(7)}, which keeps ${(o.sensitivity * 100).toFixed(1)}% sensitivity. `).join("") +
  "Choosing that threshold on the same 400 images we report on is optimistic. A real deployment would fix it on separate data.");

const logit = (s) => { const c = Math.min(Math.max(s, 1e-7), 1 - 1e-7); return Math.log(c / (1 - c)); };
// Seeded jitter so dots don't move between renders.
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const stripRows = S.map((x) => ({ group: `${NAMES[x.cohort]} · ${x.finding === "No Lung Opacity / Not Normal" ? "Other abnormality" : x.finding}`,
  expected: ["Pneumonia", "Lung Opacity"].includes(x.finding) ? "Pneumonia / opacity" : x.finding === "Normal" ? "Normal" : "Out of scope",
  logit: logit(x.score), score: x.score, id: x.source_id, j: rnd() }));
const groupOrder = [...new Set(stripRows.map((d) => d.group))].sort((a, b) => {
  const e = (g) => ({ Normal: 0, "Out of scope": 1, "Pneumonia / opacity": 2 }[stripRows.find((d) => d.group === g).expected]);
  return e(a) - e(b) || a.localeCompare(b);
});
const stripView = chart($("#strip"), {
  height: 480, title: "Every image’s score, by cohort (amber = threshold)",
  layer: [
    { data: { values: stripRows }, mark: { type: "circle", size: 14, opacity: 0.6 },
      encoding: { x: { field: "logit", type: "quantitative", scale: { domain: [-17, 17] }, title: "model logit of PNEUMONIA score (0 = score 0.5; ±16 ≈ fully saturated)" },
        y: { field: "group", type: "nominal", sort: groupOrder, title: null, axis: { labelLimit: 240 } }, yOffset: { field: "j", type: "quantitative" },
        color: { field: "expected", type: "nominal", scale: { domain: ["Normal", "Pneumonia / opacity", "Out of scope"], range: [COLORS.NORMAL, COLORS.PNEUMONIA, COLORS.OTHER] }, legend: { title: null } },
        tooltip: [{ field: "group" }, { field: "score", format: ".6f" }, { field: "id", title: "source id" }] } },
    { data: { name: "thr" }, mark: { type: "rule", color: COLORS.AMBER, strokeDash: [4, 3] }, encoding: { x: { field: "t", type: "quantitative" } } },
  ],
});
$("#strip-note").innerHTML = note("<b>Read the strip chart left to right.</b> Normal adult radiographs (teal) stretch across the whole range and many sit far right, fully saturated as PNEUMONIA. No threshold separates them cleanly from lung opacity, and the 0.5 threshold falls inside the normal cluster.");

let cmView = null, flagView = null;
async function updateThreshold() {
  const [label, t] = choices[Number(thrEl.value)];
  $("#thr-val").textContent = label;
  const m = binaryMetrics(y, sc, t);
  $("#thr-stats").innerHTML = stats([
    ["Sensitivity", pct(m.sensitivity), ci(m.sensitivity_ci)], ["Specificity", pct(m.specificity), ci(m.specificity_ci)],
    ["Precision", pct(m.precision), ci(m.precision_ci)], ["F1", m.f1.toFixed(3), "harmonic mean of precision and sensitivity"],
  ]);
  $("#cm-cap").innerHTML = `<b>RSNA</b> · 200 normal vs 200 lung-opacity adults · threshold ${t}`;
  const cm = [
    { true: "Lung opacity", pred: "PNEUMONIA", n: m.tp, kind: "correct" }, { true: "Lung opacity", pred: "NORMAL", n: m.fn, kind: "missed" },
    { true: "Normal", pred: "PNEUMONIA", n: m.fp, kind: "false alarm" }, { true: "Normal", pred: "NORMAL", n: m.tn, kind: "correct" },
  ];
  const flags = groups.map((g) => {
    const [c, f] = g.split("|"), s = S.filter((x) => x.cohort === c && x.finding === f).map((x) => x.score);
    const kk = s.filter((v) => v >= t).length, [lo, hi] = wilson(kk, s.length);
    return { group: groupLabel(g), rate: kk / s.length, lo, hi, n: s.length, kind: f === "Normal" ? "Normal" : "Out of scope" };
  }).sort((a, b) => b.rate - a.rate);
  if (!cmView) {
    cmView = await chart($("#cm"), {
      height: 240, data: { name: "cm" },
      encoding: { x: { field: "pred", type: "nominal", sort: ["NORMAL", "PNEUMONIA"], title: "Model says", axis: { orient: "top", labelAngle: 0 } },
                  y: { field: "true", type: "nominal", sort: ["Normal", "Lung opacity"], title: "RSNA label" } },
      layer: [
        { mark: { type: "rect", stroke: COLORS.INK, strokeWidth: 3 }, encoding: { color: { field: "n", type: "quantitative", scale: { range: ["#141c21", "#5c6b70"], domain: [0, 200] }, legend: null }, tooltip: [{ field: "true" }, { field: "pred" }, { field: "n" }, { field: "kind" }] } },
        { mark: { type: "text", fontSize: 26, font: "Instrument Serif", color: COLORS.BONE }, encoding: { text: { field: "n" } } },
        { mark: { type: "text", dy: 22, fontSize: 10, color: COLORS.MUTED }, encoding: { text: { field: "kind" } } },
      ],
    });
    flagView = await chart($("#flags"), {
      height: { step: 30 }, title: "False-alarm rate where no pneumonia is labelled", data: { name: "flags" },
      encoding: { y: { field: "group", type: "nominal", sort: { field: "rate", order: "descending" }, title: null, axis: { labelLimit: 260 } } },
      layer: [
        { mark: { type: "bar", height: 14, cornerRadiusEnd: 4 }, encoding: {
          x: { field: "rate", type: "quantitative", scale: { domain: [0, 1] }, axis: { format: "%" }, title: "share called PNEUMONIA (no pneumonia label)" },
          color: { field: "kind", type: "nominal", scale: { domain: ["Normal", "Out of scope"], range: [COLORS.NORMAL, COLORS.OTHER] }, legend: { title: null } },
          tooltip: [{ field: "group" }, { field: "rate", format: ".1%" }, { field: "lo", format: ".1%", title: "95% CI low" }, { field: "hi", format: ".1%", title: "95% CI high" }, { field: "n" }] } },
        { mark: { type: "rule", color: COLORS.BONE, opacity: 0.6 }, encoding: { x: { field: "lo", type: "quantitative" }, x2: { field: "hi" } } },
      ],
    });
  }
  cmView.data("cm", cm).run();
  flagView.data("flags", flags).run();
  (await rocView).data("op", [{ fpr: 1 - m.specificity, tpr: m.sensitivity, label: `threshold ${t}` }]).run();
  (await stripView).data("thr", [{ t: logit(t) }]).run();
}
thrEl.addEventListener("input", updateThreshold);
updateThreshold();

// ── 03 calibration ───────────────────────────────────────────────────────────
const cal = M.calibration.rsna_normal_vs_opacity, top = cal.bins[cal.bins.length - 1];
chart($("#cal"), {
  height: 290,
  layer: [
    { data: { values: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }, mark: { type: "line", strokeDash: [4, 4], color: COLORS.MUTED },
      encoding: { x: { field: "x", type: "quantitative", title: "mean model score in bin" }, y: { field: "y", type: "quantitative", title: "observed share with lung opacity" } } },
    { data: { values: cal.bins }, mark: { type: "circle", color: COLORS.AMBER, opacity: 1, stroke: COLORS.INK, strokeWidth: 2 },
      encoding: { x: { field: "mean_score", type: "quantitative" }, y: { field: "observed_rate", type: "quantitative" },
        size: { field: "n", type: "quantitative", scale: { range: [60, 900] }, legend: null },
        tooltip: [{ field: "lo", format: ".1f", title: "bin from" }, { field: "hi", format: ".1f", title: "bin to" }, { field: "n", title: "images" }, { field: "mean_score", format: ".3f" }, { field: "observed_rate", format: ".3f" }] } },
  ],
});
$("#cal-stats").innerHTML = stats([["ECE", cal.ece.toFixed(3), "0 = perfectly calibrated"],
  ["Top bin", `${(top.observed_rate * 100).toFixed(0)}%`, `observed, at mean score ${top.mean_score.toFixed(3)} (n=${top.n})`]]);

// ── 04 heatmaps ──────────────────────────────────────────────────────────────
const att = M.attribution.montgomery;
$("#lung-stats").innerHTML = stats([
  ["Heat inside lungs", pct(att.lungs_median, 0), `median · IQR ${pct(att.lungs_iqr[0], 0)}–${pct(att.lungs_iqr[1], 0)}`],
  ["Lungs’ share of pixels", pct(att.lungs_area_median, 0), "median"],
  ["Enrichment", `×${att.lungs_enrichment_median.toFixed(2)}`, `${pct(att.lungs_enrichment_gt1, 0)} of images above ×1`],
], "one");
chart($("#lung-hist"), {
  height: 220, title: "Lung enrichment per Montgomery image (×1 = no more than a uniform map)",
  layer: [
    { data: { values: S.filter((x) => x.cohort === "montgomery").map((x) => ({ e: x.cam_lungs / x.cam_lungs_area })) },
      mark: { type: "bar", color: COLORS.AMBER, cornerRadiusEnd: 4, binSpacing: 2 },
      encoding: { x: { field: "e", type: "quantitative", bin: { step: 0.25 }, title: "heat share ÷ area share" }, y: { aggregate: "count", type: "quantitative", title: "images" }, tooltip: [{ aggregate: "count", title: "images" }] } },
    { data: { values: [{ x: 1 }] }, mark: { type: "rule", color: COLORS.BONE, strokeDash: [4, 4] }, encoding: { x: { field: "x", type: "quantitative" } } },
  ],
});
$("#rand-stats").innerHTML = stats([
  ["Randomisation test", pct(R.pass_rate, 0), `of ${R.n} images pass (|ρ| < ${R.pass_threshold} at every stage)`],
  ["Median |ρ| after random fc", R.median_abs.rho_fc.toFixed(2), "rank correlation with the trained map"],
  ["… + random layer4, layer3", `${R.median_abs.rho_layer4.toFixed(2)} · ${R.median_abs.rho_layer3.toFixed(2)}`, "cumulative re-initialisation"],
], "one");
chart($("#rand-chart"), {
  height: 220, title: "Largest |ρ| per image across the three randomisation stages",
  layer: [
    { data: { values: R.rows }, mark: { type: "bar", color: COLORS.OTHER, cornerRadiusEnd: 4, binSpacing: 2 },
      encoding: { x: { field: "max_abs", type: "quantitative", bin: { step: 0.05 }, title: "max |Spearman ρ| vs trained map (lower is better)" },
        y: { aggregate: "count", type: "quantitative", title: "images" }, tooltip: [{ aggregate: "count", title: "images" }] } },
    { data: { values: [{ x: R.pass_threshold }] }, mark: { type: "rule", color: COLORS.BONE, strokeDash: [4, 4] }, encoding: { x: { field: "x", type: "quantitative" } } },
  ],
});
$("#att-note").innerHTML = note(`<b>What this does and doesn’t show.</b> Enrichment above ×1 means the explanation favours lung tissue beyond chance, and a passing randomisation test means the map depends on the learned weights. The ${pct(1 - R.pass_rate, 0)} of images that fail keep some resemblance after the weights are destroyed, so part of their map reflects image structure. Neither check proves the model reasons clinically. Grad-CAM is a coarse 10×10 summary of one layer.`);

// ── 05 robustness ────────────────────────────────────────────────────────────
const rob = Object.values(M.robustness).sort((a, b) => b.flip_rate - a.flip_rate);
$("#rob-sub").textContent = `Each of ${M.robustness.marker.n} external adult radiographs was re-scored after eight edits that do not change the anatomy.`;
chart($("#rob"), {
  height: { step: 30 },
  data: { values: rob.flatMap((v) => [{ edit: v.label, direction: "to PNEUMONIA", images: v.to_pneumonia, flip: v.flip_rate },
                                     { edit: v.label, direction: "to NORMAL", images: v.to_normal, flip: v.flip_rate }]) },
  mark: { type: "bar", height: 16, cornerRadiusEnd: 3, stroke: COLORS.INK, strokeWidth: 2 },
  encoding: { y: { field: "edit", type: "nominal", sort: rob.map((v) => v.label), title: null, axis: { labelLimit: 260 } },
    x: { field: "images", type: "quantitative", stack: "zero", title: "labels flipped" },
    color: { field: "direction", type: "nominal", scale: { domain: ["to NORMAL", "to PNEUMONIA"], range: [COLORS.NORMAL, COLORS.PNEUMONIA] }, legend: { title: "Flipped" } },
    order: { field: "direction", sort: "descending" },
    tooltip: [{ field: "edit" }, { field: "direction" }, { field: "images" }, { field: "flip", format: ".1%", title: "total flip rate" }] },
});
const mk = M.robustness.marker;
$("#rob-note").innerHTML = note(`<b>The asymmetry is the finding.</b> Adding a text marker, noise or inversion moves labels almost only toward PNEUMONIA (markers: ${mk.to_pneumonia} toward PNEUMONIA, ${mk.to_normal} toward NORMAL). The model treats “looks unusual” as evidence of disease. One plausible cause, which we have not verified, is that pneumonia images in the training set differ systematically in acquisition (portable films, more annotations, lower contrast). Either way it is a shortcut, not radiology.`, "warn");

// ── 06 browser parity ────────────────────────────────────────────────────────
if (P) {
  const exact = Object.entries(P.perturbations_identical);
  $("#parity").innerHTML = stats([
    ["Model input identical", `${P.letterbox_identical}/${P.images}`, "letterboxed 320×320 pixels, byte for byte"],
    ["Labels identical", `${P.labels_identical}/${P.images}`, "NORMAL / PNEUMONIA at 0.5"],
    ["Max score difference", P.max_abs_score_diff.toExponential(1), `median ${P.median_abs_score_diff.toExponential(1)}`],
    ["Max Grad-CAM difference", P.max_abs_cam_diff.toExponential(1), "on a 0–1 normalised map"],
  ]) + note(`Checked on ${P.images} evaluation images (a third re-encoded as JPEG) in headless Chrome by <code>scripts/check_web_parity.py</code>. Preprocessing is a line-by-line port of Pillow’s resampling. ${exact.map(([k2, v]) => `${k2} ${v}/${P.images}`).join(", ")} perturbed images were byte-identical too. Input-check warnings matched on ${P.warnings_identical}/${P.images}. Remaining score differences are floating-point noise from a different maths library.`, "cool");
}

// ── tables ───────────────────────────────────────────────────────────────────
const tbl = (head, rows) => `<table><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr>${rows.map((rw) => `<tr>${rw.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</table>`;
$("#tables").innerHTML =
  tbl(["Cohort", "n", "Valid test", "Sensitivity", "Specificity", "Precision", "F1", "AUC"],
      Object.entries(M.cohorts).map(([n, v]) => [esc(n), v.n, v.valid_test ? "yes" : "<b>no</b>", pct(v.sensitivity), pct(v.specificity), pct(v.precision), v.f1.toFixed(3), v.auc.toFixed(3)])) +
  `<br>` + tbl(["Group (no pneumonia label)", "n", "Called PNEUMONIA", "Rate", "95% CI"],
      Object.entries(M.flag_rates).map(([g, v]) => [esc(g), v.n, v.flagged, pct(v.rate), ci(v.rate_ci)])) +
  `<br>` + tbl(["Edit", "n", "→ PNEUMONIA", "→ NORMAL", "Flip rate"], rob.map((v) => [esc(v.label), v.n, v.to_pneumonia, v.to_normal, pct(v.flip_rate)])) +
  `<p><a href="data/scores.json" download>Download per-image scores (JSON)</a></p>`;
$("#foot").innerHTML = `Model revision ${M.model.revision.slice(0, 12)} · evaluated with torch ${M.environment.torch} · reproduce with <code>python scripts/fetch_eval_data.py && python scripts/run_evaluation.py</code>`;
