// Small shared helpers: escaping, formatting, notes, stats, metrics (mirror xraylens/evaluation.py).

export const $ = (sel, root = document) => root.querySelector(sel);
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const pct = (x, d = 1) => (Number.isFinite(x) ? `${(x * 100).toFixed(d)}%` : "—");
export const ci = ([lo, hi], d = 1) => (Number.isFinite(lo) ? `95% CI ${(lo * 100).toFixed(d)}–${(hi * 100).toFixed(d)}%` : "");
export const note = (html, kind = "") => `<div class="note ${kind}"><p>${html}</p></div>`;
export const stats = (items, cls = "") =>
  `<div class="stats ${cls}">${items.map(([k, v, c, hot]) => `<div><div class="k">${k}</div><div class="v${hot ? " hot" : ""}">${v}</div><div class="ci">${c || ""}</div></div>`).join("")}</div>`;

export async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

export async function getScores() {
  const { columns, rows } = await getJSON("data/scores.json");
  return rows.map((r) => Object.fromEntries(columns.map((c, i) => [c, r[i]])));
}

export function wilson(k, n, z = 1.96) {
  if (!n) return [NaN, NaN];
  const p = k / n, d = 1 + (z * z) / n, c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

export function binaryMetrics(y, s, t) {
  let tp = 0, fn = 0, fp = 0, tn = 0;
  y.forEach((yi, i) => { const p = s[i] >= t; if (yi && p) tp++; else if (yi) fn++; else if (p) fp++; else tn++; });
  const sens = tp / (tp + fn), spec = tn / (tn + fp), prec = tp + fp ? tp / (tp + fp) : NaN;
  return { tp, fn, fp, tn, sensitivity: sens, specificity: spec, precision: prec,
           f1: (2 * prec * sens) / (prec + sens), sensitivity_ci: wilson(tp, tp + fn),
           specificity_ci: wilson(tn, tn + fp), precision_ci: wilson(tp, tp + fp) };
}

export const VEGA_OPTS = { actions: false, renderer: "svg", config: {
  background: null, font: "JetBrains Mono", view: { stroke: null },
  axis: { labelColor: "#93a1a6", titleColor: "#93a1a6", gridColor: "rgba(236,230,214,.08)", domainColor: "rgba(236,230,214,.3)",
          tickColor: "rgba(236,230,214,.3)", labelFont: "JetBrains Mono", titleFont: "JetBrains Mono", titleFontWeight: 400,
          labelFontSize: 10, titleFontSize: 10 },
  legend: { labelColor: "#ece6d6", titleColor: "#93a1a6", labelFont: "JetBrains Mono", titleFont: "JetBrains Mono",
            titleFontWeight: 400, orient: "top" },
  title: { color: "#ece6d6", font: "Newsreader", fontWeight: 400, fontSize: 15, anchor: "start" },
  text: { color: "#ece6d6", font: "JetBrains Mono" },
} };

export const COLORS = { NORMAL: "#38a596", PNEUMONIA: "#e05c35", OTHER: "#8b7ff0", AMBER: "#ffb547", BONE: "#ece6d6", MUTED: "#93a1a6", INK: "#0b0f12" };

export async function chart(el, spec) {
  // vega-embed is loaded as a classic script (pinned, see the page <head>).
  // Wait for web fonts so Vega measures axis labels in JetBrains Mono, not the fallback.
  await document.fonts.ready;
  const full = { $schema: "https://vega.github.io/schema/vega-lite/v5.json", width: "container", autosize: { type: "fit-x", contains: "padding" }, ...spec };
  const res = await window.vegaEmbed(el, full, VEGA_OPTS);
  return res.view;
}
