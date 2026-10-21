"""Step 05: how reliable is the model? Everything here is recomputed from results/scores.csv."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from xraylens import evaluation as ev
from xraylens import ui
from xraylens.app_state import get_metrics, get_scores

ui.altair_theme()
M = get_metrics()
S = get_scores()

COHORT_NAMES = {"kermany_test": "Kermany test (pediatric, Guangzhou)", "rsna": "RSNA (adult, NIH)",
                "montgomery": "Montgomery (adult, USA)", "shenzhen": "Shenzhen (adult, China)"}

k = M["cohorts"]["kermany_test"]
r = M["cohorts"]["rsna_normal_vs_opacity"]

ui.html_block(f"""
<header class="xl-mast">
  <div class="xl-kicker">Reliability report · {len(S):,} images · 4 cohorts · every number reproducible</div>
  <div class="xl-title" style="font-size:clamp(3rem,7.5vw,6.2rem)">Two numbers,<br><em>one model</em></div>
  <p class="xl-dek">The same weights look perfect or badly miscalibrated, depending on which images you test them on.</p>
</header>
<div class="xl-hero">
  <div>
    <div class="xl-kicker" style="color:var(--muted)">Kermany “test” folder · n = {k['n']}</div>
    <div class="n"><span class="xl-strike">{k['auc']:.3f}</span> <small>AUC</small></div>
    <div class="cap">Sensitivity {ui.pct(k['sensitivity'])} · specificity {ui.pct(k['specificity'])}.
    Not a valid test: the published weights were refit on these exact images.</div>
  </div>
  <div>
    <div class="xl-kicker">RSNA adults · external · n = {r['n']}</div>
    <div class="n" style="color:var(--pneumonia)">{ui.pct(r['specificity'])}<small> specificity</small></div>
    <div class="cap">{ui.ci(r['specificity_ci'])}. Normal adult radiographs that the model correctly calls
    NORMAL at the shipped 0.5 threshold. AUC {r['auc']:.3f} ({r['auc_ci'][0]:.3f}–{r['auc_ci'][1]:.3f}).</div>
  </div>
</div>
""")

# ── 01 ───────────────────────────────────────────────────────────────────────
ui.section("01", "The leakage trap",
           "The easiest way to make a medical model look good is to test it on data it has already seen.")
a, b = st.columns([6, 5], gap="large")
with a:
    st.markdown(
        "The model card says the final weights were *“retrained on all cleaned images from the original "
        "train, test, and validation folders”* and that *“it has no independent test evaluation.”* "
        "So the 624-image Kermany test folder, the benchmark most tutorials report, is **training data** "
        "for this model. We scored it anyway, to show how misleading that number would be.")
    st.markdown(
        "Kermany images are all from children aged one to five at a single hospital. "
        "We could not find a second pediatric pneumonia set with a compatible licence in time, so our external "
        "tests use **adult** radiographs. That measures generalisation and population shift together, "
        "and we report it that way.")
with b:
    sat = M["saturation"]
    ui.stats_row([
        ("Kermany saturated", ui.pct(sat["kermany_test"], 0), "score < 0.001 or > 0.999"),
        ("RSNA saturated", ui.pct(sat["rsna"], 0), "same definition"),
        ("RSNA calibration error", f"{M['calibration']['rsna_normal_vs_opacity']['ece']:.2f}", "ECE, 10 bins"),
    ])
    ui.html_block('<div class="xl-fig">Near-0 or near-1 outputs on unfamiliar adult images show overconfidence, '
                  'not certainty. The model card itself warns that softmax outputs are not calibrated probabilities.</div>')

# ── 02 ───────────────────────────────────────────────────────────────────────
ui.section("02", "Move the threshold",
           "The shipped threshold is 0.5. Because outputs pile up near 1, useful thresholds sit extremely "
           "close to 1. Every number below is recomputed live from per-image scores.")

ops = M.get("operating_points", {})
choices = {"0.5 (shipped)": 0.5}
for t in (0.9, 0.99, 0.999, 0.9999):
    choices[str(t)] = t
for name, op in ops.items():
    choices[f"spec ≥ {name.split('_')[-1]}% on RSNA ({op['threshold']:.7f})"] = op["threshold"]
choice = st.select_slider("Decision threshold on the PNEUMONIA score", options=list(choices), value="0.5 (shipped)")
thr = choices[choice]

rs = S[(S.cohort == "rsna") & S.finding.isin(["Normal", "Lung Opacity"])]
m = ev.binary_metrics(rs.label.values, rs.score.values, thr)
c = m["confusion"]
left, right = st.columns([5, 6], gap="large")
with left:
    cm = pd.DataFrame([
        {"true": "Lung opacity", "pred": "PNEUMONIA", "n": c["tp"], "kind": "correct"},
        {"true": "Lung opacity", "pred": "NORMAL", "n": c["fn"], "kind": "missed"},
        {"true": "Normal", "pred": "PNEUMONIA", "n": c["fp"], "kind": "false alarm"},
        {"true": "Normal", "pred": "NORMAL", "n": c["tn"], "kind": "correct"},
    ])
    base = alt.Chart(cm).encode(
        x=alt.X("pred:N", sort=["NORMAL", "PNEUMONIA"], title="Model says", axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("true:N", sort=["Normal", "Lung opacity"], title="RSNA label"))
    heat = base.mark_rect(stroke=ui.INK, strokeWidth=3).encode(
        color=alt.Color("n:Q", scale=alt.Scale(range=["#141c21", "#5c6b70"]), legend=None),
        tooltip=["true", "pred", "n", "kind"])
    txt = base.mark_text(fontSize=26, font="Instrument Serif").encode(
        text="n:Q", color=alt.value(ui.BONE))
    sub = base.mark_text(dy=22, fontSize=10).encode(text="kind:N", color=alt.value(ui.MUTED))
    st.altair_chart((heat + txt + sub).properties(height=260), use_container_width=True)
    ui.html_block(f'<div class="xl-fig"><b>RSNA</b> · 200 normal vs 200 lung-opacity adults · threshold {thr:g}</div>')
with right:
    ui.stats_row([
        ("Sensitivity", ui.pct(m["sensitivity"]), ui.ci(m["sensitivity_ci"])),
        ("Specificity", ui.pct(m["specificity"]), ui.ci(m["specificity_ci"])),
        ("Precision", ui.pct(m["precision"]), ui.ci(m["precision_ci"])),
        ("F1", f"{m['f1']:.3f}", "harmonic mean of PPV and sensitivity"),
    ])
    rows = []
    for (cohort, finding), g in S.groupby(["cohort", "finding"]):
        if finding in ("Pneumonia", "Lung Opacity"):
            continue
        fr = ev.flag_rate(g.score.values, thr)
        rows.append({"group": f"{COHORT_NAMES[cohort].split(' (')[0]} · {finding.replace('No Lung Opacity / Not Normal', 'other abnormality')}",
                     "rate": fr["rate"], "lo": fr["rate_ci"][0], "hi": fr["rate_ci"][1], "n": fr["n"],
                     "kind": "Normal" if finding == "Normal" else "Out of scope"})
    fr_df = pd.DataFrame(rows).sort_values("rate", ascending=False)
    order = fr_df.group.tolist()
    bars = alt.Chart(fr_df).mark_bar(height=14, cornerRadiusEnd=4).encode(
        y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="share called PNEUMONIA (no pneumonia label)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Normal", "Out of scope"], range=[ui.NORMAL, ui.OTHER]),
                        legend=alt.Legend(title=None)),
        tooltip=["group", alt.Tooltip("rate:Q", format=".1%"), alt.Tooltip("lo:Q", format=".1%", title="95% CI low"),
                 alt.Tooltip("hi:Q", format=".1%", title="95% CI high"), "n"])
    whisk = alt.Chart(fr_df).mark_rule(color=ui.BONE, opacity=.6).encode(y=alt.Y("group:N", sort=order), x="lo:Q", x2="hi:Q")
    st.altair_chart((bars + whisk).properties(height=alt.Step(30), title="False-alarm rate where no pneumonia is labelled"),
                    use_container_width=True)

roc = M["roc"]["rsna_normal_vs_opacity"]
roc_df = pd.DataFrame({"fpr": roc["fpr"], "tpr": roc["tpr"]})
op_df = pd.DataFrame({"fpr": [1 - m["specificity"]], "tpr": [m["sensitivity"]], "label": [f"threshold {thr:g}"]})
roc_c, note_c = st.columns([5, 6], gap="large")
with roc_c:
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=ui.MUTED, strokeWidth=1).encode(x="x", y="y")
    curve = alt.Chart(roc_df).mark_line(color=ui.BONE, strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("fpr:Q", title="1 − specificity"), y=alt.Y("tpr:Q", title="sensitivity"))
    pt = alt.Chart(op_df).mark_point(size=180, filled=True, color=ui.AMBER, stroke=ui.INK, strokeWidth=2).encode(
        x="fpr:Q", y="tpr:Q", tooltip=["label", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")])
    st.altair_chart((diag + curve + pt).properties(height=300, title=f"ROC, RSNA · AUC {r['auc']:.3f}"), use_container_width=True)
with note_c:
    ui.html_block('<div style="height:2.6rem"></div>')
    ui.note(f"<b>Good ranking, bad threshold.</b> An AUC of {r['auc']:.3f} means the model usually ranks an opacity image "
            "above a normal one. The shipped 0.5 cut-off just sits in the wrong place for adults. "
            + "".join(f"For {op['specificity'] * 100:.0f}% specificity you need a threshold of {op['threshold']:.7f}, "
                      f"which keeps {op['sensitivity'] * 100:.0f}% sensitivity. " for op in ops.values())
            + "Choosing that threshold on the same 400 images we report on is optimistic. A real deployment would fix it on separate data.")

# ── 02b ───────────────────────────────────────────────────────────────────────
d = S.copy()
d["group"] = d.cohort.map(lambda x: COHORT_NAMES[x].split(" (")[0]) + " · " + d.finding.replace(
    {"No Lung Opacity / Not Normal": "Other abnormality"})
d["expected"] = np.select([d.finding.isin(["Pneumonia", "Lung Opacity"]), d.finding.eq("Normal")],
                          ["Pneumonia / opacity", "Normal"], "Out of scope")
# Logit scale spreads out the saturated tails, so 0.999 and 0.999999 can be told apart.
d["logit"] = np.log(np.clip(d.score, 1e-7, 1 - 1e-7) / (1 - np.clip(d.score, 1e-7, 1 - 1e-7)))
groups = d.groupby("group").expected.first().sort_values().index.tolist()
d["j"] = np.random.default_rng(0).uniform(size=len(d))  # seeded so dots don't move on rerun
jitter = alt.Chart(d).mark_circle(size=14, opacity=.6).encode(
    x=alt.X("logit:Q", title="model logit of PNEUMONIA score (0 = score 0.5; ±16 ≈ fully saturated)",
            scale=alt.Scale(domain=[-17, 17])),
    y=alt.Y("group:N", sort=groups, title=None, axis=alt.Axis(labelLimit=240)),
    yOffset=alt.YOffset("j:Q"),
    color=alt.Color("expected:N", scale=alt.Scale(domain=["Normal", "Pneumonia / opacity", "Out of scope"],
                                                  range=[ui.NORMAL, ui.PNEUMONIA, ui.OTHER]),
                    legend=alt.Legend(title=None)),
    tooltip=["group", alt.Tooltip("score:Q", format=".6f"), "source_id"])
tl = float(np.log(min(max(thr, 1e-7), 1 - 1e-7) / (1 - min(max(thr, 1e-7), 1 - 1e-7))))
trule = alt.Chart(pd.DataFrame({"t": [tl]})).mark_rule(color=ui.AMBER, strokeDash=[4, 3]).encode(x="t:Q")
st.altair_chart((jitter + trule).properties(height=520, title="Every image’s score, by cohort (amber = threshold)"),
                use_container_width=True)

ui.note("<b>Read the strip chart left to right.</b> Normal adult radiographs (teal) stretch across the whole range "
        "and many sit far right, fully saturated as PNEUMONIA. No threshold separates them cleanly from lung "
        "opacity, and the 0.5 threshold falls inside the normal cluster.")

# ── 03 ───────────────────────────────────────────────────────────────────────
ui.section("03", "Is a score of 0.9 really 90%?",
           "A calibrated model’s scores of 0.9 come true about 90% of the time. Bins below show how often RSNA "
           "images in each score band actually had lung opacity.")
cal = pd.DataFrame(M["calibration"]["rsna_normal_vs_opacity"]["bins"])
cal["mid"] = (cal.lo + cal.hi) / 2
cc, ct = st.columns([6, 4], gap="large")
with cc:
    ideal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color=ui.MUTED).encode(
        x=alt.X("x:Q", title="mean model score in bin"), y=alt.Y("y:Q", title="observed share with lung opacity"))
    pts = alt.Chart(cal).mark_circle(color=ui.AMBER, opacity=1, stroke=ui.INK, strokeWidth=2).encode(
        x="mean_score:Q", y="observed_rate:Q", size=alt.Size("n:Q", scale=alt.Scale(range=[60, 900]), legend=None),
        tooltip=[alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f"), "n",
                 alt.Tooltip("mean_score:Q", format=".3f"), alt.Tooltip("observed_rate:Q", format=".3f")])
    st.altair_chart((ideal + pts).properties(height=300), use_container_width=True)
with ct:
    top = cal.iloc[-1]
    ui.stats_row([("ECE", f"{M['calibration']['rsna_normal_vs_opacity']['ece']:.3f}", "0 = perfectly calibrated"),
                  ("Top bin", f"{top['observed_rate'] * 100:.0f}%", f"observed, at mean score {top['mean_score']:.3f} (n={int(top['n'])})")])
    ui.html_block('<div class="xl-fig">Most images land in the top bin, where the model claims near certainty '
                  'and is right far less often. Treat the score as a ranking, not a probability.</div>')

# ── 04 ───────────────────────────────────────────────────────────────────────
ui.section("04", "Do heatmaps land on the lungs?",
           "Montgomery County radiographs come with radiologist-drawn lung masks, so we can measure "
           "how much Grad-CAM mass falls inside the lungs compared with the lungs’ share of the image.")
att = M["attribution"]["montgomery"]
mg = S[S.cohort == "montgomery"].copy()
mg["enrichment"] = mg.cam_lungs / mg.cam_lungs_area
la, lb = st.columns([4, 6], gap="large")
with la:
    ui.stats_row([("Heat inside lungs", ui.pct(att["lungs_median"], 0), f"median · IQR {ui.pct(att['lungs_iqr'][0], 0)}–{ui.pct(att['lungs_iqr'][1], 0)}"),
                  ("Lungs’ share of pixels", ui.pct(att["lungs_area_median"], 0), "median"),
                  ("Enrichment", f"×{att['lungs_enrichment_median']:.2f}", f"{ui.pct(att['lungs_enrichment_gt1'], 0)} of images above ×1")])
with lb:
    hist = alt.Chart(mg).mark_bar(color=ui.AMBER, cornerRadiusEnd=4, binSpacing=2).encode(
        x=alt.X("enrichment:Q", bin=alt.Bin(step=0.25), title="lung enrichment (heat share ÷ area share)"),
        y=alt.Y("count():Q", title="images"), tooltip=["count()"])
    one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color=ui.BONE, strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((hist + one).properties(height=230, title="×1 = no more attention to lungs than a uniform map"),
                    use_container_width=True)
ui.note("<b>What this does and doesn’t show.</b> Enrichment above ×1 means the explanation favours lung tissue "
        "beyond chance. Values at or below ×1 mean the map spreads onto the mediastinum, the abdomen, bones or the frame. "
        "Neither proves the model reasons clinically. Grad-CAM is a coarse 10×10 summary of one layer.")

# ── 05 ───────────────────────────────────────────────────────────────────────
ui.section("05", "Robustness to edits that shouldn’t matter",
           f"Each of {M['robustness']['marker']['n']} external adult radiographs was re-scored after eight edits that do not change the anatomy.")
rb = pd.DataFrame([{"edit": v["label"], "to PNEUMONIA": v["to_pneumonia"], "to NORMAL": v["to_normal"],
                    "flip": v["flip_rate"]} for v in M["robustness"].values()]).sort_values("flip", ascending=False)
long = rb.melt(id_vars=["edit", "flip"], value_vars=["to PNEUMONIA", "to NORMAL"], var_name="direction", value_name="images")
chart = alt.Chart(long).mark_bar(height=16, cornerRadiusEnd=3, stroke=ui.INK, strokeWidth=2).encode(
    y=alt.Y("edit:N", sort=rb.edit.tolist(), title=None, axis=alt.Axis(labelLimit=260)),
    x=alt.X("images:Q", stack="zero", title="labels flipped"),
    color=alt.Color("direction:N", scale=alt.Scale(domain=["to NORMAL", "to PNEUMONIA"], range=[ui.NORMAL, ui.PNEUMONIA]),
                    legend=alt.Legend(title="Flipped")),
    order=alt.Order("direction:N", sort="descending"),
    tooltip=["edit", "direction", "images", alt.Tooltip("flip:Q", format=".1%", title="total flip rate")])
st.altair_chart(chart.properties(height=alt.Step(30)), use_container_width=True)
mk = M["robustness"]["marker"]
ui.note(f"<b>The asymmetry is the finding.</b> Adding a text marker, noise or inversion moves labels almost only toward "
        f"PNEUMONIA (markers: {mk['to_pneumonia']} toward PNEUMONIA, {mk['to_normal']} toward NORMAL). The model treats "
        "“looks unusual” as evidence of disease. One plausible cause, which we have not verified, is that pneumonia images in "
        "the training set differ systematically in acquisition (portable films, more annotations, lower contrast). "
        "Either way it is a shortcut, not radiology.", "warn")

# ── 06 ───────────────────────────────────────────────────────────────────────
ui.section("06", "What this evaluation cannot tell you")
st.markdown("""
- **No in-distribution test exists.** Every Kermany image was used to train the final weights. We can't measure pediatric performance honestly with this checkpoint.
- **External cohorts are adults.** Poor specificity on RSNA, Montgomery and Shenzhen mixes *population shift* (children → adults) with genuine generalisation error. We can't separate them.
- **Labels differ.** RSNA “Lung Opacity” is a bounding-box finding suggestive of pneumonia, not a confirmed diagnosis. Montgomery and Shenzhen labels are for tuberculosis, so their normals are the cleanest negatives we have.
- **Samples are modest.** 200 per RSNA class, 138 Montgomery and 100 Shenzhen images. Confidence intervals are shown and are wide.
- **Grad-CAM is one view.** It summarises one layer at 10×10 resolution. Passing the randomisation test shows the maps depend on the weights, not that they capture the model’s full reasoning.
- **Nothing here is clinical validation.** This is a teaching instrument for reading model behaviour critically.
""")

with st.expander("Table view · all headline metrics"):
    rows = []
    for name, v in M["cohorts"].items():
        rows.append({"cohort": name, "n": v["n"], "valid test": v["valid_test"], "sensitivity": v["sensitivity"],
                     "specificity": v["specificity"], "precision": v["precision"], "F1": v["f1"], "AUC": v["auc"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame([{"group": g, **{kk: vv for kk, vv in v.items() if kk != "rate_ci"}} for g, v in M["flag_rates"].items()]),
                 hide_index=True, use_container_width=True)
    st.dataframe(rb, hide_index=True, use_container_width=True)
    st.download_button("Download per-image scores (CSV)", S.to_csv(index=False), "xraylens_scores.csv", "text/csv")

ui.html_block(f'<div class="xl-foot">Model revision {M["model"]["revision"][:12]} · evaluated with torch {M["environment"]["torch"]} on '
              f'{M["environment"]["device"]} · reproduce with <code>python scripts/fetch_eval_data.py && python scripts/run_evaluation.py</code></div>')
