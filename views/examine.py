"""Steps 01–04: choose an image, read the output, inspect Grad-CAM, stress-test it."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from xraylens import ui
from xraylens.app_state import (GALLERY, analyse, get_gallery, get_metrics, get_model, get_scores,
                                image_key, render_overlay, stress)
from xraylens.explainability import randomization_check
from xraylens.inputs import InputError, LoadedImage, assess, load_upload

ui.altair_theme()
gallery = get_gallery()
metrics = get_metrics()
scores_df = get_scores()

ui.html_block("""
<header class="xl-mast">
  <div class="xl-kicker">An explainable-AI laboratory for chest radiographs</div>
  <div class="xl-title">XRay<em>Lens</em></div>
  <p class="xl-dek">A pneumonia classifier gives you an answer. This lab shows how it got there,
  how far you can trust it, and where it breaks.</p>
  <div class="xl-stamp">Research &amp; education only<br>Not for diagnosis</div>
</header>
""")

# ── 01 ───────────────────────────────────────────────────────────────────────
ui.section("01", "Choose a radiograph",
           "Start with the curated contact sheet. Each case was picked from the evaluation to teach "
           "one thing. You can also upload a public, de-identified research image.")

if "selected" not in st.session_state:
    st.session_state.selected = ("gallery", gallery[0]["id"]) if gallery else None

sheet, upload_col = st.columns([8, 3], gap="large")
with sheet:
    per_row = 4
    for start in range(0, len(gallery), per_row):
        cols = st.columns(per_row, gap="small")
        for col, item in zip(cols, gallery[start:start + per_row]):
            with col:
                st.image(str(GALLERY / item["thumb"]), use_container_width=True)
                ui.html_block(f'<div class="xl-sheet-cap"><b>{ui.esc(item["title"])}</b>'
                              f'{ui.esc(item["subtitle"])}</div>')
                active = st.session_state.selected == ("gallery", item["id"])
                if st.button("● Loaded" if active else "Load", key=f"g_{item['id']}",
                             use_container_width=True, type="primary" if active else "secondary"):
                    st.session_state.selected = ("gallery", item["id"])
                    st.rerun()

with upload_col:
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "dcm"],
                          help="PNG, JPEG or DICOM, up to 20 MB. Images are processed in memory and not stored.")
    if up is not None and st.session_state.get("last_upload") != up.file_id:
        st.session_state.last_upload = up.file_id
        st.session_state.selected = ("upload", up.file_id)
    ui.html_block('<div class="xl-fig">Use only public, de-identified research images. '
                  'Do not upload real patient records. DICOM tags other than modality and '
                  'view are ignored.</div>')

sel = st.session_state.selected
if sel is None:
    st.stop()

item = None
try:
    if sel[0] == "gallery":
        item = next(g for g in gallery if g["id"] == sel[1])
        img = Image.open(GALLERY / item["file"])
        img.load()
        loaded = LoadedImage(image=img, source=item["title"], warnings=assess(img))
    else:
        if up is None:
            st.session_state.selected = ("gallery", gallery[0]["id"]) if gallery else None
            st.rerun()
        loaded = load_upload(up.getvalue(), up.name)
except InputError as exc:
    ui.note(f"<b>Can’t analyse this file.</b> {ui.esc(exc)}", "warn")
    st.stop()

model, cfg = get_model()
key = image_key(loaded.image)
with st.spinner("Running ResNet-50 and Grad-CAM…"):
    res = analyse(loaded.image, key, item.get("mask") if item else None)
pred, thr = res["pred"], cfg["threshold"]
p = pred.pneumonia_score

# ── 02 ───────────────────────────────────────────────────────────────────────
ui.section("02", "Read the model’s output",
           "The model returns two softmax scores that sum to 1. A fixed threshold of 0.5 turns them into a label.")

left, right = st.columns([5, 6], gap="large")
with left:
    st.image(res["boxed"], use_container_width=True)
    w, h = loaded.image.size
    ui.html_block(f'<div class="xl-fig"><b>MODEL INPUT</b> · {w}×{h} source → letterboxed to '
                  f'{cfg["image_size"]}×{cfg["image_size"]}, black bars added · '
                  f'{ui.esc(loaded.source)}</div>')
with right:
    saturated = p < 1e-3 or p > 1 - 1e-3
    ui.score_readout(pred.label, p, thr, {
        "PNEUMONIA score": f"{p:.6f}",
        "NORMAL score": f"{pred.scores['NORMAL']:.6f}",
        "Logit margin (z₁ − z₀)": f"{res['logit_margin']:+.2f}",
        "Saturated (<0.001 or >0.999)": "yes" if saturated else "no",
    })

    rs = scores_df[scores_df.cohort == "rsna"]
    share_norm = (rs[rs.finding == "Normal"].score >= p).mean()
    share_opac = (rs[rs.finding == "Lung Opacity"].score >= p).mean()
    ui.note(f"<b>What the score is, and what it isn’t.</b> {p:.4f} is the network’s output for this image, "
            "not the probability that a patient has pneumonia. For context, from our own external test: "
            f"<b>{ui.pct(share_norm, 0)}</b> of 200 adult radiographs read as <i>normal</i> (RSNA) scored "
            f"at least this high, as did <b>{ui.pct(share_opac, 0)}</b> of 200 with lung opacity.")

if item:
    tags = "".join(f'<span class="xl-tag" style="color:{c}">{ui.esc(t)}</span>' for t, c in item["tags"])
    ui.html_block(f'<div style="margin-top:.6rem">{tags}</div>')
    ui.note(item["story"], "warn" if item.get("seen_in_training") else "cool")
for wmsg in loaded.warnings:
    ui.note(f"<b>Input check.</b> {ui.esc(wmsg)}", "warn")
if loaded.meta:
    ui.html_block('<div class="xl-fig">DICOM · ' + " · ".join(f"{k} {ui.esc(v)}" for k, v in loaded.meta.items()) + "</div>")

# ── 03 ───────────────────────────────────────────────────────────────────────
ui.section("03", "See where the model looked",
           "Grad-CAM weights the last convolutional block’s feature maps by how strongly they push "
           "the chosen class’s logit up. Bright regions influenced the output. That does not make them correct.")

c1, c2 = st.columns([3, 2])
with c1:
    target = st.radio("Explain which class", ["Predicted: " + pred.label, "NORMAL", "PNEUMONIA"],
                      horizontal=True, label_visibility="visible")
with c2:
    alpha = st.slider("Heatmap opacity", 0.0, 0.9, 0.5, 0.05)
cls = pred.label if target.startswith("Predicted") else target
cam = res["cams"][cls]
stats = res["stats"][cls]

a, b, c = st.columns([4, 4, 3], gap="medium")
with a:
    st.image(res["boxed"], use_container_width=True)
    ui.html_block('<div class="xl-fig"><b>A</b> · input as seen by the model</div>')
with b:
    st.image(render_overlay(res["boxed"], cam, alpha, res["lung"]), use_container_width=True)
    lung_txt = " · teal outline = radiologist lung mask" if res["lung"] is not None else ""
    ui.html_block(f'<div class="xl-fig"><b>B</b> · Grad-CAM for {cls}, layer4 (10×10 upsampled){lung_txt}</div>')
with c:
    def bar(label, mass, area):
        ratio = mass / area if area > 0 else float("nan")
        flag = ' style="color:var(--pneumonia)"' if (ratio == ratio and ratio > 1.3 and label != "Inside lungs") else ""
        return (f'<div><div class="k">{label}</div><div class="v"{flag}>{ui.pct(mass, 0)}</div>'
                f'<div class="ci">of heat · {ui.pct(area, 0)} of pixels · ×{ratio:.2f}</div></div>')
    rows = [bar("Black padding", stats["padding"], stats["padding_area"]),
            bar("Outer 12% frame", stats["border"], stats["border_area"])]
    if "lungs" in stats:
        rows.append(bar("Inside lungs", stats["lungs"], stats["lungs_area"]))
    ui.html_block(f'<div class="xl-stats" style="grid-template-columns:1fr">{"".join(rows)}</div>')
    ui.html_block('<div class="xl-fig">×1.00 is what a uniform, uninformative map would score. '
                  'Heat on padding or the frame (text markers, tubes, collimator edges) is a shortcut warning.</div>')

if stats["padding"] + stats["border"] > 0.5:
    ui.note("<b>Most of this map sits outside the anatomy</b>, on the black bars or the image frame. "
            "Either there is little evidence for this class, or the model is using acquisition "
            "artefacts. Neither supports a clinical reading.", "warn")

with st.expander("Sanity check: does this map depend on what the model learned?"):
    st.markdown("Adebayo et al. (2018) showed that some saliency maps look the same even when a "
                "network’s weights are randomised. Those maps act as edge detectors, not explanations. "
                "Here we destroy the learned weights from the top down and recompute Grad-CAM. "
                "A faithful map should **lose its resemblance** (rank correlation near 0).")
    if st.button("Run randomisation test", key="rand"):
        with st.spinner("Re-initialising fc → layer4 → layer3…"):
            r = randomization_check(model, res["x"], {"NORMAL": 0, "PNEUMONIA": 1}[cls], cam)
        cols = st.columns(4)
        cols[0].image(render_overlay(res["boxed"], cam, .6, None), caption="Trained weights", use_container_width=True)
        for col, stage, m in zip(cols[1:], r["stages"], r["maps"]):
            col.image(render_overlay(res["boxed"], m, .6, None),
                      caption=f"+ random {stage['randomised']} · ρ = {stage['spearman']:+.2f}", use_container_width=True)
        worst = max(abs(s["spearman"]) for s in r["stages"])
        if worst < 0.4:
            ui.note(f"<b>Passes.</b> Every randomised map is weakly correlated with the original (|ρ| ≤ {worst:.2f}), "
                    "so this heatmap depends on the learned weights. That is necessary but not sufficient: "
                    "it can still faithfully show a shortcut.", "cool")
        else:
            ui.note(f"<b>Caution.</b> A randomised model reproduces this map (|ρ| up to {worst:.2f}). "
                    "Some of what you see is image structure, not model reasoning.", "warn")

# ── 04 ───────────────────────────────────────────────────────────────────────
ui.section("04", "Stress-test this image",
           "Eight edits that do not change the anatomy a radiologist would read. A trustworthy model’s "
           "score should hardly move.")
with st.spinner("Scoring 8 perturbed copies…"):
    st_df = stress(loaded.image, key)
base = float(st_df.score.iloc[0])
st_df["flipped"] = (st_df.score >= thr) != (base >= thr)
st_df["label"] = np.where(st_df.score >= thr, "PNEUMONIA", "NORMAL")

chart_df = st_df.drop(columns=["image"]).iloc[1:].copy()
chart_df["baseline"] = base
order = chart_df.edit.tolist()
rule = alt.Chart(chart_df).mark_rule(color=ui.MUTED, strokeWidth=2).encode(
    y=alt.Y("edit:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), x="baseline:Q", x2="score:Q")
base_pt = alt.Chart(chart_df).mark_point(shape="stroke", size=160, color=ui.BONE, strokeWidth=2, angle=90).encode(
    y=alt.Y("edit:N", sort=order), x="baseline:Q")
dots = alt.Chart(chart_df).mark_circle(size=140, stroke=ui.INK, strokeWidth=2, opacity=1).encode(
    y=alt.Y("edit:N", sort=order),
    x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="PNEUMONIA score  (│ = original image)"),
    color=alt.Color("label:N", scale=alt.Scale(domain=["NORMAL", "PNEUMONIA"], range=[ui.NORMAL, ui.PNEUMONIA]),
                    legend=alt.Legend(title="Label after edit")),
    tooltip=[alt.Tooltip("edit:N", title="Edit"), alt.Tooltip("score:Q", format=".5f", title="Score"),
             alt.Tooltip("baseline:Q", format=".5f", title="Original"), alt.Tooltip("flipped:N", title="Label flipped")])
thr_rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(strokeDash=[4, 4], color=ui.BONE, opacity=.5).encode(x="t:Q")
st.altair_chart((thr_rule + rule + base_pt + dots).properties(height=300), use_container_width=True)

flips = st_df[st_df.flipped]
thumbs = st.columns(len(st_df))
for col, r in zip(thumbs, st_df.itertuples()):
    col.image(r.image, use_container_width=True)
    mark = " ⟲ flipped" if r.flipped else ""
    col.markdown(f'<div class="xl-fig"><b>{ui.esc(r.edit)}</b><br>{r.score:.4f}{mark}</div>', unsafe_allow_html=True)

rob = metrics["robustness"]
mk, inv = rob["marker"], rob["invert"]
summary = (f"<b>{len(flips)} of 8 edits flip this image’s label.</b> " if len(flips) else
           "<b>No edit flips this image’s label.</b> ")
ui.note(summary + f"Across {mk['n']} external adult radiographs, burned-in text markers flipped "
        f"{ui.pct(mk['flip_rate'])} of labels, {mk['to_pneumonia']} toward PNEUMONIA and {mk['to_normal']} "
        f"toward NORMAL. Inverting greyscale flipped {ui.pct(inv['flip_rate'])}. Edits that only add "
        "artefacts push the model one way, toward PNEUMONIA.")

ui.html_block('<div style="height:1.2rem"></div>')
st.page_link("views/reliability.py", label="Next: how reliable is this model across 1,462 images? →")
ui.html_block('<div class="xl-foot">XRayLens AI · Model: Purino/resnet50-chest-xray-pneumonia (research only) · '
              'Gallery images: Kermany et al. 2018 (CC BY 4.0) and NLM Montgomery/Shenzhen sets · '
              'Outputs are not medical advice.</div>')
