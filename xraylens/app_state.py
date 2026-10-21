"""Cached loaders shared by the Streamlit pages."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

from .explainability import attribution_stats, gradcam, overlay, peak_location
from .model import load_model, preprocess, scores_from_logits
from .perturbations import PERTURBATIONS

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "assets" / "gallery"
RESULTS = ROOT / "results"


@st.cache_resource(show_spinner="Loading ResNet-50 weights…")
def get_model():
    return load_model()


@st.cache_data
def get_gallery() -> list[dict]:
    path = GALLERY / "gallery.json"
    return json.loads(path.read_text()) if path.exists() else []


@st.cache_data
def get_metrics() -> dict:
    return json.loads((RESULTS / "metrics.json").read_text())


@st.cache_data
def get_scores() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scores.csv", keep_default_na=False, na_values=[""])


def image_key(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha1(buf.getvalue()).hexdigest()


@st.cache_data(show_spinner=False, max_entries=32)
def analyse(_image: Image.Image, key: str, lung_mask_path: str | None = None) -> dict:
    """Prediction plus Grad-CAM for both classes. ``key`` stands in for the image in the cache."""
    model, cfg = get_model()
    boxed, x = preprocess(_image, cfg)
    device = next(model.parameters()).device
    import torch
    with torch.inference_mode():
        logits = model(x.to(device))
    pred = scores_from_logits(logits, cfg)
    lung = None
    if lung_mask_path:
        mask_boxed, _ = preprocess(Image.open(ROOT / lung_mask_path), cfg)
        lung = np.asarray(mask_boxed.convert("L")) > 127
    cams, stats = {}, {}
    for idx, name in enumerate(cfg["classes"]):
        cams[name] = gradcam(model, x, idx)
        stats[name] = attribution_stats(cams[name], boxed, lung)
    lg = logits[0].float().cpu().numpy()
    return {"boxed": boxed, "x": x, "pred": pred, "cams": cams, "stats": stats, "lung": lung,
            "logit_margin": float(lg[1] - lg[0]), "peak": {k: peak_location(v) for k, v in cams.items()}}


def lung_outline(mask: np.ndarray) -> np.ndarray:
    m = Image.fromarray((mask * 255).astype(np.uint8))
    inner = np.asarray(m.filter(ImageFilter.MinFilter(5))) > 127
    return mask & ~inner


def render_overlay(boxed: Image.Image, cam: np.ndarray, alpha: float, lung: np.ndarray | None) -> Image.Image:
    out = overlay(boxed, cam, alpha=alpha)
    if lung is not None:
        arr = np.asarray(out).copy()
        arr[lung_outline(lung)] = (56, 165, 150)
        out = Image.fromarray(arr)
    return out


@st.cache_data(show_spinner=False, max_entries=16)
def stress(_image: Image.Image, key: str) -> pd.DataFrame:
    model, cfg = get_model()
    from .model import predict_batch
    names = ["original"] + list(PERTURBATIONS)
    ims = [_image] + [fn(_image) for _, fn in PERTURBATIONS.values()]
    scores = predict_batch(ims, model, cfg)
    labels = ["Original"] + [lab for lab, _ in PERTURBATIONS.values()]
    return pd.DataFrame({"key": names, "edit": labels, "score": scores, "image": ims})
