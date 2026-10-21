"""Model, Grad-CAM and metric behaviour. Model tests skip if the weights aren't downloaded."""
import math

import numpy as np
import pytest
import torch
from PIL import Image

from xraylens import evaluation as ev
from xraylens.explainability import attribution_stats, gradcam, overlay
from xraylens.model import DEFAULT_DIR, preprocess
from xraylens.perturbations import PERTURBATIONS

needs_weights = pytest.mark.skipif(not (DEFAULT_DIR / "model.safetensors").exists(),
                                   reason="run the app once (or model.download_model()) to fetch weights")


@pytest.fixture(scope="module")
def model_cfg():
    from xraylens.model import load_model
    return load_model(device=torch.device("cpu"))


@needs_weights
def test_scores_sum_to_one_and_label_follows_threshold(model_cfg, cxr_like):
    from xraylens.model import predict
    model, cfg = model_cfg
    p = predict(cxr_like, model, cfg)
    assert set(p.scores) == {"NORMAL", "PNEUMONIA"}
    assert math.isclose(sum(p.scores.values()), 1.0, abs_tol=1e-5)
    assert p.label == ("PNEUMONIA" if p.pneumonia_score >= cfg["threshold"] else "NORMAL")


@needs_weights
def test_inference_is_deterministic(model_cfg, cxr_like):
    from xraylens.model import predict_batch
    model, cfg = model_cfg
    a = predict_batch([cxr_like, cxr_like], model, cfg)
    assert np.allclose(a[0], a[1], atol=1e-6)


@needs_weights
def test_gradcam_shape_range_and_class_dependence(model_cfg, cxr_like):
    model, cfg = model_cfg
    boxed, x = preprocess(cxr_like, cfg)
    c0, c1 = gradcam(model, x, 0), gradcam(model, x, 1)
    for c in (c0, c1):
        assert c.shape == (320, 320)
        assert c.min() >= 0 and c.max() <= 1 + 1e-6
    assert not np.allclose(c0, c1)
    assert overlay(boxed, c1).size == (320, 320)


@needs_weights
def test_gradcam_leaves_model_unchanged(model_cfg, cxr_like):
    from xraylens.model import predict_batch
    model, cfg = model_cfg
    before = predict_batch([cxr_like], model, cfg)
    gradcam(model, preprocess(cxr_like, cfg)[1], 1)
    assert np.allclose(before, predict_batch([cxr_like], model, cfg))
    assert all(len(m._forward_hooks) == 0 for m in model.modules())


def test_attribution_stats_on_known_maps():
    img = Image.new("L", (320, 320), 0)
    img.paste(Image.new("L", (320, 240), 120), (0, 40))  # letterbox bars top and bottom
    cam = np.zeros((320, 320), np.float32)
    cam[:40] = 1.0                                       # all heat on the top bar
    s = attribution_stats(cam, img)
    assert s["padding"] == pytest.approx(1.0)
    assert s["padding_area"] == pytest.approx(80 / 320)
    lung = np.zeros((320, 320), bool)
    lung[120:200, 60:260] = True
    cam = lung.astype(np.float32)
    s = attribution_stats(cam, img, lung)
    assert s["lungs"] == pytest.approx(1.0) and s["padding"] == 0


def test_perturbations_preserve_type(cxr_like):
    for key, (_, fn) in PERTURBATIONS.items():
        out = fn(cxr_like)
        assert isinstance(out, Image.Image), key
        assert min(out.size) >= 64


def test_metrics_against_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .2, .6, .3, .1, .05])
    m = ev.binary_metrics(y, s, 0.5)
    assert m["confusion"] == {"tp": 3, "fn": 1, "fp": 1, "tn": 3}
    assert m["sensitivity"] == pytest.approx(.75) and m["specificity"] == pytest.approx(.75)
    assert m["precision"] == pytest.approx(.75) and m["f1"] == pytest.approx(.75)
    assert m["auc"] == pytest.approx(14 / 16)  # 0.2 positive beats only 2 of 4 negatives
    lo, hi = m["sensitivity_ci"]
    assert lo < .75 < hi


def test_wilson_matches_known_value():
    lo, hi = ev.wilson(77, 200)  # 38.5%
    assert lo == pytest.approx(.3197, abs=1e-3) and hi == pytest.approx(.4547, abs=1e-3)


def test_calibration_perfect_model_has_zero_ece():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < s).astype(int)
    assert ev.calibration(y, s)["ece"] < 0.02
    assert ev.calibration(np.zeros(100, int), np.ones(100))["ece"] == pytest.approx(1.0)
