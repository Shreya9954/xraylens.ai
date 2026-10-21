"""Metrics used by both the offline evaluation script and the live dashboard.

Everything is computed from per-image scores, so the dashboard can recompute any
metric at any threshold without touching the model.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, int]:
    pred = score >= threshold
    y = y.astype(bool)
    return {"tp": int((pred & y).sum()), "fn": int((~pred & y).sum()),
            "fp": int((pred & ~y).sum()), "tn": int((~pred & ~y).sum())}


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict:
    """Sensitivity, specificity, PPV, F1, balanced accuracy and AUC with 95% CIs."""
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    c = confusion(y, score, threshold)
    tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
    out = {"n": int(len(y)), "positives": int(y.sum()), "negatives": int((1 - y).sum()),
           "threshold": threshold, "confusion": c}

    def prop(k, n, name):
        out[name] = k / n if n else math.nan
        out[name + "_ci"] = wilson(k, n)

    prop(tp, tp + fn, "sensitivity")
    prop(tn, tn + fp, "specificity")
    prop(tp, tp + fp, "precision")
    prop(tp + tn, len(y), "accuracy")
    s, p = out["sensitivity"], out["precision"]
    out["f1"] = 2 * p * s / (p + s) if p + s and not math.isnan(p) else math.nan
    out["balanced_accuracy"] = (out["sensitivity"] + out["specificity"]) / 2
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, score))
        out["auc_ci"] = bootstrap_auc(y, score)
    else:
        out["auc"], out["auc_ci"] = math.nan, (math.nan, math.nan)
    return out


def bootstrap_auc(y: np.ndarray, score: np.ndarray, reps: int = 1000, seed: int = 0):
    """Percentile bootstrap, stratified so every resample has both classes."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def flag_rate(score: np.ndarray, threshold: float = 0.5) -> dict:
    """Share of images called PNEUMONIA, for cohorts where none should be."""
    score = np.asarray(score)
    k, n = int((score >= threshold).sum()), len(score)
    return {"n": n, "flagged": k, "rate": k / n if n else math.nan, "rate_ci": wilson(k, n)}


def roc_points(y, score):
    fpr, tpr, thr = roc_curve(y, score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": np.clip(thr, 0, 1).tolist()}


def calibration(y, score, bins: int = 10) -> dict:
    """Reliability-diagram bins and expected calibration error (ECE)."""
    y, score = np.asarray(y), np.asarray(score)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(score[m].mean()), float(y[m].mean())
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": int(m.sum()), "mean_score": conf, "observed_rate": acc})
        ece += m.mean() * abs(conf - acc)
    return {"bins": rows, "ece": float(ece)}


def saturation_share(score, eps: float = 1e-3) -> float:
    """Fraction of outputs pinned within eps of 0 or 1."""
    score = np.asarray(score)
    return float(((score < eps) | (score > 1 - eps)).mean())
