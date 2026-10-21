"""Score every evaluation image and write results/ (committed).

Outputs
-------
results/scores.csv     one row per image: cohort, source id, label, finding, model
                       score, Grad-CAM attribution shares, perturbed scores.
results/metrics.json   headline metrics per cohort, calibration, robustness.

Run ``python scripts/fetch_eval_data.py`` first. Deterministic for a given device.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xraylens import evaluation as ev  # noqa: E402
from xraylens.explainability import attribution_stats, gradcam  # noqa: E402
from xraylens.model import REPO_ID, REVISION, load_model, predict_batch, preprocess  # noqa: E402
from xraylens.perturbations import PERTURBATIONS  # noqa: E402

EVAL, OUT = ROOT / "data" / "eval", ROOT / "results"
BATCH = 16


def score_files(paths, model, cfg, transform=None):
    scores = []
    for i in range(0, len(paths), BATCH):
        ims = [Image.open(EVAL / p) for p in paths[i:i + BATCH]]
        if transform:
            ims = [transform(im) for im in ims]
        scores.append(predict_batch(ims, model, cfg))
    return np.concatenate(scores)


def main():
    torch.manual_seed(0)
    t0 = time.time()
    model, cfg = load_model()
    df = pd.read_csv(EVAL / "manifest.csv", keep_default_na=False)
    print(f"{len(df)} images on {next(model.parameters()).device}")

    df["score"] = score_files(df.file.tolist(), model, cfg)

    # Grad-CAM attribution for the predicted class on every external image.
    cams = []
    for r in df.itertuples():
        im = Image.open(EVAL / r.file)
        boxed, x = preprocess(im, cfg)
        target = int(r.score >= cfg["threshold"])
        cam = gradcam(model, x, target)
        lung = None
        if r.mask:
            mask_boxed, _ = preprocess(Image.open(EVAL / r.mask), cfg)
            lung = np.asarray(mask_boxed.convert("L")) > 127
        cams.append(attribution_stats(cam, boxed, lung))
    cam_df = pd.DataFrame(cams).add_prefix("cam_")
    df = pd.concat([df, cam_df], axis=1)
    print(f"grad-cam done {time.time() - t0:.0f}s")

    # Robustness: rescore external adult cohorts under each perturbation.
    ext = df.cohort.isin(["rsna", "montgomery", "shenzhen"])
    for key, (_, fn) in PERTURBATIONS.items():
        df.loc[ext, f"pert_{key}"] = score_files(df.loc[ext, "file"].tolist(), model, cfg, fn)
    print(f"perturbations done {time.time() - t0:.0f}s")

    thr = cfg["threshold"]
    k = df[df.cohort == "kermany_test"]
    rsna = df[df.cohort == "rsna"]
    rsna_main = rsna[rsna.finding.isin(["Normal", "Lung Opacity"])]
    metrics = {
        "model": {"repo": REPO_ID, "revision": REVISION, "threshold": thr},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(next(model.parameters()).device)},
        "cohorts": {
            "kermany_test": {**ev.binary_metrics(k.label.values, k.score.values, thr),
                             "valid_test": False,
                             "note": "Model was refit on these images. Memorisation, not generalisation."},
            "rsna_normal_vs_opacity": {**ev.binary_metrics(rsna_main.label.values, rsna_main.score.values, thr),
                                       "valid_test": True},
        },
        "flag_rates": {},
        "calibration": {"rsna_normal_vs_opacity": ev.calibration(rsna_main.label.values, rsna_main.score.values)},
        "saturation": {c: ev.saturation_share(g.score.values) for c, g in df.groupby("cohort")},
        "attribution": {},
        "robustness": {},
    }
    for (cohort, finding), g in df.groupby(["cohort", "finding"]):
        if finding == "Pneumonia" or finding == "Lung Opacity":
            continue
        metrics["flag_rates"][f"{cohort}:{finding}"] = ev.flag_rate(g.score.values, thr)
    for cohort, g in df.groupby("cohort"):
        a = {"padding_median": float(g.cam_padding.median()),
             "border_median": float(g.cam_border.median()),
             "border_area_median": float(g.cam_border_area.median()),
             "padding_or_border_gt_50pct": float(((g.cam_padding + g.cam_border) > .5).mean())}
        if g.cam_lungs.notna().any():
            a["lungs_median"] = float(g.cam_lungs.median())
            a["lungs_iqr"] = [float(g.cam_lungs.quantile(.25)), float(g.cam_lungs.quantile(.75))]
            a["lungs_area_median"] = float(g.cam_lungs_area.median())
            # Enrichment > 1 means the map concentrates on lungs more than chance.
            enrich = g.cam_lungs / g.cam_lungs_area
            a["lungs_enrichment_median"] = float(enrich.median())
            a["lungs_enrichment_gt1"] = float((enrich > 1).mean())
        metrics["attribution"][cohort] = a
    e = df[ext]
    base_pred = e.score >= thr
    for key, (label, _) in PERTURBATIONS.items():
        p = e[f"pert_{key}"]
        metrics["robustness"][key] = {
            "label": label, "n": int(len(e)),
            "flip_rate": float(((p >= thr) != base_pred).mean()),
            "flip_rate_ci": ev.wilson(int(((p >= thr) != base_pred).sum()), len(e)),
            "median_abs_shift": float((p - e.score).abs().median()),
            "to_pneumonia": int(((p >= thr) & ~base_pred).sum()),
            "to_normal": int(((p < thr) & base_pred).sum()),
        }
    rs = rsna_main
    metrics["operating_points"] = {}
    for target in (0.8, 0.9):
        # Lowest threshold reaching the target specificity (scores are saturated, so
        # thresholds can sit extremely close to 1).
        for t in sorted(set(rs.score.values)):
            c = ev.confusion(rs.label.values, rs.score.values, float(t))
            spec, sens = c["tn"] / (c["tn"] + c["fp"]), c["tp"] / (c["tp"] + c["fn"])
            if spec >= target:
                metrics["operating_points"][f"rsna_spec_{int(target * 100)}"] = {
                    "threshold": float(t), "sensitivity": sens, "specificity": spec}
                break
    metrics["roc"] = {"rsna_normal_vs_opacity": ev.roc_points(rsna_main.label.values, rsna_main.score.values)}
    metrics["runtime_seconds"] = round(time.time() - t0, 1)

    OUT.mkdir(exist_ok=True)
    df.drop(columns=["file", "mask"]).to_csv(OUT / "scores.csv", index=False, float_format="%.9g")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps({k2: {m: v[m] for m in ("n", "sensitivity", "specificity", "auc")}
                      for k2, v in metrics["cohorts"].items()}, indent=1))
    print(json.dumps(metrics["flag_rates"], indent=1))
    print(json.dumps(metrics["attribution"], indent=1))
    print(json.dumps({k2: (v["flip_rate"], v["to_pneumonia"], v["to_normal"]) for k2, v in metrics["robustness"].items()}, indent=1))
    print(json.dumps(metrics["operating_points"], indent=1))
    print(json.dumps(metrics["saturation"], indent=1), "ECE", metrics["calibration"]["rsna_normal_vs_opacity"]["ece"])


if __name__ == "__main__":
    main()
