"""Build the static data the web app needs (web/data, web/gallery).

* metrics.json, scores.json   evaluation results (from scripts/run_evaluation.py)
* gallery/                    teaching cases (from scripts/build_gallery.py), plus
                              PyTorch-precomputed outputs so pages render before the
                              in-browser model has downloaded: logits, raw 10x10
                              CAMs, stress-test scores and randomisation-test maps
* randomisation.json          the randomisation sanity check run across a sample of
                              evaluation images (it needs to re-initialise weights,
                              which the browser can't do with the fixed ONNX graph)

Run after export_onnx.py. Deterministic.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from xraylens.model import load_model, predict_batch, preprocess  # noqa: E402
from xraylens.perturbations import PERTURBATIONS  # noqa: E402

WEB = ROOT / "web"


def raw_cams(model, x):
    """Logits and un-normalised CAMs (2, 10, 10), the same quantities the ONNX graph emits."""
    store = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    with torch.no_grad():
        logits = model(x)
    h.remove()
    w = model.fc.weight
    diff = torch.stack([w[0] - w[1], w[1] - w[0]])
    cams = F.relu(torch.einsum("ck,bkhw->bchw", diff, store["a"]))[0]
    return logits[0].tolist(), cams.detach().numpy()


def upsample(cam10):
    t = F.interpolate(torch.from_numpy(cam10)[None, None], size=(320, 320), mode="bilinear", align_corners=False)[0, 0].numpy()
    return t / t.max() if t.max() > 1e-12 else np.zeros_like(t)


def randomisation(model, x, target, seed=0):
    """Same cascade and statistic as explainability.randomization_check, but keeps raw 10x10 maps."""
    torch.manual_seed(seed)
    _, base = raw_cams(model, x)
    ref = upsample(base[target])
    probe = copy.deepcopy(model).eval()
    out = []
    for name, module in (("fc", probe.fc), ("layer4", probe.layer4), ("layer3", probe.layer3)):
        for m in module.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        _, cams = raw_cams(probe, x)
        rho = spearmanr(ref.ravel()[::7], upsample(cams[target]).ravel()[::7]).statistic
        out.append({"randomised": name, "spearman": float(np.nan_to_num(rho)),
                    "cam": np.round(cams[target], 6).ravel().tolist()})
    return out


def main():
    model, cfg = load_model(device=torch.device("cpu"))
    data, gal_out = WEB / "data", WEB / "gallery"
    data.mkdir(parents=True, exist_ok=True)
    if gal_out.exists():
        shutil.rmtree(gal_out)
    gal_out.mkdir()

    shutil.copy(ROOT / "results" / "metrics.json", data / "metrics.json")
    if (ROOT / "results" / "web_parity.json").exists():
        shutil.copy(ROOT / "results" / "web_parity.json", data / "web_parity.json")
    s = pd.read_csv(ROOT / "results" / "scores.csv", keep_default_na=False, na_values=[""])
    cols = ["cohort", "source_id", "label", "finding", "score", "cam_lungs", "cam_lungs_area"]
    rows = s[cols].astype(object).where(s[cols].notna(), None).values.tolist()
    (data / "scores.json").write_text(json.dumps({"columns": cols, "rows": rows}, separators=(",", ":")))

    gallery = json.loads((ROOT / "assets" / "gallery" / "gallery.json").read_text())
    for item in gallery:
        for key in ("file", "thumb"):
            shutil.copy(ROOT / "assets" / "gallery" / item[key], gal_out / item[key])
        if item["mask"]:
            shutil.copy(ROOT / item["mask"], gal_out / Path(item["mask"]).name)
            item["mask"] = Path(item["mask"]).name
        im = Image.open(ROOT / "assets" / "gallery" / item["file"])
        im.load()
        _, x = preprocess(im, cfg)
        logits, cams = raw_cams(model, x)
        pred = int(torch.tensor(logits).softmax(0)[1] >= cfg["threshold"])
        stress_ims = [im] + [fn(im) for _, fn in PERTURBATIONS.values()]
        stress_scores = predict_batch(stress_ims, model, cfg).tolist()
        item["precomputed"] = {
            "logits": logits,
            "cams": [np.round(cams[c], 6).ravel().tolist() for c in (0, 1)],
            "stress": dict(zip(["original"] + list(PERTURBATIONS), stress_scores)),
            "randomisation": {cfg["classes"][c]: randomisation(model, x, c) for c in (0, 1)},
            "predicted": cfg["classes"][pred],
        }
        print(f"{item['id']:26s} score={torch.tensor(logits).softmax(0)[1]:.6f}")
    (gal_out / "gallery.json").write_text(json.dumps(gallery, separators=(",", ":")))

    # Population randomisation test: 60 RSNA + 60 Montgomery images, predicted class.
    man = pd.read_csv(ROOT / "data" / "eval" / "manifest.csv", keep_default_na=False)
    sample = pd.concat([man[man.cohort == c].sample(60, random_state=3) for c in ("rsna", "montgomery")])
    rows = []
    for r in sample.itertuples():
        im = Image.open(ROOT / "data" / "eval" / r.file)
        _, x = preprocess(im, cfg)
        logits, _ = raw_cams(model, x)
        target = int(logits[1] > logits[0])
        st = randomisation(model, x, target)
        rows.append({"cohort": r.cohort, **{f"rho_{d['randomised']}": d["spearman"] for d in st}})
    df = pd.DataFrame(rows)
    df["max_abs"] = df[["rho_fc", "rho_layer4", "rho_layer3"]].abs().max(axis=1)
    summary = {"n": len(df), "pass_threshold": 0.4,
               "pass_rate": float((df.max_abs < 0.4).mean()),
               "median_abs": {c: float(df[c].abs().median()) for c in ("rho_fc", "rho_layer4", "rho_layer3")},
               "rows": df.round(4).to_dict(orient="records")}
    (data / "randomisation.json").write_text(json.dumps(summary, separators=(",", ":")))
    print("randomisation pass rate", summary["pass_rate"], summary["median_abs"])


if __name__ == "__main__":
    main()
