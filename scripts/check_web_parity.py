"""Check that the browser pipeline (web/js) reproduces the Python pipeline.

Serves web/ locally with cross-origin-isolation headers, opens a harness page in
headless Chrome, and compares, image by image:
  * letterboxed 320x320 model input        (SHA-1 of bytes, must be identical)
  * deterministic perturbations             (SHA-1, must be identical)
  * input-check warnings                    (must be identical)
  * PNEUMONIA score and Grad-CAM            (max abs difference)

Usage: python scripts/check_web_parity.py [--n 60]
Needs data/eval (scripts/fetch_eval_data.py) and playwright.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import io
import json
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from xraylens.explainability import gradcam  # noqa: E402
from xraylens.inputs import assess  # noqa: E402
from xraylens.model import letterbox, load_model, predict, preprocess  # noqa: E402
from xraylens.perturbations import crop_center, hflip, invert, low_contrast, rotate10  # noqa: E402

WEB = ROOT / "web"
WORK = WEB / "_parity"
EXACT = {"hflip": hflip, "rotate10": rotate10, "crop": crop_center, "contrast": low_contrast, "invert": invert}


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        super().end_headers()

    def log_message(self, *a):
        pass


def sha(im: Image.Image) -> str:
    return hashlib.sha1(im.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    model, cfg = load_model()
    man = pd.read_csv(ROOT / "data" / "eval" / "manifest.csv", keep_default_na=False)
    sample = pd.concat([g.sample(a.n // 4, random_state=1) for _, g in man.groupby("cohort")])
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir()
    cases = []
    for i, r in enumerate(sample.itertuples()):
        im = Image.open(ROOT / "data" / "eval" / r.file)
        im.load()
        # Every third image as JPEG to exercise the JPEG decoder too.
        if i % 3 == 0:
            name = f"{i:03d}.jpg"
            im.convert("L").save(WORK / name, quality=90)
        else:
            name = f"{i:03d}.png"
            im.save(WORK / name)
        im = Image.open(WORK / name)
        im.load()
        boxed, x = preprocess(im, cfg)
        p = predict(im, model, cfg)
        cam = gradcam(model, x, 1)
        cases.append({
            "file": name, "cohort": r.cohort,
            "boxed": sha(letterbox(im, 320)), "score": p.pneumonia_score,
            "pert": {k: sha(fn(im)) for k, fn in EXACT.items()},
            "warnings": assess(im),
            "cam": [round(float(v), 6) for v in cam[::16, ::16].ravel()],  # 20x20 subsample
        })
    (WORK / "expected.json").write_text(json.dumps(cases))
    (WORK / "index.html").write_text(HARNESS)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/_parity/index.html"
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.on("console", lambda m: print("  [browser]", m.text) if m.type == "error" else None)
        page.goto(url)
        page.wait_for_function("window.__done === true", timeout=600_000)
        res = page.evaluate("window.__results")
        info = page.evaluate("window.__info")
        browser.close()
    server.shutdown()
    shutil.rmtree(WORK)

    exp = {c["file"]: c for c in cases}
    boxed_ok = sum(r["boxed"] == exp[r["file"]]["boxed"] for r in res)
    pert_ok = {k: sum(r["pert"][k] == exp[r["file"]]["pert"][k] for r in res) for k in EXACT}
    warn_ok = sum(r["warnings"] == exp[r["file"]]["warnings"] for r in res)
    dscore = np.array([abs(r["score"] - exp[r["file"]]["score"]) for r in res])
    dcam = np.array([np.abs(np.array(r["cam"]) - np.array(exp[r["file"]]["cam"])).max() for r in res])
    labels_ok = sum((r["score"] >= .5) == (exp[r["file"]]["score"] >= .5) for r in res)
    n = len(res)
    report = {"images": n, "browser": info, "letterbox_identical": boxed_ok,
              "perturbations_identical": pert_ok, "warnings_identical": warn_ok,
              "labels_identical": labels_ok, "max_abs_score_diff": float(dscore.max()),
              "median_abs_score_diff": float(np.median(dscore)), "max_abs_cam_diff": float(dcam.max())}
    print(json.dumps(report, indent=1))
    for r in res:
        e = exp[r["file"]]
        if r["boxed"] != e["boxed"] or r["warnings"] != e["warnings"]:
            print("  mismatch", r["file"], e["cohort"], "boxed" if r["boxed"] != e["boxed"] else "", r["warnings"], e["warnings"])
    (ROOT / "results" / "web_parity.json").write_text(json.dumps(report, indent=2))
    ok = boxed_ok == n and all(v == n for v in pert_ok.values()) and labels_ok == n and dscore.max() < 1e-4
    sys.exit(0 if ok else 1)


HARNESS = """<!doctype html><meta charset="utf-8"><script type="module">
import { decodeBlob, letterbox, assess, PERTURBATIONS, upsampleCam } from "../js/imaging.js";
import { loadModel, infer, backendInfo } from "../js/model.js";
const hex = async (u8) => [...new Uint8Array(await crypto.subtle.digest("SHA-1", u8))].map(b => b.toString(16).padStart(2, "0")).join("");
const cases = await (await fetch("expected.json")).json();
await loadModel();
window.__info = { ...backendInfo(), userAgent: navigator.userAgent };
const out = [];
for (const c of cases) {
  const img = await decodeBlob(await (await fetch(c.file)).blob());
  // decodeBlob returns RGB; PIL keeps greyscale JPEG/PNG as "L" then pads in RGB, which matches.
  const r = { file: c.file, boxed: await hex(letterbox(img, 320).data), pert: {} };
  for (const [k, , fn, exact] of PERTURBATIONS) if (exact) r.pert[k] = await hex(fn(img).data);
  try { r.warnings = assess(img); } catch (e) { r.warnings = ["ERROR " + e.message]; }
  const [res] = await infer([img]);
  r.score = res.pneumonia;
  const cam = upsampleCam(res.cams[1], res.camDims[0], res.camDims[1], 320, 320);
  r.cam = []; for (let y = 0; y < 320; y += 16) for (let x = 0; x < 320; x += 16) r.cam.push(cam[y * 320 + x]);
  out.push(r);
}
window.__results = out; window.__done = true;
</script>"""

if __name__ == "__main__":
    main()
