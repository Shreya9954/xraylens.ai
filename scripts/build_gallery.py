"""Pick teaching cases from the evaluation by rule and copy them into assets/gallery/.

Only redistributable sources are used: Kermany (CC BY 4.0) and the NLM Montgomery
and Shenzhen sets. RSNA images are never copied. Every pick is a deterministic
rule over results/scores.csv, so the gallery can be regenerated.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from xraylens.model import load_model, predict  # noqa: E402

EVAL, OUT = ROOT / "data" / "eval", ROOT / "assets" / "gallery"
AMBER, PNEU, NORM, OTHER, MUTED = "#ffb547", "#e05c35", "#38a596", "#8b7ff0", "#93a1a6"


def thumb(im: Image.Image) -> Image.Image:
    """Uniform 4:5 contact-sheet frame so every card in the sheet lines up."""
    return ImageOps.pad(im.convert("L"), (240, 300), color=11)


def main():
    s = pd.read_csv(ROOT / "results" / "scores.csv", keep_default_na=False, na_values=[""])
    man = pd.read_csv(EVAL / "manifest.csv", keep_default_na=False)
    s = s.merge(man[["cohort", "source_id", "file", "mask"]], on=["cohort", "source_id"])
    s["offbody"] = s.cam_padding + s.cam_border
    k, mg, sz = (s[s.cohort == c] for c in ("kermany_test", "montgomery", "shenzhen"))

    picks = [
        ("textbook-normal", k[(k.finding == "Normal") & (k.score < 1e-3)].sort_values("offbody").iloc[0],
         "Textbook normal", "Child · Kermany", True,
         "This child’s radiograph is part of the model’s <b>training set</b>. A near-zero score here shows memory, "
         "not skill. Compare it with the adult normals in this sheet."),
        ("textbook-pneumonia", k[(k.finding == "Pneumonia") & (k.score > .999)].sort_values("offbody").iloc[0],
         "Textbook pneumonia", "Child · Kermany", True,
         "Labelled pneumonia, <b>seen in training</b>, scored with saturated confidence. The heatmap sits on the "
         "anatomy, so this is the explanation everyone hopes for. Now look at the next case."),
        ("right-answer-wrong-place", k[(k.finding == "Pneumonia") & (k.score > .99)].sort_values("offbody").iloc[-1],
         "Right answer, wrong place?", "Child · Kermany", True,
         "Correct label, but most Grad-CAM mass falls on the image frame and padding rather than the lungs. "
         "A right answer from the wrong evidence is how shortcut learning hides inside high accuracy."),
        ("adult-false-alarm", mg[(mg.finding == "Normal") & (mg.score > .99)].sort_values("cam_lungs").iloc[0],
         "Healthy adult, flagged", "Adult · Montgomery", False,
         "An adult radiograph read as <b>normal</b> by radiologists, scored as pneumonia with near-certainty. "
         "The teal outline is the radiologist’s lung mask. Note where the heat peaks: on the burned-in "
         "<b>“PA ERECT”</b> marker, not the lungs. In step 04, the tighter crop that removes the marker "
         "turns the verdict to NORMAL. This is shortcut learning caught in the act."),
        ("adult-true-normal", mg[(mg.finding == "Normal") & (mg.score < .05)].sort_values("cam_lungs").iloc[-1],
         "Healthy adult, cleared", "Adult · Montgomery", False,
         "A normal adult the model gets right. Use the stress test below to see how easily that verdict flips."),
        ("tb-flagged", mg[(mg.finding == "Tuberculosis") & (mg.score > .99)].sort_values("cam_lungs").iloc[-1],
         "Tuberculosis, called pneumonia", "Adult · Montgomery", False,
         "Tuberculosis is a disease this model was never taught. It can only answer NORMAL or PNEUMONIA, so "
         "abnormal lungs of any cause get pushed into one of two boxes."),
        ("tb-missed", sz[(sz.finding == "Tuberculosis") & (sz.score < .01)].sort_values("score").iloc[0],
         "Tuberculosis, called normal", "Adult · Shenzhen", False,
         "The same disease from another hospital, now scored <b>normal</b> with confidence. A binary classifier "
         "has no way to say “something else is wrong”."),
    ]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    items = []
    for gid, row, title, subtitle, seen, story in picks:
        src = EVAL / row.file
        dst = f"{gid}.png"
        shutil.copy(src, OUT / dst)
        thumb(Image.open(src)).save(OUT / f"{gid}_thumb.jpg", quality=88)
        mask = ""
        if row["mask"]:
            mask = f"assets/gallery/{gid}_lungs.png"
            shutil.copy(EVAL / row["mask"], ROOT / mask)
        tags = [("seen in training", PNEU)] if seen else [("external · never seen", NORM)]
        tags.append((f"label: {row.finding.lower()}", MUTED))
        if row.cohort in ("montgomery",):
            tags.append(("radiologist lung mask", NORM))
        licence = "CC BY 4.0, Kermany et al. 2018" if row.cohort == "kermany_test" else "NLM public release, Jaeger et al. 2014"
        items.append({"id": gid, "file": dst, "thumb": f"{gid}_thumb.jpg", "mask": mask, "title": title,
                      "subtitle": subtitle, "seen_in_training": seen, "story": story, "tags": tags,
                      "source": f"{row.cohort}:{row.source_id}", "licence": licence, "label": row.finding})
        print(f"{gid:26s} {row.cohort}:{row.source_id:18s} score={row.score:.6f} offbody={row.offbody:.2f}")

    # A deliberately out-of-domain input. Generated here, so there are no licence questions.
    rng = np.random.default_rng(7)
    im = Image.new("L", (512, 420), 30)
    d = ImageDraw.Draw(im)
    for i in range(14):
        r = 20 + i * 16
        d.ellipse((256 - r, 210 - r, 256 + r, 210 + r), outline=int(80 + 10 * i), width=6)
    for _ in range(40):
        x, y = rng.integers(0, 512), rng.integers(0, 420)
        d.rectangle((x, y, x + rng.integers(8, 40), y + rng.integers(8, 40)), fill=int(rng.integers(90, 230)))
    im = im.filter(ImageFilter.GaussianBlur(1.2))
    im.save(OUT / "not-an-xray.png")
    thumb(im).save(OUT / "not-an-xray_thumb.jpg", quality=88)
    model, cfg = load_model()
    p = predict(im, model, cfg)
    items.append({"id": "not-an-xray", "file": "not-an-xray.png", "thumb": "not-an-xray_thumb.jpg", "mask": "",
                  "title": "Not an X-ray", "subtitle": "Synthetic pattern", "seen_in_training": False,
                  "story": f"Rings and rectangles, no anatomy at all. The model still answers "
                           f"<b>{p.label}</b> ({p.pneumonia_score:.4f}). A classifier has no “I don’t know” output. "
                           "Recognising an unsuitable input is the job of the system around it.",
                  "tags": [("out of domain", OTHER), ("generated", MUTED)], "source": "generated",
                  "licence": "Generated by scripts/build_gallery.py", "label": "n/a"})
    print(f"{'not-an-xray':26s} generated score={p.pneumonia_score:.6f}")
    (OUT / "gallery.json").write_text(json.dumps(items, indent=2))


if __name__ == "__main__":
    main()
