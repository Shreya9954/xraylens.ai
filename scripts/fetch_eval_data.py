"""Download and downsample the evaluation cohorts into data/eval/ (git-ignored).

Cohorts
-------
kermany_test   624 pediatric images, the original Kermany "test" folder.
               WARNING: the final model was refit on ALL Kermany images, test folder
               included. Scores here measure memorisation, not generalisation. We
               keep it only to show how large that gap is.
rsna           Adult frontal radiographs (RSNA Pneumonia Detection Challenge 2018,
               derived from NIH ChestX-ray14). Fixed-seed sample of N per class:
               "Normal", "Lung Opacity" and "No Lung Opacity / Not Normal".
montgomery     138 adult radiographs (NLM, Montgomery County TB programme) with
               radiologist-drawn lung masks. Normal vs TB, used for (a) adult
               specificity and (b) measuring how much Grad-CAM mass lands in lungs.
shenzhen       Fixed-seed sample of adult radiographs (NLM / Shenzhen No.3 Hospital),
               normal vs TB. Out-of-scope disease probe.

Images are resized so the long side is 512 px. That is above the model's 320 px
input, so the letterboxed tensor is unchanged in practice, and it keeps disk use
small. Nothing here is committed: only manifests of IDs and per-image scores.
"""
from __future__ import annotations

import argparse
import io
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "eval"
LONG_SIDE = 512

RSNA = ("Baldezo313/rsna-pneumonia-dataset", "a7f1dad69625cb4d02b9ef8ee849d5c0e6fc24b4")
KERMANY = ("hf-vision/chest-xray-pneumonia", "c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc")
NLM = "Famatsu123/montgomery-shenzhen-tuberculosis-cxr"


def shrink(im: Image.Image) -> Image.Image:
    im = im.convert("L")
    s = LONG_SIDE / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS) if s < 1 else im


def dicom_to_image(path: str) -> Image.Image:
    import pydicom
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def fetch_kermany() -> pd.DataFrame:
    p = hf_hub_download(KERMANY[0], "data/test-00000-of-00001.parquet", repo_type="dataset",
                        revision=KERMANY[1], local_dir=RAW / "kermany")
    df = pd.read_parquet(p)
    out = OUT / "kermany_test"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in df.iterrows():
        name = f"kermany_{i:04d}.png"
        shrink(Image.open(io.BytesIO(r.image["bytes"]))).save(out / name)
        rows.append({"cohort": "kermany_test", "file": f"kermany_test/{name}",
                     "source_id": f"test[{i}]", "label": int(r.label),
                     "finding": ["Normal", "Pneumonia"][int(r.label)]})
    return pd.DataFrame(rows)


def fetch_rsna(per_class: int, seed: int) -> pd.DataFrame:
    api = HfApi()
    info = pd.read_csv(hf_hub_download(RSNA[0], "stage_2_detailed_class_info.csv", repo_type="dataset",
                                       revision=RSNA[1], local_dir=RAW / "rsna"))
    info = info.drop_duplicates("patientId")  # one row per bounding box upstream
    where = {}
    for d in ("stage_2_train_images_0", "stage_2_train_images_1", "stage_2_train_images_2"):
        for f in api.list_repo_tree(RSNA[0], path_in_repo=d, repo_type="dataset", revision=RSNA[1]):
            where[Path(f.path).stem] = f.path
    info = info[info.patientId.isin(where)]
    picks = pd.concat([g.sample(min(per_class, len(g)), random_state=seed)
                       for _, g in info.groupby("class", sort=True)])
    out = OUT / "rsna"
    out.mkdir(parents=True, exist_ok=True)

    def one(row):
        name = f"{row.patientId}.png"
        if not (out / name).exists():
            dcm = hf_hub_download(RSNA[0], where[row.patientId], repo_type="dataset",
                                  revision=RSNA[1], local_dir=RAW / "rsna")
            shrink(dicom_to_image(dcm)).save(out / name)
            Path(dcm).unlink()  # keep disk use bounded
        finding = row["class"]
        # Binary task: Lung Opacity -> positive, Normal -> negative,
        # "Not Normal" (other abnormality) -> negative for pneumonia but tracked separately.
        return {"cohort": "rsna", "file": f"rsna/{name}", "source_id": row.patientId,
                "label": int(finding == "Lung Opacity"), "finding": finding}

    with ThreadPoolExecutor(8) as pool:
        return pd.DataFrame(pool.map(one, [r for _, r in picks.iterrows()]))


def fetch_nlm(shenzhen_n: int, seed: int) -> pd.DataFrame:
    api = HfApi()
    sha = api.dataset_info(NLM).sha
    mont = sorted(f.path for f in api.list_repo_tree(NLM, path_in_repo="MontgomerySet/CXR_png",
                                                     repo_type="dataset", revision=sha)
                  if f.path.endswith(".png"))
    shen = sorted(f.path for f in api.list_repo_tree(NLM, path_in_repo="ChinaSet_AllFiles/CXR_png",
                                                     repo_type="dataset", revision=sha)
                  if f.path.endswith(".png"))
    rng = random.Random(seed)
    shen_pos = [p for p in shen if p.endswith("_1.png")]
    shen_neg = [p for p in shen if p.endswith("_0.png")]
    shen = rng.sample(shen_neg, shenzhen_n // 2) + rng.sample(shen_pos, shenzhen_n // 2)
    (OUT / "montgomery").mkdir(parents=True, exist_ok=True)
    (OUT / "shenzhen").mkdir(parents=True, exist_ok=True)

    def get(path):
        local = hf_hub_download(NLM, path, repo_type="dataset", revision=sha, local_dir=RAW / "nlm")
        im = Image.open(local)
        im.load()
        Path(local).unlink()
        return im

    def one(path):
        cohort = "montgomery" if path.startswith("Montgomery") else "shenzhen"
        stem = Path(path).stem
        tb = int(stem.endswith("_1"))
        im = get(path)
        small = shrink(im)
        small.save(OUT / cohort / f"{stem}.png")
        row = {"cohort": cohort, "file": f"{cohort}/{stem}.png", "source_id": stem,
               "label": 0,  # neither TB nor normal is "pneumonia" in this binary task
               "finding": "Tuberculosis" if tb else "Normal", "mask": ""}
        if cohort == "montgomery":
            masks = [np.asarray(get(f"MontgomerySet/ManualMask/{side}/{stem}.png").convert("L")
                                .resize(small.size, Image.Resampling.NEAREST)) > 127
                     for side in ("leftMask", "rightMask")]
            name = f"{stem}_lungs.png"
            Image.fromarray(((masks[0] | masks[1]) * 255).astype(np.uint8)).save(OUT / cohort / name)
            row["mask"] = f"{cohort}/{name}"
        return row

    with ThreadPoolExecutor(6) as pool:
        return pd.DataFrame(pool.map(one, mont + shen)), sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsna-per-class", type=int, default=200)
    ap.add_argument("--shenzhen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [fetch_kermany(), fetch_rsna(a.rsna_per_class, a.seed)]
    nlm, sha = fetch_nlm(a.shenzhen, a.seed)
    parts.append(nlm)
    manifest = pd.concat(parts, ignore_index=True)
    manifest["mask"] = manifest.get("mask", "").fillna("")
    manifest.to_csv(OUT / "manifest.csv", index=False)
    print(manifest.groupby(["cohort", "finding"]).size().to_string())
    print("NLM revision", sha)


if __name__ == "__main__":
    main()
