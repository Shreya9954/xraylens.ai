"""Controlled image edits for stress-testing the classifier.

None of these edits change the anatomy a radiologist would read. Large shifts in
the model's score therefore point to sensitivity to acquisition artefacts, not
clinically meaningful findings.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


def _gray(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im).convert("L")


def hflip(im):
    return ImageOps.mirror(_gray(im))


def invert(im):
    return ImageOps.invert(_gray(im))


def rotate10(im):
    return _gray(im).rotate(10, resample=Image.Resampling.BILINEAR, fillcolor=0)


def low_contrast(im):
    return ImageEnhance.Contrast(_gray(im)).enhance(0.55)


def blur(im):
    g = _gray(im)
    return g.filter(ImageFilter.GaussianBlur(radius=max(g.size) / 300))


def noise(im, sigma: float = 18.0, seed: int = 0):
    arr = np.asarray(_gray(im), dtype=np.float32)
    arr += np.random.default_rng(seed).normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def marker(im):
    """Burn a laterality-style 'L' marker and a tag into the top corner.

    Hospital-specific markers are a well-documented shortcut in chest X-ray models
    (Zech et al., PLOS Medicine 2018).
    """
    g = _gray(im).copy()
    d = ImageDraw.Draw(g)
    size = max(12, g.width // 14)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    x, y = int(g.width * 0.78), int(g.height * 0.04)
    d.text((x, y), "L", fill=255, font=font)
    d.text((int(g.width * 0.04), y), "PORTABLE", fill=235, font=font)
    return g


def crop_center(im):
    """Tight crop to the central 70%, as in a differently collimated exposure."""
    g = _gray(im)
    w, h = g.size
    return g.crop((int(w * .15), int(h * .15), int(w * .85), int(h * .85)))


PERTURBATIONS: dict[str, tuple[str, Callable[[Image.Image], Image.Image]]] = {
    "hflip": ("Mirror left–right", hflip),
    "rotate10": ("Rotate 10°", rotate10),
    "crop": ("Tighter collimation (70% crop)", crop_center),
    "contrast": ("Low contrast (×0.55)", low_contrast),
    "blur": ("Mild blur", blur),
    "noise": ("Sensor noise (σ = 18)", noise),
    "marker": ("Burned-in text markers", marker),
    "invert": ("Inverted greyscale", invert),
}
