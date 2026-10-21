"""Grad-CAM for the ResNet-50 classifier, plus checks on whether a heatmap is trustworthy.

Implementation notes
--------------------
* Target layer is ``model.layer4[-1]`` (the last bottleneck block, 2048 x 10 x 10
  at 320 px input). That is the deepest spatial feature map, which is standard for
  ResNet Grad-CAM (Selvaraju et al., 2017).
* We back-propagate the *logit difference* z_target - z_other rather than the
  softmax score. This model's softmax is saturated (scores of 0.0000/1.0000 are
  common), and the softmax gradient vanishes at saturation, so a softmax-based
  CAM would be numerically empty exactly on the most confident images.
* ``randomization_check`` implements the model-parameter randomisation test of
  Adebayo et al., "Sanity Checks for Saliency Maps" (NeurIPS 2018): if the map
  barely changes when learned weights are destroyed, it is not explaining the
  model.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from scipy.stats import spearmanr


def gradcam(model, x: torch.Tensor, target: int, layer=None) -> np.ndarray:
    """Return a Grad-CAM map in [0, 1] at the input resolution (H, W)."""
    layer = layer if layer is not None else model.layer4[-1]
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        device = next(model.parameters()).device
        x = x.to(device).requires_grad_(False)
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x)
            other = 1 - target
            (logits[0, target] - logits[0, other]).backward()
    finally:
        handle.remove()

    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)          # global-average-pooled grads
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, h, w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].float().cpu().numpy()
    peak = cam.max()
    return cam / peak if peak > 1e-12 else np.zeros_like(cam)


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45,
            cmap: str = "inferno") -> Image.Image:
    """Blend a CAM onto a (letterboxed) image. Weak activations stay transparent."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    heat = colormaps[cmap](cam)[..., :3].astype(np.float32)
    a = (alpha * np.clip(cam * 1.4, 0, 1))[..., None]
    out = base * (1 - a) + heat * a
    return Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))


def content_mask(image: Image.Image) -> np.ndarray:
    """True where the letterboxed image has real pixels (False on black padding).

    Padding is exact black from ``ImageOps.pad``; a tiny tolerance absorbs bilinear
    edges. Genuine pure-black pixels inside an X-ray are rare and sparse.
    """
    arr = np.asarray(image.convert("L"))
    rows = arr.max(axis=1) > 2
    cols = arr.max(axis=0) > 2
    mask = np.zeros_like(arr, dtype=bool)
    if rows.any() and cols.any():
        r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
        c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
        mask[r0:r1, c0:c1] = True
    return mask


def attribution_stats(cam: np.ndarray, image: Image.Image,
                      lung_mask: np.ndarray | None = None) -> dict[str, float]:
    """Where does the heatmap's mass go?

    * ``padding``  – share on the black letterbox bars (never informative).
    * ``border``   – share in the outer 12% frame of the real image, where text
      markers, rulers, tubes and collimator edges live (classic shortcut zones).
    * ``lungs``    – share inside a radiologist lung mask, when one is supplied.

    Each share comes with a matching ``*_area`` baseline.
    """
    total = float(cam.sum()) or 1.0
    content = content_mask(image)
    # ``*_area`` is the share of pixels in each region: the value a uniform,
    # uninformative heatmap would score. Compare mass against area, never alone.
    stats = {"padding": float(cam[~content].sum()) / total,
             "padding_area": float((~content).mean())}

    ys, xs = np.where(content)
    if len(ys):
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mh, mw = int((r1 - r0) * 0.12), int((c1 - c0) * 0.12)
        inner = np.zeros_like(content)
        inner[r0 + mh:r1 - mh, c0 + mw:c1 - mw] = True
        stats["border"] = float(cam[content & ~inner].sum()) / total
        stats["border_area"] = float((content & ~inner).mean())
    else:
        stats["border"] = stats["border_area"] = 0.0
    if lung_mask is not None:
        lung_mask = lung_mask.astype(bool)
        stats["lungs"] = float(cam[lung_mask].sum()) / total
        stats["lungs_area"] = float(lung_mask.mean())
    return stats


def peak_location(cam: np.ndarray) -> tuple[int, int]:
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return int(y), int(x)


def randomization_check(model, x: torch.Tensor, target: int, cam: np.ndarray,
                        seed: int = 0) -> dict:
    """Adebayo et al. cascading randomisation, top-down.

    Re-initialises fc, then layer4, then layer3 (cumulatively) and recomputes
    Grad-CAM each time. Returns Spearman rank correlation with the original map
    for each stage. A trustworthy explanation should decorrelate quickly.
    """
    torch.manual_seed(seed)
    probe = copy.deepcopy(model).eval()
    stages = [("fc", [probe.fc]), ("layer4", [probe.layer4]), ("layer3", [probe.layer3])]
    results, maps = [], []
    for name, modules in stages:
        for module in modules:
            for m in module.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        rand_cam = gradcam(probe, x, target, layer=probe.layer4[-1])
        rho = spearmanr(cam.ravel()[::7], rand_cam.ravel()[::7]).statistic
        results.append({"randomised": name, "spearman": float(np.nan_to_num(rho))})
        maps.append(rand_cam)
    del probe
    return {"stages": results, "maps": maps}
