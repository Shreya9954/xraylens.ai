"""Model loading, preprocessing and inference for the Purino ResNet-50 classifier.

Preprocessing mirrors the model card's ``inference.py`` exactly (EXIF transpose,
RGB, aspect-preserving letterbox pad to 320x320 with black, ImageNet
normalisation). Any deviation here would silently change the model's outputs, so
``tests/test_preprocessing.py`` pins this behaviour.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torchvision import models
from torchvision import transforms as T

REPO_ID = "Purino/resnet50-chest-xray-pneumonia"
# Pin the exact Hub commit so results are reproducible even if the repo changes.
REVISION = "da2a7724bfa53ca6693fd4e1783b211963ed2a55"
MODEL_FILES = ("config.json", "model.safetensors")
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models" / "resnet50-chest-xray-pneumonia"


@dataclass(frozen=True)
class Prediction:
    label: str
    scores: dict[str, float]
    threshold: float

    @property
    def pneumonia_score(self) -> float:
        return self.scores["PNEUMONIA"]

    @property
    def margin(self) -> float:
        """Distance of the pneumonia score from the decision threshold (0–0.5)."""
        return abs(self.pneumonia_score - self.threshold)


def download_model(folder: Path = DEFAULT_DIR) -> Path:
    """Fetch config and weights from the Hugging Face Hub if they are not cached."""
    folder = Path(folder)
    if all((folder / f).exists() for f in MODEL_FILES):
        return folder
    from huggingface_hub import hf_hub_download

    folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        hf_hub_download(REPO_ID, name, revision=REVISION, local_dir=folder)
    return folder


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS gives ~5x on Apple silicon; Grad-CAM backward works there too.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(folder: Path = DEFAULT_DIR, device: torch.device | None = None):
    folder = download_model(folder)
    config = json.loads((folder / "config.json").read_text())
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, config["num_classes"])
    model.load_state_dict(load_file(str(folder / "model.safetensors")))
    model.to(device or pick_device()).eval()
    return model, config


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """EXIF-transpose, convert to RGB and pad to a centred square, as in training."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                        color=(0, 0, 0), centering=(0.5, 0.5))


def to_tensor(image: Image.Image, config: dict) -> torch.Tensor:
    """Letterboxed PIL image -> normalised (1, 3, H, W) tensor."""
    tf = T.Compose([T.ToTensor(), T.Normalize(config["mean"], config["std"])])
    return tf(image).unsqueeze(0)


def preprocess(image: Image.Image, config: dict) -> tuple[Image.Image, torch.Tensor]:
    boxed = letterbox(image, config["image_size"])
    return boxed, to_tensor(boxed, config)


def scores_from_logits(logits: torch.Tensor, config: dict) -> Prediction:
    probs = logits.softmax(1)[0].detach().float().cpu().tolist()
    scores = dict(zip(config["classes"], probs))
    label = config["classes"][int(probs[1] >= config["threshold"])]
    return Prediction(label=label, scores=scores, threshold=config["threshold"])


@torch.inference_mode()
def predict(image: Image.Image, model, config: dict) -> Prediction:
    _, x = preprocess(image, config)
    device = next(model.parameters()).device
    return scores_from_logits(model(x.to(device)), config)


@torch.inference_mode()
def predict_batch(images: list[Image.Image], model, config: dict) -> np.ndarray:
    """Return the PNEUMONIA softmax score for each image (used by evaluation)."""
    device = next(model.parameters()).device
    x = torch.cat([preprocess(im, config)[1] for im in images]).to(device)
    return model(x).softmax(1)[:, 1].float().cpu().numpy()
