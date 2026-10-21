import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def cxr_like():
    """Synthetic 'radiograph': dark lungs either side of a bright mediastinum, wider than tall."""
    h, w = 300, 400
    y, x = np.mgrid[0:h, 0:w]
    arr = np.full((h, w), 40, np.float32)
    body = ((x - w / 2) / (w * .45)) ** 2 + ((y - h * .55) / (h * .5)) ** 2 < 1
    arr[body] = 170
    for cx in (w * .32, w * .68):
        lung = ((x - cx) / (w * .13)) ** 2 + ((y - h * .5) / (h * .32)) ** 2 < 1
        arr[lung] = 70
    arr[:, int(w * .46):int(w * .54)][body[:, int(w * .46):int(w * .54)]] = 220
    arr += np.random.default_rng(0).normal(0, 6, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "L")
