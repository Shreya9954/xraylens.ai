"""Preprocessing must match the model card's inference.py exactly."""
import io

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from torchvision import transforms as T

from xraylens.inputs import InputError, assess, load_upload
from xraylens.model import letterbox, preprocess

CFG = {"image_size": 320, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def reference(image):
    """Verbatim from Purino/resnet50-chest-xray-pneumonia inference.py."""
    size = CFG["image_size"]
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR,
                         color=(0, 0, 0), centering=(0.5, 0.5))
    return T.Compose([T.ToTensor(), T.Normalize(CFG["mean"], CFG["std"])])(image)


def test_matches_model_card_reference(cxr_like):
    boxed, x = preprocess(cxr_like, CFG)
    assert x.shape == (1, 3, 320, 320)
    assert boxed.size == (320, 320)
    torch.testing.assert_close(x[0], reference(cxr_like))


def test_letterbox_pads_with_black_and_keeps_aspect(cxr_like):
    boxed = np.asarray(letterbox(cxr_like, 320))
    # 400x300 -> 320x240 content, 40 px bars top and bottom
    assert boxed[:38].max() == 0 and boxed[-38:].max() == 0
    assert boxed[160].max() > 0


def test_grayscale_rgba_and_16bit_all_become_rgb(cxr_like):
    for im in (cxr_like, cxr_like.convert("RGBA"), cxr_like.convert("I;16")):
        assert preprocess(im, CFG)[1].shape == (1, 3, 320, 320)


def _png_bytes(im, fmt="PNG"):
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def test_upload_roundtrip(cxr_like):
    loaded = load_upload(_png_bytes(cxr_like), "chest.png")
    assert loaded.image.size == cxr_like.size
    assert loaded.warnings == []


@pytest.mark.parametrize("data,name,msg", [
    (b"", "x.png", "empty"),
    (b"not an image at all", "x.png", "not a readable image"),
    (b"%PDF-1.4", "report.pdf", "not supported"),
])
def test_bad_uploads_raise_friendly_errors(data, name, msg):
    with pytest.raises(InputError, match=msg):
        load_upload(data, name)


def test_truncated_image_rejected(cxr_like):
    data = _png_bytes(cxr_like)
    with pytest.raises(InputError):
        load_upload(data[: len(data) // 2], "cut.png")


def test_tiny_and_blank_images_rejected():
    with pytest.raises(InputError, match="at least"):
        assess(Image.new("L", (40, 40), 128))
    with pytest.raises(InputError, match="blank"):
        assess(Image.new("L", (400, 400), 128))


def test_warnings_for_colour_inverted_and_odd_aspect(cxr_like):
    colour = Image.merge("RGB", [cxr_like, cxr_like.point(lambda v: 255 - v), cxr_like])
    assert any("colour" in w for w in assess(colour))
    assert any("inverted" in w for w in assess(ImageOps.invert(cxr_like)))
    wide = cxr_like.resize((900, 300))
    assert any("aspect" in w for w in assess(wide))


def test_dicom_upload(cxr_like, tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "x.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
    arr = np.asarray(cxr_like, dtype=np.uint16) * 16
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME1"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.Modality, ds.ViewPosition, ds.PatientName = "CR", "PA", "Should^NotAppear"
    ds.PixelData = (4095 - arr).astype(np.uint16).tobytes()  # stored inverted (MONOCHROME1)
    ds.save_as(tmp_path / "x.dcm", enforce_file_format=True)
    loaded = load_upload((tmp_path / "x.dcm").read_bytes(), "x.dcm")
    assert loaded.meta == {"Modality": "CR", "ViewPosition": "PA"}
    assert "PatientName" not in loaded.meta
    # MONOCHROME1 must be flipped back so bone is white: centre brighter than corner
    out = np.asarray(loaded.image)
    assert out[150, 200] > out[5, 5]
