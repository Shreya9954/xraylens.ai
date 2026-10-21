"""Export the classifier to an ONNX graph for in-browser inference (web/).

Two properties matter:

1. The graph outputs the logits AND raw class-activation maps. For this
   architecture (layer4 -> global average pool -> fc), Grad-CAM at layer4 equals
   CAM with fc-weight differences, exactly:
       d(z_t - z_o)/dA_k(i,j) = (W[t,k] - W[o,k]) / (H*W)      for every (i, j)
   so Grad-CAM's channel weights are (W[t]-W[o])/(H*W), and the map is
   ReLU(sum_k (W[t,k]-W[o,k]) * A_k) up to a positive scale that max-normalisation
   removes. The browser therefore needs no gradients. verify() checks this
   against xraylens.explainability.gradcam.

2. The weights are NOT redistributed. Every initializer is rewritten as ONNX
   external data pointing at a byte range inside the model author's original
   model.safetensors, which the browser downloads directly from Hugging Face.
   The .onnx file we ship holds only the graph (a few hundred KB).
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from xraylens.explainability import gradcam  # noqa: E402
from xraylens.model import DEFAULT_DIR, load_model, preprocess  # noqa: E402

OUT = ROOT / "web" / "model"


class WithCAM(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        a = n.layer4(n.layer3(n.layer2(n.layer1(x))))            # (B, 2048, 10, 10)
        logits = n.fc(torch.flatten(n.avgpool(a), 1))
        w = n.fc.weight                                          # (2, 2048)
        diff = torch.stack([w[0] - w[1], w[1] - w[0]])           # class c vs the other
        cams = torch.relu(torch.einsum("ck,bkhw->bchw", diff, a))  # un-normalised CAMs
        return logits, cams


def safetensors_offsets(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def main():
    model, cfg = load_model(device=torch.device("cpu"))
    wrapped = WithCAM(model).eval()
    x = torch.zeros(1, 3, 320, 320)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "_tmp.onnx"
    torch.onnx.export(wrapped, (x,), str(tmp), input_names=["input"], output_names=["logits", "cams"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "cams": {0: "batch"}},
                      opset_version=17, do_constant_folding=False, dynamo=False)
    m = onnx.load(str(tmp), load_external_data=True)
    tmp.unlink()
    for extra in OUT.glob("_tmp.onnx*"):
        extra.unlink()

    st_path = DEFAULT_DIR / "model.safetensors"
    header, base = safetensors_offsets(st_path)
    tensors = load_file(str(st_path))
    by_bytes = {}
    for name, arr in tensors.items():
        by_bytes.setdefault(arr.tobytes(), []).append(name)

    kept, mapped = [], 0
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        names = by_bytes.get(np.ascontiguousarray(arr).tobytes())
        if not names or arr.dtype != np.float32:
            kept.append(init.name)  # small constants created by the exporter stay inline
            continue
        meta = header[names[0]]
        start, end = meta["data_offsets"]
        assert meta["dtype"] == "F32" and end - start == arr.nbytes
        init.ClearField("raw_data")
        del init.float_data[:]
        init.data_location = onnx.TensorProto.EXTERNAL
        del init.external_data[:]
        for k, v in (("location", "model.safetensors"), ("offset", str(base + start)), ("length", str(arr.nbytes))):
            e = init.external_data.add()
            e.key, e.value = k, v
        mapped += 1
    onnx.save(m, str(OUT / "xraylens.onnx"))
    size = (OUT / "xraylens.onnx").stat().st_size
    print(f"mapped {mapped} initializers to model.safetensors; {len(kept)} inline: {kept[:6]}")
    print(f"graph file {size / 1024:.0f} KB")
    verify(model, cfg)


def verify(model, cfg):
    """ONNX (reading the original safetensors) must match PyTorch logits and Grad-CAM."""
    import shutil

    import onnxruntime as ort
    from PIL import Image

    work = OUT / "_verify"
    work.mkdir(exist_ok=True)
    shutil.copy(OUT / "xraylens.onnx", work / "xraylens.onnx")
    os.link(DEFAULT_DIR / "model.safetensors", work / "model.safetensors")  # ORT rejects symlinks
    sess = ort.InferenceSession(str(work / "xraylens.onnx"), providers=["CPUExecutionProvider"])
    worst_logit, worst_cam = 0.0, 0.0
    gallery = sorted((ROOT / "assets" / "gallery").glob("*.png"))
    gallery = [g for g in gallery if not g.stem.endswith("_lungs")]
    for p in gallery:
        _, xt = preprocess(Image.open(p), cfg)
        logits, cams = sess.run(None, {"input": xt.numpy()})
        with torch.no_grad():
            ref = model(xt).numpy()
        worst_logit = max(worst_logit, float(np.abs(logits - ref).max()))
        for c in (0, 1):
            g = gradcam(model, xt, c)
            cam = torch.nn.functional.interpolate(torch.from_numpy(cams[:, c:c + 1]), size=(320, 320),
                                                  mode="bilinear", align_corners=False)[0, 0].numpy()
            cam = cam / cam.max() if cam.max() > 1e-12 else np.zeros_like(cam)
            worst_cam = max(worst_cam, float(np.abs(cam - g).max()))
    shutil.rmtree(work)
    print(f"verify: max |Δlogit| = {worst_logit:.2e}, max |ΔCAM| = {worst_cam:.2e} over {len(gallery)} gallery files")
    assert worst_logit < 1e-3 and worst_cam < 1e-3


if __name__ == "__main__":
    main()
