"""Extract the 512x512 emap matrix baked into inswapper_128.onnx.

The inswapper network expects its source-face input to be the ArcFace
embedding after `latent = embedding @ emap` (and L2 renormalisation).
The emap is stored inside the ONNX graph as a float32 initializer of shape
(512, 512). We pull it out and save it as a raw little-endian f32 blob
that the Rust binary loads at startup.

Run from the project root:

    python scripts/extract_emap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx

root = Path(__file__).resolve().parent.parent
model_path = root / "models" / "inswapper_128.onnx"
out_path = root / "models" / "emap.bin"

model = onnx.load(str(model_path))
matches = [
    (init.name, onnx.numpy_helper.to_array(init))
    for init in model.graph.initializer
    if onnx.numpy_helper.to_array(init).dtype == np.float32
    and onnx.numpy_helper.to_array(init).shape == (512, 512)
]
if not matches:
    sys.exit("no (512,512) f32 initializer found — is this the right model?")

name, emap = matches[-1]
print(f"found emap as initializer {name!r}, shape={emap.shape}")
emap.astype(np.float32).tofile(str(out_path))
print(f"wrote {out_path}")
