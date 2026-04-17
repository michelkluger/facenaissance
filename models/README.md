# ONNX model files

Drop the following ONNX files into this directory. All come from the
[InsightFace](https://github.com/deepinsight/insightface) project.

| File | Purpose | Source |
|------|---------|--------|
| `det_10g.onnx`          | Face detection + 5-point landmarks (SCRFD 10GF) | InsightFace model zoo (`buffalo_l` bundle) |
| `w600k_r50.onnx`        | ArcFace 512-d embedding             | InsightFace model zoo (`buffalo_l` bundle) |
| `inswapper_128.onnx`    | Face-swap (target + source → swapped) | `insightface` python package / model zoo |

## Fastest way to get them

If you have Python available, this one-liner pulls the two `buffalo_l`
models into your cache, and you can copy them here:

```bash
pip install insightface onnxruntime
python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1)"
# Then copy ~/.insightface/models/buffalo_l/{scrfd_2.5g_bnkps.onnx,w600k_r50.onnx} here.
```

`inswapper_128.onnx` is distributed with the `insightface` Python package
(the model is ~530 MB and will be downloaded the first time you call
`insightface.model_zoo.get_model('inswapper_128.onnx')`). Its licence is
non-commercial; only use this app for personal/research purposes.

## Helper scripts

- `scripts/download_models.ps1` — Windows PowerShell
- `scripts/download_models.sh`  — macOS / Linux

Both shell out to Python and copy the models into place.
