#!/usr/bin/env bash
# Fetch the ONNX models required by facenaissance into ./models/.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
models="$root/models"
mkdir -p "$models"

echo "Using Python to fetch InsightFace buffalo_l bundle (scrfd + arcface)..."
python - <<PY
import insightface, shutil, os, pathlib
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)
src = pathlib.Path(os.path.expanduser('~/.insightface/models/buffalo_l'))
dst = pathlib.Path("$models")
for name in ['scrfd_2.5g_bnkps.onnx', 'w600k_r50.onnx']:
    shutil.copy(src / name, dst / name)
    print('  ->', dst / name)
PY

echo "Fetching inswapper_128.onnx (~530 MB)..."
python - <<PY
import insightface, shutil, pathlib
m = insightface.model_zoo.get_model('inswapper_128.onnx')
src = pathlib.Path(m.model_file)
dst = pathlib.Path("$models") / 'inswapper_128.onnx'
shutil.copy(src, dst)
print('  ->', dst)
PY

echo
echo "All models downloaded. You can now run: cargo run --release"
