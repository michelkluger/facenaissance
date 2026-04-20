# Fetch the ONNX models required by facenaissance into ./models/.
# Run from the project root:   powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1

$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$models = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $models | Out-Null

Write-Host "Using Python to fetch InsightFace buffalo_l bundle (scrfd + arcface)..."
python -c @"
import insightface, shutil, os, pathlib
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)
src = pathlib.Path(os.path.expanduser('~/.insightface/models/buffalo_l'))
dst = pathlib.Path(r'$models')
for name in ['scrfd_2.5g_bnkps.onnx', 'w600k_r50.onnx']:
    shutil.copy(src / name, dst / name)
    print('  ->', dst / name)
"@

Write-Host "Fetching inswapper_128.onnx (this is ~530 MB and may take a while)..."
python -c @"
import insightface, shutil, pathlib, os
m = insightface.model_zoo.get_model('inswapper_128.onnx')
src = pathlib.Path(m.model_file)
dst = pathlib.Path(r'$models') / 'inswapper_128.onnx'
shutil.copy(src, dst)
print('  ->', dst)
"@

Write-Host "`nAll models downloaded. You can now run: cargo run --release"
