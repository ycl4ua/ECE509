$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"

& $python -m wanet_repro.train `
  --dataset cifar10 `
  --epochs 200 `
  --batch_size 128 `
  --data_root (Join-Path $root "data") `
  --out_dir (Join-Path $root "results") `
  --device cuda `
  --num_workers 0
