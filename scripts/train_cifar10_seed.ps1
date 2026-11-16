param(
    [int]$Seed = 0,
    [int]$Epochs = 200
)

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"

& $python -m wanet_repro.train `
  --dataset cifar10 `
  --epochs $Epochs `
  --batch_size 128 `
  --seed $Seed `
  --data_root (Join-Path $root "data") `
  --out_dir (Join-Path $root "results") `
  --device cuda `
  --num_workers 0
