param(
    [string]$Dataset = "cifar10",
    [int]$Seed = 0
)

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"

& $python -m wanet_repro.train `
  --dataset $Dataset `
  --epochs 200 `
  --batch_size 128 `
  --attack_mode all2all `
  --seed $Seed `
  --tag all2all `
  --data_root (Join-Path $root "data") `
  --out_dir (Join-Path $root "results") `
  --device cuda `
  --num_workers 0
