param(
    [double]$RhoA = 0.05,
    [double]$S = 0.5,
    [int]$Seed = 0
)

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"
$tag = "v2_rho${RhoA}_s${S}"

& $python -m wanet_repro.train `
  --dataset cifar10 `
  --epochs 200 `
  --batch_size 128 `
  --rho_a $RhoA `
  --s $S `
  --seed $Seed `
  --tag $tag `
  --data_root (Join-Path $root "data") `
  --out_dir (Join-Path $root "results") `
  --device cuda `
  --num_workers 0
