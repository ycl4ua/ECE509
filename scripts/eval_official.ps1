param([string]$Dataset = "mnist")

if ($Dataset -notin @("mnist", "cifar10")) {
    Write-Error "Only mnist and cifar10 are wired up so far (GTSRB needs its dataset loader)."
    exit 1
}

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"
$checkpoint = Join-Path $root "checkpoints_official\checkpoints\${Dataset}_all2one_morph.pth.tar"
$out = Join-Path $root "results\official_${Dataset}_eval.json"

& $python -m wanet_repro.eval `
  --checkpoint $checkpoint `
  --dataset $Dataset `
  --data_root (Join-Path $root "data") `
  --device cuda `
  --out $out
