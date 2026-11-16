param(
    [string]$Checkpoint,
    [string]$Dataset = "cifar10",
    [string]$Prefix = "official_cifar10",
    [int]$NcRuns = 3,
    [int]$NcSamples = 1000,
    [int]$NcEpochs = 20
)

$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"
$data = Join-Path $root "data"
$results = Join-Path $root "results"

& $python -m wanet_repro.defenses.neural_cleanse `
  --checkpoint $Checkpoint --dataset $Dataset --data_root $data --device cuda `
  --n_samples $NcSamples --epochs $NcEpochs --runs $NcRuns `
  --out (Join-Path $results "${Prefix}_nc.json")

& $python -m wanet_repro.defenses.strip `
  --checkpoint $Checkpoint --dataset $Dataset --data_root $data --device cuda `
  --n_samples 100 --n_superimpose 50 `
  --out (Join-Path $results "${Prefix}_strip.json")

& $python -m wanet_repro.defenses.fine_pruning `
  --checkpoint $Checkpoint --dataset $Dataset --data_root $data --device cuda `
  --n_levels 9 `
  --out (Join-Path $results "${Prefix}_fp.json")
