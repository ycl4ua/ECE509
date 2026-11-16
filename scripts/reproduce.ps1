param(
    [ValidateSet("Quick", "Full")]
    [string]$Mode = "Quick",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"
$data = Join-Path $root "data"
$results = Join-Path $root "results"
$figures = Join-Path $root "figures"
$ckpts = Join-Path $root "checkpoints_official\checkpoints"
New-Item -ItemType Directory -Force -Path $results, $figures | Out-Null

function Invoke-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

function Test-Skip {
    param([string]$Path)
    if ((Test-Path $Path) -and (-not $Force)) {
        Write-Host "Skip existing $Path"
        return $true
    }
    return $false
}

# ---------------------------------------------------------------------------
# 1. Evaluate the released checkpoints with our independent code.
# ---------------------------------------------------------------------------
foreach ($dataset in @("mnist", "cifar10", "gtsrb")) {
    $checkpoint = Join-Path $ckpts "${dataset}_all2one_morph.pth.tar"
    if (-not (Test-Path $checkpoint)) {
        Write-Warning "Missing released checkpoint: $checkpoint"
        continue
    }
    $out = Join-Path $results "official_${dataset}_eval.json"
    if (Test-Skip $out) { continue }
    Invoke-Python -Arguments @(
        "-m", "wanet_repro.eval",
        "--checkpoint", $checkpoint,
        "--dataset", $dataset,
        "--data_root", $data,
        "--device", "cuda",
        "--out", $out
    )
}

# ---------------------------------------------------------------------------
# 2. Defenses on the released MNIST and CIFAR-10 checkpoints.
# ---------------------------------------------------------------------------
foreach ($dataset in @("mnist", "cifar10")) {
    $checkpoint = Join-Path $ckpts "${dataset}_all2one_morph.pth.tar"
    if (-not (Test-Path $checkpoint)) { continue }

    $nc = Join-Path $results "official_${dataset}_nc.json"
    if (-not (Test-Skip $nc)) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.defenses.neural_cleanse",
            "--checkpoint", $checkpoint,
            "--dataset", $dataset,
            "--data_root", $data,
            "--device", "cuda",
            "--n_samples", "1000",
            "--epochs", "20",
            "--runs", "3",
            "--out", $nc
        )
    }

    $strip = Join-Path $results "official_${dataset}_strip.json"
    if (-not (Test-Skip $strip)) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.defenses.strip",
            "--checkpoint", $checkpoint,
            "--dataset", $dataset,
            "--data_root", $data,
            "--device", "cuda",
            "--n_samples", "100",
            "--n_superimpose", "50",
            "--out", $strip
        )
    }

    $fp = Join-Path $results "official_${dataset}_fp.json"
    if (-not (Test-Skip $fp)) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.defenses.fine_pruning",
            "--checkpoint", $checkpoint,
            "--dataset", $dataset,
            "--data_root", $data,
            "--device", "cuda",
            "--n_levels", "9",
            "--out", $fp
        )
    }
}

# ---------------------------------------------------------------------------
# 3. Figures.
# ---------------------------------------------------------------------------
$trigger_mnist = Join-Path $figures "trigger_mnist.png"
if (-not (Test-Skip $trigger_mnist)) {
    Invoke-Python -Arguments @(
        "-m", "wanet_repro.analysis.plots", "trigger",
        "--checkpoint", (Join-Path $ckpts "mnist_all2one_morph.pth.tar"),
        "--dataset", "mnist",
        "--data-root", $data,
        "--out", $trigger_mnist
    )
}

$trigger_cifar = Join-Path $figures "trigger_cifar10.png"
if (-not (Test-Skip $trigger_cifar)) {
    Invoke-Python -Arguments @(
        "-m", "wanet_repro.analysis.plots", "trigger",
        "--checkpoint", (Join-Path $ckpts "cifar10_all2one_morph.pth.tar"),
        "--dataset", "cifar10",
        "--data-root", $data,
        "--out", $trigger_cifar
    )
}

foreach ($dataset in @("mnist", "cifar10")) {
    $fp_json = Join-Path $results "official_${dataset}_fp.json"
    if (Test-Path $fp_json) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.plots", "fine-pruning",
            "--input", $fp_json,
            "--out", (Join-Path $figures "fine_pruning_${dataset}.png")
        )
    }
    $strip_json = Join-Path $results "official_${dataset}_strip.json"
    if (Test-Path $strip_json) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.plots", "strip",
            "--input", $strip_json,
            "--out", (Join-Path $figures "strip_${dataset}.png")
        )
    }
}

# ---------------------------------------------------------------------------
# 4. Full mode: train the main and ablation runs.
# ---------------------------------------------------------------------------
if ($Mode -eq "Full") {
    $train_runs = @(
        @("--dataset", "mnist", "--epochs", "100", "--seed", "0"),
        @("--dataset", "mnist", "--epochs", "100", "--seed", "1"),
        @("--dataset", "mnist", "--epochs", "100", "--seed", "2"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "1"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "2"),
        @("--dataset", "gtsrb", "--epochs", "200", "--seed", "0"),
        @("--dataset", "gtsrb", "--epochs", "200", "--seed", "1"),
        @("--dataset", "gtsrb", "--epochs", "200", "--seed", "2"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--rho_n", "0", "--tag", "no_noise"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--rho_a", "0", "--rho_n", "0", "--tag", "clean"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--attack_mode", "all2all"),
        @("--dataset", "mnist", "--epochs", "100", "--seed", "0", "--attack_mode", "all2all"),
        @("--dataset", "gtsrb", "--epochs", "200", "--seed", "0", "--attack_mode", "all2all"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--poison_mode", "precomputed", "--tag", "v1_precomputed"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--model", "resnet18", "--poison_mode", "precomputed", "--tag", "v1_precomputed"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--model", "mobilenetv2", "--poison_mode", "precomputed", "--tag", "v1_precomputed"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--model", "vgg16", "--poison_mode", "precomputed", "--tag", "v1_precomputed"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--rho_a", "0.05", "--tag", "v2_rho0.05"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--rho_a", "0.02", "--tag", "v2_rho0.02"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--rho_a", "0.01", "--tag", "v2_rho0.01"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--s", "0.35", "--tag", "v2_s0.35"),
        @("--dataset", "cifar10", "--epochs", "200", "--seed", "0", "--s", "0.25", "--tag", "v2_s0.25"),
        @("--dataset", "fashionmnist", "--epochs", "100", "--seed", "0")
    )
    foreach ($run in $train_runs) {
        Invoke-Python -Arguments (@(
            "-m", "wanet_repro.train",
            "--batch_size", "128",
            "--data_root", $data,
            "--out_dir", $results,
            "--device", "cuda",
            "--num_workers", "0"
        ) + $run)
    }

    # Training curves for every completed run.
    foreach ($run_dir in Get-ChildItem $results -Directory) {
        $res_path = Join-Path $run_dir.FullName "results.json"
        if (-not (Test-Path $res_path)) { continue }
        $curve_path = Join-Path $figures ($run_dir.Name + "_curve.png")
        if (Test-Path $curve_path) { continue }
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.plots", "training",
            "--input", $res_path,
            "--out", $curve_path
        )
    }

    # CelebA checkpoint evaluation (requires the parquet mirror in data/celeba_hf).
    $celeba_ckpt = Join-Path $ckpts "celeba_all2one_morph.pth.tar"
    $celeba_parquet = Join-Path $data "celeba_hf"
    if ((Test-Path $celeba_ckpt) -and (Test-Path $celeba_parquet)) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.eval_celeba_parquet",
            "--checkpoint", $celeba_ckpt,
            "--parquet_dir", $celeba_parquet,
            "--device", "cuda",
            "--out", (Join-Path $results "official_celeba_eval.json")
        )
    } else {
        Write-Warning "CelebA checkpoint or parquet shards missing; skipping CelebA evaluation."
    }

    # Test-time preprocessing and pixel-wise cross-check.
    $official_cifar = Join-Path $ckpts "cifar10_all2one_morph.pth.tar"
    if (Test-Path $official_cifar) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.robustness",
            "--checkpoint", $official_cifar,
            "--dataset", "cifar10",
            "--data_root", $data,
            "--device", "cuda",
            "--n_samples", "5000",
            "--out", (Join-Path $results "robustness_official_cifar10.json")
        )
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.crosscheck",
            "--checkpoint", $official_cifar,
            "--dataset", "cifar10",
            "--data_root", $data,
            "--device", "cuda",
            "--out", (Join-Path $results "crosscheck_official_cifar10.json")
        )
    }
    $our_cifar = Join-Path (Join-Path $results "cifar10_preact18_all2one_seed0") "checkpoint.pth.tar"
    if (Test-Path $our_cifar) {
        Invoke-Python -Arguments @(
            "-m", "wanet_repro.analysis.robustness",
            "--checkpoint", $our_cifar,
            "--dataset", "cifar10",
            "--data_root", $data,
            "--device", "cuda",
            "--n_samples", "5000",
            "--out", (Join-Path $results "robustness_ours_cifar10.json")
        )
    }
}

Write-Host "Reproduction ($Mode) complete."
