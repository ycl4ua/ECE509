# Resumable experiment queue for the ECE 509 WaNet reproduction.
#
# The queue waits for the currently running CIFAR-10 seed 2 run to finish,
# then executes the remaining required experiments in priority order.  A run
# is skipped when its results.json already exists, so the script can be
# restarted at any time without redoing completed work.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_experiment_queue.ps1

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root 'src'
$py = 'C:\workplace\_gpu_env\Scripts\python.exe'
$data = Join-Path $root 'data'
$results = Join-Path $root 'results'
$figures = Join-Path $root 'figures'
New-Item -ItemType Directory -Force -Path $results, $figures | Out-Null
$statusLog = Join-Path $results 'queue_log.txt'

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    Write-Output $line
    Add-Content -Path $statusLog -Value $line -Encoding utf8
}

function Invoke-Python([string[]]$argList, [string]$stdout, [string]$stderr) {
    $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $root `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    return $p.ExitCode
}

Log '=== experiment queue started ==='

# ---------------------------------------------------------------------------
# 0. Wait for the CIFAR-10 seed 2 run that is already in progress.
# ---------------------------------------------------------------------------
$seed2Dir = Join-Path $results 'cifar10_preact18_all2one_seed2'
$seed2Res = Join-Path $seed2Dir 'results.json'
if (-not (Test-Path $seed2Res)) {
    Log 'Waiting for CIFAR-10 seed 2 to finish...'
    while (-not (Test-Path $seed2Res)) {
        $proc = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'wanet_repro\.train' -and $_.CommandLine -match '--seed 2(\s|$)' })
        if ($proc.Count -eq 0) {
            Start-Sleep -Seconds 60
            if (-not (Test-Path $seed2Res)) {
                Log 'CIFAR-10 seed 2 is not running and has no results.json; continuing with the queue.'
                break
            }
        }
        Start-Sleep -Seconds 120
    }
}
if (Test-Path $seed2Res) {
    $s2 = Get-Content $seed2Res -Raw | ConvertFrom-Json
    Log ("seed 2 finished: clean={0} attack={1} noise={2} best_epoch={3}" -f $s2.best_clean_acc, $s2.best_attack_acc, $s2.best_noise_acc, $s2.best_epoch)
}

# ---------------------------------------------------------------------------
# 1. Required experiments, in priority order.
# ---------------------------------------------------------------------------
$runs = @(
    # Complete the 3-seed main table for GTSRB.
    @{Name='gtsrb_preact18_all2one_seed1'; Dataset='gtsrb'; Epochs=200; Seed=1; Model='preact18'; Eval=$true},
    @{Name='gtsrb_preact18_all2one_seed2'; Dataset='gtsrb'; Epochs=200; Seed=2; Model='preact18'; Eval=$true},

    # All-to-all claim (C5).
    @{Name='cifar10_preact18_all2all_seed0'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; AttackMode='all2all'; Eval=$true},
    @{Name='mnist_mnist_all2all_seed0'; Dataset='mnist'; Epochs=100; Seed=0; Model='mnist'; AttackMode='all2all'; Eval=$true},

    # V1: data-only attacker (precomputed poison) and unknown victim architectures.
    @{Name='cifar10_preact18_all2one_seed0_v1_precomputed'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; PoisonMode='precomputed'; Tag='v1_precomputed'; Eval=$true},
    @{Name='cifar10_resnet18_all2one_seed0_v1_precomputed'; Dataset='cifar10'; Epochs=200; Seed=0; Model='resnet18'; PoisonMode='precomputed'; Tag='v1_precomputed'; Eval=$true},
    @{Name='cifar10_mobilenetv2_all2one_seed0_v1_precomputed'; Dataset='cifar10'; Epochs=200; Seed=0; Model='mobilenetv2'; PoisonMode='precomputed'; Tag='v1_precomputed'; Eval=$true},
    @{Name='cifar10_vgg16_all2one_seed0_v1_precomputed'; Dataset='cifar10'; Epochs=200; Seed=0; Model='vgg16'; PoisonMode='precomputed'; Tag='v1_precomputed'; Eval=$true},

    # V2: tighter poisoning / warping budgets.
    @{Name='cifar10_preact18_all2one_seed0_v2_rho0.05'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; RhoA=0.05; Tag='v2_rho0.05'; Eval=$true},
    @{Name='cifar10_preact18_all2one_seed0_v2_rho0.02'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; RhoA=0.02; Tag='v2_rho0.02'; Eval=$true},
    @{Name='cifar10_preact18_all2one_seed0_v2_rho0.01'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; RhoA=0.01; Tag='v2_rho0.01'; Eval=$true},
    @{Name='cifar10_preact18_all2one_seed0_v2_s0.35'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; S=0.35; Tag='v2_s0.35'; Eval=$true},
    @{Name='cifar10_preact18_all2one_seed0_v2_s0.25'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; S=0.25; Tag='v2_s0.25'; Eval=$true},

    # New dataset promised in the proposal.
    @{Name='fashionmnist_mnist_all2one_seed0'; Dataset='fashionmnist'; Epochs=100; Seed=0; Model='mnist'; Eval=$true},

    # Same-pipeline clean baseline (last; optional if time runs out).
    @{Name='cifar10_preact18_all2one_seed0_clean'; Dataset='cifar10'; Epochs=200; Seed=0; Model='preact18'; RhoA=0.0; RhoN=0.0; Tag='clean'; Eval=$false}
)

foreach ($spec in $runs) {
    $name = $spec.Name
    $runDir = Join-Path $results $name
    $resPath = Join-Path $runDir 'results.json'
    if (Test-Path $resPath) {
        Log ("SKIP {0} (results.json already exists)" -f $name)
        continue
    }

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $batchSize = 128
    if ($spec.ContainsKey('BatchSize')) { $batchSize = [int]$spec.BatchSize }

    $trainArgs = @(
        '-u', '-m', 'wanet_repro.train',
        '--dataset', [string]$spec.Dataset,
        '--epochs', [string]$spec.Epochs,
        '--batch_size', [string]$batchSize,
        '--seed', [string]$spec.Seed,
        '--data_root', $data,
        '--out_dir', $results,
        '--device', 'cuda',
        '--num_workers', '0'
    )
    if ($spec.ContainsKey('Model'))       { $trainArgs += @('--model', [string]$spec.Model) }
    if ($spec.ContainsKey('AttackMode'))  { $trainArgs += @('--attack_mode', [string]$spec.AttackMode) }
    if ($spec.ContainsKey('RhoA'))        { $trainArgs += @('--rho_a', [string]$spec.RhoA) }
    if ($spec.ContainsKey('RhoN'))        { $trainArgs += @('--rho_n', [string]$spec.RhoN) }
    if ($spec.ContainsKey('S'))           { $trainArgs += @('--s', [string]$spec.S) }
    if ($spec.ContainsKey('PoisonMode'))  { $trainArgs += @('--poison_mode', [string]$spec.PoisonMode) }
    if ($spec.ContainsKey('Tag'))         { $trainArgs += @('--tag', [string]$spec.Tag) }

    Log ("START {0} ({1} epochs, seed {2}, batch {3})" -f $name, $spec.Epochs, $spec.Seed, $batchSize)
    $trainLog = Join-Path $runDir 'train.log'
    $trainErr = Join-Path $runDir 'train.err.log'
    $code = Invoke-Python -argList $trainArgs -stdout $trainLog -stderr $trainErr

    if (-not (Test-Path $resPath)) {
        Log ("FAIL {0} (exit {1}); see {2}" -f $name, $code, $trainErr)
        continue
    }

    $p = Get-Content $resPath -Raw | ConvertFrom-Json
    Log ("DONE {0}: clean={1} attack={2} noise={3} best_epoch={4}" -f $name, $p.best_clean_acc, $p.best_attack_acc, $p.best_noise_acc, $p.best_epoch)

    if ($spec.Eval) {
        $evalArgs = @(
            '-u', '-m', 'wanet_repro.eval',
            '--checkpoint', (Join-Path $runDir 'checkpoint.pth.tar'),
            '--dataset', [string]$spec.Dataset,
            '--data_root', $data,
            '--device', 'cuda',
            '--out', (Join-Path $runDir 'eval.json')
        )
        if ($spec.ContainsKey('Model'))      { $evalArgs += @('--model', [string]$spec.Model) }
        if ($spec.ContainsKey('AttackMode')) { $evalArgs += @('--attack_mode', [string]$spec.AttackMode) }
        if ($spec.ContainsKey('S'))          { $evalArgs += @('--s', [string]$spec.S) }
        $evalCode = Invoke-Python -argList $evalArgs -stdout (Join-Path $runDir 'eval.log') -stderr (Join-Path $runDir 'eval.err.log')
        if ($evalCode -ne 0) { Log ("EVAL_FAIL {0} (exit {1})" -f $name, $evalCode) }
    }

    $plotArgs = @(
        '-m', 'wanet_repro.analysis.plots', 'training',
        '--input', $resPath,
        '--out', (Join-Path $figures ($name + '_curve.png'))
    )
    $plotCode = Invoke-Python -argList $plotArgs -stdout (Join-Path $runDir 'plot.log') -stderr (Join-Path $runDir 'plot.err.log')
    if ($plotCode -ne 0) { Log ("PLOT_FAIL {0} (exit {1})" -f $name, $plotCode) }
}

Log '=== experiment queue complete ==='
$done = 0
$failed = 0
foreach ($spec in $runs) {
    $resPath = Join-Path (Join-Path $results $spec.Name) 'results.json'
    if (Test-Path $resPath) { $done++ } else { $failed++ }
}
Log ("Summary: {0} runs have results.json, {1} missing" -f $done, $failed)
