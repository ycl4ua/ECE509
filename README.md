# ECE 509 — Reproducing WaNet

Track A (Reproduction) of **WaNet — Imperceptible Warping-based Backdoor Attack**
(Tuan Anh Nguyen and Anh Tuan Tran, ICLR 2021).

**Team:** Yicheng Leng, Zhuo Wang

**Repository:** https://github.com/ycl4ua/ECE509

## Status (2026-11-16)

- [x] One-page proposal (`proposal/proposal.pdf`, English)
- [x] Environment: Python 3.11.9, torch 2.2.2+cu121, torchvision 0.17.2,
      kornia 0.7.2, numpy 1.26.4; CUDA works on the RTX 4070 Laptop GPU.
- [x] From-scratch implementation of the warping field, poisoning pipeline,
      noise mode, training loop, and evaluation.
- [x] Official pretrained checkpoints evaluated **with our own code**:
      MNIST clean 99.52 / attack 99.86 / noise 98.4–98.5 (paper: 99.52 / 99.86 / 98.20);
      CIFAR-10 clean 94.14 / attack 99.55 / noise 92.1–92.3 (paper: 94.15 / 99.55 / 93.55);
      GTSRB clean 98.99 / attack 98.96 / noise 99.03
      (paper: 98.97 / 98.78 / 98.01; README: 98.87 / 99.33 / 98.01).
- [x] Official CelebA checkpoint: clean 79.01 / attack 99.04 / noise 77.60
      (paper: 78.99 / 99.33 / 76.74), evaluated on the FLWRLabs parquet mirror
      of the official 19,962-image test split.
- [x] Neural Cleanse (MNIST, official checkpoint): anomaly index 1.99 at
      20 epochs / 1000 samples (threshold 2, consistent with the paper);
      37.4 at 5 epochs / 500 samples, showing the defense budget matters.
- [x] STRIP (MNIST): clean entropy 0.338 ± 0.119 vs backdoor 0.358 ± 0.125;
      the distributions overlap, consistent with the paper's "STRIP fails".
- [x] Fine-Pruning (MNIST): clean accuracy collapses from 99.52 to 9.89 while
      attack accuracy stays ~99.98 at 63/64 pruned filters; pruning cannot
      remove the backdoor without destroying utility.
- [x] Main training runs: CIFAR-10 seeds 0–2 (200 epochs) mean 94.37±0.16 /
      99.63±0.19 / 92.18±0.08; MNIST seeds 0–2 (100 epochs) mean 99.24±0.03 /
      99.04±0.23 / 98.76±0.17; GTSRB seeds 0–2 (200 epochs) mean 98.80±0.28 /
      99.75±0.01 / 98.47±0.23.
- [x] CIFAR-10 Neural Cleanse on our checkpoint: Anomaly Index 2.40
      (marginally detected).
- [x] No-noise ablation: clean 94.53 / attack 99.73 / noise 75.24; noise
      accuracy drops 17 pp vs. the main model, confirming the mechanism.
- [x] Neural Cleanse comparisons: official CIFAR-10 0.89 (passes), no-noise
      5.14 (detected), our main model 2.40 (marginal). Confirms noise mode is
      necessary to evade NC.
- [x] All-to-all: CIFAR-10 94.39 / 93.40 / 92.12; MNIST 99.30 / 98.73 / 98.31;
      GTSRB 98.73 / 98.05 / 98.73.
- [x] V1 data-only / architecture transfer: ASR 97.01 (PreActResNet18),
      96.16 (ResNet18), 90.02 (VGG16), 62.39 (MobileNetV2).
- [x] V2 tighter budgets: ASR 98.56 at rho_a=0.05, 96.23 at 0.02, 84.57 at
      0.01; 97.53 at s=0.35, 85.45 at s=0.25.
- [x] Fashion-MNIST: clean 91.80 / attack 77.32 / noise 90.44.
- [x] Clean baseline (CIFAR-10): clean 94.56.
- [x] Test-time preprocessing robustness (CIFAR-10): JPEG-50 drops the official
      ASR from 99.48 to 14.98; resizing to 24 and back drops it to 72.52.
- [x] Pixel-wise cross-check against the official warping/poisoning code:
      max abs diff 0.0 on triggered and poisoned images; 0 / 256 label
      mismatches.
- [x] Final report (`report/report.pdf`, compiled from `report/report.tex`).
- [ ] Demo and presentation.

## Repository layout

```
ECE509_WaNet/
  proposal/            one-page proposal (Markdown + PDF)
  src/wanet_repro/     our independent re-implementation
    warp.py            warping-field generation and grid_sample
    poison.py          on-the-fly and fixed-dataset poisoning
    data.py            datasets, transforms, post-warp augmentation
    models.py          NetC_MNIST, PreActResNet18, victim models
    train.py           training entry point
    eval.py            evaluation entry point
  configs/             experiment configuration files (to be added)
  scripts/             PowerShell entry points
  docs/
    assumptions.md     paper vs. official-code discrepancies and choices
    claims.md          Track A claims table and reproduction status
    results.md         consolidated results log
    compute.md         hardware, timings, and estimates
  report/
    report.tex         IEEEtran final-report skeleton
    references.bib     bibliography
  results/             checkpoints, metrics, and JSON outputs
  figures/             generated figures
  third_party/
    wanet_official/    official implementation (AGPL-3.0), reference only
  checkpoints_official/  released paper checkpoints (downloaded data)
```

## Quick start

```powershell
$env:PYTHONPATH = "C:\workplace\ECE509_WaNet\src"
$python = "C:\workplace\_gpu_env\Scripts\python.exe"

# Evaluate a released checkpoint with our code
& $python -m wanet_repro.eval `
  --checkpoint C:\workplace\ECE509_WaNet\checkpoints_official\checkpoints\cifar10_all2one_morph.pth.tar `
  --dataset cifar10 --data_root C:\workplace\ECE509_WaNet\data --device cuda

# Train a reduced-epoch reproduction run
& $python -m wanet_repro.train --dataset cifar10 --epochs 200 `
  --data_root C:\workplace\ECE509_WaNet\data --out_dir C:\workplace\ECE509_WaNet\results `
  --device cuda --num_workers 0
```

The same commands are wrapped in `scripts/train_mnist.ps1`,
`scripts/train_cifar10.ps1`, and `scripts/eval_official.ps1`.

## Reproduce

- Quick (released checkpoints + defenses + figures):
  `.\scripts\reproduce.ps1 -Mode Quick`
- Full (train the main and ablation runs, then evaluate):
  `.\scripts\reproduce.ps1 -Mode Full`
- Individual entry points: `scripts/train_mnist.ps1`,
  `scripts/train_cifar10.ps1`, `scripts/train_all2all.ps1`,
  `scripts/train_no_noise.ps1`, `scripts/train_clean.ps1`,
  `scripts/train_v1_precomputed.ps1`, `scripts/train_v2_budget.ps1`,
  `scripts/run_defenses.ps1`.

The Full mode takes many GPU-hours; see `docs/compute.md` for the estimate.

## Notes

- `third_party/wanet_official` is the official AGPL-3.0 implementation. It is
  kept only as a reference for cross-checking; the artifact is
  `src/wanet_repro`, written independently from the paper description.
- The released checkpoints are downloaded data from the paper's Google Drive
  link and are not redistributed in this repository.
- The official code pins `torch==1.5.0`, which cannot run on the RTX 4070
  (Ada, compute capability 8.9). Our implementation targets
  `torch 2.2.2+cu121`; see `docs/assumptions.md` for every compatibility
  decision and paper/code discrepancy.
- Measured training speed: MNIST ~24 s/epoch, CIFAR-10 ~44 s/epoch on the
  RTX 4070 Laptop (8 GB). A 200-epoch CIFAR-10 run is ~2.4 h; 3 seeds ~7 h.
  See `docs/compute.md` for the full estimate.
