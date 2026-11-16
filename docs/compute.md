# Compute notes

## Hardware

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM, compute capability 8.9,
  driver 555.97 (CUDA 12.5). The laptop power limit is modest; expect slower
  epochs than a desktop 4070.
- CPU: 13th Gen Intel Core i9-13900H, 14 cores / 20 threads.
- RAM: 31.6 GB.
- Disk: C: 41 GB free (datasets and checkpoints are small; see below).
- Note: the desktop currently uses ~5 GB of VRAM (browser, chat, wallpaper
  engine). Close heavy applications before long runs.

## Software

- Python 3.11.9 (`C:\workplace\_gpu_env`)
- torch 2.2.2+cu121, torchvision 0.17.2+cu121, kornia 0.7.2, numpy 1.26.4
- CUDA available: yes; device recognized as RTX 4070 Laptop GPU.

## Measured timings (our implementation)

| Dataset | Model | Batch | Epochs measured | Time per epoch | Notes |
|---|---|---:|---:|---:|---|
| MNIST | NetC_MNIST | 128 | 2 | 24.9 s / 23.6 s | includes evaluation |
| CIFAR-10 | PreActResNet18 | 128 | 1 | 43.8 s | includes evaluation |

## Estimates

| Experiment | Estimate |
|---|---:|
| MNIST, 100 epochs | ~40 min |
| MNIST, 1000 epochs | ~6.7 h (converges earlier) |
| CIFAR-10, 200 epochs | ~2.4 h/run |
| CIFAR-10, 200 epochs x 3 seeds | ~7.3 h |
| CIFAR-10, 1000 epochs | ~12.2 h/run |
| CIFAR-10, 1000 epochs x 3 seeds | ~36.5 h |
| GTSRB, 200 epochs | ~2 h/run (estimate; 39k images) |
| Neural Cleanse, CIFAR-10, 3 runs | ~4–8 h (estimate) |
| Fine-Pruning, CIFAR-10 | ~1–2 h |
| STRIP, CIFAR-10 | < 1 h |

Core reproduction with reduced-epoch runs: roughly 20–40 GPU-hours.
Full 1000-epoch runs for all seeds: roughly 50–70 GPU-hours. Both are
feasible over one to two weeks of overnight runs; the GPU is not the
bottleneck for the course schedule.

## Storage

- MNIST: ~11 MB; CIFAR-10: ~170 MB; CIFAR-100: ~170 MB; GTSRB: ~300 MB;
  CelebA: ~1.4 GB.
- Released checkpoints: 4 files, ~90 MB each (MNIST is ~4.7 MB).
- Our checkpoints: ~90 MB per CIFAR-10/GTSRB run; keep only the best.
- Total expected: < 5 GB, well within the 41 GB free.

## Actual compute (2026-11-13)

- Main and variation queue: ~30.6 GPU-hours.
- GTSRB all-to-all: ~3.3 GPU-hours.
- Earlier main, ablation, and defense runs: ~11 GPU-hours.
- Evaluation, robustness, and cross-check scripts: ~2 GPU-hours.
- Total: roughly 47 GPU-hours on the RTX 4070 Laptop.
