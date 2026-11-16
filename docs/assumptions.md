# Assumptions, discrepancies, and reproduction decisions

This document records every detail that the paper leaves unspecified or that
differs between the paper and the official implementation. Track A requires
documenting guessed details, so this file is part of the artifact.

## 1. Noise mode: Gaussian vs uniform

- **Paper (Sec. 3.3):** the noise-mode warping field is formed by "adding
  Gaussian noise to M".
- **Official code:** `ins = torch.rand(..., 2) * 2 - 1`, i.e. uniform noise in
  [-1, 1], divided by the image height.
- **Our choice:** `--noise_type uniform` by default (matches the official code
  and the released checkpoints); `--noise_type gaussian` is available for the
  paper-text variant.
- **Impact:** noise-mode accuracy is random regardless; the choice may affect
  stealth/defense results. We will report both if time permits.

## 2. Training length

- **Paper:** "trained until convergence"; no epoch count is given.
- **Official code:** `--n_iters 1000` (1000 epochs), SGD lr 0.01 with a 0.1
  decay every 100 epochs, milestones [100, 200, 300, 400].
- **Our choice:** the released checkpoints are used to reproduce the exact
  paper numbers. For our own 3-seed runs we use a reduced schedule
  (200 epochs by default, same milestones) and report the gap. If time allows,
  we will run 1000 epochs for the main CIFAR-10 configuration.
- **Impact:** reduced-epoch models may have lower clean accuracy and attack
  success; the gap is reported explicitly.

## 3. Poisoning semantics

- **Paper (Sec. 3.1):** a subset of the training set of ratio rho_a is
  *replaced* by backdoor pairs (B(x), c(y)).
- **Official code:** each batch is split on the fly: the first `int(bs*rho_a)`
  samples are warped and relabeled, the next `int(num_bd * rho_n/rho_a)`
  samples are noise-warped and keep their labels, and the rest are clean.
  The same sample can therefore be clean in one epoch and poisoned in another.
- **Our choice:** `--poison_mode online` matches the official code (used for
  the main reproduction); `--poison_mode precomputed` builds a fixed poisoned
  dataset and is used for the data-only attacker variation (V1).

## 4. Warping-field details

- Control points: uniform random in [-1, 1], shape (1, 2, k, k).
- Normalization: divide by the mean absolute value (paper Eq. 4); matches the
  official code.
- Upsampling: bicubic to the image size (paper Eq. 5); matches the official
  code.
- Grid combination: `(identity_grid + s * field / input_height) * grid_rescale`,
  clipped to [-1, 1]; matches the official code.
- Coordinate order: `grid_sample` expects (x, y) in the last dimension. The
  official code stacks `(y, x)` from a default `meshgrid`; our implementation
  uses `indexing="ij"` and stacks `(xx, yy)`, which is equivalent.
- `k = 4`, `s = 0.5`, `rho_a = 0.1`, `rho_n = 0.2`, target label 0 (all-to-one).

## 5. Data augmentation

- **Paper:** random crop and random rotation (plus random horizontal flip on
  CIFAR-10) during training; none at evaluation.
- **Official code:** augmentation is applied *after* the trigger is injected,
  using kornia (`RandomCrop` p=0.8, `RandomRotation` p=0.5,
  `RandomHorizontalFlip` p=0.5 on CIFAR-10).
- **Our choice:** same post-warp augmentation via kornia 0.7.2. For the
  data-only variation, the victim applies its standard augmentation to the
  fixed poisoned dataset, which can partially disrupt the warp.

## 6. Architectures

- MNIST: `NetC_MNIST` (3 conv blocks + 2 FC; paper Table 2 and official code).
- CIFAR-10 / GTSRB: `PreActResNet18`.
- CelebA: `ResNet18` (official `classifier_models.ResNet18`, not torchvision's
  ImageNet stem). Not yet implemented in our code; CelebA is a later stretch.
- Our MNIST and PreActResNet18 state-dict keys match the official checkpoints;
  this is verified by loading the released MNIST and CIFAR-10 checkpoints and
  reproducing clean/attack accuracy exactly.

## 7. Paper vs README numbers

- GTSRB: paper reports clean 98.97 / attack 98.78; the README reports
  98.87 / 99.33. We will report both and our own evaluation of the released
  checkpoint.
- CelebA: the paper text says 79.77% clean, while the table says 78.99%.
- Noise-mode accuracy is random and varies between runs; the README notes this.
  We will report mean and std over repeated evaluations where possible.

## 8. Environment and compatibility

- The official `requirements.txt` pins `torch==1.5.0`, `torchvision==0.6.0`,
  `numpy==1.16.6`, etc. `torch==1.5.0` cannot run on the RTX 4070 Laptop
  (Ada, compute capability 8.9; CUDA 10.x does not support it).
- We use Python 3.11.9, `torch==2.2.2+cu121`, `torchvision==0.17.2+cu121`,
  `numpy==1.26.4`, and `kornia==0.7.2`.
- The official code imports `kornia` but does not list it in
  `requirements.txt`; we install it explicitly.
- `utils/utils.py` calls `stty size`, which fails on Windows; our copy of the
  official code is patched to fall back to `shutil.get_terminal_size()`.
- `eval.py` in the official code references undefined `best_clean_acc` /
  `best_bd_acc` when `cross_ratio == 0`; patched in our reference copy.
- The official repo's install command says `requirements.py` while the file is
  `requirements.txt`; documented as a reproducibility issue.

## 9. Checkpoint compatibility

The released checkpoints are dictionaries with `netC`, `identity_grid`,
`noise_grid`, `best_clean_acc`, `best_bd_acc`, and `epoch_current`. Our
evaluation code loads them directly and uses the stored grids. Verified:

| Dataset | Clean | Attack | Noise | Paper (clean/attack/noise) |
|---|---:|---:|---:|---|
| MNIST | 99.52 | 99.86 | 98.4–98.5 | 99.52 / 99.86 / 98.20 |
| CIFAR-10 | 94.14 | 99.55 | 92.1–92.3 | 94.15 / 99.55 / 93.55 |
| GTSRB | 98.99 | 98.96 | 99.03 | 98.97 / 98.78 / 98.01 (paper); 98.87 / 99.33 / 98.01 (README) |

## 10. Neural Cleanse budget sensitivity

The paper reports that WaNet's Anomaly Index is below the detection threshold
of 2 on all datasets. Our re-implementation reproduces this qualitatively, but
the result is sensitive to the scan budget:

| Scan budget | Anomaly Index (MNIST official checkpoint) |
|---|---:|
| 5 epochs, 500 samples | 37.4 (detected) |
| 20 epochs, 1000 samples | 1.99 (passes, just below threshold) |

The official code uses 20 epochs and 10 repeated scans over the full test set;
our reduced scan is cheaper. We will report 3 runs at a 20-epoch budget and
treat the 5-epoch result as evidence that the defense's conclusion depends on
how much optimization it is allowed. This is a useful critique point: the
"passes Neural Cleanse" claim is not budget-invariant.

## 11. Defense implementation details and results (MNIST official checkpoint)

### Neural Cleanse
- Re-implemented from the paper/official reference; the pattern is mapped into
  the normalized input space before blending (the official `Normalize` step).
- Reduced scan: 20 epochs, 1000 test samples, 3 runs planned (the official
  code uses 20 epochs and 10 scans over the full test set).
- Result: Anomaly Index **1.99** at 20 epochs / 1000 samples (threshold 2,
  passes), **37.4** at 5 epochs / 500 samples (detected). The conclusion is
  budget-sensitive; report both.

### STRIP
- Superimpose clean images in [0, 1] space and then normalize, matching the
  official code; use softmax entropy (the official code uses sigmoid on logits,
  which is not a proper multi-class entropy).
- Result (MNIST): clean entropy 0.338 ± 0.119; backdoor entropy
  0.358 ± 0.125; distributions overlap, so STRIP does not separate WaNet
  backdoor inputs from clean inputs. This is consistent with the paper.

### Fine-Pruning
- The official code is pruning-only (no fine-tuning after pruning); our
  implementation follows that and samples pruning levels instead of sweeping
  every filter. MNIST: prune filters of `layer3.conv1` by mean activation and
  adjust `linear6`.
- Result (MNIST): clean/attack accuracy at 0, 35, 42, 56, 63 pruned filters:
  (99.52, 99.86), (96.90, 99.84), (89.89, 99.59), (35.42, 100.00),
  (9.89, 99.98). Clean utility collapses before the attack does, so pruning
  cannot mitigate the backdoor. Consistent with the paper.

## 12. Not reproduced (optional)

- Human inspection study (40 participants, 25 GTSRB images).
- Physical screen-recapture attack (98% clean / 96% ASR).
- Spectral-signature defense, GradCAM analysis, and the 8-warping-field
  stability study.
- All-to-all on CelebA (no released all-to-all checkpoint; training it was
  outside the project scope).

## 13. CelebA data source

The torchvision Google Drive download is rate-limited ("too many users have
viewed or downloaded this file"). We evaluated the released CelebA checkpoint
on the FLWRLabs parquet mirror of the official test split: 19,962 images, the
same 40 attributes, and the paper's three balanced attributes (Heavy_Makeup,
Smiling, Mouth_Slightly_Open). Images are resized to 64x64, matching the
paper. Result: clean 79.01 / attack 99.04 / noise 77.60 (paper: 78.99 / 99.33
/ 76.74).

## 14. Test-time preprocessing and cross-check

The proposal's robustness check applies random crop, 0.75x resampling, and
JPEG-50 after the trigger. On the official CIFAR-10 checkpoint, JPEG-50 drops
ASR from 99.48 to 14.98 and 0.75x resampling drops it to 72.52; random
cropping is mild (98.31). The reduced-epoch model is more fragile. The
pixel-wise cross-check against the official evaluation/training formulas gives
zero difference on triggered images, the poisoning split, and seed-matched
noise-mode images, and zero label mismatches.
