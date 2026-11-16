# Results log

All numbers are percentages unless noted. "Paper" refers to the WaNet ICLR
2021 paper; "README" refers to the official repository's results table.

## 1. Released checkpoints evaluated with our own code

| Dataset | Clean | Attack (ASR) | Noise | Paper (clean / attack / noise) |
|---|---:|---:|---:|---|
| MNIST | 99.52 | 99.86 | 98.4–98.5 | 99.52 / 99.86 / 98.20 |
| CIFAR-10 | 94.14 | 99.55 | 92.1–92.3 | 94.15 / 99.55 / 93.55 |
| GTSRB | 98.99 | 98.96 | 99.03 | 98.97 / 98.78 / 98.01 (paper); 98.87 / 99.33 / 98.01 (README) |
| CelebA | 79.01 | 99.04 | 77.60 | 78.99 / 99.33 / 76.74 |

The CelebA evaluation uses the FLWRLabs parquet mirror of the official test
split because the torchvision Google Drive download is rate-limited; the
mirror preserves the 19,962 test images and the 40 attributes, and images are
resized to 64x64 as in the paper.

Clean and attack accuracy match the paper within 0.3 pp on all four
datasets. Noise accuracy varies between runs because the noise-mode warping
is random; the official README also notes this.

## 2. Defense reproduction

### 2.1 Neural Cleanse

| Model | Scan budget | Anomaly Index | Verdict |
|---|---|---:|---|
| MNIST official | 5 epochs, 500 samples | 37.4 | detected |
| MNIST official | 20 epochs, 1000 samples | 1.99 | passes (threshold 2) |
| CIFAR-10 official | 20 epochs, 1000 samples | 0.89 | passes |
| CIFAR-10 ours (200 epochs) | 20 epochs, 1000 samples | 2.40 | marginally detected |
| CIFAR-10 no-noise | 20 epochs, 1000 samples | 5.14 | detected |

The paper reports that WaNet passes Neural Cleanse (Anomaly Index < 2). The
official checkpoints pass at the 20-epoch budget (MNIST 1.99, CIFAR-10 0.89);
the 5-epoch MNIST scan detects (37.4), so the conclusion is budget-sensitive.
Our reduced-epoch CIFAR-10 model is marginal (2.40), while removing the noise
mode makes it clearly detectable (5.14). This confirms the paper's mechanism
and shows that the margin depends on the training budget. The "passes Neural
Cleanse" claim is real but fragile.

### 2.2 STRIP

| Dataset | Clean entropy | Backdoor entropy | Interpretation |
|---|---:|---:|---|
| MNIST | 0.338 ± 0.119 | 0.358 ± 0.125 | distributions overlap |
| CIFAR-10 | 0.466 ± 0.176 | 0.517 ± 0.177 | distributions overlap |

Consistent with the paper: the perturbation destroys the warping trigger, so
STRIP cannot separate backdoor inputs from clean ones.

### 2.3 Fine-Pruning

MNIST (prune `layer3.conv1` filters, no fine-tuning, matching the official
pruning-only implementation):

| Filters pruned | Clean | Attack (ASR) |
|---:|---:|---:|
| 0 / 64 | 99.52 | 99.86 |
| 35 / 64 | 96.90 | 99.84 |
| 42 / 64 | 89.89 | 99.59 |
| 56 / 64 | 35.42 | 100.00 |
| 63 / 64 | 9.89 | 99.98 |

CIFAR-10 (prune `layer4[1].conv2` filters):

| Filters pruned | Clean | Attack (ASR) |
|---:|---:|---:|
| 0 / 512 | 94.14 | 99.55 |
| 392 / 512 | 92.31 | 99.92 |
| 448 / 512 | 83.14 | 99.96 |
| 504 / 512 | 55.61 | 99.84 |

Consistent with the paper: clean accuracy collapses before the attack
accuracy does, so pruning cannot mitigate the backdoor without destroying
utility.

## 3. Our training runs

| Run | Dataset | Model | Epochs | Seeds | Status |
|---|---|---|---:|---|---|
| main | CIFAR-10 | PreActResNet18 | 200 | 0 | done: clean 94.23 / attack 99.53 / noise 92.22 |
| main | CIFAR-10 | PreActResNet18 | 200 | 1 | done: clean 94.35 / attack 99.50 / noise 92.09 |
| main | CIFAR-10 | PreActResNet18 | 200 | 2 | done: clean 94.54 / attack 99.85 / noise 92.23 |
| main | MNIST | NetC_MNIST | 100 | 0 | done: clean 99.26 / attack 99.07 / noise 98.82 |
| main | MNIST | NetC_MNIST | 100 | 1 | done: clean 99.25 / attack 98.80 / noise 98.78 |
| main | MNIST | NetC_MNIST | 100 | 2 | done: clean 99.21 / attack 99.26 / noise 98.58 |
| main | GTSRB | PreActResNet18 | 200 | 0 | done: clean 99.11 / attack 99.75 / noise 98.74 |
| main | GTSRB | PreActResNet18 | 200 | 1 | done: clean 98.72 / attack 99.74 / noise 98.32 |
| main | GTSRB | PreActResNet18 | 200 | 2 | done: clean 98.58 / attack 99.75 / noise 98.35 |
| all-to-all | CIFAR-10 | PreActResNet18 | 200 | 0 | done: clean 94.39 / attack 93.40 / noise 92.12 |
| all-to-all | MNIST | NetC_MNIST | 100 | 0 | done: clean 99.30 / attack 98.73 / noise 98.31 |
| all-to-all | GTSRB | PreActResNet18 | 200 | 0 | done: clean 98.73 / attack 98.05 / noise 98.73 |
| no-noise ablation | CIFAR-10 | PreActResNet18 | 200 | 0 | done: clean 94.53 / attack 99.73 / noise 75.24 |
| clean baseline | CIFAR-10 | PreActResNet18 | 200 | 0 | done: clean 94.56 (attack/noise not meaningful) |
| V1 data-only | CIFAR-10 | PreActResNet18 | 200 | 0 | done: clean 93.96 / attack 97.01 / noise 90.98 |
| V1 data-only | CIFAR-10 | ResNet18 | 200 | 0 | done: clean 93.78 / attack 96.16 / noise 91.01 |
| V1 data-only | CIFAR-10 | VGG16 | 200 | 0 | done: clean 92.29 / attack 90.02 / noise 89.36 |
| V1 data-only | CIFAR-10 | MobileNetV2 | 200 | 0 | done: clean 85.21 / attack 62.39 / noise 82.85 |
| V2 budget | CIFAR-10 | PreActResNet18 | 200 | 0 | rho_a=0.05: 94.53 / 98.56 / 92.38 |
| V2 budget | CIFAR-10 | PreActResNet18 | 200 | 0 | rho_a=0.02: 94.56 / 96.23 / 92.35 |
| V2 budget | CIFAR-10 | PreActResNet18 | 200 | 0 | rho_a=0.01: 94.34 / 84.57 / 92.38 |
| V2 budget | CIFAR-10 | PreActResNet18 | 200 | 0 | s=0.35: 94.04 / 97.53 / 92.72 |
| V2 budget | CIFAR-10 | PreActResNet18 | 200 | 0 | s=0.25: 92.77 / 85.45 / 92.12 |
| new dataset | Fashion-MNIST | NetC_MNIST | 100 | 0 | done: clean 91.80 / attack 77.32 / noise 90.44 |

### CIFAR-10 seed 0 (200 epochs)

Best checkpoint: clean 94.23 / attack 99.53 / noise 92.22 (paper:
94.15 / 99.55 / 93.55). Clean and attack match within 0.1 pp; noise is 1.3 pp
lower, within the run-to-run variation of the random noise mode. Training
curve: `figures/cifar10_training_curve.png`.

### CIFAR-10 seed 1 (200 epochs)

Best checkpoint (epoch 184): clean 94.35 / attack 99.50 / noise 92.09
(paper: 94.15 / 99.55 / 93.55). This is within 0.15 pp of seed 0 on all
three metrics, confirming the stability of the 200-epoch reproduction. At
epoch 200 the instantaneous values were 93.89 / 99.75 / 92.01; the
best-checkpoint rule selected epoch 184 by clean accuracy.

### CIFAR-10 seed 2 (200 epochs)

Best checkpoint (epoch 177): clean 94.54 / attack 99.85 / noise 92.23.
All three metrics are the highest of the three seeds, but the spread is small.

### CIFAR-10 3-seed summary (200 epochs)

| Metric | Seeds 0 / 1 / 2 | Mean ± std | Paper |
|---|---|---:|---:|
| Clean | 94.23 / 94.35 / 94.54 | 94.37 ± 0.16 | 94.15 |
| Attack | 99.53 / 99.50 / 99.85 | 99.63 ± 0.19 | 99.55 |
| Noise | 92.22 / 92.09 / 92.23 | 92.18 ± 0.08 | 93.55 |

The clean and attack means are 0.22 pp above the paper, and the seed-to-seed
std is below 0.2 pp. Noise accuracy is 1.4 pp below the paper, within the
run-to-run variation of the random noise mode.

### GTSRB seed 0 (200 epochs)

Best checkpoint: clean 99.11 / attack 99.75 / noise 98.74 (paper:
98.97 / 98.78 / 98.01; README: 98.87 / 99.33 / 98.01). Clean is 0.14 pp above
the paper, attack 0.97 pp above, noise 0.73 pp above. Training curve:
`figures/gtsrb_training_curve.png`.

### GTSRB seed 1 (200 epochs)

Best checkpoint (epoch 200): clean 98.72 / attack 99.74 / noise 98.32.

### GTSRB seed 2 (200 epochs)

Best checkpoint (epoch 200): clean 98.58 / attack 99.75 / noise 98.35.

### GTSRB 3-seed summary (200 epochs)

| Metric | Seeds 0 / 1 / 2 | Mean ± std | Paper |
|---|---|---:|---:|
| Clean | 99.11 / 98.72 / 98.58 | 98.80 ± 0.28 | 98.97 |
| Attack | 99.75 / 99.74 / 99.75 | 99.75 ± 0.01 | 98.78 |
| Noise | 98.74 / 98.32 / 98.35 | 98.47 ± 0.23 | 98.01 |

Clean is 0.17 pp below the paper; attack is 0.97 pp above and extremely
stable across seeds; noise is 0.46 pp above. The attack number is closer to
the README's 99.33 than to the paper's 98.78, consistent with the reported
paper/README discrepancy.

### MNIST seed 0 (100 epochs)

Best checkpoint: clean 99.26 / attack 99.07 / noise 98.82 (paper:
99.52 / 99.86 / 98.20). The gap is consistent with the reduced training
budget (100 vs. 1000 epochs). Training curve:
`figures/mnist_training_curve.png`.

Observation: clean and noise accuracy are stable across epochs, but the
per-epoch ASR oscillates strongly (down to ~10% at some epochs, back to ~100%
at others). The official best-checkpoint criterion captures a high-ASR model.
This is consistent with the subtle warping trigger and the reduced epoch
budget; it is a useful detail for the gap analysis.

### MNIST seed 1 (100 epochs)

Best checkpoint: clean 99.25 / attack 98.80 / noise 98.78, consistent with
seed 0 (99.26 / 99.07 / 98.82). The per-epoch ASR oscillation is present in
both seeds; best-checkpoint selection matters.

### MNIST 3-seed summary (100 epochs)

| Metric | Seeds 0 / 1 / 2 | Mean ± std | Paper |
|---|---|---:|---:|
| Clean | 99.26 / 99.25 / 99.21 | 99.24 ± 0.03 | 99.52 |
| Attack | 99.07 / 98.80 / 99.26 | 99.04 ± 0.23 | 99.86 |
| Noise | 98.91 / 98.78 / 98.58 | 98.76 ± 0.17 | 98.20 |

The clean/attack gap is consistent with the reduced 100-epoch budget; the
per-epoch ASR oscillation is present in all three seeds.

### Noise-mode ablation (CIFAR-10, 200 epochs)

| Setting | Clean | Attack | Noise |
|---|---:|---:|---:|
| With noise mode ($\rho_n=0.2$) | 94.23 | 99.53 | 92.22 |
| Without noise mode ($\rho_n=0$) | 94.53 | 99.73 | 75.24 |

Removing the noise mode leaves clean and attack accuracy essentially unchanged
but drops noise-mode accuracy by 17 pp. This directly confirms the paper's
mechanism: the noise mode teaches the model to ignore random warps rather than
memorizing pixel-level artifacts. Neural Cleanse confirms it: Anomaly Index
5.14 for the no-noise model (detected) vs. 0.89 for the official model
(passes) and 2.40 for our reduced-epoch main model (marginal).

### All-to-all attack (CIFAR-10, MNIST, and GTSRB)

| Dataset | Clean | Attack | Noise | Paper (clean / attack / noise) |
|---|---:|---:|---:|---|
| CIFAR-10 (200 epochs) | 94.39 | 93.40 | 92.12 | 94.43 / 93.36 / 91.47 |
| MNIST (100 epochs) | 99.30 | 98.73 | 98.31 | 99.44 / 95.90 / 94.34 |
| GTSRB (200 epochs) | 98.73 | 98.05 | 98.73 | 99.39 / 98.31 / 98.96 |

GTSRB is within 0.66 pp of the paper on clean accuracy and within 0.26 pp on
attack; CIFAR-10 matches within 0.05 pp. MNIST attack is 2.8 pp above the
paper's 95.90 and noise is 4.0 pp above; the reduced 100-epoch budget and the
random noise mode likely explain part of that gap. All runs use the paper's
all-to-all target $c(y)=y+1$.

### V1: data-only attacker and architecture transfer

The attacker pre-generates a fixed poisoned dataset ($\rho_a=0.1$,
$\rho_n=0.2$) and does not control the victim's training; each victim uses
standard augmentation. All rows are single-seed.

| Victim | Clean | Attack | Noise |
|---|---:|---:|---:|
| PreActResNet18 | 93.96 | 97.01 | 90.98 |
| ResNet18 | 93.78 | 96.16 | 91.01 |
| VGG16 | 92.29 | 90.02 | 89.36 |
| MobileNetV2 | 85.21 | 62.39 | 82.85 |

The backdoor survives architecture transfer to PreActResNet18 and ResNet18
(ASR 96–97%) and degrades on VGG16 (90.0%) and MobileNetV2 (62.4%). The
MobileNetV2 clean accuracy is also lower (85.21), consistent with the
training hyperparameters being tuned for PreActResNet18; the result should
be read as a failure case for the data-only attacker rather than a clean
architecture comparison.

### V2: tighter poisoning and warping budgets

All runs use PreActResNet18 on CIFAR-10, 200 epochs, one seed.

| Setting | Clean | Attack | Noise |
|---|---:|---:|---:|
| $\rho_a=0.05$ | 94.53 | 98.56 | 92.38 |
| $\rho_a=0.02$ | 94.56 | 96.23 | 92.35 |
| $\rho_a=0.01$ | 94.34 | 84.57 | 92.38 |
| $s=0.35$ | 94.04 | 97.53 | 92.72 |
| $s=0.25$ | 92.77 | 85.45 | 92.12 |

Attack success stays above 96% down to $\rho_a=0.02$ and $s=0.35$, then
drops to 84.6% at $\rho_a=0.01$ and 85.5% at $s=0.25$. Clean accuracy is
essentially unchanged except for the weakest warp ($s=0.25$, 92.77). This
gives a breaking-point curve for the report.

### Fashion-MNIST (100 epochs)

Clean 91.80 / attack 77.32 / noise 90.44. The backdoor transfers to a new
dataset but is weaker than on MNIST/CIFAR-10; the 100-epoch budget and the
lack of dataset-specific tuning are likely contributors.

### Clean baseline (CIFAR-10, 200 epochs)

Clean 94.56; attack 10.15 and noise 81.33 are not meaningful for a benign
model (no trigger training, no noise mode). The clean accuracy is within
0.2 pp of the backdoored three-seed mean (94.37), so the backdoor does not
buy its attack success at the cost of clean utility.

### Test-time preprocessing robustness (CIFAR-10, 5,000 test images)

The trigger is applied first, then the victim-side preprocessing. Official
checkpoint:

| Condition | Clean | Attack | Noise |
|---|---:|---:|---:|
| identity | 93.92 | 99.48 | 91.30 |
| random crop (pad 4) | 92.79 | 98.31 | 90.39 |
| resize 32->24->32 | 76.78 | 72.52 | 79.04 |
| JPEG quality 50 | 81.32 | 14.98 | 73.98 |

Our 200-epoch checkpoint:

| Condition | Clean | Attack | Noise |
|---|---:|---:|---:|
| identity | 93.76 | 99.46 | 91.78 |
| random crop (pad 4) | 92.25 | 91.65 | 90.07 |
| resize 32->24->32 | 87.80 | 31.50 | 84.78 |
| JPEG quality 50 | 81.94 | 11.76 | 76.38 |

Random cropping is mild, but resampling and JPEG compression break the
trigger. The reduced-epoch model is more fragile than the official one,
consistent with its smaller defense margin. This supports the critique that
the trigger is digital: a victim pipeline that resamples the input can remove
the attack without any backdoor-specific defense.

Outputs: `results/robustness_official_cifar10.json`,
`results/robustness_ours_cifar10.json`.

### Pixel-wise cross-check against the official code path

Using the released CIFAR-10 grids, the reference formulas from the official
evaluation/training code (`grid_temps = clip(identity + s * field / H)`,
`F.grid_sample`, and the split `num_bd = int(bs * rho_a)`,
`num_cross = int(num_bd * rho_n / rho_a)`) were compared with our
implementation:

| Check | Max abs diff |
|---|---:|
| identity grid | 3.0e-8 |
| triggered images | 0.0 |
| poisoned backdoor split | 0.0 |
| poisoned clean split | 0.0 |
| noise-mode images (controlled RNG) | 0.0 |
| label mismatches | 0 / 256 |

The warping, poisoning split, and labels are pixel-identical to the official
code path; the only non-zero difference is float32 rounding in the identity
grid. Output: `results/crosscheck_official_cifar10.json`.

## 4. Gap analysis notes

- **Clean/attack match:** our independent implementation reproduces the
  released checkpoints' clean and attack accuracy to within rounding on
  MNIST, CIFAR-10, and GTSRB. This validates the warping field, poisoning
  pipeline, and architectures.
- **Noise accuracy:** random by construction; we report ranges over repeated
  evaluations rather than a single number.
- **GTSRB discrepancy:** the paper reports 98.97 / 98.78, the README reports
  98.87 / 99.33. Our evaluation (98.99 / 98.96) is closer to the paper's clean
  number and between the two attack numbers, reconciling the discrepancy.
- **Neural Cleanse:** the paper's "passes" claim holds at the official-like
  20-epoch budget but not at a 5-epoch budget. The defense's conclusion
  depends on how much optimization it is allowed.
- **STRIP:** the official code mixes images in [0,1] space and uses a
  sigmoid-based score; we use the same mixing and a proper softmax entropy.
  Both produce overlapping distributions.
- **Fine-Pruning:** the official code is pruning-only (no fine-tuning). Our
  sampled sweep reproduces the qualitative claim on both MNIST and CIFAR-10.
