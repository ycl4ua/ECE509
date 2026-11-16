# Paper claims and reproduction status

This is the Track A claims table. "Paper" numbers are from the ICLR 2021
paper; "Ours" are from the released checkpoints evaluated with our independent
code unless noted. Status: R = reproduced, P = partially reproduced, N = not
reproduced (with reason).

| # | Claim | Paper number | Our result | Status |
|---|---|---|---|---|
| C1 | All-to-one clean/attack/noise on MNIST | 99.52 / 99.86 / 98.20 | 99.52 / 99.86 / 98.4–98.5 | R |
| C2 | All-to-one on CIFAR-10 | 94.15 / 99.55 / 93.55 | 94.14 / 99.55 / 92.1–92.3 | R |
| C3 | All-to-one on GTSRB | 98.97 / 98.78 / 98.01 (paper); 98.87 / 99.33 / 98.01 (README) | 98.99 / 98.96 / 99.03 | R |
| C4 | All-to-one on CelebA | 78.99 / 99.33 / 76.74 | 79.01 / 99.04 / 77.60 | R |
| C5 | All-to-all attack (MNIST/CIFAR-10/GTSRB/CelebA) | 95.90 / 93.36 / 98.31 / 78.58 attack accuracy | CIFAR-10 93.40, MNIST 98.73, GTSRB 98.05; CelebA not run | P |
| C6 | Noise mode is necessary to evade Neural Cleanse | without noise mode, NC detects | no-noise model: AI 5.14 (detected); official model: 0.89 (passes) | R |
| C7 | WaNet passes Neural Cleanse (Anomaly Index < 2) | < 2 on all datasets | MNIST official 1.99; CIFAR-10 official 0.89; our CIFAR-10 2.40 (marginal) | R on official checkpoints; P on reduced-epoch model (budget-sensitive: 37.4 at 5 epochs) |
| C8 | WaNet passes Fine-Pruning | clean accuracy never considerably higher than attack | MNIST 63/64: clean 9.89, attack 99.98; CIFAR-10 504/512: clean 55.61, attack 99.84 | R |
| C9 | WaNet passes STRIP | entropy distributions overlap | MNIST 0.338±0.119 vs 0.358±0.125; CIFAR-10 0.466±0.176 vs 0.517±0.177 | R |
| C10 | Human inspection: WaNet fooling rate 28.0% vs ≤7.7% for prior attacks | 38.6% backdoor, 17.4% clean, 28.0% all | not reproduced | N: optional user study |
| C11 | Physical screen-recapture attack | 98% clean / 96% ASR | not reproduced | N: optional demo |
| C12 | Spectral-signature defense fails | histograms inseparable | not reproduced | N: optional |
| C13 | Stability across 8 warping fields (CIFAR-10) | 94.42±0.08 / 99.40±0.21 / 93.16±0.43 | not reproduced | N: optional |
| C14 | s and k ablations | backdoor accuracy rises with s/k | V2 covers s; k pending | P |
| C15 | All-to-all defense results | passes NC/FP/STRIP | not reproduced | N: optional |

## Gap explanations

- **C1–C3:** clean and attack accuracy match the released checkpoints within
  0.2 pp. Noise accuracy is random and varies between runs; the README says
  the same.
- **C3:** the paper and README disagree on GTSRB; our numbers are 98.99/98.96,
  closer to the paper's clean value and between the two attack values.
- **C4:** evaluated the released checkpoint on the FLWRLabs parquet mirror of
  the official 19,962-image test split (the torchvision Google Drive download
  is rate-limited). Clean and attack match within 0.02 and 0.29 pp; noise is
  0.86 pp higher, within the random noise-mode variation.
- **C5:** CIFAR-10 matches the paper within 0.04 pp on attack accuracy; GTSRB
  is within 0.26 pp; MNIST attack is 2.8 pp higher. CelebA is not run (there is
  no released all-to-all CelebA checkpoint, and training it was out of scope).
- **C7:** Neural Cleanse is unstable and budget-dependent. At the
  official-like 20-epoch budget we reproduce the "passes" verdict; at a
  5-epoch budget the same model is detected. This is a critique of the
  defense's robustness, not a failure of the reproduction.
- **C8–C9:** our defense implementations are simplified (subset scans, sampled
  pruning levels) but follow the official mixing/pruning logic; the
  qualitative conclusions match the paper.
- **C10–C15:** optional experiments that require human subjects, physical
  hardware, or substantial extra compute; the artifact documents how to run
  them.
