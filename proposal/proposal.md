# Reproducing WaNet: An Imperceptible Warping-based Backdoor Attack

**Team:** Yicheng Leng, Zhuo Wang · **Track:** A (Reproduction) · ECE 509 · November 2026

## Problem

Outsourced training and third-party model zoos are common, but a malicious trainer can deliver a backdoored model that behaves normally on clean inputs yet misclassifies inputs carrying a secret trigger. WaNet (ICLR 2021) uses a subtle elastic image warp instead of patch- or noise-based triggers and adds a "noise mode" that teaches the model to ignore random warps, which makes the backdoor hard for humans and defenses to detect. We will reproduce its main results, re-implement its core warping pipeline, and test whether those claims hold under a weaker adversary and tighter budgets.

## Threat model sketch

**System:** data are collected and annotated, training is outsourced, and the resulting weights are delivered to a victim who deploys them for inference. **Assets and goal:** the attacker wants a model that classifies clean inputs normally but predicts a target label ŷ whenever a fixed warping field M is applied; clean validation accuracy should stay close to a benign model (at most a 1% drop). **Capabilities and knowledge:** as in the paper, the attacker controls training, chooses M, injects warped samples labeled ŷ and noise-mode samples labeled y, and delivers the weights (ρa=0.1, ρn=0.2, s=0.5, k=4); they have white-box knowledge of the pipeline and M but no access to the victim's private validation set. **Assumptions and scope:** the victim deploys the model as-is and validates only on clean data, and the trigger is applied digitally at inference; adaptive defenses, purely physical settings where the attacker cannot control the digital image pipeline, and inference-time model modification are out of scope. The threat is realistic for outsourced training and model zoos, and the paper's screen-recapture demo supports physical plausibility.

## Approach

We will reproduce the all-to-one results on MNIST, CIFAR-10, and GTSRB, and evaluate the released CelebA checkpoint, reporting clean/attack/noise accuracy over three seeds. The core warping pipeline (k×k control grid, mean-absolute normalization, bicubic upsampling, clipping, and grid_sample) will be re-implemented from scratch, together with the poisoning pipeline and noise mode; poisoned images will be cross-checked pixel-wise against the official code. We will also reproduce Neural Cleanse with and without noise mode (Anomaly Index, three runs to account for its optimization variance) and, if time permits, Fine-Pruning and STRIP. Beyond the paper's settings, we will test a new dataset (Fashion-MNIST) and alternative victim models (ResNet-18, MobileNetV2, and VGG-16), and evaluate robustness to test-time preprocessing (crop/resize/JPEG). Two variations probe the threat model: (V1) a data-only attacker that pre-generates a poisoned dataset without controlling training, with architectures unknown to the attacker; and (V2) tighter budgets (ρa ∈ {0.01,0.02,0.05,0.1}, s ∈ {0.25,0.35,0.5}). The report will critique the paper's strong "total training control" assumption, its reliance on digital trigger insertion (the physical demo is screen recapture), the scope of its human study, and its non-adaptive defense evaluation.

## Artifact and evaluation plan

We will deliver a repository with a pinned environment, one-command reproduction, and a demo under three minutes. The evaluation will cover MNIST, CIFAR-10, GTSRB, and Fashion-MNIST (optionally CIFAR-100), and report clean accuracy (CA), attack success rate (ASR), noise accuracy, the Neural Cleanse Anomaly Index, Fine-Pruning clean/attack curves, and STRIP entropy. Baselines are the paper's numbers, the released checkpoints, and the official code. Experiments will run on one RTX 4070 Laptop GPU (8 GB) and take roughly 30–60 GPU-hours, using the released checkpoints for exact numbers and reduced-epoch training for the three-seed runs.

## Milestones and fallback

- Submit the proposal; modernize the environment; evaluate the official checkpoints; establish MNIST/CIFAR-10 baselines.
- Complete the CIFAR-10/MNIST three-seed runs, all-to-all, and the re-implementation cross-check.
- Run Neural Cleanse and the noise-mode ablation, Fine-Pruning/STRIP, and the GTSRB experiments.
- Run Fashion-MNIST and the new victim models, V1/V2; produce the figures and report draft.
- Finalize the report, demo, presentation, and artifact.

**Fallbacks:** if 1000-epoch training is too slow, we will use the released checkpoints plus reduced-epoch runs and report the gap. If GTSRB is unavailable, we will substitute MNIST/CIFAR-10/Fashion-MNIST; if the defenses are too costly, we will reproduce Neural Cleanse only; and if time runs short, we will drop the optional robustness check and user study.

## References

[1] T. A. Nguyen and A. T. Tran. WaNet — Imperceptible Warping-based Backdoor Attack. ICLR 2021.

[2] B. Wang et al. Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks. IEEE S&P 2019.

[3] K. Liu et al. Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. RAID 2018.

[4] Y. Gao et al. STRIP: A Defence Against Trojan Attacks on Deep Neural Networks. ACSAC 2019.
