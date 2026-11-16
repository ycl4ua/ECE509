"""Poisoning pipelines: on-the-fly batches and fixed poisoned datasets."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .warp import warp_batch, warp_batch_with_noise


def target_labels(
    targets: torch.Tensor,
    attack_mode: str,
    target_label: int,
    num_classes: int,
) -> torch.Tensor:
    if attack_mode == "all2one":
        return torch.full_like(targets, target_label)
    if attack_mode == "all2all":
        return torch.remainder(targets + 1, num_classes)
    raise ValueError(f"Unknown attack_mode: {attack_mode!r}")


def target_label_scalar(label: int, attack_mode: str, target_label: int, num_classes: int) -> int:
    if attack_mode == "all2one":
        return target_label
    if attack_mode == "all2all":
        return (label + 1) % num_classes
    raise ValueError(f"Unknown attack_mode: {attack_mode!r}")


def poison_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    field: torch.Tensor,
    identity_grid: torch.Tensor,
    *,
    rho_a: float,
    rho_n: float,
    s: float,
    attack_mode: str,
    target_label: int,
    num_classes: int,
    noise_type: str = "uniform",
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Poison a batch the way the official training loop does.

    The first ``rho_a`` fraction of the batch is warped and relabeled to the
    target; the next ``rho_a * rho_n / rho_a`` fraction is warped with noise
    and keeps its clean label; the rest is clean.
    """
    batch = inputs.shape[0]
    num_bd = int(batch * rho_a)
    cross_ratio = (rho_n / rho_a) if rho_a > 0 else 0.0
    num_cross = min(int(num_bd * cross_ratio), batch - num_bd)

    if num_bd > 0:
        inputs_bd = warp_batch(inputs[:num_bd], field, s=s, identity_grid=identity_grid)
        targets_bd = target_labels(targets[:num_bd], attack_mode, target_label, num_classes)
    else:
        inputs_bd = inputs[:0]
        targets_bd = targets[:0]

    if num_cross > 0:
        inputs_cross = warp_batch_with_noise(
            inputs[num_bd : num_bd + num_cross],
            field,
            s=s,
            identity_grid=identity_grid,
            noise_type=noise_type,
        )
    else:
        inputs_cross = inputs[:0]

    total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[num_bd + num_cross :]], dim=0)
    total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
    return total_inputs, total_targets, num_bd, num_cross


class PoisonedDataset(Dataset):
    """A fixed poisoned dataset for the data-only attacker variation.

    Each sample is assigned one of three modes with probabilities
    ``rho_a`` (backdoor), ``rho_n`` (noise) and the remainder (clean).  The
    assignment is seeded and fixed for the whole training run, so a victim
    that does not control training receives a static poisoned dataset.
    """

    def __init__(
        self,
        base: Dataset,
        field: torch.Tensor,
        identity_grid: torch.Tensor,
        *,
        rho_a: float,
        rho_n: float,
        s: float,
        attack_mode: str,
        target_label: int,
        num_classes: int,
        noise_type: str = "uniform",
        seed: int = 0,
    ):
        self.base = base
        self.field = field
        self.identity_grid = identity_grid
        self.rho_a = rho_a
        self.rho_n = rho_n
        self.s = s
        self.attack_mode = attack_mode
        self.target_label = target_label
        self.num_classes = num_classes
        self.noise_type = noise_type

        count = len(base)
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        modes = np.zeros(count, dtype=np.int8)
        n_bd = int(count * rho_a)
        n_noise = int(count * rho_n)
        modes[order[:n_bd]] = 1
        modes[order[n_bd : n_bd + n_noise]] = 2
        self.modes = modes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        mode = int(self.modes[index])
        if mode == 1:
            image = warp_batch(
                image.unsqueeze(0), self.field, s=self.s, identity_grid=self.identity_grid
            )[0]
            label = target_label_scalar(label, self.attack_mode, self.target_label, self.num_classes)
        elif mode == 2:
            image = warp_batch_with_noise(
                image.unsqueeze(0),
                self.field,
                s=self.s,
                identity_grid=self.identity_grid,
                noise_type=self.noise_type,
            )[0]
        return image, label
