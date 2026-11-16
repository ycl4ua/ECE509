"""Pixel-wise cross-check of our warping and poisoning against the official logic.

The reference formulas are taken from the official evaluation and training
code paths (``grid_temps = (identity_grid + s * noise_grid / H)`` followed by
``F.grid_sample``).  The official model weights are not imported; only the
released grids and images are used, so the check isolates the trigger and
poisoning implementation.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from ..data import build_dataset, build_transform, get_spec
from ..poison import poison_batch
from ..utils import resolve_device, save_json
from ..warp import make_identity_grid, warp_batch, warp_batch_with_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-check WaNet warping against the official formula.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "gtsrb"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--rho_a", type=float, default=0.1)
    parser.add_argument("--rho_n", type=float, default=0.2)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def mean_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity_grid = checkpoint["identity_grid"].to(device)
    field = checkpoint["noise_grid"].to(device)

    dataset = build_dataset(args.dataset, args.data_root, False, build_transform(args.dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)
    batch, _, height, width = inputs.shape

    # --- 1. Identity grid: our constructor vs. the released grid. -----------
    ours_identity = make_identity_grid(height, width, device=device)
    identity_diff = max_diff(ours_identity, identity_grid)

    # --- 2. Deterministic trigger: our warp vs. the official eval formula. --
    grid = (identity_grid + args.s * field / height) * 1.0
    grid = torch.clamp(grid, -1.0, 1.0)
    bd_ref = F.grid_sample(inputs, grid.repeat(batch, 1, 1, 1), align_corners=True)
    bd_ours = warp_batch(inputs, field, s=args.s, identity_grid=identity_grid)

    # --- 3. Poisoning split: labels and the deterministic backdoor part. ---
    poisoned_inputs, poisoned_targets, num_bd, num_cross = poison_batch(
        inputs,
        targets,
        field,
        identity_grid,
        rho_a=args.rho_a,
        rho_n=args.rho_n,
        s=args.s,
        attack_mode="all2one",
        target_label=args.target_label,
        num_classes=spec.num_classes,
        noise_type="uniform",
    )
    expected_targets = torch.cat(
        [torch.full_like(targets[:num_bd], args.target_label), targets[num_bd:]]
    )
    label_mismatches = int((poisoned_targets != expected_targets).sum().item())
    bd_split_diff = max_diff(poisoned_inputs[:num_bd], bd_ref[:num_bd])
    clean_split_diff = max_diff(
        poisoned_inputs[num_bd + num_cross :], inputs[num_bd + num_cross :]
    )

    # --- 4. Noise mode: controlled generator on both sides. ----------------
    start, stop = num_bd, num_bd + num_cross
    ours_generator = torch.Generator(device="cpu").manual_seed(1234)
    ref_generator = torch.Generator(device="cpu").manual_seed(1234)
    cross_ours = warp_batch_with_noise(
        inputs[start:stop],
        field,
        s=args.s,
        identity_grid=identity_grid,
        noise_type="uniform",
        generator=ours_generator,
    )
    noise = (
        torch.rand(stop - start, height, width, 2, generator=ref_generator) * 2.0 - 1.0
    ).to(device)
    grid2 = torch.clamp(
        grid.repeat(stop - start, 1, 1, 1) + noise / height, -1.0, 1.0
    )
    cross_ref = F.grid_sample(inputs[start:stop], grid2, align_corners=True)

    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "batch_size": batch,
        "s": args.s,
        "rho_a": args.rho_a,
        "rho_n": args.rho_n,
        "num_bd": num_bd,
        "num_cross": num_cross,
        "identity_grid_max_abs_diff": identity_diff,
        "trigger_max_abs_diff": max_diff(bd_ours, bd_ref),
        "trigger_mean_abs_diff": mean_diff(bd_ours, bd_ref),
        "poison_bd_max_abs_diff": bd_split_diff,
        "poison_clean_max_abs_diff": clean_split_diff,
        "poison_label_mismatches": label_mismatches,
        "noise_max_abs_diff": max_diff(cross_ours, cross_ref),
        "noise_mean_abs_diff": mean_diff(cross_ours, cross_ref),
    }
    save_json(args.out, payload)

    print(f"identity grid      max|diff| = {payload['identity_grid_max_abs_diff']:.3e}")
    print(f"trigger            max|diff| = {payload['trigger_max_abs_diff']:.3e}  mean = {payload['trigger_mean_abs_diff']:.3e}")
    print(f"poison backdoor    max|diff| = {payload['poison_bd_max_abs_diff']:.3e}")
    print(f"poison clean part  max|diff| = {payload['poison_clean_max_abs_diff']:.3e}")
    print(f"poison labels      mismatches = {payload['poison_label_mismatches']} / {batch}")
    print(f"noise mode         max|diff| = {payload['noise_max_abs_diff']:.3e}  mean = {payload['noise_mean_abs_diff']:.3e}")


if __name__ == "__main__":
    main()
