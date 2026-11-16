"""Neural Cleanse (Wang et al., IEEE S&P 2019), simplified re-implementation.

For each candidate target label, optimize a mask/pattern pair that turns clean
inputs into that label.  A backdoored model tends to need a much smaller mask
for its true target label than for the other labels.  The anomaly index is

    (median(L1) - min(L1)) / (1.4826 * MAD(L1))

with a detection threshold of 2.  The official implementation is used as a
reference; this version uses a subset of the test set and a fixed epoch budget
to keep the defense affordable on one 8 GB GPU.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..data import build_dataloaders, get_spec
from ..models import build_model
from ..utils import gpu_info, resolve_device, save_json


class RegressionModel(nn.Module):
    def __init__(
        self,
        classifier: nn.Module,
        input_shape: Tuple[int, int, int],
        mean: Tuple[float, ...] = (0.0,),
        std: Tuple[float, ...] = (1.0,),
        eps: float = 1e-7,
    ):
        super().__init__()
        channels, height, width = input_shape
        self.mask_tanh = nn.Parameter(torch.ones(1, 1, height, width))
        self.pattern_tanh = nn.Parameter(torch.ones(1, channels, height, width))
        self.classifier = classifier
        self.eps = eps
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def raw_mask(self) -> torch.Tensor:
        return torch.tanh(self.mask_tanh) / (2 + self.eps) + 0.5

    def raw_pattern(self) -> torch.Tensor:
        return torch.tanh(self.pattern_tanh) / (2 + self.eps) + 0.5

    def normalized_pattern(self) -> torch.Tensor:
        # The classifier consumes normalized inputs, so the optimized pattern
        # must be mapped into the same space (matches the official Normalize).
        return (self.raw_pattern() - self.mean) / self.std

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = self.raw_mask()
        pattern = self.normalized_pattern()
        return self.classifier((1 - mask) * inputs + mask * pattern)


def scan_model(
    classifier: nn.Module,
    loader,
    input_shape: Tuple[int, int, int],
    target_label: int,
    device: torch.device,
    *,
    mean: Tuple[float, ...] = (0.0,),
    std: Tuple[float, ...] = (1.0,),
    epochs: int = 15,
    lr: float = 0.1,
    init_cost: float = 1e-3,
    atk_succ_threshold: float = 98.0,
    patience: int = 5,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    regression = RegressionModel(classifier, input_shape, mean=mean, std=std).to(device)
    optimizer = torch.optim.Adam(regression.parameters(), lr=lr, betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    cost = init_cost
    cost_up_counter = 0
    cost_down_counter = 0
    best_reg = float("inf")
    best_mask: Optional[torch.Tensor] = None
    best_pattern: Optional[torch.Tensor] = None

    for _ in range(epochs):
        total = 0
        correct = 0
        for inputs, _ in loader:
            inputs = inputs.to(device)
            targets = torch.full((inputs.shape[0],), target_label, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            predictions = regression(inputs)
            loss_ce = criterion(predictions, targets)
            loss_reg = regression.raw_mask().norm(p=1)
            loss = loss_ce + cost * loss_reg
            loss.backward()
            optimizer.step()

            total += inputs.shape[0]
            correct += (predictions.argmax(1) == targets).sum().item()

        accuracy = 100.0 * correct / total
        with torch.no_grad():
            reg_value = regression.raw_mask().norm(p=1).item()
            if accuracy >= atk_succ_threshold and reg_value < best_reg:
                best_reg = reg_value
                best_mask = regression.raw_mask().detach().cpu()
                best_pattern = regression.raw_pattern().detach().cpu()

        # Official cost schedule: increase the regularizer when the attack
        # succeeds, decrease it when it fails.
        if accuracy >= atk_succ_threshold:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1

        if cost_up_counter >= patience:
            cost_up_counter = 0
            cost *= 2.0
        elif cost_down_counter >= patience:
            cost_down_counter = 0
            cost /= 2.0**1.5

    if best_mask is None:
        best_mask = regression.raw_mask().detach().cpu()
        best_pattern = regression.raw_pattern().detach().cpu()
        best_reg = best_mask.norm(p=1).item()

    return best_reg, best_mask, best_pattern


def anomaly_index(norms: List[float]) -> Tuple[float, float, float]:
    values = torch.tensor(norms, dtype=torch.float32)
    median = torch.median(values)
    mad = 1.4826 * torch.median(torch.abs(values - median))
    if mad.item() == 0:
        return float("inf"), float(median), float(mad)
    score = torch.abs(torch.min(values) - median) / mad
    return float(score), float(median), float(mad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural Cleanse on a WaNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset", default="cifar10", choices=["mnist", "fashionmnist", "cifar10", "cifar100", "gtsrb"]
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--save_masks", default=None, help="Directory for mask/pattern PNGs.")
    return parser.parse_args()


def default_model(dataset: str) -> str:
    return "mnist" if dataset in ("mnist", "fashionmnist") else "preact18"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)
    model_name = args.model or default_model(args.dataset)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classifier = build_model(model_name, num_classes=spec.num_classes, input_channels=spec.channels)
    classifier.load_state_dict(checkpoint["netC"])
    classifier.to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad = False

    _, test_loader, _ = build_dataloaders(
        args.dataset,
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # A deterministic subset keeps the scan affordable and reproducible.
    dataset = test_loader.dataset
    n_samples = min(args.n_samples, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    all_runs = []
    for run in range(args.runs):
        norms = []
        masks = []
        patterns = []
        for label in range(spec.num_classes):
            reg_value, mask, pattern = scan_model(
                classifier,
                loader,
                (spec.channels, spec.input_size, spec.input_size),
                label,
                device,
                mean=spec.mean,
                std=spec.std,
                epochs=args.epochs,
            )
            norms.append(reg_value)
            masks.append(mask)
            patterns.append(pattern)
            print(f"run {run + 1}/{args.runs} label {label:2d}: L1={reg_value:.4f}")
        score, median, mad = anomaly_index(norms)
        all_runs.append({"norms": norms, "anomaly_index": score, "median": median, "mad": mad})
        print(f"run {run + 1}/{args.runs}: anomaly_index={score:.4f} (median={median:.4f}, mad={mad:.4f})")

        if args.save_masks:
            import torchvision.utils as vutils

            os.makedirs(args.save_masks, exist_ok=True)
            for label, (mask, pattern) in enumerate(zip(masks, patterns)):
                vutils.save_image(mask, os.path.join(args.save_masks, f"run{run}_mask_{label}.png"), normalize=True)
                vutils.save_image(pattern, os.path.join(args.save_masks, f"run{run}_pattern_{label}.png"), normalize=True)

    scores = [run["anomaly_index"] for run in all_runs]
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "model": model_name,
        "n_samples": n_samples,
        "epochs_per_label": args.epochs,
        "runs": all_runs,
        "anomaly_index_mean": sum(scores) / len(scores),
        "anomaly_index_std": (sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)) ** 0.5,
        "gpu": gpu_info(),
    }
    print(f"Anomaly index: {summary['anomaly_index_mean']:.4f} +/- {summary['anomaly_index_std']:.4f}")
    if args.out:
        save_json(args.out, summary)


if __name__ == "__main__":
    main()
