"""Evaluate a trained WaNet checkpoint (ours or the official one)."""

from __future__ import annotations

import argparse
import json

import torch

from .data import build_dataloaders, get_spec
from .models import build_model
from .poison import target_labels
from .utils import gpu_info, resolve_device, save_json
from .warp import make_identity_grid, make_warp_field, warp_batch, warp_batch_with_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a WaNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["mnist", "fashionmnist", "cifar10", "cifar100", "gtsrb", "celeba"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--attack_mode", default="all2one", choices=["all2one", "all2all"])
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--noise_type", default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional path to save metrics as JSON.")
    return parser.parse_args()


def default_model(dataset: str) -> str:
    if dataset in ("mnist", "fashionmnist"):
        return "mnist"
    if dataset == "celeba":
        return "celeba_resnet18"
    return "preact18"


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    field: torch.Tensor,
    identity_grid: torch.Tensor,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
):
    model.eval()
    total = 0
    clean_correct = attack_correct = noise_correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += inputs.shape[0]

        logits = model(inputs)
        clean_correct += (logits.argmax(1) == targets).sum().item()

        attacked = warp_batch(inputs, field, s=args.s, identity_grid=identity_grid)
        bd_targets = target_labels(targets, args.attack_mode, args.target_label, num_classes)
        attack_correct += (model(attacked).argmax(1) == bd_targets).sum().item()

        noisy = warp_batch_with_noise(
            inputs, field, s=args.s, identity_grid=identity_grid, noise_type=args.noise_type
        )
        noise_correct += (model(noisy).argmax(1) == targets).sum().item()

    return {
        "clean_acc": 100.0 * clean_correct / total,
        "attack_acc": 100.0 * attack_correct / total,
        "noise_acc": 100.0 * noise_correct / total,
        "num_samples": total,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)
    model_name = args.model or default_model(args.dataset)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_name, num_classes=spec.num_classes, input_channels=spec.channels)
    model.load_state_dict(checkpoint["netC"])
    model.to(device)

    identity_grid = checkpoint.get("identity_grid")
    field = checkpoint.get("noise_grid")
    if identity_grid is None:
        identity_grid = make_identity_grid(spec.input_size, spec.input_size)
    if field is None:
        field = make_warp_field(args.k, spec.input_size, spec.input_size, seed=args.seed)
    identity_grid = identity_grid.to(device)
    field = field.to(device)

    _, test_loader, _ = build_dataloaders(
        args.dataset,
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    metrics = evaluate(model, test_loader, field, identity_grid, args, spec.num_classes, device)
    metrics.update({"checkpoint": args.checkpoint, "dataset": args.dataset, "model": model_name, "gpu": gpu_info()})
    print(json.dumps(metrics, indent=2))
    if args.out:
        save_json(args.out, metrics)


if __name__ == "__main__":
    main()
