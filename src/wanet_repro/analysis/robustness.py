"""Evaluate WaNet under test-time preprocessing.

The trigger is applied first, then the image is processed the way a victim
pipeline might process it (random crop, resampling, JPEG compression).  This
is the robustness check promised in the proposal.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.io import decode_jpeg, encode_jpeg

from ..data import build_dataset, build_transform, get_spec
from ..models import build_model
from ..poison import target_labels
from ..utils import resolve_device, save_json
from ..warp import warp_batch, warp_batch_with_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-time preprocessing robustness for WaNet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--attack_mode", default="all2one", choices=["all2one", "all2all"])
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--noise_type", default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--crop_repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def default_model(dataset: str) -> str:
    return "mnist" if dataset in ("mnist", "fashionmnist") else "preact18"


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    field: torch.Tensor,
    identity_grid: torch.Tensor,
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total = clean_correct = attack_correct = noise_correct = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            attack_targets = target_labels(
                targets, args.attack_mode, args.target_label, num_classes
            )

            attacked = warp_batch(inputs, field, s=args.s, identity_grid=identity_grid)
            noisy = warp_batch_with_noise(
                inputs,
                field,
                s=args.s,
                identity_grid=identity_grid,
                noise_type=args.noise_type,
            )

            for images, correct_kind in (
                (inputs, "clean"),
                (attacked, "attack"),
                (noisy, "noise"),
            ):
                processed = preprocess(images)
                logits = model(processed)
                predictions = logits.argmax(1)
                if correct_kind == "clean":
                    clean_correct += (predictions == targets).sum().item()
                elif correct_kind == "attack":
                    attack_correct += (predictions == attack_targets).sum().item()
                else:
                    noise_correct += (predictions == targets).sum().item()

            total += inputs.shape[0]

    return {
        "clean": 100.0 * clean_correct / total,
        "attack": 100.0 * attack_correct / total,
        "noise": 100.0 * noise_correct / total,
        "n_samples": total,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)
    model_name = args.model or default_model(args.dataset)

    dataset = build_dataset(
        args.dataset, args.data_root, False, build_transform(args.dataset)
    )
    if args.n_samples < len(dataset):
        dataset = Subset(dataset, range(args.n_samples))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_name, num_classes=spec.num_classes, input_channels=spec.channels)
    model.load_state_dict(checkpoint["netC"])
    model.to(device)
    identity_grid = checkpoint["identity_grid"].to(device)
    field = checkpoint["noise_grid"].to(device)

    mean = torch.tensor(spec.mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(spec.std, device=device).view(1, -1, 1, 1)

    def denormalize(x: torch.Tensor) -> torch.Tensor:
        return (x * std + mean).clamp(0.0, 1.0)

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std

    def identity(x: torch.Tensor) -> torch.Tensor:
        return x

    def resize(x: torch.Tensor, scale: float = 0.75) -> torch.Tensor:
        y = denormalize(x)
        height, width = y.shape[-2:]
        small = (max(1, int(height * scale)), max(1, int(width * scale)))
        y = F.interpolate(y, size=small, mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
        return normalize(y)

    def jpeg(x: torch.Tensor, quality: int = 50) -> torch.Tensor:
        y = denormalize(x)
        decoded: List[torch.Tensor] = []
        for index in range(y.shape[0]):
            image = (y[index] * 255.0).round().to(torch.uint8).cpu()
            buffer = encode_jpeg(image, quality=quality)
            decoded.append(decode_jpeg(buffer))
        y = torch.stack(decoded).to(device).float() / 255.0
        return normalize(y)

    def crop_factory(seed: int) -> Callable[[torch.Tensor], torch.Tensor]:
        def crop(x: torch.Tensor) -> torch.Tensor:
            y = denormalize(x)
            height, width = y.shape[-2:]
            pad = 4
            y = F.pad(y, (pad, pad, pad, pad), mode="reflect")
            rng = random.Random(seed)
            crops = []
            for index in range(y.shape[0]):
                top = rng.randint(0, 2 * pad)
                left = rng.randint(0, 2 * pad)
                crops.append(y[index, :, top : top + height, left : left + width])
            return normalize(torch.stack(crops))
        return crop

    results: Dict[str, Dict[str, float]] = {}
    results["identity"] = evaluate(
        model, loader, field, identity_grid, identity, args, spec.num_classes, device
    )
    results["resize_0.75"] = evaluate(
        model, loader, field, identity_grid, lambda x: resize(x, 0.75), args, spec.num_classes, device
    )
    results["jpeg_50"] = evaluate(
        model, loader, field, identity_grid, lambda x: jpeg(x, 50), args, spec.num_classes, device
    )

    crop_runs = [
        evaluate(
            model, loader, field, identity_grid, crop_factory(seed), args, spec.num_classes, device
        )
        for seed in range(args.crop_repeats)
    ]
    results["crop"] = {
        key: sum(run[key] for run in crop_runs) / len(crop_runs)
        for key in ("clean", "attack", "noise")
    }
    results["crop"]["n_samples"] = crop_runs[0]["n_samples"]
    results["crop_runs"] = crop_runs

    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_samples": results["identity"]["n_samples"],
        "s": args.s,
        "attack_mode": args.attack_mode,
        "results": results,
    }
    save_json(args.out, payload)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'condition':<12} {'clean':>7} {'attack':>7} {'noise':>7}")
    for name in ("identity", "crop", "resize_0.75", "jpeg_50"):
        row = results[name]
        print(f"{name:<12} {row['clean']:7.2f} {row['attack']:7.2f} {row['noise']:7.2f}")


if __name__ == "__main__":
    main()
