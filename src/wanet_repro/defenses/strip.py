"""STRIP (Gao et al., ACSAC 2019), simplified re-implementation.

For each input, superimpose several clean images and measure the entropy of
the model's predictions.  A backdoored input tends to keep a stable (low
entropy) prediction under perturbation, while a clean input becomes uncertain.
The WaNet paper reports that its trigger is destroyed by the perturbation, so
clean and backdoor entropy distributions overlap.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from ..data import build_dataloaders, get_spec
from ..models import build_model
from ..utils import gpu_info, resolve_device, save_json
from ..warp import warp_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STRIP entropy analysis of a WaNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "fashionmnist", "cifar10", "cifar100", "gtsrb"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_superimpose", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def default_model(dataset: str) -> str:
    return "mnist" if dataset in ("mnist", "fashionmnist") else "preact18"


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log2(probabilities + 1e-12)).sum(dim=1)


@torch.no_grad()
def strip_scores(
    model: torch.nn.Module,
    dataset,
    field: torch.Tensor,
    identity_grid: torch.Tensor,
    args: argparse.Namespace,
    spec,
    device: torch.device,
    backdoor: bool,
) -> list[float]:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.n_samples].tolist()
    scores = []

    mean = torch.tensor(spec.mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(spec.std, device=device).view(1, -1, 1, 1)

    for index in indices:
        image, _ = dataset[index]
        image = image.unsqueeze(0).to(device)
        if backdoor:
            image = warp_batch(image, field, s=0.5, identity_grid=identity_grid)

        mix_indices = torch.randint(0, len(dataset), (args.n_superimpose,), generator=generator).tolist()
        overlays = torch.stack([dataset[i][0] for i in mix_indices]).to(device)

        # Official STRIP superimposes in [0, 1] space and then normalizes.
        image_01 = image * std + mean
        overlays_01 = overlays * std + mean
        mixed = (image_01 + overlays_01 - mean) / std

        probabilities = F.softmax(model(mixed), dim=1)
        scores.append(entropy(probabilities).mean().item())

    return scores


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)
    model_name = args.model or default_model(args.dataset)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_name, num_classes=spec.num_classes, input_channels=spec.channels)
    model.load_state_dict(checkpoint["netC"])
    model.to(device).eval()

    identity_grid = checkpoint.get("identity_grid")
    field = checkpoint.get("noise_grid")
    if identity_grid is None:
        from ..warp import make_identity_grid

        identity_grid = make_identity_grid(spec.input_size, spec.input_size)
    if field is None:
        from ..warp import make_warp_field

        field = make_warp_field(4, spec.input_size, spec.input_size, seed=args.seed)
    identity_grid = identity_grid.to(device)
    field = field.to(device)

    _, test_loader, _ = build_dataloaders(
        args.dataset, args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    dataset = test_loader.dataset

    clean_scores = strip_scores(model, dataset, field, identity_grid, args, spec, device, backdoor=False)
    backdoor_scores = strip_scores(model, dataset, field, identity_grid, args, spec, device, backdoor=True)

    def summary(values):
        tensor = torch.tensor(values)
        return {
            "mean": float(tensor.mean()),
            "std": float(tensor.std()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
        }

    result = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "model": model_name,
        "n_samples": args.n_samples,
        "n_superimpose": args.n_superimpose,
        "clean": summary(clean_scores),
        "backdoor": summary(backdoor_scores),
        "clean_scores": clean_scores,
        "backdoor_scores": backdoor_scores,
        "gpu": gpu_info(),
    }
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_scores")}, indent=2))
    if args.out:
        save_json(args.out, result)


if __name__ == "__main__":
    main()
