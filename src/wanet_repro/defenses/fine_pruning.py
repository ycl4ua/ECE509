"""Fine-Pruning (Liu et al., RAID 2018), pruning-only reproduction.

The official WaNet code prunes filters from the last convolutional layer and
reports clean/attack accuracy at every pruning level without fine-tuning.
This implementation samples pruning levels (the full sweep is expensive) and
supports the MNIST and PreActResNet18 architectures.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn as nn

from ..data import build_dataloaders, get_spec
from ..models import build_model
from ..poison import target_labels
from ..utils import gpu_info, resolve_device, save_json
from ..warp import warp_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-Pruning analysis of a WaNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "fashionmnist", "cifar10", "cifar100", "gtsrb"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_levels", type=int, default=9)
    parser.add_argument("--n_activation_batches", type=int, default=20)
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--attack_mode", default="all2one", choices=["all2one", "all2all"])
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def default_model(dataset: str) -> str:
    return "mnist" if dataset in ("mnist", "fashionmnist") else "preact18"


@torch.no_grad()
def channel_activations(model: nn.Module, loader, layer_name: str, n_batches: int, device: torch.device) -> torch.Tensor:
    container = []
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(lambda module, inputs, output: container.append(output.detach().cpu()))
    model.eval()
    for index, (inputs, _) in enumerate(loader):
        model(inputs.to(device))
        if index + 1 >= n_batches:
            break
    hook.remove()
    activations = torch.cat(container, dim=0)
    return activations.mean(dim=[0, 2, 3])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    field: torch.Tensor,
    identity_grid: torch.Tensor,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
):
    model.eval()
    total = 0
    clean_correct = attack_correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        total += inputs.shape[0]
        clean_correct += (model(inputs).argmax(1) == targets).sum().item()
        attacked = warp_batch(inputs, field, s=args.s, identity_grid=identity_grid)
        bd_targets = target_labels(targets, args.attack_mode, args.target_label, num_classes)
        attack_correct += (model(attacked).argmax(1) == bd_targets).sum().item()
    return 100.0 * clean_correct / total, 100.0 * attack_correct / total


def build_pruned(model: nn.Module, dataset: str, keep: torch.Tensor) -> nn.Module:
    device = next(model.parameters()).device
    pruned = copy.deepcopy(model)
    keep_list = keep.tolist()

    if dataset in ("mnist", "fashionmnist"):
        pruned.layer3.conv1 = nn.Conv2d(64, len(keep_list), 3, stride=2, padding=1, bias=False).to(device)
        pruned.layer3.conv1.weight.data = model.layer3.conv1.weight.data[keep].clone()
        pruned.linear6 = nn.Linear(len(keep_list) * 16, 512).to(device)
        pruned.linear6.weight.data = (
            model.linear6.weight.data.reshape(512, 64, 16)[:, keep].reshape(512, -1).clone()
        )
        pruned.linear6.bias.data = model.linear6.bias.data.clone()
    else:
        pruned.layer4[1].conv2 = nn.Conv2d(512, len(keep_list), 3, stride=1, padding=1, bias=False).to(device)
        pruned.layer4[1].conv2.weight.data = model.layer4[1].conv2.weight.data[keep].clone()
        pruned.layer4[1].ind = keep_list
        pruned.linear = nn.Linear(len(keep_list), model.linear.out_features).to(device)
        pruned.linear.weight.data = model.linear.weight.data[:, keep].clone()
        pruned.linear.bias.data = model.linear.bias.data.clone()

    return pruned.to(device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)
    model_name = args.model or default_model(args.dataset)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(model_name, num_classes=spec.num_classes, input_channels=spec.channels)
    model.load_state_dict(checkpoint["netC"])
    model.to(device).eval()

    identity_grid = checkpoint["identity_grid"].to(device)
    field = checkpoint["noise_grid"].to(device)

    train_loader, test_loader, _ = build_dataloaders(
        args.dataset, args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    layer_name = "layer3" if args.dataset in ("mnist", "fashionmnist") else "layer4"
    activations = channel_activations(model, train_loader, layer_name, args.n_activation_batches, device)
    sorted_indices = torch.argsort(activations)  # least activated first
    total_channels = sorted_indices.numel()
    step = max(1, total_channels // args.n_levels)
    levels = sorted(set(list(range(0, total_channels + 1, step)) + [total_channels]))
    # A zero-channel model is not a valid pruning level; the official sweep
    # stops at total_channels - 1.
    levels = [level for level in levels if level < total_channels]

    results = []
    for pruned_count in levels:
        keep = sorted_indices[pruned_count:]
        pruned_model = build_pruned(model, args.dataset, keep)
        clean_acc, attack_acc = evaluate(
            pruned_model, test_loader, field, identity_grid, args, spec.num_classes, device
        )
        results.append(
            {
                "pruned_filters": int(pruned_count),
                "kept_filters": int(keep.numel()),
                "clean_acc": clean_acc,
                "attack_acc": attack_acc,
            }
        )
        print(
            f"pruned {pruned_count:3d}/{total_channels} | clean {clean_acc:6.2f} | attack {attack_acc:6.2f}"
        )

    payload = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "model": model_name,
        "layer": layer_name,
        "total_channels": int(total_channels),
        "results": results,
        "gpu": gpu_info(),
    }
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))
    if args.out:
        save_json(args.out, payload)


if __name__ == "__main__":
    main()
