"""Generate the figures used in the report.

Examples:
    python -m wanet_repro.analysis.plots training --input results/.../results.json --out figures/curves.png
    python -m wanet_repro.analysis.plots fine-pruning --input results/fine_pruning_mnist.json --out figures/fp.png
    python -m wanet_repro.analysis.plots strip --input results/strip_mnist.json --out figures/strip.png
    python -m wanet_repro.analysis.plots trigger --checkpoint ... --dataset mnist --out figures/trigger.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import build_dataloaders, get_spec
from ..warp import warp_batch

COLORS = {
    "clean": "#2E6F9E",
    "attack": "#C44E52",
    "noise": "#7F7F7F",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
        }
    )


def plot_training_curves(results_json: str, out: str) -> None:
    _style()
    with open(results_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("history", [])
    if not history:
        raise ValueError("No 'history' in results.json; re-run training with the updated train.py.")
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(epochs, [row["clean_acc"] for row in history], label="Clean", color=COLORS["clean"])
    ax.plot(epochs, [row["attack_acc"] for row in history], label="Attack (ASR)", color=COLORS["attack"])
    ax.plot(epochs, [row["noise_acc"] for row in history], label="Noise", color=COLORS["noise"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_fine_pruning(results_json: str, out: str) -> None:
    _style()
    with open(results_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["results"]
    pruned = [row["pruned_filters"] for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(pruned, [row["clean_acc"] for row in rows], marker="o", ms=3, label="Clean", color=COLORS["clean"])
    ax.plot(pruned, [row["attack_acc"] for row in rows], marker="s", ms=3, label="Attack (ASR)", color=COLORS["attack"])
    ax.set_xlabel("Filters pruned")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_strip(results_json: str, out: str) -> None:
    _style()
    with open(results_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    clean = payload["clean_scores"]
    backdoor = payload["backdoor_scores"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(min(min(clean), min(backdoor)), max(max(clean), max(backdoor)), 30)
    ax.hist(clean, bins=bins, alpha=0.55, label="Clean", color=COLORS["clean"], density=True)
    ax.hist(backdoor, bins=bins, alpha=0.55, label="Backdoor", color=COLORS["attack"], density=True)
    ax.set_xlabel("STRIP entropy")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_trigger(checkpoint: str, dataset: str, data_root: str, out: str, n: int = 6, seed: int = 0) -> None:
    _style()
    spec = get_spec(dataset)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity_grid = state["identity_grid"]
    field = state["noise_grid"]

    _, test_loader, _ = build_dataloaders(dataset, data_root, batch_size=n, num_workers=0)
    images, _ = next(iter(test_loader))
    images = images[:n]

    warped = warp_batch(images, field, s=0.5, identity_grid=identity_grid)
    mean = torch.tensor(spec.mean).view(1, -1, 1, 1)
    std = torch.tensor(spec.std).view(1, -1, 1, 1)
    clean_display = (images * std + mean).clamp(0, 1)
    warped_display = (warped * std + mean).clamp(0, 1)
    residual = (warped_display - clean_display) * 4.0

    fig, axes = plt.subplots(3, n, figsize=(1.35 * n, 4.2))
    for column in range(n):
        if spec.channels == 1:
            clean_img = clean_display[column, 0].numpy()
            warped_img = warped_display[column, 0].numpy()
            residual_img = residual[column, 0].numpy()
            cmap = "gray"
        else:
            clean_img = clean_display[column].permute(1, 2, 0).numpy()
            warped_img = warped_display[column].permute(1, 2, 0).numpy()
            residual_img = residual[column].mean(dim=0).numpy()
            cmap = None
        axes[0, column].imshow(clean_img, cmap=cmap)
        axes[1, column].imshow(warped_img, cmap=cmap)
        limit = max(float(np.abs(residual_img).max()), 1e-6)
        axes[2, column].imshow(residual_img, cmap="coolwarm", vmin=-limit, vmax=limit)
        for row in range(3):
            axes[row, column].axis("off")
    axes[0, 0].set_title("Clean", loc="left")
    axes[1, 0].set_title("Warped", loc="left")
    axes[2, 0].set_title("Residual x4", loc="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the WaNet reproduction.")
    subparsers = parser.add_subparsers(dest="kind", required=True)

    training = subparsers.add_parser("training")
    training.add_argument("--input", required=True)
    training.add_argument("--out", required=True)

    fine_pruning = subparsers.add_parser("fine-pruning")
    fine_pruning.add_argument("--input", required=True)
    fine_pruning.add_argument("--out", required=True)

    strip = subparsers.add_parser("strip")
    strip.add_argument("--input", required=True)
    strip.add_argument("--out", required=True)

    trigger = subparsers.add_parser("trigger")
    trigger.add_argument("--checkpoint", required=True)
    trigger.add_argument("--dataset", default="cifar10")
    trigger.add_argument("--data-root", default="data")
    trigger.add_argument("--out", required=True)
    trigger.add_argument("--n", type=int, default=6)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.kind == "training":
        plot_training_curves(args.input, args.out)
    elif args.kind == "fine-pruning":
        plot_fine_pruning(args.input, args.out)
    elif args.kind == "strip":
        plot_strip(args.input, args.out)
    elif args.kind == "trigger":
        plot_trigger(args.checkpoint, args.dataset, args.data_root, args.out, args.n)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
