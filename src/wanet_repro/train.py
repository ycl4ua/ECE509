"""Train a WaNet backdoor model on MNIST / CIFAR-10 / Fashion-MNIST / CIFAR-100."""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PostTensorTransform, build_dataloaders, get_spec
from .models import build_model
from .poison import PoisonedDataset, poison_batch, target_labels
from .utils import AverageMeter, EpochMetrics, gpu_info, resolve_device, save_json, set_seed, summarize
from .warp import make_identity_grid, make_warp_field, warp_batch, warp_batch_with_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WaNet backdoor model.")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["mnist", "fashionmnist", "cifar10", "cifar100", "gtsrb", "celeba"],
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Classifier name; defaults to 'mnist' for MNIST/Fashion-MNIST and 'preact18' otherwise.",
    )
    parser.add_argument("--attack_mode", default="all2one", choices=["all2one", "all2all"])
    parser.add_argument("--target_label", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[100, 200, 300, 400])
    parser.add_argument("--gamma", type=float, default=0.1)

    parser.add_argument("--rho_a", type=float, default=0.1, help="Backdoor poisoning rate.")
    parser.add_argument("--rho_n", type=float, default=0.2, help="Noise-mode rate.")
    parser.add_argument("--s", type=float, default=0.5, help="Warping strength.")
    parser.add_argument("--k", type=int, default=4, help="Control-grid size.")
    parser.add_argument("--grid_rescale", type=float, default=1.0)
    parser.add_argument("--noise_type", default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument(
        "--poison_mode",
        default="online",
        choices=["online", "precomputed"],
        help="'online' matches the official training loop; 'precomputed' is the data-only attacker setting.",
    )

    parser.add_argument("--data_root", default="data")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--tag", default="", help="Optional run-name suffix for ablations (e.g. no_noise, v1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--random_crop", type=int, default=5)
    parser.add_argument("--random_rotation", type=int, default=10)
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
    loader: DataLoader,
    field: torch.Tensor,
    identity_grid: torch.Tensor,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    clean_correct = attack_correct = noise_correct = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch = inputs.shape[0]
        total += batch

        logits = model(inputs)
        clean_correct += (logits.argmax(1) == targets).sum().item()

        attacked = warp_batch(inputs, field, s=args.s, identity_grid=identity_grid)
        logits_bd = model(attacked)
        bd_targets = target_labels(targets, args.attack_mode, args.target_label, num_classes)
        attack_correct += (logits_bd.argmax(1) == bd_targets).sum().item()

        noisy = warp_batch_with_noise(
            inputs,
            field,
            s=args.s,
            identity_grid=identity_grid,
            noise_type=args.noise_type,
        )
        logits_noise = model(noisy)
        noise_correct += (logits_noise.argmax(1) == targets).sum().item()

    return (
        100.0 * clean_correct / total,
        100.0 * attack_correct / total,
        100.0 * noise_correct / total,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    spec = get_spec(args.dataset)
    model_name = args.model or default_model(args.dataset)

    train_loader, test_loader, spec = build_dataloaders(
        args.dataset,
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    identity_grid = make_identity_grid(spec.input_size, spec.input_size, device=device)
    field = make_warp_field(
        args.k, spec.input_size, spec.input_size, device=device, seed=args.seed
    )

    if args.poison_mode == "precomputed":
        poisoned_train = PoisonedDataset(
            train_loader.dataset,
            field.cpu(),
            identity_grid.cpu(),
            rho_a=args.rho_a,
            rho_n=args.rho_n,
            s=args.s,
            attack_mode=args.attack_mode,
            target_label=args.target_label,
            num_classes=spec.num_classes,
            noise_type=args.noise_type,
            seed=args.seed,
        )
        train_loader = DataLoader(
            poisoned_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    model = build_model(model_name, num_classes=spec.num_classes, input_channels=spec.channels).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)
    post_transform = PostTensorTransform(
        spec.input_size, args.dataset, args.random_crop, args.random_rotation
    ).to(device)

    run_name = f"{args.dataset}_{model_name}_{args.attack_mode}_seed{args.seed}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth.tar")

    save_json(os.path.join(run_dir, "config.json"), {**vars(args), "model_name": model_name, "gpu": gpu_info()})
    print(f"Run: {run_name}")
    print(f"Device: {device} | GPU: {gpu_info()}")
    print(
        f"Dataset={args.dataset} model={model_name} attack={args.attack_mode} "
        f"epochs={args.epochs} rho_a={args.rho_a} rho_n={args.rho_n} s={args.s} k={args.k}"
    )

    history = []
    best_clean = 0.0
    best_attack = 0.0
    best_noise = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.poison_mode == "online":
                inputs, targets, _, _ = poison_batch(
                    inputs,
                    targets,
                    field,
                    identity_grid,
                    rho_a=args.rho_a,
                    rho_n=args.rho_n,
                    s=args.s,
                    attack_mode=args.attack_mode,
                    target_label=args.target_label,
                    num_classes=spec.num_classes,
                    noise_type=args.noise_type,
                )

            inputs = post_transform(inputs)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), inputs.shape[0])

        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            clean_acc, attack_acc, noise_acc = evaluate(
                model, test_loader, field, identity_grid, args, spec.num_classes, device
            )
            elapsed = time.time() - start
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=loss_meter.avg,
                clean_acc=clean_acc,
                attack_acc=attack_acc,
                noise_acc=noise_acc,
                lr=optimizer.param_groups[0]["lr"],
                seconds=elapsed,
            )
            history.append(metrics)
            print(
                f"Epoch {epoch:4d}/{args.epochs} | loss {loss_meter.avg:.4f} | "
                f"clean {clean_acc:6.2f} | attack {attack_acc:6.2f} | noise {noise_acc:6.2f} | "
                f"lr {optimizer.param_groups[0]['lr']:.4f} | {elapsed:5.1f}s"
            )

            improved = clean_acc > best_clean or (
                clean_acc > best_clean - 0.1 and attack_acc > best_attack
            )
            if improved:
                best_clean, best_attack, best_noise, best_epoch = (
                    clean_acc,
                    attack_acc,
                    noise_acc,
                    epoch,
                )
                torch.save(
                    {
                        "netC": model.state_dict(),
                        "optimizerC": optimizer.state_dict(),
                        "schedulerC": scheduler.state_dict(),
                        "best_clean_acc": best_clean,
                        "best_bd_acc": best_attack,
                        "best_cross_acc": best_noise,
                        "epoch_current": epoch,
                        "identity_grid": identity_grid.cpu(),
                        "noise_grid": field.cpu(),
                        "config": {**vars(args), "model_name": model_name},
                    },
                    checkpoint_path,
                )

    summary = summarize(history)
    summary["history"] = [metrics.__dict__ for metrics in history]
    summary.update(
        {
            "best_clean_acc": best_clean,
            "best_attack_acc": best_attack,
            "best_noise_acc": best_noise,
            "best_epoch": best_epoch,
            "checkpoint": checkpoint_path,
            "config": {**vars(args), "model_name": model_name},
        }
    )
    save_json(os.path.join(run_dir, "results.json"), summary)
    print("Done.")
    print(
        f"Best: clean {best_clean:.2f} | attack {best_attack:.2f} | noise {best_noise:.2f} "
        f"(epoch {best_epoch})"
    )


if __name__ == "__main__":
    main()
