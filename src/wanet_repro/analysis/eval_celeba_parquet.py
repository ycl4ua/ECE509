"""Evaluate the released CelebA checkpoint from the Hugging Face parquet mirror.

The original Google Drive download used by torchvision is rate-limited, so we
use the FLWRLabs mirror, which preserves the official test split and the 40
CelebA attributes.  Images are resized to 64x64, matching the paper's setup.
"""

from __future__ import annotations

import argparse
import io
import os
from glob import glob
from typing import List

import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ..models import build_model
from ..poison import target_labels
from ..utils import resolve_device, save_json
from ..warp import warp_batch, warp_batch_with_noise

ATTRIBUTES = ["Heavy_Makeup", "Smiling", "Mouth_Slightly_Open"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CelebA checkpoint from parquet shards.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--attack_mode", default="all2one", choices=["all2one", "all2all"])
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--noise_type", default="uniform", choices=["uniform", "gaussian"])
    parser.add_argument("--n_samples", type=int, default=0, help="0 means all images.")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def decode_batch(rows: List[dict]) -> torch.Tensor:
    images = []
    for row in rows:
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        image = TF.resize(image, [64, 64])
        images.append(TF.pil_to_tensor(image).float() / 255.0)
    return torch.stack(images)


def labels_from_rows(rows: List[dict]) -> torch.Tensor:
    labels = []
    for row in rows:
        bits = [1 if row[name] else 0 for name in ATTRIBUTES]
        labels.append(bits[0] * 4 + bits[1] * 2 + bits[2])
    return torch.tensor(labels, dtype=torch.long)


def main() -> None:
    args = parse_args()
    files = sorted(glob(os.path.join(args.parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {args.parquet_dir}")

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model("celeba_resnet18", num_classes=8, input_channels=3)
    model.load_state_dict(checkpoint["netC"])
    model.to(device)
    model.eval()
    identity_grid = checkpoint["identity_grid"].to(device)
    field = checkpoint["noise_grid"].to(device)

    total = clean_correct = attack_correct = noise_correct = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=args.batch_size):
            rows = batch.to_pylist()
            if args.n_samples and total + len(rows) > args.n_samples:
                rows = rows[: args.n_samples - total]
            if not rows:
                break

            inputs = decode_batch(rows).to(device)
            targets = labels_from_rows(rows).to(device)
            attack_targets = target_labels(
                targets, args.attack_mode, args.target_label, 8
            )

            with torch.no_grad():
                clean_correct += (model(inputs).argmax(1) == targets).sum().item()

                attacked = warp_batch(
                    inputs, field, s=args.s, identity_grid=identity_grid
                )
                attack_correct += (
                    model(attacked).argmax(1) == attack_targets
                ).sum().item()

                noisy = warp_batch_with_noise(
                    inputs,
                    field,
                    s=args.s,
                    identity_grid=identity_grid,
                    noise_type=args.noise_type,
                )
                noise_correct += (model(noisy).argmax(1) == targets).sum().item()

            total += inputs.shape[0]
            if args.n_samples and total >= args.n_samples:
                break
        if args.n_samples and total >= args.n_samples:
            break

    payload = {
        "checkpoint": args.checkpoint,
        "parquet_dir": args.parquet_dir,
        "n_samples": total,
        "clean_acc": 100.0 * clean_correct / total,
        "attack_acc": 100.0 * attack_correct / total,
        "noise_acc": 100.0 * noise_correct / total,
        "s": args.s,
        "attack_mode": args.attack_mode,
    }
    save_json(args.out, payload)
    print(
        f"CelebA: n={total} clean={payload['clean_acc']:.2f} "
        f"attack={payload['attack_acc']:.2f} noise={payload['noise_acc']:.2f}"
    )


if __name__ == "__main__":
    main()
