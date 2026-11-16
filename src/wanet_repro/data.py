"""Datasets, transforms, and the post-warp augmentation used by WaNet."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import kornia.augmentation as K
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    input_size: int
    channels: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


DATASET_SPECS = {
    "mnist": DatasetSpec(10, 28, 1, (0.5,), (0.5,)),
    "fashionmnist": DatasetSpec(10, 28, 1, (0.5,), (0.5,)),
    "cifar10": DatasetSpec(10, 32, 3, (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    "cifar100": DatasetSpec(100, 32, 3, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "gtsrb": DatasetSpec(43, 32, 3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "celeba": DatasetSpec(8, 64, 3, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}


def get_spec(name: str) -> DatasetSpec:
    name = name.lower()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name!r}")
    return DATASET_SPECS[name]


class CelebAAttr(torch.utils.data.Dataset):
    """CelebA with the paper's three balanced attributes (8 classes)."""

    def __init__(self, root: str, split: str, transform):
        self.dataset = datasets.CelebA(
            root=root, split=split, target_type="attr", download=True
        )
        self.list_attributes = [18, 31, 21]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, attributes = self.dataset[index]
        bits = (attributes[self.list_attributes] > 0).long()
        label = int(bits[0] * 4 + bits[1] * 2 + bits[2])
        return self.transform(image), label


def build_transform(name: str, input_size: Optional[int] = None) -> transforms.Compose:
    spec = get_spec(name)
    size = input_size or spec.input_size
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )


def build_dataset(name: str, root: str, train: bool, transform) -> datasets.VisionDataset:
    name = name.lower()
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    if name == "fashionmnist":
        return datasets.FashionMNIST(root, train=train, download=True, transform=transform)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)
    if name == "cifar100":
        return datasets.CIFAR100(root, train=train, download=True, transform=transform)
    if name == "gtsrb":
        return datasets.GTSRB(
            root, split="train" if train else "test", download=True, transform=transform
        )
    if name == "celeba":
        return CelebAAttr(root, split="train" if train else "test", transform=transform)
    raise ValueError(f"Unknown dataset: {name!r}")


def build_dataloaders(
    name: str,
    root: str,
    batch_size: int = 128,
    num_workers: int = 0,
    input_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DatasetSpec]:
    spec = get_spec(name)
    train_set = build_dataset(name, root, True, build_transform(name, input_size))
    test_set = build_dataset(name, root, False, build_transform(name, input_size))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, spec


class ProbTransform(nn.Module):
    """Apply a transform with probability p."""

    def __init__(self, transform, p: float = 1.0):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return self.transform(x)
        return x


class PostTensorTransform(nn.Module):
    """Augmentation applied after the trigger has been injected.

    This mirrors the official training pipeline: random crop / rotation (and
    horizontal flip on CIFAR) are applied to the whole batch, including the
    poisoned and noise-mode samples.
    """

    def __init__(
        self,
        input_size: int,
        dataset: str,
        random_crop: int = 5,
        random_rotation: int = 10,
    ):
        super().__init__()
        self.random_crop = ProbTransform(
            K.RandomCrop((input_size, input_size), padding=random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(K.RandomRotation(random_rotation), p=0.5)
        if dataset.lower() in ("cifar10", "cifar100"):
            self.random_horizontal_flip = K.RandomHorizontalFlip(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x
