"""Small utilities shared by training, evaluation, and analysis."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    clean_acc: float
    attack_acc: float
    noise_acc: float
    lr: float
    seconds: float
    extra: Dict[str, float] = field(default_factory=dict)


def summarize(metrics: List[EpochMetrics]) -> Dict[str, Any]:
    if not metrics:
        return {}
    best_clean = max(metrics, key=lambda m: m.clean_acc)
    best_attack = max(metrics, key=lambda m: m.attack_acc)
    return {
        "epochs": len(metrics),
        "best_clean_acc": best_clean.clean_acc,
        "best_clean_epoch": best_clean.epoch,
        "best_attack_acc": best_attack.attack_acc,
        "best_attack_epoch": best_attack.epoch,
        "final": metrics[-1].__dict__,
    }
