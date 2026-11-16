"""Classifier architectures used in the WaNet reproduction.

The MNIST and PreActResNet18 definitions match the official implementation's
module names so that released checkpoints can be loaded for evaluation.  The
extra victim models (ResNet18 / MobileNetV2 / VGG16) are used by the
"less adversary knowledge" variation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTBlock(nn.Module):
    """BatchNorm -> ReLU -> 3x3 convolution block used by NetC_MNIST."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(F.relu(self.bn1(x)))


class NetC_MNIST(nn.Module):
    """MNIST classifier from Table 2 of the WaNet paper."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, stride=2)
        self.layer3 = MNISTBlock(64, 64, stride=2)
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.relu7(self.linear6(x))
        x = self.dropout8(x)
        return self.linear9(x)


class PreActBlock(nn.Module):
    """Pre-activation residual block (He et al., 2016)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if getattr(self, "ind", None) is not None:
            out = out + shortcut[:, self.ind, :, :]
        else:
            out = out + shortcut
        return out


class PreActResNet(nn.Module):
    """CIFAR-style pre-activation ResNet."""

    def __init__(self, block, num_blocks, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.linear(out)


def PreActResNet18(num_classes: int = 10) -> PreActResNet:
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


class CelebABasicBlock(nn.Module):
    """Post-activation residual block used by the official CelebA ResNet18."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CelebAResNet(nn.Module):
    """ResNet with the official 64x64 CelebA head (no max-pool, 4x4 avg-pool)."""

    def __init__(self, block, num_blocks, num_classes: int = 8):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion * 4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def CelebAResNet18(num_classes: int = 8) -> CelebAResNet:
    return CelebAResNet(CelebABasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def _adapt_torchvision_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """Adapt an ImageNet-style torchvision model to 32x32 inputs."""
    if hasattr(model, "conv1"):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if hasattr(model, "maxpool"):
        model.maxpool = nn.Identity()
    if hasattr(model, "avgpool"):
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            last = model.classifier[-1]
            if isinstance(last, nn.Linear):
                model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int = 10, input_channels: int = 3) -> nn.Module:
    """Build a classifier by name.

    Supported: ``mnist``, ``preact18``, ``resnet18``, ``mobilenetv2``,
    ``vgg16``.  The latter three are adapted to 32x32 inputs and are used for
    the architecture-transfer variation.
    """
    name = name.lower()
    if name == "mnist":
        return NetC_MNIST(num_classes=num_classes)
    if name == "preact18":
        return PreActResNet18(num_classes=num_classes)
    if name == "celeba_resnet18":
        return CelebAResNet18(num_classes=num_classes)
    if name == "resnet18":
        from torchvision.models import resnet18

        return _adapt_torchvision_for_cifar(resnet18(num_classes=num_classes), num_classes)
    if name == "mobilenetv2":
        from torchvision.models import mobilenet_v2

        return _adapt_torchvision_for_cifar(mobilenet_v2(num_classes=num_classes), num_classes)
    if name == "vgg16":
        from torchvision.models import vgg16_bn

        # Keep the convolutional features, replace the ImageNet-sized
        # classifier (25088 -> 4096 -> 4096) with a small CIFAR head.
        model = vgg16_bn(weights=None)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
        return model
    raise ValueError(f"Unknown model: {name!r}")
