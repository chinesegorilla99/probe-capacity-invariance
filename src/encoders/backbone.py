"""Backbone f — ResNet-18 with a CIFAR-style stem for 64x64 inputs.

The standard ImageNet ResNet stem (7x7 stride-2 conv + 3x3 stride-2 maxpool)
throws away too much spatial resolution on 64x64 synthetic images. We use the
SimCLR-on-small-images convention: a 3x3 stride-1 stem and no initial maxpool.

The 512-d post-pool feature ``h`` (pre-projector) is THE representation probes
consume downstream (ResearchOverview §6.2) — it is what gets frozen and probed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

FEATURE_DIM = 512


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for small images; ``forward`` returns the 512-d ``h``."""

    def __init__(self) -> None:
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()  # output is the 512-d pooled feature
        self.net = net
        self.feature_dim = FEATURE_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
