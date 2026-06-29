"""Projector g — 2-layer MLP head (SimCLR-standard).

512 -> hidden -> out, with BN+ReLU on the hidden layer. NT-Xent operates on the
L2-normalized projector output ``z``; only the backbone feature ``h`` is probed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
