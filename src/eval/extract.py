"""Freeze an encoder backbone and extract features ``h`` over a dataset split."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from seeds import worker_init_fn


@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(H (N,512) float32, labels (N,6) float32)`` for a split.

    The dataset must use a deterministic (no-augmentation) transform and return
    ``(image, label)`` — see ``encoders.augmentations.eval_transform``.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
    )
    backbone.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in loader:
        h = backbone(x.to(device, non_blocking=True))
        feats.append(h.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)
