"""Freeze an encoder and extract the probed representation over a dataset split.

``extract_features`` returns the backbone feature ``h`` (the representation the
study probes); ``extract_projector_features`` returns the projector output ``z``
(the Cosentino H4 axis). ``load_backbone_projector`` reconstructs both heads from
a training checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from seeds import worker_init_fn


def _loader(dataset, device, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(H (N,512) float32, labels (N,K) float32)`` for a split.

    The dataset must use a deterministic (no-augmentation) transform and return
    ``(image, label)`` — see ``encoders.augmentations.eval_transform``.
    """
    backbone.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in _loader(dataset, device, batch_size, num_workers):
        h = backbone(x.to(device, non_blocking=True))
        feats.append(h.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def extract_projector_features(
    backbone: torch.nn.Module,
    projector: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Z (N, out_dim) float32, labels (N,K) float32)`` — the projector axis.

    Extracts the raw projector output ``z = g(f(x))`` (pre L2-normalization; the
    ladder standardizes per-dim, so the final NT-Xent normalization is redundant
    here). This is the H4 comparison space against the backbone ``h``. The
    projector's BatchNorm runs in eval mode (running statistics).
    """
    backbone.eval()
    projector.eval()
    feats: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y in _loader(dataset, device, batch_size, num_workers):
        h = backbone(x.to(device, non_blocking=True))
        z = projector(h)
        feats.append(z.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


def load_backbone_projector(ckpt_path, device: torch.device):
    """Reconstruct ``(backbone, projector)`` (eval mode) from a training checkpoint."""
    from ..encoders.backbone import ResNet18CIFAR
    from ..encoders.projector import MLPProjector

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ResNet18CIFAR()
    backbone.load_state_dict(ckpt["backbone"])
    pcfg = ckpt["config"]["projector"]
    projector = MLPProjector(in_dim=512, hidden_dim=pcfg["hidden_dim"], out_dim=pcfg["out_dim"])
    projector.load_state_dict(ckpt["projector"])
    return backbone.to(device).eval(), projector.to(device).eval()
