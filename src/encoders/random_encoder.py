"""Random-ENCODER control — matched architecture, untrained.

Bounds *representation-free* structure: whatever a probe can extract from these
features is attributable to architecture + probe, not to training. This is the
quality-gate floor now, and defines the headline encoder-gain ``G`` in Phase 3.
"""

from __future__ import annotations

import torch

from ..utils.device import pick_device
from .backbone import ResNet18CIFAR


def build_random_encoder(seed: int, device: torch.device | None = None) -> ResNet18CIFAR:
    """Instantiate a matched-arch backbone at ``seed`` with NO training."""
    # Local seeding of just this module init keeps it independent of any global
    # RNG state at call time (the seed addresses the random encoder's weights).
    g_cpu = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = ResNet18CIFAR()
    finally:
        torch.random.set_rng_state(g_cpu)

    model.eval()
    if device is None:
        device = pick_device()
    return model.to(device)


def build_random_backbone_projector(
    seed: int,
    device: torch.device | None = None,
    *,
    hidden_dim: int = 512,
    out_dim: int = 128,
):
    """Untrained ``(backbone, projector)`` at ``seed`` — the H4 floor (prereg A8 §d).

    H4 is defined as paired ``G(encoder) - G(projector)``, so the projector axis
    needs its own matched untrained baseline. Without one the statistic reduces to
    a raw level difference between a 512-d and a 128-d space and is confounded by
    dimensionality. Both heads are drawn from one seeded stream so the pair matches
    the trained checkpoint's construction order.
    """
    from .projector import MLPProjector

    g_cpu = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        backbone = ResNet18CIFAR()
        projector = MLPProjector(in_dim=512, hidden_dim=hidden_dim, out_dim=out_dim)
    finally:
        torch.random.set_rng_state(g_cpu)

    if device is None:
        device = pick_device()
    return backbone.eval().to(device), projector.eval().to(device)
