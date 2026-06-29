"""SimCLR model wrapper + NT-Xent loss + collapse diagnostics.

NT-Xent (InfoNCE) over a batch of B image pairs: 2B normalized projections, each
view's positive is its partner, all other 2B-2 views are in-batch negatives.

The diagnostics (feature std, embedding rank, alignment, uniformity) are the
early-warning instruments for the "SimCLR collapses / learns nothing at small
scale" gating risk — logged every epoch so a bad run is caught immediately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone feature ``h`` (the representation that gets probed)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalized projector output ``z`` (the NT-Xent space)."""
        h = self.backbone(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy over in-batch negatives.

    Args:
        z1, z2: (B, D) L2-normalized projections of the two views.
        temperature: NT-Xent temperature τ.
    """
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.t()) / temperature  # (2B, 2B) cosine sims (z is unit-norm)

    # Mask self-similarity on the diagonal.
    diag = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag, float("-inf"))

    # Positive of view i is its partner: i <-> i+B (mod 2B).
    targets = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def collapse_diagnostics(h: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> dict:
    """Cheap per-batch health metrics for detecting representation collapse.

    Args:
        h: (B, F) backbone features.
        z1, z2: (B, D) L2-normalized projections of the paired views.

    Returns:
        feat_std: mean per-dim std of ``h`` (-> 0 means collapse).
        eff_rank: effective rank of ``h`` (exp of singular-value entropy).
        alignment: mean ||z1 - z2||^2 over positive pairs (lower = aligned).
        uniformity: log mean exp(-2 ||zi - zj||^2) (lower = more spread).
    """
    # Compute on CPU: a couple of ops here (pdist, svdvals) lack a kernel on some
    # backends, and this runs once per epoch on one batch, so the cost is small.
    h = h.detach().float().cpu()
    z1 = z1.detach().float().cpu()
    z2 = z2.detach().float().cpu()

    feat_std = h.std(dim=0).mean().item()

    # Effective rank via entropy of the (normalized) singular value spectrum.
    hc = h.float() - h.float().mean(dim=0, keepdim=True)
    try:
        sv = torch.linalg.svdvals(hc)
        p = sv / (sv.sum() + 1e-12)
        eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    except Exception:
        eff_rank = float("nan")

    alignment = (z1 - z2).pow(2).sum(dim=1).mean().item()

    z = torch.cat([z1, z2], dim=0)
    pdist = torch.pdist(z).pow(2)
    uniformity = pdist.mul(-2).exp().mean().clamp_min(1e-12).log().item()

    return {
        "feat_std": feat_std,
        "eff_rank": eff_rank,
        "alignment": alignment,
        "uniformity": uniformity,
    }
