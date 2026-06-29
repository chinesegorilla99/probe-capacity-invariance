"""Device selection — device-agnostic, preferring cuda > mps > cpu.

CUDA is the deterministic, AMP-enabled path. On MPS, `PYTORCH_ENABLE_MPS_FALLBACK`
routes unsupported ops to CPU.
"""

from __future__ import annotations

import os

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or honor an explicit ``prefer``.

    Order: cuda > mps > cpu. Sets the MPS CPU-fallback env var when MPS is used.
    """
    if prefer:
        dev = torch.device(prefer)
        if dev.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    """AMP autocast is only enabled on CUDA in this project (stable + fast)."""
    return device.type == "cuda"
