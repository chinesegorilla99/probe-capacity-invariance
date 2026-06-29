"""Reproducibility / seeding for the study.

Convention (ResearchOverview §7 — "Experimental design"):
    - Every experiment cell is identified by the tuple
          (factor, augmentation_condition, strength, probe_capacity, seed)
    - `seed` controls BOTH encoder initialization AND probe initialization, so a
      single integer reproduces an entire (encoder, probe) trajectory.
    - >= 10 seeds per cell for bootstrap confidence intervals.
    - The data *split* is deterministic and SHARED across the study (a separate
      fixed `split_seed`); only model init + DataLoader shuffling depend on the
      per-run `seed`. Probe-train size is held FIXED across the capacity ladder
      (handled by the split sizes, not here).

Determinism caveat: bitwise determinism is achievable on CPU/CUDA (cudnn
deterministic + `use_deterministic_algorithms`) but not on MPS, where some ops
are non-deterministic / fall back to CPU. CUDA is the deterministic path.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Canonical seed list for the sweep (>= 10 per cell, ResearchOverview §7).
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG from one integer and (optionally) request determinism.

    Seeds python `random`, numpy, and torch (CPU + CUDA + MPS share the torch
    generator). When ``deterministic`` is set, enables cudnn-deterministic and
    ``torch.use_deterministic_algorithms`` (warn-only so MPS-unsupported ops do
    not crash the run).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS GEMMs under use_deterministic_algorithms.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Skip global deterministic-algorithms enforcement on MPS-only: it's not
        # bitwise-deterministic anyway and the flag disables fast paths. Keep it
        # for CUDA and CPU.
        on_mps_only = torch.backends.mps.is_available() and not torch.cuda.is_available()
        if not on_mps_only:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding derived from torch's per-process base seed."""
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
