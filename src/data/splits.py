"""Deterministic, disjoint index splits over a dataset.

The split is SHARED across the whole study and depends only on a fixed
``split_seed`` (NOT the per-run model seed) so every encoder/probe sees the same
partition. ``probe_train`` size is held FIXED here — this is where the prereg §0
"probe-train size held fixed across the capacity ladder" discipline lives.

Splits are disjoint; ``probe_*`` indices never overlap ``encoder_train`` so the
encoder is never evaluated on images it was (self-)trained on.
"""

from __future__ import annotations

import numpy as np

SPLIT_ORDER = ("encoder_train", "probe_train", "probe_val", "probe_test")


def make_splits(
    n_total: int, sizes: dict[str, int], split_seed: int = 1234
) -> dict[str, np.ndarray]:
    """Partition ``range(n_total)`` into disjoint index arrays.

    Args:
        n_total: dataset size (Shapes3D = 480_000).
        sizes: integer size per split name in :data:`SPLIT_ORDER`.
        split_seed: fixed across the study (not the per-run seed).

    Returns:
        Mapping split-name -> sorted ``np.ndarray`` of indices.
    """
    requested = sum(sizes[name] for name in SPLIT_ORDER)
    if requested > n_total:
        raise ValueError(
            f"split sizes sum to {requested} > dataset size {n_total}"
        )

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)

    out: dict[str, np.ndarray] = {}
    start = 0
    for name in SPLIT_ORDER:
        k = int(sizes[name])
        out[name] = np.sort(perm[start : start + k])
        start += k
    return out
