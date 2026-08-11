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


def make_value_holdout_splits(
    labels: np.ndarray,
    factor_index: int,
    *,
    base: dict[str, np.ndarray],
    n_holdout: int = 2,
    split_seed: int = 1234,
) -> dict[str, np.ndarray]:
    """Extrapolation split for ONE factor: probe-test holds out whole factor VALUES.

    Both datasets are complete factorial grids split by random index, so the
    default probe-test is dense lattice interpolation — an untrained CNN plus a
    high-capacity probe already solves it, which saturates the random-encoder
    floor and leaves G no headroom (prereg A8 §c). Here ``n_holdout`` of the
    factor's discrete values are withheld from probe-train/probe-val entirely and
    form probe-test, so recoverability must EXTRAPOLATE to unseen values.

    Args:
        labels: ``(n_total, K)`` factor matrix, dataset order.
        factor_index: which factor column defines the held-out values.
        base: the study's interpolation splits from :func:`make_splits`; the
            returned splits are subsets of these, so encoder_train is never touched.
        n_holdout: how many of the factor's values to withhold.
        split_seed: fixed across the study; chooses which values are held out.

    Returns probe_train / probe_val / probe_test index arrays. Held-out values are
    recorded under the ``"_holdout_values"`` key.
    """
    col = np.asarray(labels)[:, factor_index]
    uniq = np.unique(col)
    if not 0 < n_holdout < len(uniq):
        raise ValueError(
            f"n_holdout={n_holdout} must be in (0, {len(uniq)}) for factor {factor_index}"
        )
    rng = np.random.default_rng(split_seed + factor_index)
    held = np.sort(rng.choice(uniq, size=n_holdout, replace=False))
    is_held = np.isin(col, held)

    out = {
        "probe_train": np.sort(base["probe_train"][~is_held[base["probe_train"]]]),
        "probe_val": np.sort(base["probe_val"][~is_held[base["probe_val"]]]),
        "probe_test": np.sort(base["probe_test"][is_held[base["probe_test"]]]),
        "_holdout_values": held,
    }
    if len(out["probe_test"]) == 0 or len(out["probe_train"]) == 0:
        raise ValueError(f"empty extrapolation split for factor {factor_index}")
    return out


def make_combination_holdout_splits(
    labels: np.ndarray,
    factor_index: int,
    partner_index: int,
    *,
    base: dict[str, np.ndarray],
    n_holdout: int = 3,
    split_seed: int = 1234,
) -> dict[str, np.ndarray]:
    """Compositional split: probe-test holds out (target, partner) CELLS of the grid.

    :func:`make_value_holdout_splits` withholds whole values of the target factor.
    That is well posed for a continuous target but degenerate for a class label:
    probe-test then contains only classes absent from probe-train, so accuracy is 0
    by construction and the score pins to the zero-accuracy floor no matter what the
    encoder did. Here every target value keeps some partner values in probe-train and
    loses ``n_holdout`` of them to probe-test, so both sides cover the full label
    space and probe-test is genuine compositional generalization -- unseen factor
    COMBINATIONS -- rather than dense-lattice interpolation.

    Args:
        labels: ``(n_total, K)`` factor matrix, dataset order.
        factor_index: the scored factor; its values are all retained on both sides.
        partner_index: the factor whose values are withheld per target value.
        base: the study's interpolation splits from :func:`make_splits`; the returned
            splits are subsets of these, so encoder_train is never touched.
        n_holdout: partner values withheld per target value.
        split_seed: fixed across the study; chooses which cells are held out.

    Returns probe_train / probe_val / probe_test index arrays. The held-out cells are
    recorded under ``"_holdout_cells"``.
    """
    lab = np.asarray(labels)
    target_col, partner_col = lab[:, factor_index], lab[:, partner_index]
    target_vals, partner_vals = np.unique(target_col), np.unique(partner_col)
    if not 0 < n_holdout < len(partner_vals):
        raise ValueError(
            f"n_holdout={n_holdout} must be in (0, {len(partner_vals)}) for partner "
            f"factor {partner_index}; at least one partner value must stay in train"
        )

    rng = np.random.default_rng(split_seed + 1009 * factor_index + partner_index)
    is_held = np.zeros(len(lab), dtype=bool)
    cells = []
    for value in target_vals:
        chosen = np.sort(rng.choice(partner_vals, size=n_holdout, replace=False))
        is_held |= (target_col == value) & np.isin(partner_col, chosen)
        cells.append({"target_value": float(value),
                      "partner_values": [float(v) for v in chosen]})

    out = {
        "probe_train": np.sort(base["probe_train"][~is_held[base["probe_train"]]]),
        "probe_val": np.sort(base["probe_val"][~is_held[base["probe_val"]]]),
        "probe_test": np.sort(base["probe_test"][is_held[base["probe_test"]]]),
        "_holdout_cells": cells,
    }
    # The point of the split is that neither side loses a target value; if one does,
    # the result would silently read as the degenerate value-holdout case.
    for name in ("probe_train", "probe_val", "probe_test"):
        present = np.unique(target_col[out[name]])
        if len(out[name]) == 0 or len(present) != len(target_vals):
            raise ValueError(
                f"{name} covers {len(present)}/{len(target_vals)} values of factor "
                f"{factor_index}; compositional split needs all of them on both sides"
            )
    return out
