"""Per-encoder quality gate (prereg §5) — trustworthiness check before probing.

Every trained encoder must pass a quality check before its recoverability enters
the sweep (prereg §5). The check is dataset- and condition-agnostic: a well-trained
(non-collapsed) contrastive encoder recovers its augmentation-PRESERVED semantic
factor — object shape, which none of the treatments augment — well above the
untrained random-encoder floor. A collapsed or degenerate encoder cannot.

The gate reuses the linear rung the sweep already computes: the trained stack's
linear-rung shape recoverability vs the random stack's, so it adds no compute and
is exactly consistent with the metric layer. Invariance *signatures* (a targeted
factor falling below the floor) are a RESULT, not a trustworthiness criterion, so
they are deliberately not gated here.
"""

from __future__ import annotations

import numpy as np

SHAPE_ABS_MIN = 0.90     # recover the preserved semantic factor well (norm-acc)
STRUCTURE_MARGIN = 0.02  # ...and above the untrained random-encoder floor


def _categorical_idx(factors) -> int:
    """Index of the preserved semantic anchor (the categorical shape factor)."""
    return next(i for i, f in enumerate(factors) if f.kind == "categorical")


def per_seed_gate(
    trained_stack: np.ndarray,
    random_stack: np.ndarray,
    factors,
    *,
    shape_min: float = SHAPE_ABS_MIN,
    margin: float = STRUCTURE_MARGIN,
) -> list[dict]:
    """Gate each trained encoder seed from the linear-rung shape recoverability.

    Args:
        trained_stack: [S_t, F, R] trained-encoder recoverability.
        random_stack:  [S_r, F, R] random-encoder floor.
        factors:       the dataset's factor tuple.

    Returns one dict per trained seed: shape recoverability, the floor, and PASS.
    """
    si = _categorical_idx(factors)
    floor = float(random_stack.mean(0)[si, 0])  # linear-rung shape floor
    out = []
    for s in range(trained_stack.shape[0]):
        recov = float(trained_stack[s, si, 0])
        out.append(
            {
                "seed_index": s,
                "structure_factor": factors[si].name,
                "shape_recoverability": recov,
                "floor": floor,
                "passed": bool(recov >= shape_min and recov > floor + margin),
            }
        )
    return out


def gate_summary(seed_gates: list[dict]) -> dict:
    """Collapse per-seed gate results into a pass/fail summary for the sweep meta."""
    failed = [g["seed_index"] for g in seed_gates if not g["passed"]]
    return {
        "criteria": {"shape_abs_min": SHAPE_ABS_MIN, "structure_margin": STRUCTURE_MARGIN},
        "n_encoders": len(seed_gates),
        "n_passed": len(seed_gates) - len(failed),
        "failed_seed_indices": failed,
        "all_passed": len(failed) == 0,
        "per_seed": seed_gates,
    }
