"""Metric layer (prereg §3-4) — encoder gain G, probe selectivity S, epsilon_G, flips.

Operates on the probe-capacity ladder (ladder.py). The instrument is split into a
compute layer (fit the ladder per encoder / factor -> per-seed recoverability stacks)
and a stats layer (paired seed bootstrap for G/S CIs, the random-vs-random epsilon_G
null, the dual-gate case classification, and the invariance flip count) so the stats
are testable without re-running probes.

Frozen definitions (prereg §3-4):
    R(F,c)        recoverability of factor F at rung c (norm-acc | R^2)
    S(F,c)        = R(real labels) - R(random labels)      [bounds the probe]
    G(F,c)        = R(trained enc) - R(random enc)          [headline; bounds structure]
    epsilon_G(F,c) upper tail of the random-vs-random-encoder null on G  [init-noise band]
    invariant iff G <= epsilon_G ; genuine recovery iff G > epsilon_G AND S > 0
    flip          invariance boolean changes between the linear rung and the top rung

epsilon_G interpretation (impl choice, flagged for the Phase-3 stats review): the
prereg fixes "upper bound of the bootstrap 95% CI of G under the random-vs-random
null." We read that as the upper limit of a two-sided 95% band on the null G
distribution per (F,c) — i.e. how large a gain init noise alone can produce.
"""

from __future__ import annotations

import numpy as np

from ..data.shapes3d import FACTORS
from .ladder import LADDER, fit_ladder

RUNG_NAMES = tuple(r.name for r in LADDER)


# --- compute layer -----------------------------------------------------------

def permute_labels(Y: np.ndarray, seed: int) -> np.ndarray:
    """Random-LABEL control (Hewitt-Liang): permute each factor column independently."""
    rng = np.random.default_rng(seed)
    out = np.array(Y, copy=True)
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


def ladder_recoverability(feats: dict, *, seed: int, device="cpu", permute=False, **kw):
    """Fit the ladder for every factor on one encoder's features.

    ``feats`` maps split name -> (H, Y) with splits probe_train / probe_val / probe_test.
    Returns recov[F, R] normalized recoverability; params[F, R]; metrics[F] ("r2"|"norm_acc").
    """
    Htr, Ytr = feats["probe_train"]
    Hva, Yva = feats["probe_val"]
    Hte, Yte = feats["probe_test"]
    if permute:  # random-label control task
        Ytr, Yva, Yte = permute_labels(Ytr, seed), permute_labels(Yva, seed), permute_labels(Yte, seed)

    F, R = len(FACTORS), len(LADDER)
    recov, params = np.zeros((F, R)), np.zeros((F, R), int)
    metrics = []
    for fi, fac in enumerate(FACTORS):
        col = fac.index
        if fac.kind == "categorical":
            ytr, yva, yte = (np.rint(Y[:, col]).astype(int) for Y in (Ytr, Yva, Yte))
        else:
            ytr, yva, yte = (Y[:, col].astype(np.float32) for Y in (Ytr, Yva, Yte))
        rungs = fit_ladder(Htr, ytr, Hva, yva, Hte, yte, fac.kind, fac.chance,
                           seed=seed, device=device, **kw)
        recov[fi] = [r["recoverability"] for r in rungs]
        params[fi] = [r["params"] for r in rungs]
        metrics.append(rungs[0]["metric"])
    return recov, params, metrics


def stack_runs(runs: list[tuple[dict, int]], *, device="cpu", permute=False, **kw) -> np.ndarray:
    """Recoverability stack [n_seeds, F, R] over (features, seed) runs of one encoder role."""
    return np.stack([
        ladder_recoverability(feats, seed=seed, device=device, permute=permute, **kw)[0]
        for feats, seed in runs
    ])


# --- stats layer -------------------------------------------------------------

def _boot_mean_ci(x: np.ndarray, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean along axis 0. x: [S, ...] -> (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    s = x.shape[0]
    means = np.stack([x[rng.integers(0, s, s)].mean(0) for _ in range(n_boot)])
    lo = np.percentile(means, 100 * alpha / 2, axis=0)
    hi = np.percentile(means, 100 * (1 - alpha / 2), axis=0)
    return x.mean(0), lo, hi


def paired_gain(trained_stack: np.ndarray, random_stack: np.ndarray, **kw):
    """G = R(trained) - R(random), paired per seed. Returns dict of mean/lo/hi [F,R]."""
    s = min(trained_stack.shape[0], random_stack.shape[0])
    g = trained_stack[:s] - random_stack[:s]
    mean, lo, hi = _boot_mean_ci(g, **kw)
    return {"mean": mean, "lo": lo, "hi": hi}


def selectivity(real_stack: np.ndarray, perm_stack: np.ndarray, **kw):
    """S = R(real) - R(random labels), paired per seed on the trained encoder."""
    s = min(real_stack.shape[0], perm_stack.shape[0])
    sv = real_stack[:s] - perm_stack[:s]
    mean, lo, hi = _boot_mean_ci(sv, **kw)
    return {"mean": mean, "lo": lo, "hi": hi}


def epsilon_g(random_stack: np.ndarray, n_boot=2000, alpha=0.05, seed=0) -> np.ndarray:
    """epsilon_G[F,R]: upper limit of a 95% band on the random-vs-random null of G.

    Pool all ordered seed pairs i!=j as null gains (R_i - R_j), bootstrap the pool,
    and take the upper (1-alpha/2) percentile of the bootstrap distribution per (F,R).
    Needs >= 2 random-encoder seeds; returns NaN otherwise (fall back to fixed 0.05).
    """
    s = random_stack.shape[0]
    if s < 2:
        return np.full(random_stack.shape[1:], np.nan)
    i, j = np.triu_indices(s, k=1)
    deltas = np.concatenate([random_stack[i] - random_stack[j],
                             random_stack[j] - random_stack[i]])  # symmetric null
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    boot = np.stack([deltas[rng.integers(0, n, n)].mean(0) for _ in range(n_boot)])
    return np.percentile(boot, 100 * (1 - alpha / 2), axis=0)


def classify(g: float, s: float, eps: float) -> str:
    """Dual-gate case (prereg §4)."""
    if g <= eps:
        return "suppressed" if g < 0 else "invariant"  # G<0 reported distinctly
    return "genuine" if s > 0 else "dead_zone"


def flip_count(inv_bool: np.ndarray) -> dict:
    """Invariance boolean G<=eps flipping between the linear rung (0) and top (-1)."""
    flipped = inv_bool[:, 0] != inv_bool[:, -1]
    factors = [FACTORS[i].name for i in np.where(flipped)[0]]
    return {"n_flips": int(flipped.sum()), "flipped_factors": factors}


EPS_FIXED = 0.05  # sensitivity-only fixed threshold (prereg §4, FIX 2/5)


def build_report(trained_stack, random_stack, real_stack, perm_stack, *, eps_boot=2000):
    """Assemble the per-(factor, rung) G / S / epsilon_G table + case grid + flip counts.

    ``real_stack`` / ``perm_stack`` are trained-encoder recoverability with true / permuted
    labels (for S). Uses the data-derived epsilon_G when >=2 random seeds, else the fixed
    0.05 sensitivity threshold. Returns a JSON-friendly dict.
    """
    G = paired_gain(trained_stack, random_stack, n_boot=eps_boot)
    S = selectivity(real_stack, perm_stack, n_boot=eps_boot)
    eps = epsilon_g(random_stack, n_boot=eps_boot)
    eps_used = np.where(np.isnan(eps), EPS_FIXED, eps)
    eps_source = "fixed_0.05" if np.isnan(eps).all() else "random_vs_random_null"

    F, R = G["mean"].shape
    inv_primary = G["mean"] <= eps_used
    inv_fixed = G["mean"] <= EPS_FIXED

    table = {}
    for fi, fac in enumerate(FACTORS):
        table[fac.name] = [
            {
                "rung": RUNG_NAMES[ri],
                "G": float(G["mean"][fi, ri]),
                "G_ci": [float(G["lo"][fi, ri]), float(G["hi"][fi, ri])],
                "S": float(S["mean"][fi, ri]),
                "S_ci": [float(S["lo"][fi, ri]), float(S["hi"][fi, ri])],
                "epsilon_G": float(eps_used[fi, ri]),
                "invariant": bool(inv_primary[fi, ri]),
                "case": classify(G["mean"][fi, ri], S["mean"][fi, ri], eps_used[fi, ri]),
            }
            for ri in range(R)
        ]
    return {
        "epsilon_source": eps_source,
        "n_seeds": {"trained": int(trained_stack.shape[0]), "random": int(random_stack.shape[0])},
        "rungs": list(RUNG_NAMES),
        "table": table,
        "flips_primary": flip_count(inv_primary),
        "flips_fixed_0.05": flip_count(inv_fixed),
    }
