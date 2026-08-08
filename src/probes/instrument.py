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

from ..data.shapes3d import FACTORS  # default factor set (Shapes3D)
from .ladder import LADDER, fit_ladder

RUNG_NAMES = tuple(r.name for r in LADDER)


# --- compute layer -----------------------------------------------------------

def control_task_labels(Ys, seed: int, factors=FACTORS) -> list[np.ndarray]:
    """Random-LABEL control task (Hewitt-Liang), value-level (prereg A8 §b).

    Permutes each factor's VALUE -> TARGET map once and applies the SAME map to
    every split, so the control label is a function of the image and a
    high-capacity probe can actually fit it. Row-wise shuffling per split (the
    superseded ``permute_labels``) is unlearnable at probe-test by construction,
    which collapsed S to R(real) and made the §4 dead-zone case unreachable.

    For a CATEGORICAL factor a value permutation is a pure relabeling, so S == 0
    identically; A8 §b discloses this. Every targeted factor in the study is
    continuous, where the permutation destroys the target's ordinal structure.

    Args:
        Ys: label arrays for the splits, each ``(N, K)``, sharing a column order.
        seed: draws the per-factor value permutation.
        factors: the dataset's factor tuple (``fac.index`` selects the column).

    Returns one control-label array per input split, in the same order.
    """
    rng = np.random.default_rng(seed)
    outs = [np.array(Y, copy=True) for Y in Ys]
    for fac in factors:
        col = fac.index
        uniq = np.unique(np.concatenate([np.asarray(Y[:, col]) for Y in Ys]))
        mapped = uniq[rng.permutation(len(uniq))]
        for out in outs:
            out[:, col] = mapped[np.searchsorted(uniq, out[:, col])]
    return outs


def permute_labels(Y: np.ndarray, seed: int) -> np.ndarray:
    """SUPERSEDED by :func:`control_task_labels` (prereg A8 §b) — kept for provenance.

    Permutes each factor column's rows independently, so the control label is not
    a function of the image and the control task cannot be learned at probe-test.
    Retained only so the pre-A8 behaviour stays reproducible; not used by the
    instrument.
    """
    rng = np.random.default_rng(seed)
    out = np.array(Y, copy=True)
    for j in range(out.shape[1]):
        out[:, j] = out[rng.permutation(out.shape[0]), j]
    return out


def ladder_recoverability(feats: dict, *, seed: int, factors=FACTORS, device="cpu", permute=False, **kw):
    """Fit the ladder for every factor on one encoder's features.

    ``feats`` maps split name -> (H, Y) with splits probe_train / probe_val / probe_test.
    ``factors`` is the dataset's factor tuple (defaults to Shapes3D). Returns
    recov[F, R] normalized recoverability; params[F, R]; metrics[F] ("r2"|"norm_acc").
    """
    Htr, Ytr = feats["probe_train"]
    Hva, Yva = feats["probe_val"]
    Hte, Yte = feats["probe_test"]
    if permute:  # random-label control task (value-level, split-consistent; A8 §b)
        Ytr, Yva, Yte = control_task_labels((Ytr, Yva, Yte), seed, factors)

    F, R = len(factors), len(LADDER)
    recov, params = np.zeros((F, R)), np.zeros((F, R), int)
    metrics = []
    for fi, fac in enumerate(factors):
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


def stack_runs(runs: list[tuple[dict, int]], *, factors=FACTORS, device="cpu", permute=False, **kw) -> np.ndarray:
    """Recoverability stack [n_seeds, F, R] over (features, seed) runs of one encoder role."""
    return np.stack([
        ladder_recoverability(feats, seed=seed, factors=factors, device=device, permute=permute, **kw)[0]
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


def _null_gain_pool(random_stack: np.ndarray) -> np.ndarray:
    """Random-vs-random null gains: all ordered seed pairs (R_i - R_j), i != j."""
    s = random_stack.shape[0]
    i, j = np.triu_indices(s, k=1)
    return np.concatenate([random_stack[i] - random_stack[j],
                           random_stack[j] - random_stack[i]])  # symmetric null


def epsilon_g(random_stack: np.ndarray, n_boot=2000, alpha=0.05, seed=0,
              method: str = "null_quantile") -> np.ndarray:
    """epsilon_G[F,R] — the init-noise band on G (prereg §4 as amended by A8 §a).

    ``method="null_quantile"`` (PRIMARY, A8 §a): the (1-alpha/2) empirical quantile
    of the random-vs-random null gain distribution — "how large a gain init noise
    alone can produce", which is what §4's gloss describes.

    ``method="ci_mean"`` (SENSITIVITY ONLY): the frozen §4 reading, a bootstrap CI
    of the null pool's MEAN. The pool is antisymmetric, so that mean is identically
    zero for any input and the result is the standard error of an estimator of zero:
    ~10x too small at S=12, and shrinking as 1/sqrt(S(S-1)) so more control seeds
    make readouts MORE likely to be called recovered. Retained co-reported, never
    primary.

    Needs >= 2 random-encoder seeds; returns NaN otherwise (fall back to fixed 0.05).
    """
    if random_stack.shape[0] < 2:
        return np.full(random_stack.shape[1:], np.nan)
    deltas = _null_gain_pool(random_stack)
    if method == "null_quantile":
        return np.percentile(deltas, 100 * (1 - alpha / 2), axis=0)
    if method != "ci_mean":
        raise ValueError(f"unknown method {method!r}; use 'null_quantile' or 'ci_mean'")
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    boot = np.stack([deltas[rng.integers(0, n, n)].mean(0) for _ in range(n_boot)])
    return np.percentile(boot, 100 * (1 - alpha / 2), axis=0)


MIN_HEADROOM = 0.02  # A8 §c: below this the G~ denominator is floored and flagged


def headroom_gain(trained_stack: np.ndarray, random_stack: np.ndarray,
                  *, min_headroom: float = MIN_HEADROOM, **kw) -> dict:
    """G~ = G / (1 - mean random floor) — headroom-normalized gain (prereg A8 §c).

    A per-readout monotone rescaling of G expressing gain as a fraction of the
    headroom the normalized scale actually leaves above the untrained floor. Where
    the floor is near the ceiling, raw G is compressed toward 0 by construction and
    "G <= epsilon_G" cannot separate absence from lack of headroom. Co-primary with
    G, never a replacement. Returns mean/lo/hi [F,R] plus the headroom used.
    """
    s = min(trained_stack.shape[0], random_stack.shape[0])
    headroom = 1.0 - random_stack.mean(0)                       # [F, R]
    denom = np.maximum(headroom, min_headroom)
    gt = (trained_stack[:s] - random_stack[:s]) / denom
    mean, lo, hi = _boot_mean_ci(gt, **kw)
    return {"mean": mean, "lo": lo, "hi": hi,
            "headroom": headroom, "headroom_floored": headroom < min_headroom}


def classify(g: float, s: float, eps: float) -> str:
    """Dual-gate case (prereg §4)."""
    if g <= eps:
        return "suppressed" if g < 0 else "invariant"  # G<0 reported distinctly
    return "genuine" if s > 0 else "dead_zone"


def flip_count(inv_bool: np.ndarray, factors=FACTORS) -> dict:
    """Invariance boolean G<=eps flipping between the linear rung (0) and top (-1)."""
    flipped = inv_bool[:, 0] != inv_bool[:, -1]
    names = [factors[i].name for i in np.where(flipped)[0]]
    return {"n_flips": int(flipped.sum()), "flipped_factors": names}


def flip_bootstrap(g_perseed: np.ndarray, eps, n_boot=2000, seed=0, factors=FACTORS) -> dict:
    """Seed-resampling uncertainty of the flip count at a FIXED eps threshold (A1 §c).

    Resamples the paired per-seed G stack [S,F,R] with the same rng pattern as
    _boot_mean_ci (shared draws with the G/S CIs), recomputes the mean-G
    invariance boolean per draw, and counts linear-vs-top flips. Threshold (eps)
    uncertainty is carried separately by the epsilon_G diagnostics. Returns the
    summary plus the raw per-draw counts under "_draws" (for study-level sums).
    """
    rng = np.random.default_rng(seed)
    s = g_perseed.shape[0]
    eps = np.broadcast_to(np.asarray(eps, float), g_perseed.shape[1:])
    draws = np.empty(n_boot, int)
    flip_frac = np.zeros(g_perseed.shape[1])
    for b in range(n_boot):
        gm = g_perseed[rng.integers(0, s, s)].mean(0)
        inv = gm <= eps
        flipped = inv[:, 0] != inv[:, -1]
        draws[b] = int(flipped.sum())
        flip_frac += flipped
    flip_frac /= n_boot
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_flips_mean": float(draws.mean()),
        "n_flips_ci95": [float(lo), float(hi)],
        "per_factor_flip_fraction": {factors[i].name: float(flip_frac[i])
                                     for i in range(g_perseed.shape[1])},
        "_draws": draws,
    }


EPS_FIXED = 0.05  # sensitivity-only fixed threshold (prereg §4, FIX 2/5)

SATURATION_LEVEL = 0.90  # A4: null-saturated iff mean random-floor recoverability >= this


def null_saturation(random_stack: np.ndarray, level: float = SATURATION_LEVEL) -> dict:
    """A4 null-headroom diagnostic: floor means, per-rung saturation, flip-endpoint flag.

    Where the matched random-encoder floor sits near the ceiling of the normalized
    scale, G is compressed toward 0 by construction and "G <= epsilon_G" cannot
    distinguish absence from lack of headroom. A readout is null-saturated for the
    flip statistic iff the mean floor is >= ``level`` at the linear rung or the top
    rung (the two rungs the flip boolean compares). Reporting only — enters no
    decision rule; the saturation-excluded flip variants are co-reported.
    """
    floor_mean = random_stack.mean(0)  # [F, R]
    sat = floor_mean >= level
    return {
        "floor_mean": floor_mean,
        "saturated": sat,
        "flip_endpoint_saturated": sat[:, 0] | sat[:, -1],
    }


def build_report(trained_stack, random_stack, real_stack, perm_stack, *, factors=FACTORS, eps_boot=2000):
    """Assemble the per-(factor, rung) G / S / epsilon_G table + case grid + flip counts.

    ``real_stack`` / ``perm_stack`` are trained-encoder recoverability with true / permuted
    labels (for S). ``factors`` is the dataset's factor tuple (defaults to Shapes3D). Uses
    the data-derived epsilon_G when >=2 random seeds, else the fixed 0.05 sensitivity
    threshold. Returns a JSON-friendly dict.
    """
    G = paired_gain(trained_stack, random_stack, n_boot=eps_boot)
    S = selectivity(real_stack, perm_stack, n_boot=eps_boot)
    Gt = headroom_gain(trained_stack, random_stack, n_boot=eps_boot)   # A8 §c co-primary
    eps = epsilon_g(random_stack, n_boot=eps_boot, method="null_quantile")   # A8 §a primary
    eps_ci = epsilon_g(random_stack, n_boot=eps_boot, method="ci_mean")      # frozen §4, sensitivity
    eps_used = np.where(np.isnan(eps), EPS_FIXED, eps)
    eps_ci_used = np.where(np.isnan(eps_ci), EPS_FIXED, eps_ci)
    eps_source = "fixed_0.05" if np.isnan(eps).all() else "random_vs_random_null_quantile"
    sat = null_saturation(random_stack)

    F, R = G["mean"].shape
    inv_primary = G["mean"] <= eps_used
    inv_ci_mean = G["mean"] <= eps_ci_used
    inv_fixed = G["mean"] <= EPS_FIXED

    table = {}
    for fi, fac in enumerate(factors):
        table[fac.name] = [
            {
                "rung": RUNG_NAMES[ri],
                "G": float(G["mean"][fi, ri]),
                "G_ci": [float(G["lo"][fi, ri]), float(G["hi"][fi, ri])],
                "G_headroom_norm": float(Gt["mean"][fi, ri]),
                "G_headroom_norm_ci": [float(Gt["lo"][fi, ri]), float(Gt["hi"][fi, ri])],
                "headroom": float(Gt["headroom"][fi, ri]),
                "S": float(S["mean"][fi, ri]),
                "S_ci": [float(S["lo"][fi, ri]), float(S["hi"][fi, ri])],
                "epsilon_G": float(eps_used[fi, ri]),
                "epsilon_G_ci_mean": float(eps_ci_used[fi, ri]),
                "epsilon_G_fixed": EPS_FIXED,
                "null_saturated": bool(sat["saturated"][fi, ri]),
                "invariant": bool(inv_primary[fi, ri]),
                "case": classify(G["mean"][fi, ri], S["mean"][fi, ri], eps_used[fi, ri]),
            }
            for ri in range(R)
        ]
    return {
        "epsilon_source": eps_source,
        "epsilon_primary": "null_quantile (A8 §a)",
        "n_seeds": {"trained": int(trained_stack.shape[0]), "random": int(random_stack.shape[0])},
        "rungs": list(RUNG_NAMES),
        "table": table,
        "flips_primary": flip_count(inv_primary, factors),
        "flips_epsilon_ci_mean": flip_count(inv_ci_mean, factors),
        "flips_fixed_0.05": flip_count(inv_fixed, factors),
        "flip_endpoint_saturated": {
            factors[fi].name: bool(sat["flip_endpoint_saturated"][fi]) for fi in range(F)
        },
    }
