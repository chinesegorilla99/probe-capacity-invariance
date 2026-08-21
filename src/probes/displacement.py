"""Counterfactual displacement spectrum — a probe-free invariance instrument.

Both datasets are COMPLETE factorial grids, so for any factor F, any pair of its
values (v, v'), and any assignment z of the remaining factors, both images
g(z, F=v) and g(z, F=v') exist. Exact counterfactual pairs with context held fixed
are therefore available without any model, and the encoder's response to changing
F alone can be measured directly:

    Delta_F(z; v -> v') = h(g(z, F=v')) - h(g(z, F=v))

collected over sampled contexts into a displacement matrix M_F and whitened by the
embedding covariance fit on probe-train. Three statistics, all probe-free:

    m_total  = E_z ||Delta_F(z)||    how far the encoder moves at all, in WHITENED
               units — i.e. in standard deviations of the representation itself.
               Whitening fixes the per-direction variance at 1 and a matched readout
               over an r-dimensional displacement gains exactly sqrt(r), so this is a
               DISCRIMINABILITY INDEX, free of the embedding width and of the number
               of directions the factor occupies. A15 (b) supplies the constant A12
               left as "<< 1": m_star is calibrated on a fixture of known Bayes
               decodability, and the verdict is eps-DESTROYED AT RESOLUTION m_star,
               never "the information is absent" — the encoders are deterministic, so
               a context-aware function separates any nonzero displacement.
    m_shared = ||E_z Delta_F(z)||    the component a single fixed readout sees, so
               this is what bounds LINEAR decodability.
    rho_F    = sigma_1^2 / sum_i sigma_i^2   spectral concentration of the field.
    r_F      = (sum sigma_i^2)^2 / sum sigma_i^4   participation ratio.

Reading, with no probe involved:

    m_total < m_star                      -> eps-destroyed at resolution m_star
    m_total large, m_shared ~ m_total     -> linearly accessible
    m_total large, m_shared << m_total    -> present but context-entangled; a
                                             readout needs capacity

Amendment A12 (2026-08-13) corrected two errors in A11's definitions, before any
displacement number existed. (i) A11 (b) keyed the certificate on the DEFICIT
m_F(random) - m_F(trained), which is relative: an encoder moving half as far as an
untrained one cleared it while the factor stayed a perfect linear ramp, so it
certified reduction, not destruction. The scale is now absolute, supplied by the
whitening. (ii) rho_F was described as bounding linear decodability; it does not.
Two contexts moving +u and -u put all energy on one direction (rho_F = 1) while
cancelling exactly, so no single readout sees them. m_shared is the bound.

Correctness requirements, all enforced here:

 1. rho_F is NECESSARY, not sufficient, for linear decodability: a single-direction
    displacement whose projection is non-monotone in v still defeats a linear probe.
    ``monotonicity`` reports the fraction of contexts whose projection onto sigma_1
    is monotone in the factor's ordered values, and the claim is stated as a bound.
 2. sigma_i^2 is biased upward at small context counts. The spectrum basis is FIT on
    one context sample and the reported statistics are EVALUATED on a disjoint
    held-out sample, with a bootstrap over held-out contexts.
 3. ``epsilon_m`` (the random-vs-random null on the m_F deficit, the A8 (a)
    estimator) is retained and reported, but as a REDUCTION test only: it answers
    "does this encoder move less than an untrained one", not "is the factor gone".
    The destruction reading keys on the absolute whitened m_total (A12).
 4. Whitening is fit on PROBE-TRAIN and applied by the identical procedure to every
    encoder role. A15 (d) caps it at the leading VAR_FRACTION of embedding variance
    and projects the rest out, because the superseded 1e-6 floor RETAINED near-null
    directions and amplified them by up to 1e3 — by an amount set by each role's own
    dimensional collapse, across exactly the roles the certificate compares. The cost
    is that affine invariance becomes approximate (exact under orthogonal maps), and
    it is bounded by the preregistered var_fraction sweep rather than assumed.
 5. The displacement measures the FACTOR, not the augmentation. Prereg A4 (d)(2)'s
    arm-to-perturbed-factor map applies at every interpretation: an arm perturbs a
    SET of factors, so "the augmentation targeting F" is always read against it.
 6. Contexts are drawn from a fixed seed and depend only on (factor, seed,
    n_contexts), so the identical context set is reused across factors, encoders and
    roles and every comparison is paired.

BLIND GUARD. m_F / rho_F / r_F on a TRAINED encoder at a TARGETED factor are
trained-encoder targeted-factor quantities, and the confirmatory blind is intact.
Random-encoder and pixel-reference roles are disclosable (A4 precedent) and run by
default. Scoring a trained checkpoint requires --trained-encoders together with
--amendment, naming the dated prereg amendment that preregisters the
(1 - rho_F) -> Delta_G link; without it this module refuses to load one.

    python -m src.probes.displacement --dataset shapes3d --cell reference \
        --random-seed 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn as nn

from ..data.registry import get_dataset
from ..data.splits import make_splits
from ..encoders.augmentations import eval_transform
from ..encoders.random_encoder import build_random_backbone_projector
from ..eval.extract import extract_features, load_backbone_projector
from ..utils.config import load_config
from ..utils.device import pick_device

ALPHA = 0.05
N_CONTEXTS = 512          # sampled contexts per factor; split fit / eval in half
N_BOOT = 2000
CONTEXT_SEED = 20260813   # fixed: requirement 6 (identical context set everywhere)
EIG_FLOOR = 1e-6          # whitening eigenvalue floor, relative to the largest
VAR_FRACTION = 0.99       # A15 (d): embedding-variance fraction the whitener retains
CTX_PAIR_SEED = 20260819  # A15 (b): fixed context pairing for the m_ctx reference
R2_ABSENT = 0.05          # A15 (b): fixture Bayes R^2 at or below which F is absent
R2_DECODABLE = 0.90       # A15 (b): fixture Bayes R^2 at or above which F is decodable


# --- the factorial grid ---------------------------------------------------------

def grid_codes(labels: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Integer codes per label column, plus each column's sorted unique values.

    Works for any complete factorial dataset without assuming a stride order:
    columns are coded by rank, so a mixed-radix code addresses the grid directly.
    A constant column (dSprites carries a constant colour column) codes to 0 with
    radix 1 and drops out of the addressing harmlessly.
    """
    codes = np.empty(labels.shape, np.int64)
    values = []
    for c in range(labels.shape[1]):
        uniq, inv = np.unique(labels[:, c], return_inverse=True)
        codes[:, c] = inv
        values.append(uniq)
    return codes, values


def _strides(values: list[np.ndarray]) -> np.ndarray:
    radices = np.array([len(v) for v in values], np.int64)
    return np.concatenate([np.cumprod(radices[::-1])[::-1][1:], [np.int64(1)]])


def grid_index(codes: np.ndarray, values: list[np.ndarray]) -> np.ndarray:
    """Row index of every grid point, addressed by its mixed-radix code.

    Returns ``lookup`` of length prod(n_values) with ``lookup[code] = row``. Raises
    if the grid is incomplete, because every guarantee this module makes — exact
    counterfactual pairs, context held fixed — rests on completeness.
    """
    radices = np.array([len(v) for v in values], np.int64)
    total = int(np.prod(radices))
    if total != codes.shape[0]:
        raise ValueError(
            f"not a complete factorial grid: prod(n_values)={total} but "
            f"{codes.shape[0]} rows. Exact counterfactual pairs are unavailable.")
    flat = codes @ _strides(values)
    lookup = np.full(total, -1, np.int64)
    lookup[flat] = np.arange(codes.shape[0], dtype=np.int64)
    if (lookup < 0).any():
        raise ValueError("factorial grid has duplicate or missing cells")
    return lookup


def sample_contexts(values: list[np.ndarray], factor_col: int, n_contexts: int,
                    seed: int = CONTEXT_SEED) -> np.ndarray:
    """Context codes (all columns except ``factor_col``), drawn from a FIXED seed.

    Depends only on (factor_col, n_contexts, seed) and the grid shape, so every
    encoder, role and seed sees the identical context set (requirement 6). The
    ``factor_col`` entry is a placeholder overwritten by the value sweep.
    """
    rng = np.random.default_rng((seed, factor_col))
    radices = [len(v) for v in values]
    return np.stack([rng.integers(0, r, n_contexts) if c != factor_col
                     else np.zeros(n_contexts, np.int64)
                     for c, r in enumerate(radices)], axis=1)


def sweep_rows(ctx: np.ndarray, lookup: np.ndarray, values: list[np.ndarray],
               factor_col: int) -> np.ndarray:
    """Dataset rows for every context x every value of the swept factor.

    Returns ``[n_contexts, n_values]`` of row indices: entry (i, k) is the image
    identical to context i except that the swept factor takes its k-th value.
    """
    st = _strides(values)
    n_vals = len(values[factor_col])
    base = (ctx @ st) - ctx[:, factor_col] * st[factor_col]
    flat = base[:, None] + np.arange(n_vals)[None, :] * st[factor_col]
    return lookup[flat]


# --- whitening ------------------------------------------------------------------

def whitener(H: np.ndarray, var_fraction: float = VAR_FRACTION,
             eig_floor: float = EIG_FLOOR, return_info: bool = False):
    """Sigma^{-1/2} on the leading ``var_fraction`` of embedding variance (A15 d).

    Returns a ``[k, d]`` map, so ``X @ W.T`` lands in the retained k-dimensional
    whitened subspace. Directions outside it are PROJECTED OUT, not floored.

    Flooring at a fixed relative epsilon (the superseded rule) left near-null
    directions in the subspace and multiplied them by up to ``eig_floor**-0.5``.
    Contrastive features are dimensionally collapsed to a degree that differs
    between trained and random encoders, so that amplification differed across
    exactly the roles the certificate compares. A variance-fraction rank cap is
    role-adaptive instead: each role keeps the directions it actually populates.
    Cross-role comparability is carried by the ratios in ``certificate``, which are
    formed within a single role's subspace.

    Exact invariance holds under orthogonal maps; under a general invertible map the
    retained rank can move, so affine invariance is approximate and is bounded by
    the ``var_fraction`` sensitivity sweep the amendment requires.
    """
    X = np.asarray(H, np.float64)
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"whitening needs more probe-train samples than dimensions, got "
            f"{X.shape[0]} x {X.shape[1]}. Below that the covariance is singular, the "
            f"retained directions are sample noise, and ||Delta|| is an artifact of "
            f"the estimator rather than a property of the encoder.")
    X = X - X.mean(0, keepdims=True)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    total = max(w.sum(), 1e-30)
    k = int(np.searchsorted(np.cumsum(w) / total, var_fraction) + 1)
    k = max(1, min(k, int((w > eig_floor * max(w[0], 1e-30)).sum())))
    W = (V[:, :k] / np.sqrt(w[:k])).T
    if not return_info:
        return W
    return W, {"rank": k, "d": int(X.shape[1]), "var_fraction": float(var_fraction),
               "variance_retained": float(w[:k].sum() / total),
               "condition_number": float(w[0] / max(w[k - 1], 1e-30))}


# --- the spectrum ---------------------------------------------------------------

def displacements(Hsweep: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Whitened consecutive-value displacements. [n_ctx, n_vals, d] -> [n_ctx, n_vals-1, d].

    Consecutive differences along the factor's ordered values are the discrete
    tangent field of the counterfactual sweep, and the same sweep supplies the
    monotonicity check, so no extra images are extracted for it.
    """
    return np.diff(np.asarray(Hsweep, np.float64), axis=1) @ W.T


def spectrum_stats(fit: np.ndarray, evl: np.ndarray) -> dict:
    """m_F, rho_F, r_F with the basis FIT on one context sample, EVALUATED on another.

    ``fit`` / ``evl`` are ``[n, d]`` stacks of whitened displacements from disjoint
    context samples. sigma_i^2 estimated and evaluated on the same sample is biased
    upward (requirement 2), so the right singular basis comes from ``fit`` and the
    reported energy distribution is ``evl``'s energy in that fixed basis. The
    in-sample rho is returned alongside so the size of that bias is visible.
    """
    fit = np.asarray(fit, np.float64)
    evl = np.asarray(evl, np.float64)
    _, s_fit, Vt = np.linalg.svd(fit, full_matrices=False)
    e = ((evl @ Vt.T) ** 2).sum(0)
    tot = e.sum()
    return {
        "m": float(np.linalg.norm(evl, axis=1).mean()),
        "rho": float(e[0] / tot) if tot > 0 else float("nan"),
        "r_eff": float(tot ** 2 / (e ** 2).sum()) if tot > 0 else float("nan"),
        "rho_in_sample": float(s_fit[0] ** 2 / (s_fit ** 2).sum()),
        "n_fit": int(fit.shape[0]),
        "n_eval": int(evl.shape[0]),
        "_Vt": Vt,
    }


def bootstrap_stats(fit: np.ndarray, evl_by_ctx: np.ndarray, n_boot: int = N_BOOT,
                    alpha: float = ALPHA, seed: int = 0) -> dict:
    """Percentile CIs for m/rho/r_eff, resampling held-out CONTEXTS (requirement 2).

    Contexts, not individual displacements, are the resampling unit: displacements
    drawn from one context share that context and are not independent.
    """
    _, _, Vt = np.linalg.svd(np.asarray(fit, np.float64), full_matrices=False)
    evl_by_ctx = np.asarray(evl_by_ctx, np.float64)
    E = (evl_by_ctx @ Vt.T) ** 2                     # [n_ctx, n_disp, k]
    N = np.linalg.norm(evl_by_ctx, axis=2)           # [n_ctx, n_disp]
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    draws = {"m": [], "rho": [], "r_eff": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        e = E[idx].sum((0, 1))
        tot = e.sum()
        draws["m"].append(N[idx].mean())
        draws["rho"].append(e[0] / tot if tot > 0 else np.nan)
        draws["r_eff"].append(tot ** 2 / (e ** 2).sum() if tot > 0 else np.nan)
    return {k: [float(np.percentile(v, 100 * alpha / 2)),
                float(np.percentile(v, 100 * (1 - alpha / 2)))]
            for k, v in draws.items()}


def displacement_scale(D: np.ndarray) -> dict:
    """Whitened displacement magnitudes — the ABSOLUTE scale the certificate needs (A12).

    A11 (b) keyed the certificate on the deficit m_F(random) - m_F(trained), which is
    RELATIVE: an encoder moving half as far as an untrained one clears it while the
    factor stays a perfect linear ramp, so it certified reduction, not destruction.
    Whitening supplies the absolute scale for free — after it, the embedding has unit
    variance in EVERY direction, so a displacement measured in whitened units is
    measured in standard deviations of the representation itself:

        m_total  = E_z ||Delta||       total movement, any probe
        m_shared = ||E_z Delta||       the component a single fixed readout sees

    ``m_shared << m_total`` means the movement is real but context-dependent, which is
    the regime that needs probe capacity.

    A15 (b) supersedes A12 (a)'s ``m_total << 1`` reading of these magnitudes. Both are
    reported unchanged as descriptive statistics, but neither adjudicates destruction on
    its own: ``m_total`` is a norm over the retained rank, so it is not comparable across
    factors whose effects occupy different numbers of directions, and the encoders are
    DETERMINISTIC, so there is no noise ball for a pair to land inside. What bounds a
    probe is confusability — an F-displacement is unusable when it is small beside the
    movement the encoder shows for a context change. ``certificate`` forms that ratio.
    """
    D = np.asarray(D, np.float64)
    flat = D.reshape(-1, D.shape[-1])
    per_step_shared = np.linalg.norm(D.mean(0), axis=1)     # [n_steps]
    return {
        "m_total": float(np.linalg.norm(flat, axis=1).mean()),
        "m_shared": float(per_step_shared.mean()),
    }


def shared_direction_fraction(D: np.ndarray) -> float:
    """||E_z Delta||^2 / E_z||Delta||^2 per step, averaged — bounds LINEAR readability (A12).

    rho_F measures how concentrated the displacement SPECTRUM is, which is not the same
    as how consistent the displacement DIRECTION is across contexts: two contexts moving
    +u and -u put all their energy on one direction (rho_F = 1) while cancelling exactly,
    so no single linear readout sees them. A linear probe applies one fixed w to every
    context, so what bounds it is the SHARED component of the displacement field.
    """
    D = np.asarray(D, np.float64)
    num = (D.mean(0) ** 2).sum(1)          # [n_steps]
    den = (D ** 2).sum(2).mean(0)          # [n_steps]
    return float(np.nanmean(np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)))


def context_scale(Hsweep: np.ndarray, W: np.ndarray,
                  seed: int = CTX_PAIR_SEED) -> float:
    """m_ctx — how far the encoder moves when the CONTEXT changes at matched F.

    An instrument check on the whitening, not the certificate's denominator (A15 b).
    It costs no extra images: the sweep array already holds every context at every value
    of F, so pairing context i with a fixed derangement partner at the SAME value of F
    gives displacements that hold F fixed and change everything else.

    Whitening fixes the per-direction variance at 1, so on a k-dimensional retained
    subspace m_ctx must approach sqrt(2k) once a context change decorrelates the
    embedding. A large shortfall means the whitening is not delivering unit variance per
    direction, and m_total's reading as a discriminability index fails with it. Reported
    as ``whitening_check = m_ctx / sqrt(2k)``, which should sit near 1.
    """
    H = np.asarray(Hsweep, np.float64) @ W.T          # [n_ctx, n_vals, k]
    n = H.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    partner = (np.arange(n) + rng.integers(1, n)) % n  # fixed derangement, no self-pairs
    return float(np.linalg.norm(H - H[partner], axis=2).mean())


def certificate(m_total: float, m_shared: float, m_ctx: float, rank: int,
                m_star: float | None = None) -> dict:
    """The eps-DESTRUCTION reading of the whitened magnitude (A15 b).

    A12 (a) was right that ``m_total`` is the scale, and A15 (b) supplies the constant it
    never fixed. Whitening sets the per-direction variance to 1, and a readout matched to
    an r-dimensional displacement gains exactly sqrt(r), so a whitened ``m_total`` IS a
    discriminability index: it is already free of the embedding width and of the number
    of directions the factor occupies. Dividing it by a context reference would put a
    sqrt(rank) dependence back in, which is why m_ctx is an instrument check here and not
    a denominator.

    A factor is eps-DESTROYED AT RESOLUTION ``m_star`` iff m_total < m_star. The claim
    that licenses is "no probe separates F at discriminability m_star on this encoder",
    NOT "the information is absent": the encoders are deterministic, so a context-aware
    function separates any nonzero displacement, and no magnitude certifies absolute
    destruction. ``m_star`` is not a constant of this module — ``calibrate_m_star`` fixes
    it on a fixture with known decodability before any real encoder is scored, and a
    verdict is refused without it.
    """
    ok = m_ctx is not None and np.isfinite(m_ctx) and m_ctx > 0
    expected = float(np.sqrt(2 * max(int(rank), 1)))
    out = {
        "d_prime": float(m_total),
        "shared_fraction": float(m_shared / m_total) if m_total > 0 else float("nan"),
        "whitening_check": float(m_ctx / expected) if ok else float("nan"),
        "m_ctx_expected": expected,
        "m_star": m_star,
    }
    out["verdict"] = ("uncalibrated" if m_star is None or not np.isfinite(m_total)
                      else "eps_destroyed" if m_total < m_star else "moves")
    return out


def calibrate_m_star(dims: int = 512, n_ctx: int = 256, n_vals: int = 10,
                     amplitudes=None,
                     seed: int = 0) -> dict:
    """Fix m_star on a fixture whose decodability is known exactly (A15 b).

    The fixture is A12 (a)'s family, ``h = z + a*v*u``: an isotropic context plus a
    consistent ramp of amplitude ``a`` along one direction. Its decodability is known in
    closed form and is BAYES-optimal, not linear-probe-specific — the context component
    along u is irreducible noise for any predictor — so the calibration cannot smuggle in
    the probe-capacity question the study exists to ask: R^2 = a^2 / (a^2 + 1).

    ``m_star`` is the geometric midpoint between the smallest m_total of a known-decodable
    level (R^2 >= 0.9) and the largest of a known-absent one (R^2 <= 0.05). Levels between
    those bands are deliberately unclassified and define the resolution the certificate
    declines to adjudicate; that band is reported. If the classes overlap there is no
    threshold, ``m_star`` is None, and the certificate is reported as uncalibratable
    rather than given a chosen number.
    """
    if amplitudes is None:
        # the exact R^2 = 0.05 and R^2 = 0.9 amplitudes are grid points, so the reported
        # band is the R^2 definition's own resolution and not an artifact of sampling
        amplitudes = (0.0, *np.geomspace(1e-3, 10.0, 31),
                      *[float(np.sqrt(r / (1 - r))) for r in (R2_ABSENT, R2_DECODABLE)])
    rng = np.random.default_rng(seed)
    u = np.zeros(dims); u[0] = 1.0
    vals = np.arange(n_vals, dtype=float)
    vals = (vals - vals.mean()) / vals.std()
    rows = []
    for a in amplitudes:
        H = rng.normal(size=(n_ctx, 1, dims)) + a * vals[None, :, None] * u[None, None, :]
        sc = displacement_scale(np.diff(H, axis=1))
        rows.append({"amplitude": float(a), "r2": float(a ** 2 / (a ** 2 + 1.0)),
                     "m_total": sc["m_total"], "m_shared": sc["m_shared"]})
    tol = 1e-9   # the boundary amplitudes are grid points; keep them in their own class
    hi = [r["m_total"] for r in rows if r["r2"] >= R2_DECODABLE - tol]
    lo = [r["m_total"] for r in rows if r["r2"] <= R2_ABSENT + tol]
    m_star = None
    if hi and lo and max(lo) < min(hi):
        m_star = float(np.sqrt(max(lo) * min(hi))) if max(lo) > 0 else float(min(hi) / 2)
    return {
        "m_star": m_star,
        "separable": m_star is not None,
        "decodable_min_m": float(min(hi)) if hi else None,
        "absent_max_m": float(max(lo)) if lo else None,
        "unadjudicated_band": [float(max(lo)), float(min(hi))] if hi and lo else None,
        "dims": int(dims), "n_contexts": int(n_ctx), "n_values": int(n_vals),
        "rows": rows,
        "note": "m_star is the geometric midpoint between the smallest whitened m_total "
                "of a known-decodable fixture level (Bayes R2 >= 0.9) and the largest of "
                "a known-absent one (R2 <= 0.05). m_total is width-free, so the "
                "calibration transfers across embedding widths and retained ranks. "
                "Overlap returns None and the certificate is reported as uncalibratable, "
                "never given a picked number.",
    }


def monotonicity(Hsweep: np.ndarray, W: np.ndarray, Vt: np.ndarray) -> dict:
    """Requirement 1: is the projection onto sigma_1 monotone in the factor's values?

    rho_F ~ 1 says the effect lives on one direction; it does NOT say a linear probe
    can read the value off it. A projection that rises then falls is single-direction
    and linearly undecodable. Reports the fraction of contexts whose projection is
    monotone and the mean |Spearman| over contexts, so rho_F is quoted as a bound
    rather than as a prediction.
    """
    proj = (np.asarray(Hsweep, np.float64) @ W.T) @ Vt[0]   # [n_ctx, n_vals]
    d = np.diff(proj, axis=1)
    mono = (d >= 0).all(1) | (d <= 0).all(1)
    k = proj.shape[1]
    ranks = np.argsort(np.argsort(proj, axis=1), axis=1).astype(float)
    order = np.arange(k, dtype=float)
    rc = ranks - ranks.mean(1, keepdims=True)
    oc = order - order.mean()
    denom = np.sqrt((rc ** 2).sum(1) * (oc ** 2).sum())
    rho_s = np.where(denom > 0, (rc @ oc) / np.maximum(denom, 1e-30), 0.0)
    return {
        "monotone_fraction": float(mono.mean()),
        "mean_abs_spearman": float(np.abs(rho_s).mean()),
        "note": "rho_F bounds linear decodability, it does not imply it: a "
                "single-direction but non-monotone projection defeats a linear probe",
    }


def epsilon_m(null_m, alpha: float = ALPHA) -> float:
    """epsilon_m — the init-noise band on the m_F deficit (requirement 3).

    ``null_m`` is m_F per random-encoder seed. The null pool is every ordered pair
    difference m_i - m_j, and epsilon_m is its (1-alpha/2) quantile: exactly the
    estimator prereg A8 (a) fixed for epsilon_G and A10 (c) for epsilon_D.

    A12 (a) retired this as a DESTRUCTION rule and A15 (a) removes the retired wording
    from the artifact: it is relative, so it fires on an encoder that halves a
    displacement which remains a perfect linear ramp. It is retained as a REDUCTION
    test, which is worth reporting and is not a certificate. Fewer than two random seeds
    gives no null and returns NaN.
    """
    m = np.asarray(null_m, float).ravel()
    if m.size < 2:
        return float("nan")
    i, j = np.triu_indices(m.size, k=1)
    pool = np.concatenate([m[i] - m[j], m[j] - m[i]])
    return float(np.percentile(pool, 100 * (1 - alpha / 2)))


def build_external_encoder(name: str, device):
    """An ImageNet-pretrained backbone as an extra encoder role (A13 external check).

    The study's encoders are small and trained at 64x64, which is the standard "does
    this survive at scale" objection. Running the SAME counterfactual pairs through a
    public pretrained checkpoint keeps the exact-pair property the factorial grid
    supplies while changing the encoder entirely, so the geometry claim is tested
    outside this study's own training pipeline. Diagnostic: it enters no confirmatory
    family (its augmentation recipe is not ours and is not controlled).
    """
    import torch.nn as nn
    from torchvision import models
    fn = getattr(models, name)
    net = fn(weights="IMAGENET1K_V1")
    net.fc = nn.Identity()
    return net.eval().to(device)


# --- driver ---------------------------------------------------------------------

def analyze_role(Hsweeps: dict, W: np.ndarray, n_boot: int = N_BOOT, seed: int = 0,
                 m_star: float | None = None) -> dict:
    """Per-factor spectrum for one encoder, from its per-factor sweep embeddings."""
    out = {}
    for fname, Hs in Hsweeps.items():
        half = Hs.shape[0] // 2
        if half < 1 or Hs.shape[0] - half < 1:
            raise ValueError(f"{fname}: need >=2 contexts to split fit/eval")
        d_all = displacements(Hs, W)                    # [n_ctx, n_vals-1, d]
        fit = d_all[:half].reshape(-1, d_all.shape[-1])
        evl_ctx = d_all[half:]
        st = spectrum_stats(fit, evl_ctx.reshape(-1, d_all.shape[-1]))
        Vt = st.pop("_Vt")
        st["ci"] = bootstrap_stats(fit, evl_ctx, n_boot=n_boot, seed=seed)
        st["monotonicity"] = monotonicity(Hs[half:], W, Vt)
        st.update(displacement_scale(evl_ctx))                      # A12 magnitudes
        st["shared_direction_fraction"] = shared_direction_fraction(evl_ctx)  # A12
        st["m_ctx"] = context_scale(Hs[half:], W)                   # A15 (b) instrument check
        st.update(certificate(st["m_total"], st["m_shared"], st["m_ctx"], W.shape[0], m_star))
        out[fname] = st
    return out


def _transform(control: str | None):
    """Eval transform, optionally under a hue-REDUCTION control (A15 c).

    A13 (c) called luminance greyscale a ground-truth control on the grounds that it
    removes hue from the input. Measured on raw Shapes3D images with one context and
    object_hue swept over its ten values, it does not: hue at fixed saturation and value
    maps to different luminances, leaving a per-pixel range of 16.6/255 on average,
    15.5% of pixels moving by more than 1/255, and the ten values pairwise separated by
    L2 >= 61.4. A CNN recovers hue from that.

    ``value`` takes max(R,G,B), the HSV value channel, which is hue-invariant at fixed
    saturation and value by construction and measures 16x tighter (per-pixel range
    1.06/255, min pairwise L2 2.0) but is still not zero, because shading, specular
    highlights and compression break the constant-S,V assumption.

    Neither is a provable zero, so neither sets a threshold. Both are reported as
    strong-reduction controls: a certificate that fails to fire under them is broken,
    firing under them is necessary and not sufficient. The threshold comes from
    ``calibrate_m_star``, where the answer is known exactly.
    """
    base = eval_transform()
    if not control:
        return base
    from torchvision import transforms as T
    if control == "grayscale":
        return T.Compose([base, T.Grayscale(num_output_channels=3)])
    if control == "value":
        return T.Compose([base, T.Lambda(lambda t: t.max(dim=0, keepdim=True)
                                         .values.expand_as(t).clone())])
    raise ValueError(f"unknown control transform: {control!r}")


def _extract(model, spec, rows, path, device, bs, nw, in_memory, grayscale=None):
    ds = spec.cls(np.asarray(rows), transform=_transform(grayscale), path=path,
                  return_label=True, in_memory=in_memory)
    H, _ = extract_features(model, ds, device, bs, nw)
    return np.asarray(H, np.float32)


def _sweep_features(model, spec, rows_by_factor, path, device, bs, nw, in_memory,
                    grayscale=None):
    """Extract embeddings for every swept row ONCE, then scatter back per factor."""
    flat = np.concatenate([r.ravel() for r in rows_by_factor.values()])
    uniq, inv = np.unique(flat, return_inverse=True)
    H = _extract(model, spec, uniq, path, device, bs, nw, in_memory, grayscale)
    out, off = {}, 0
    for fname, rows in rows_by_factor.items():
        n = rows.size
        out[fname] = H[inv[off:off + n]].reshape(*rows.shape, H.shape[1])
        off += n
    return out


def _load_m_star(path) -> float | None:
    """Read the calibrated m_star, or None if it was never calibrated (A15 b).

    None is not an error here: the random and pixel roles are worth measuring before a
    threshold exists. It propagates to verdict "uncalibrated", so an artifact can never
    carry a destruction verdict that no calibration backs.
    """
    if not path or not Path(path).exists():
        return None
    cal = json.loads(Path(path).read_text())
    return cal.get("m_star")


def run(args) -> dict:
    # The blind guard fires BEFORE any other work: a guard that only trips after the
    # config, the device and the dataset have been resolved can be defeated by an
    # unrelated failure earlier in the call, and is not a guard.
    if getattr(args, "trained_encoders", None) and not getattr(args, "amendment", None):
        raise SystemExit(
            "BLIND GUARD: scoring a trained checkpoint produces trained-encoder "
            "targeted-factor values. Pass --amendment <id> naming the dated prereg "
            "amendment that preregisters the (1 - rho_F) -> Delta_G link, and file it "
            "FIRST. Random-encoder and pixel roles need no flag.")

    control = "value" if getattr(args, "value_channel", False) else (
        "grayscale" if getattr(args, "grayscale", False) else None)
    m_star = _load_m_star(getattr(args, "threshold_file", None))

    cfg = load_config(args.config)
    spec = get_dataset(args.dataset)
    device = pick_device(args.device)
    path = args.data_path or spec.default_path

    ds_all = spec.cls(np.arange(spec.n_total), transform=None, path=path, return_label=True)
    labels_all = np.asarray(ds_all.labels)
    del ds_all
    codes, values = grid_codes(labels_all)
    lookup = grid_index(codes, values)
    base = make_splits(spec.n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])

    factors = [f for f in spec.factors if not args.factors or f.name in args.factors]
    rows_by_factor = {
        f.name: sweep_rows(
            sample_contexts(values, f.index, args.n_contexts, args.context_seed),
            lookup, values, f.index)
        for f in factors
    }
    print(f"[disp] {spec.name}: {len(factors)} factor(s), {args.n_contexts} contexts, "
          f"{sum(r.size for r in rows_by_factor.values())} counterfactual images")

    roles: dict[str, list] = {
        "random": [(f"seed{s}", build_random_backbone_projector(s, device)[0])
                   for s in args.random_seed],
    }
    if args.pixel_reference:
        roles["pixels"] = [("identity", nn.Flatten(start_dim=1).to(device))]
    if args.external_encoder:
        roles["external"] = [(n, build_external_encoder(n, device))
                             for n in args.external_encoder]
    if args.trained_encoders:
        roles["trained"] = [(Path(p).parent.name, load_backbone_projector(p, device)[0])
                            for p in args.trained_encoders]

    out: dict = {
        "kind": "counterfactual_displacement_spectrum",
        "dataset": spec.name,
        "cell": args.cell,
        "n_contexts": int(args.n_contexts),
        "reduction_control": control,
        "reduction_control_note": "A15 (c): greyscale and value-channel REDUCE hue in "
                                  "the input, measurably but not to zero. Neither is a "
                                  "ground truth and neither sets a threshold.",
        "m_star": m_star,
        "m_star_source": getattr(args, "threshold_file", None) if m_star is not None
                           else "uncalibrated: run --calibrate-threshold first",
        "var_fraction": float(args.var_fraction),
        "context_seed": int(args.context_seed),
        "alpha": ALPHA,
        "blind": {
            "trained_roles_scored": bool(args.trained_encoders),
            "amendment": args.amendment,
            "note": "random-encoder and pixel-reference values are disclosable (A4 "
                    "precedent); trained-encoder targeted-factor values are not, and "
                    "require the amendment named above",
        },
        "interpretation_pin": "prereg A4 (d)(2): each arm perturbs a SET of factors, "
                              "so a displacement is read against the arm-to-factor map, "
                              "never as 'the augmentation targeting F' alone",
        "roles": {},
    }

    for role, models in roles.items():
        out["roles"][role] = {}
        for tag, model in models:
            Htr = _extract(model, spec, base["probe_train"], path, device,
                           args.batch_size, args.num_workers, args.in_memory, control)
            W, winfo = whitener(Htr, args.var_fraction, return_info=True)
            del Htr
            Hs = _sweep_features(model, spec, rows_by_factor, path, device,
                                 args.batch_size, args.num_workers, args.in_memory,
                                 control)
            scored = analyze_role(Hs, W, args.n_boot, m_star=m_star)
            del Hs
            out["roles"][role][tag] = {"whitening": winfo, "factors": scored}
            for fname, st in scored.items():
                print(f"[disp] {role:8s} {tag:16s} {fname:12s} "
                      f"d_prime={st['d_prime']:.4f} shared={st['shared_fraction']:.3f} "
                      f"[{st['verdict']}] wcheck={st['whitening_check']:.2f} "
                      f"m={st['m']:.4f} rho={st['rho']:.4f} r_eff={st['r_eff']:.2f} "
                      f"mono={st['monotonicity']['monotone_fraction']:.2f} "
                      f"rank={winfo['rank']}")

    rand = out["roles"].get("random", {})
    out["epsilon_m"] = ({f.name: epsilon_m([rand[t]["factors"][f.name]["m"] for t in rand])
                         for f in factors} if len(rand) >= 2 else {})
    out["epsilon_m_note"] = (
        "epsilon_m is the 97.5th percentile of the random-vs-random null on the m_F "
        "deficit, the estimator A8 (a) fixed for epsilon_G and A10 (c) for epsilon_D. "
        "REDUCTION TEST ONLY: m_F(random) - m_F(trained) > epsilon_m says this encoder "
        "moves less than an untrained one, nothing more. A12 (a) retired it as a "
        "destruction rule after it returned DESTROYED for a factor that is a perfect "
        "linear ramp. Destruction is adjudicated by the m_total / m_star certificate "
        "(A15 b), reported per factor as 'verdict'."
        if out["epsilon_m"] else
        "epsilon_m unavailable: needs >= 2 random-encoder seeds")

    out_path = Path(args.out_root) / f"{args.cell}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[disp] wrote {out_path}")
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/probe/ladder.yaml")
    ap.add_argument("--dataset", default="shapes3d", choices=["shapes3d", "dsprites"])
    ap.add_argument("--cell", default="reference",
                    help="output name: results/displacement/<cell>.json")
    ap.add_argument("--factors", nargs="*", default=None, help="default: every factor")
    ap.add_argument("--n-contexts", type=int, default=N_CONTEXTS)
    ap.add_argument("--context-seed", type=int, default=CONTEXT_SEED)
    ap.add_argument("--random-seed", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--trained-encoders", nargs="*", default=None,
                    help="backbone.pt paths; BLIND-GATED, requires --amendment")
    ap.add_argument("--amendment", default=None,
                    help="dated prereg amendment preregistering the (1-rho_F) -> Delta_G link")
    ap.add_argument("--grayscale", action="store_true",
                    help="hue-REDUCTION control (A15 c): luminance greyscale. Reduces "
                         "hue, does not remove it; not a ground truth")
    ap.add_argument("--value-channel", action="store_true",
                    help="tighter hue-REDUCTION control (A15 c): max(R,G,B), "
                         "hue-invariant at fixed saturation and value; still not zero")
    ap.add_argument("--var-fraction", type=float, default=VAR_FRACTION,
                    help="A15 (d) whitening rank cap; sweep {0.95, 0.99, 0.999} as the "
                         "preregistered sensitivity")
    ap.add_argument("--threshold-file", default="results/displacement/"
                                                "threshold_calibration.json",
                    help="A15 (b) calibrated m_star; without it every verdict is "
                         "'uncalibrated'")
    ap.add_argument("--calibrate-threshold", action="store_true",
                    help="A15 (b): calibrate m_star on the known-answer fixture and "
                         "write --threshold-file, then exit. Run this FIRST")
    ap.add_argument("--external-encoder", nargs="*", default=None,
                    help="ImageNet-pretrained torchvision backbones as an extra role "
                         "(e.g. resnet18 resnet50); diagnostic, no confirmatory family")
    ap.add_argument("--pixel-reference", action="store_true", default=True)
    ap.add_argument("--no-pixel-reference", dest="pixel_reference", action="store_false")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--out-root", default="results/displacement")
    args = ap.parse_args()
    if args.calibrate_threshold:
        cal = calibrate_m_star()
        p = Path(args.threshold_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cal, indent=2))
        print(f"[disp] m_star = {cal['m_star']} (separable={cal['separable']}); "
              f"wrote {p}")
        return
    run(args)


if __name__ == "__main__":
    _main()
