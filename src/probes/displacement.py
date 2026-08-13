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
               m_total << 1 means the counterfactual pair lands inside the
               representation's own noise ball in every direction, so no probe of
               any capacity can separate them; with INDEPENDENTLY sampled factors,
               h is then a function of the context alone and carries nothing about
               F. That is the destruction certificate.
    m_shared = ||E_z Delta_F(z)||    the component a single fixed readout sees, so
               this is what bounds LINEAR decodability.
    rho_F    = sigma_1^2 / sum_i sigma_i^2   spectral concentration of the field.
    r_F      = (sum sigma_i^2)^2 / sum sigma_i^4   participation ratio.

Reading, with no probe involved:

    m_total << 1                          -> destroyed; genuine invariance
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
    encoder role. Without it the statistic is only orthogonally invariant, not
    affinely, which is exactly the coordinate-stability property a probe hierarchy
    is supposed to have.
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

def whitener(H: np.ndarray, eig_floor: float = EIG_FLOOR) -> np.ndarray:
    """Sigma^{-1/2} from probe-train embeddings (requirement 4).

    Eigenvalues below ``eig_floor`` times the largest are floored rather than
    inverted: a near-null direction carries no signal, and amplifying it would let
    numerical noise dominate every displacement norm.
    """
    X = np.asarray(H, np.float64)
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"whitening needs more probe-train samples than dimensions, got "
            f"{X.shape[0]} x {X.shape[1]}. Below that the covariance is singular, the "
            f"floored directions dominate, and ||Delta|| is an artifact of the floor "
            f"rather than a property of the encoder.")
    X = X - X.mean(0, keepdims=True)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, eig_floor * max(w.max(), 1e-30))
    return (V / np.sqrt(w)) @ V.T


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

    ``m_total << 1`` means the counterfactual pair lands inside the representation's own
    noise ball in every direction, so no probe of any capacity can separate them; because
    the generative factors are sampled INDEPENDENTLY, h is then a function of the context
    alone and carries no information about F. ``m_shared << m_total`` means the movement
    is real but context-dependent, which is the regime that needs probe capacity.

    A probe picks its own direction, so the noise that bounds it is the context spread
    ALONG the displacement — not the total variance summed over every dimension. That is
    why the scale is a whitened magnitude and not a variance share.
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
    estimator prereg A8 (a) fixed for epsilon_G and A10 (c) for epsilon_D, so
    "destroyed" (m deficit > epsilon_m) is adjudicated on the same footing as
    "suppressed". Fewer than two random seeds gives no null and returns NaN.
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

def analyze_role(Hsweeps: dict, W: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> dict:
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
        st.update(displacement_scale(evl_ctx))                      # A12 certificate scale
        st["shared_direction_fraction"] = shared_direction_fraction(evl_ctx)  # A12
        out[fname] = st
    return out


def _transform(grayscale: bool):
    """Eval transform, optionally grayscaled.

    A12 ground-truth control: converting to greyscale removes hue from the INPUT, so
    every hue factor is provably absent by construction and m_total MUST collapse. It
    is the only case where the right answer is known independently of the encoder, and
    it validates the destruction certificate on real data rather than on a fixture.
    """
    base = eval_transform()
    if not grayscale:
        return base
    from torchvision import transforms as T
    return T.Compose([base, T.Grayscale(num_output_channels=3)])


def _extract(model, spec, rows, path, device, bs, nw, in_memory, grayscale=False):
    ds = spec.cls(np.asarray(rows), transform=_transform(grayscale), path=path,
                  return_label=True, in_memory=in_memory)
    H, _ = extract_features(model, ds, device, bs, nw)
    return np.asarray(H, np.float32)


def _sweep_features(model, spec, rows_by_factor, path, device, bs, nw, in_memory,
                    grayscale=False):
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
        "grayscale_control": bool(args.grayscale),
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
                           args.batch_size, args.num_workers, args.in_memory,
                           args.grayscale)
            W = whitener(Htr)
            del Htr
            Hs = _sweep_features(model, spec, rows_by_factor, path, device,
                                 args.batch_size, args.num_workers, args.in_memory,
                                 args.grayscale)
            out["roles"][role][tag] = analyze_role(Hs, W, args.n_boot)
            del Hs
            for fname, st in out["roles"][role][tag].items():
                print(f"[disp] {role:8s} {tag:16s} {fname:12s} "
                      f"m={st['m']:.4f} rho={st['rho']:.4f} r_eff={st['r_eff']:.2f} "
                      f"mono={st['monotonicity']['monotone_fraction']:.2f}")

    rand = out["roles"].get("random", {})
    out["epsilon_m"] = ({f.name: epsilon_m([rand[t][f.name]["m"] for t in rand])
                         for f in factors} if len(rand) >= 2 else {})
    out["epsilon_m_note"] = (
        "epsilon_m is the 97.5th percentile of the random-vs-random null on the m_F "
        "deficit, the estimator A8 (a) fixed for epsilon_G and A10 (c) for epsilon_D. "
        "A factor is DESTROYED iff m_F(random) - m_F(trained) > epsilon_m: the encoder "
        "maps the counterfactual pair to the same point, so no probe of any capacity "
        "can recover F."
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
                    help="A12 ground-truth control: greyscale the input, so hue is "
                         "provably absent and m_total MUST collapse on every hue factor")
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
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
