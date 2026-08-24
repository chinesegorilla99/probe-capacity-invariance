"""H1-H4 statistics layer (prereg §6-7) — per-cell reports + the confirm/refute table.

Consumes the run_sweep contract, one directory per cell:

    results/probes/<condition>_<strength>/{stacks.npz, meta.json}
        stacks.npz  trained [S_t,F,R] · random [S_r,F,R] · perm [S_t,F,R]
                    · projector [S_t,F,R]

and produces:

    <cell>/hypothesis_report.json    per-cell G/S/epsilon_G table, dual-gate cases,
                                     flip counts, epsilon diagnostics, H1-H4 components
    results/probes/hypotheses.json   the prereg §6 confirm/refute table assembled over
                                     every cell present (absent cells skipped, not fatal)

Frozen statistical rules (prereg §4/§6/§7):
  * All paired quantities (G, S, Delta_G, H2/H4 differences) are per-seed
    differences; trained seed i pairs with random-encoder seed i (seed controls
    both encoder and probe init). Gate-failed encoders are excluded first (D022).
  * The G and S gates share ONE paired-seed bootstrap: every CI comes from
    instrument's mean bootstrap with a common rng seed over a common seed axis,
    so the resampled index draws are identical across gates and quantities.
  * epsilon_G = CI-of-mean of the random-vs-random-encoder null
    (instrument.epsilon_g, reading pinned by D021); requires >=10 random seeds,
    else verdicts are flagged diagnostic-only (D020). The fixed 0.05 threshold is
    co-reported everywhere as sensitivity only (FIX 2/5).
  * Wilcoxon signed-rank for all paired comparisons; Holm across the stated
    family — H2: within-type pairs x condition cells; H4: targeted factors x
    rungs x condition cells. R^2 and normalized-accuracy factors are never
    pooled or rank-compared (FIX 1): H2 pairs are within-type only.
  * Probe-TEST recoverability only (upstream of this layer, by construction).

Amendment A10 §c (2026-08-12) — the invariance boolean and the flip statistic are
TWO-SIDED, on the training-induced deficit D = R(random) - R(trained):
  * suppressed iff D > epsilon_D, recovered iff D <= epsilon_D, where epsilon_D is
    the (1-alpha/2) quantile of the random-vs-random null DEFICIT distribution
    (instrument.epsilon_d, the A8 §a estimator in the deficit direction).
  * epsilon-invariant (H3) iff suppressed at EVERY rung; the headline
    verdict-stability flip iff suppressed at the linear rung AND recovered at the
    top rung. Delta_G, H1, H2, H4, S, epsilon_G, G~, the A6 null-saturation gate
    and the A3 exclusion are unchanged.
  * The frozen one-sided boolean (G <= epsilon_G) and both frozen flip variants
    (primary epsilon_G, fixed 0.05) are CO-REPORTED as sensitivities in every
    table under "*_frozen" keys, so a reader sees what the frozen rule returned.

Interpretation pins (adopted into prereg text by Amendment A1, 2026-07-13):
  * H1's "> epsilon_G" threshold for Delta_G uses the same CI-of-mean estimator
    applied to the random-vs-random null of the capacity gap itself
    (epsilon_g on the random stack's own top-minus-linear gap).
  * H3 confirm keys on the §4 point boolean G <= epsilon_G at every rung; the
    suppressed sub-case (G < 0) is reported distinctly, never merged, and the §6
    parenthetical noise band (G CI within +-epsilon_G) is co-reported.
  * Q13 watch-item (D021): winsorized-null and MAD-robust epsilon_G are computed
    as DIAGNOSTICS only; adopting either requires a dated prereg amendment.
    G itself is never clipped or winsorized.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .instrument import (
    EPS_FIXED,
    RUNG_NAMES,
    SATURATION_LEVEL,
    _boot_mean_ci,
    _deficit_flip_mask,
    _frozen_flip_mask,
    build_report,
    epsilon_d,
    epsilon_g,
    flip_bootstrap,
    null_saturation,
    paired_deficit,
    paired_gain,
    selectivity,
)

N_BOOT = 2000
ALPHA = 0.05
MIN_SEEDS = 10  # prereg §0 / D020

# Augmentation-targeted factors per condition (H4).
TARGETED_FACTORS = {
    "color": ("floor_hue", "wall_hue", "object_hue"),
    "position": ("pos_x", "pos_y"),
    "orientation": ("orientation",),
    "scale": ("scale",),
    "control": (),
}

# Realized grid (prereg Amendment A1: the strong cross-section; A5 (2026-07-28)
# dropped orientation_strong — 2D-rotation augmentation is construct-invalid for
# the Shapes3D azimuth factor and yields no §5-gate-passing encoder) — used only
# to mark the assembled table provisional while cells are missing.
EXPECTED_CELLS = ("color_strong", "position_strong", "control_strong")

# Pre-registered MAXIMAL strong-slice family and the pre-verdict exclusions applied
# to it, disclosed (assemble -> "holm_family") so the Holm correction is seen to span
# the REALIZED family BY DESIGN — cells removed on a shape-gate / construct basis
# before any verdict — not by post-hoc pruning of inconvenient cells. A shrinking
# family that lifts power on survivors is the exact appearance this disclosure exists
# to defuse; the correction is over tests actually run, and every drop is logged here.
DESIGNED_STRONG_CELLS = ("color_strong", "position_strong", "orientation_strong",
                         "scale_strong", "control_strong")
PRE_VERDICT_EXCLUSIONS = {
    "scale_strong": "Amendment A7 (2026-08-07): the bounded (0.75, 1.25) retrain scored "
        "0/12 on the §5 shape gate (linear-rung anchor 0.464-0.507 vs a 0.90 threshold "
        "and a ~0.829 floor), which is the <6/12 branch of the rule pre-committed in "
        "A6(d) on 2026-07-28, BEFORE the retrain ran. The anchor deficit does not close "
        "with probe capacity (0.483 -> 0.609 against a floor rising 0.829 -> 0.986), so "
        "the gate reads FAIL-DESTROYED under A8(b) and the exclusion is gate-licensed "
        "(arm-level, pre-verdict, gate orthogonal to H1-H4). A7(d) freezes the realized "
        "grid at three cells: no further cell may be excluded for any reason.",
    "orientation_strong": "Amendment A5 (2026-07-28): 2D in-plane rotation (SO(2)) is "
        "construct-invalid for the Shapes3D 3D-azimuth (SO(3)) factor and drives the "
        "shape anchor below the random-encoder floor, so no §5-gate-passing encoder "
        "exists (arm-level exclusion, pre-verdict, gate orthogonal to H1-H4).",
    "(dsprites, orientation)": "Amendment A3 (2026-07-15): non-identifiable readout for "
        "symmetric shapes; excluded from confirmatory families everywhere, retained as "
        "a tagged diagnostic (readout-level exclusion, pre-verdict).",
}

# Q16 / Amendment A3 (2026-07-15): non-identifiable readouts. dSprites orientation
# is the full [0, 2*pi) circle and is unrecoverable for the symmetric shapes (square
# 90deg-periodic, ellipse 180deg-periodic), so no encoder can recover it and any
# G<=eps_G verdict on it is vacuous. Excluded from ALL confirmatory families (H1-H3)
# and the headline flip count; RETAINED per-factor as a labeled diagnostic. Keyed on
# (dataset, factor) so Shapes3D orientation (bounded arc, identifiable) is unaffected.
DIAGNOSTIC_ONLY_READOUTS = frozenset({("dsprites", "orientation")})


def _floor_below_zero_names(cell) -> set[str]:
    """A14 (d)(4): a continuous readout whose random-encoder floor is below 0 at ANY
    reported rung is excluded from confirmatory families on A3's vacuity logic.

    A baseline worse than the probe-test mean is not a usable reference for G or D at
    any capacity, so the readout cannot support a confirmatory claim. A14 (d) added the
    lower bound after observing that A10 (b)'s gate tested only the ceiling and would
    pass a floor of -0.89.
    """
    floors = cell.random_stats.mean(axis=0)          # [F, R], mean over random seeds
    return {f.name for fi, f in enumerate(cell.factors)
            if f.kind == "continuous" and bool((floors[fi] < 0).any())}


CALIBRATION_ROOT = Path("results/calibration")


def _a8e_upper_bias(cell, calib_root: Path = CALIBRATION_ROOT) -> dict:
    """A14 (g): the residual A8 (e) linear-rung gap at the pin, quoted numerically.

    The Adam linear rung underfits the convex solver by ``linear_rung_gap``. The gap
    is positive at the pin on both datasets, which inflates Delta_G in the
    H1-confirming direction, so A14 (g) requires it to travel with every reported
    Delta_G rather than sit in a calibration file. The pin is read from the cell's
    own meta, never assumed.
    """
    meta = json.loads((cell.path / "meta.json").read_text())
    pin = (meta.get("probe_regime"), meta.get("probe_train_size"), meta.get("probe_steps"))
    path = Path(calib_root) / f"calibration_{cell.dataset}.json"
    if not path.exists():
        return {"available": False, "pin": list(pin), "source": str(path),
                "note": "A14 (g) bias UNQUANTIFIED: calibration artifact absent"}
    rows = [r for r in json.loads(path.read_text()).get("results", [])
            if (r.get("regime"), r.get("probe_train_size"), r.get("probe_steps")) == pin
            and r.get("encoder_role") == "random"]
    if len(rows) != 1:
        return {"available": False, "pin": list(pin), "source": str(path),
                "note": f"A14 (g) bias UNQUANTIFIED: {len(rows)} calibration rows match "
                        f"pin {pin} at encoder_role 'random', expected exactly 1"}
    r = rows[0]
    return {"available": True, "pin": list(pin), "source": str(path),
            "delta_g_upper_bias": float(r["linear_rung_gap"]),
            "linear_rung_ladder_adam": float(r["linear_rung_ladder_adam"]),
            "linear_rung_closed_form": float(r["linear_rung_closed_form"]),
            "n_seeds": r.get("n_seeds"),
            "note": "A14 (g): upper bias on Delta_G in the H1-confirming direction. The "
                    "Adam linear rung underfits the convex solver by this much at the "
                    "pin; the 4000-step rows close it to ~0.001, so it is an "
                    "optimizer-budget artifact, bounded, and stated with every Delta_G"}


def _diagnostic_only_names(cell) -> set[str]:
    return {f.name for f in cell.factors
            if (cell.dataset, f.name) in DIAGNOSTIC_ONLY_READOUTS}


def _apply_q16(report: dict, h1: dict, h2: dict, h3: dict, diag_names: set[str]) -> None:
    """Drop non-identifiable diagnostic readouts from the confirmatory families and
    the flip count, tagging their per-factor rows; the per-factor table is retained."""
    if not diag_names:
        return
    report["diagnostic_only_factors"] = sorted(diag_names)
    for key in ("flips_two_sided", "flips_primary", "flips_fixed_0.05"):
        fl = report[key]
        fl["flipped_factors"] = [f for f in fl["flipped_factors"] if f not in diag_names]
        fl["n_flips"] = len(fl["flipped_factors"])
    h1["confirmed_factors"] = [f for f in h1["confirmed_factors"] if f not in diag_names]
    h1["confirmed_factors_fixed_0.05"] = [f for f in h1["confirmed_factors_fixed_0.05"]
                                          if f not in diag_names]
    h1["confirmed"] = bool(h1["confirmed_factors"])
    h2["pairs"] = [p for p in h2["pairs"] if not (set(p["pair"]) & diag_names)]
    for key in ("confirmed_factors", "confirmed_factors_frozen"):
        h3[key] = [f for f in h3[key] if f not in diag_names]
    h3["confirmed"] = bool(h3["confirmed_factors"])
    h3["confirmed_frozen"] = bool(h3["confirmed_factors_frozen"])
    for rows in (h1["per_factor"], h3["per_factor"]):
        for row in rows:
            if row["factor"] in diag_names:
                row["diagnostic_only"] = True


# Frozen §4 case labels that read as "invariant" under the SUPERSEDED one-sided
# rule; retained for the co-reported sensitivity verdict only (A10 §c). The
# primary cases are the two-sided pair {"suppressed", "recovered"}.
_FROZEN_INVARIANT_CASES = {"invariant", "suppressed"}


@dataclass(frozen=True)
class FactorMeta:
    name: str
    kind: str  # "continuous" | "categorical"
    index: int
    n_values: int
    cyclic: bool


@dataclass
class Cell:
    name: str
    path: Path
    dataset: str
    condition: str
    strength: str
    factors: tuple[FactorMeta, ...]
    rungs: tuple[str, ...]
    rung_params: dict         # factor -> per-rung trainable-param count (capacity axis)
    trained: np.ndarray       # [n, F, R] gate-passed trained-encoder h
    perm: np.ndarray          # [n, F, R] trained encoder, permuted labels
    projector: np.ndarray     # [n, F, R] trained-encoder projector features
    random_stats: np.ndarray  # [S_r, F, R]; first n rows are seed-paired to trained
    trained_seeds: list[int]
    random_seeds: list[int]
    warnings: list[str]
    # [S_r, F, R] untrained-projector floor; None for pre-A8 sweeps (H4 then falls
    # back to the uncorrected level difference and records a warning).
    random_projector: np.ndarray | None = None


def load_cell(cell_dir: str | Path) -> Cell:
    """Load one sweep cell, apply gate exclusions, and seed-pair the random floor."""
    path = Path(cell_dir)
    meta = json.loads((path / "meta.json").read_text())
    npz = np.load(path / "stacks.npz")
    required = ("trained", "random", "perm", "projector")
    missing = [k for k in required if k not in npz]
    if missing:
        raise ValueError(f"stacks.npz missing arrays {missing}")
    trained, random_full, perm, projector = (np.asarray(npz[k], float) for k in required)
    random_projector = (np.asarray(npz["random_projector"], float)
                        if "random_projector" in npz else None)   # A8 §d

    factors = tuple(
        FactorMeta(f["name"], f["kind"], f["index"], f["n_values"], bool(f.get("cyclic", False)))
        for f in meta["factors"]
    )
    rungs = tuple(meta["rungs"])
    if rungs != RUNG_NAMES:
        raise ValueError(f"rungs {rungs} do not match the locked ladder {RUNG_NAMES}")
    F, R = len(factors), len(rungs)
    for k, arr in (("trained", trained), ("random", random_full),
                   ("perm", perm), ("projector", projector)):
        if arr.shape[1:] != (F, R):
            raise ValueError(f"{k} shape {arr.shape} does not match (S, {F}, {R})")
    if trained.shape[0] != perm.shape[0] or trained.shape[0] != projector.shape[0]:
        raise ValueError("trained / perm / projector seed counts differ")

    warnings: list[str] = []
    trained_seeds = list(meta["seeds"]["trained"])
    random_seeds = list(meta["seeds"]["random"])

    # Gate-failed encoders never enter the stats (prereg §5 / D022) — EXCEPT the
    # control-aug baseline, which is exempt from the encoder-quality gate: by
    # design it is a minimal-augmentation encoder expected to sit at the random
    # floor, so its near-random recoverability is the intended datum, not a
    # training failure (prereg Amendment A2, 2026-07-14). Retain all its encoders.
    failed = sorted(set(meta.get("quality_gate", {}).get("failed_seed_indices", [])))
    if meta.get("condition") == "control" and failed:
        warnings.append(
            f"control-aug: {len(failed)} encoder(s) below the shape gate RETAINED "
            "(gate exempt per Amendment A2); near-random recoverability is the baseline datum"
        )
        failed = []
    if failed:
        keep = [i for i in range(trained.shape[0]) if i not in failed]
        trained, perm, projector = trained[keep], perm[keep], projector[keep]
        trained_seeds = [trained_seeds[i] for i in keep]
        warnings.append(f"excluded {len(failed)} gate-failed encoder(s) at seed idx {failed}")

    # Pair the random floor per seed VALUE; reorder so the paired rows come first
    # (paired_gain slices [:n]) while epsilon_g still sees every random seed.
    if set(trained_seeds) <= set(random_seeds):
        order = [random_seeds.index(s) for s in trained_seeds]
        rest = [i for i in range(len(random_seeds)) if i not in order]
        random_stats = random_full[order + rest]
    else:
        warnings.append("trained/random seed values do not align; paired positionally")
        n = min(len(trained_seeds), random_full.shape[0])
        trained, perm, projector = trained[:n], perm[:n], projector[:n]
        trained_seeds = trained_seeds[:n]
        random_stats = random_full

    if trained.shape[0] < MIN_SEEDS:
        warnings.append(f"only {trained.shape[0]} trained seeds (<{MIN_SEEDS}: under-powered, prereg §0)")
    if random_full.shape[0] < MIN_SEEDS:
        warnings.append(f"only {random_full.shape[0]} random seeds (<{MIN_SEEDS}): "
                        "primary epsilon_G diagnostic-only (D020)")

    return Cell(
        name=f"{meta['condition']}_{meta['strength']}",
        path=path,
        dataset=meta["dataset"],
        condition=meta["condition"],
        strength=meta["strength"],
        factors=factors,
        rungs=rungs,
        rung_params=dict(meta.get("rung_params_h", {})),
        trained=trained,
        perm=perm,
        projector=projector,
        random_stats=random_stats,
        trained_seeds=trained_seeds,
        random_seeds=random_seeds,
        warnings=warnings,
        random_projector=random_projector,
    )


# --- shared statistical helpers ----------------------------------------------

def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone-enforced)."""
    p = np.asarray(pvals, float)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def _wilcoxon_p(x: np.ndarray, alternative: str) -> float:
    """Wilcoxon signed-rank p on paired per-seed differences; all-zero -> 1.0."""
    x = np.asarray(x, float)
    if x.size == 0 or np.allclose(x, 0):
        return 1.0
    return float(wilcoxon(x, alternative=alternative).pvalue)


def epsilon_diagnostics(random_stack, eps_used, g_mean, factor_names, rung_names,
                        n_boot=N_BOOT, seed=0) -> dict:
    """Q13 watch-item (D021): robust epsilon_G alternatives, DIAGNOSTIC ONLY.

    Mirrors instrument.epsilon_g's random-vs-random null pool, then derives
    (a) a winsorized-null epsilon (pool clipped at its own 2.5/97.5 percentiles
    before the mean bootstrap) and (b) a MAD-based robust epsilon. Neither is
    applied to any verdict: adopting one requires a dated prereg amendment.
    """
    s = random_stack.shape[0]
    if s < 2:
        return {"available": False}
    i, j = np.triu_indices(s, k=1)
    pool = np.concatenate([random_stack[i] - random_stack[j],
                           random_stack[j] - random_stack[i]])  # [P, F, R]
    n = pool.shape[0]
    wins = np.clip(pool, np.percentile(pool, 2.5, axis=0), np.percentile(pool, 97.5, axis=0))
    rng = np.random.default_rng(seed)
    boot = np.stack([wins[rng.integers(0, n, n)].mean(0) for _ in range(n_boot)])
    eps_wins = np.percentile(boot, 97.5, axis=0)
    z975 = 1.959963984540054
    med = np.median(pool, axis=0)
    mad_sigma = 1.4826 * np.median(np.abs(pool - med), axis=0)
    eps_mad = med + z975 * mad_sigma / np.sqrt(n)

    disagreements = []
    for fi, fname in enumerate(factor_names):
        for ri, rname in enumerate(rung_names):
            v_primary = bool(g_mean[fi, ri] <= eps_used[fi, ri])
            v_wins = bool(g_mean[fi, ri] <= eps_wins[fi, ri])
            v_mad = bool(g_mean[fi, ri] <= eps_mad[fi, ri])
            if v_primary != v_wins or v_primary != v_mad:
                disagreements.append({
                    "factor": fname, "rung": rname,
                    "invariant_primary": v_primary,
                    "invariant_winsorized_null": v_wins,
                    "invariant_mad": v_mad,
                })

    def per_factor(a):
        return {f: [float(x) for x in a[fi]] for fi, f in enumerate(factor_names)}

    return {
        "available": True,
        "null_sd": per_factor(pool.std(axis=0)),
        "epsilon_primary": per_factor(eps_used),
        "epsilon_winsorized_null": per_factor(eps_wins),
        "epsilon_mad": per_factor(eps_mad),
        "verdict_disagreements": disagreements,
        "watch_item_triggered": bool(disagreements),
        "note": "diagnostic only (Q13/D021): adopting a robust epsilon_G requires "
                "a dated prereg amendment; G itself is never clipped",
    }


# --- per-cell analysis --------------------------------------------------------

def analyze_cell(cell: Cell, n_boot: int = N_BOOT) -> dict:
    """Per-cell G/S/epsilon_G report + the cell's H1-H4 components.

    Keys starting with "_" carry per-seed arrays for study-level assembly and
    are stripped before the per-cell report is written.
    """
    names = [f.name for f in cell.factors]
    n = cell.trained.shape[0]
    diag_names = _diagnostic_only_names(cell)                    # Q16 / Amendment A3
    diag_names |= _floor_below_zero_names(cell)                  # Amendment A14 (d)(4)
    conf = np.array([nm not in diag_names for nm in names], bool)  # confirmatory-factor mask

    # Per-seed paired quantities. All CIs below resample the same n-seed axis
    # with instrument's common bootstrap rng -> shared draws across gates.
    g = cell.trained - cell.random_stats[:n]     # per-seed G [n,F,R]
    dfc = cell.random_stats[:n] - cell.trained   # per-seed deficit D = -G (A10 §c)
    dg = g[:, :, -1] - g[:, :, 0]                # per-seed capacity gap [n,F]
    # H4 = paired G(enc) - G(proj) (prereg §6). The floor does NOT cancel: h is 512-d
    # and z is 128-d, so each axis needs its own untrained baseline (A8 §d).
    if cell.random_projector is not None:
        d_h4 = ((cell.trained - cell.random_stats[:n])
                - (cell.projector - cell.random_projector[:n]))
    else:
        d_h4 = cell.trained - cell.projector
        cell.warnings.append(
            "H4 fallback: no random_projector stack (pre-A8 sweep), so G(enc)-G(proj) "
            "is an uncorrected 512-d vs 128-d level difference — re-sweep to fix.")

    G = paired_gain(cell.trained, cell.random_stats, n_boot=n_boot)
    D = paired_deficit(cell.trained, cell.random_stats, n_boot=n_boot)   # A10 §c
    S = selectivity(cell.trained, cell.perm, n_boot=n_boot)
    eps = epsilon_g(cell.random_stats, n_boot=n_boot)
    eps_used = np.where(np.isnan(eps), EPS_FIXED, eps)
    eps_d = epsilon_d(cell.random_stats, n_boot=n_boot)                  # A10 §c
    eps_d_used = np.where(np.isnan(eps_d), EPS_FIXED, eps_d)

    report = build_report(cell.trained, cell.random_stats, cell.trained, cell.perm,
                          factors=cell.factors, eps_boot=n_boot)
    eps_diag = epsilon_diagnostics(cell.random_stats, eps_used, G["mean"],
                                   names, list(cell.rungs), n_boot=n_boot)

    # Absolute recoverability levels (A1 §c): co-reported so "invariant" over a
    # near-ceiling random floor is never read as "factor absent."
    levels = {}
    for lname, arr in (("trained", cell.trained), ("random_floor", cell.random_stats),
                       ("projector", cell.projector)):
        m, lo, hi = _boot_mean_ci(arr, n_boot=n_boot)
        levels[lname] = {
            fac.name: {"mean": [float(x) for x in m[fi]],
                       "lo": [float(x) for x in lo[fi]],
                       "hi": [float(x) for x in hi[fi]]}
            for fi, fac in enumerate(cell.factors)
        }

    # Null-headroom diagnostic (A4): floor saturation flags; reporting only.
    ns = null_saturation(cell.random_stats)
    sat_flip = ns["flip_endpoint_saturated"]
    null_sat = {
        "level": SATURATION_LEVEL,
        "floor_mean": {fac.name: [float(x) for x in ns["floor_mean"][fi]]
                       for fi, fac in enumerate(cell.factors)},
        "saturated_by_rung": {fac.name: [bool(b) for b in ns["saturated"][fi]]
                              for fi, fac in enumerate(cell.factors)},
        "saturated_factors_flip": [fac.name for fi, fac in enumerate(cell.factors)
                                   if sat_flip[fi]],
        "note": "A4: diagnostic only; saturation-excluded flip variants co-reported; "
                "no decision rule changes",
    }

    # Flip-count seed-bootstrap uncertainty (A1 §c) at fixed eps thresholds;
    # raw draws kept under "_" for the study-level sum in assemble(). The A4
    # saturation-excluded variants use the same machinery on the masked factor set.
    flip_unc, flip_draws = {}, {}
    conf_factors = tuple(f for f in cell.factors if f.name not in diag_names)  # Q16 / A3
    excl = conf & ~sat_flip                                                    # A4
    excl_factors = tuple(f for fi, f in enumerate(cell.factors) if excl[fi])
    eps_fix = np.full_like(eps_used, EPS_FIXED)
    variants = [
        # A10 §c primary: suppressed at the linear rung AND recovered at the top,
        # read off the per-seed DEFICIT stack against epsilon_D.
        ("two_sided", dfc, eps_d_used, conf, conf_factors, _deficit_flip_mask),
        ("two_sided_excl_null_saturated", dfc, eps_d_used, excl, excl_factors,
         _deficit_flip_mask),
        # frozen §4/§6 sensitivities: the one-sided boolean changing either way.
        ("primary", g, eps_used, conf, conf_factors, _frozen_flip_mask),
        ("fixed_0.05", g, eps_fix, conf, conf_factors, _frozen_flip_mask),
        ("primary_excl_null_saturated", g, eps_used, excl, excl_factors, _frozen_flip_mask),
        ("fixed_0.05_excl_null_saturated", g, eps_fix, excl, excl_factors, _frozen_flip_mask),
    ]
    for key, stat, eps_arr, mask, facs, rule in variants:
        fb = flip_bootstrap(stat[:, mask], eps_arr[mask], n_boot=n_boot,
                            factors=facs, rule=rule)
        flip_draws[key] = fb.pop("_draws")
        flip_unc[key] = fb

    # H1 — capacity dependence: Delta_G CI > 0 and > its own random-vs-random
    # null band, with the S gate open at the top rung.
    dg_mean, dg_lo, dg_hi = _boot_mean_ci(dg, n_boot=n_boot)
    rand_gap = cell.random_stats[:, :, -1] - cell.random_stats[:, :, 0]
    eps_dg = epsilon_g(rand_gap[:, :, None], n_boot=n_boot)[:, 0]
    eps_dg = np.where(np.isnan(eps_dg), EPS_FIXED, eps_dg)
    h1_rows = []
    for fi, fac in enumerate(cell.factors):
        s_top_lo = float(S["lo"][fi, -1])
        h1_rows.append({
            "factor": fac.name,
            "delta_g": float(dg_mean[fi]),
            "delta_g_ci": [float(dg_lo[fi]), float(dg_hi[fi])],
            # A13 (a) computes the link statistic WITHIN each seed, so the per-seed
            # Delta_G is that test's dependent variable and has to leave this module.
            "delta_g_per_seed": [float(v) for v in dg[:, fi]],
            "epsilon_delta_g": float(eps_dg[fi]),
            "s_top_ci_lo": s_top_lo,
            "p_wilcoxon_greater": _wilcoxon_p(dg[:, fi], "greater"),
            "g_non_decreasing": bool(np.all(np.diff(G["mean"][fi]) >= 0)),
            "confirmed": bool(dg_lo[fi] > 0 and dg_lo[fi] > eps_dg[fi] and s_top_lo > 0),
            "confirmed_fixed_0.05": bool(dg_lo[fi] > 0 and dg_lo[fi] > EPS_FIXED and s_top_lo > 0),
        })
    # Closure fraction kappa(F) = 1 - D(top)/D(linear) (A13). The deficit training
    # caused, read across the ladder: kappa ~ 1 means a deficit present at the linear
    # rung CLOSES with capacity (the information was made inaccessible, not removed);
    # kappa ~ 0 means it does not close (the information is destroyed). This is the
    # study's own absent-versus-inaccessible axis made quantitative, and it is what
    # promotes the A5/A7 excluded arms from attrition to a measured dissociation.
    # kappa is only meaningful where a deficit EXISTS at the linear rung to close, so
    # it is defined exactly where the A10 (c) boolean reads "suppressed" there. Off a
    # near-zero denominator the ratio is numerically explosive and substantively empty.
    d_lin, d_top = D["mean"][:, 0], D["mean"][:, -1]
    has_deficit = d_lin > eps_d_used[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(has_deficit, 1.0 - d_top / d_lin, np.nan)
    for row, fi in zip(h1_rows, range(len(cell.factors))):
        row["deficit_linear"] = float(d_lin[fi])
        row["deficit_top"] = float(d_top[fi])
        row["closure_fraction_kappa"] = float(kappa[fi])
        row["suppressed_at_linear"] = bool(has_deficit[fi])
        row["closure_reading"] = (
            "undefined (no deficit at the linear rung to close)" if not np.isfinite(kappa[fi])
            else "closes with capacity (inaccessible, not absent)" if kappa[fi] >= 0.5
            else "does not close (destroyed)" if kappa[fi] <= 0.0
            else "partial closure")

    # A14 (g): every reported Delta_G carries the A8 (e) upper bias numerically.
    a8e = _a8e_upper_bias(cell)
    for row in h1_rows:
        row["delta_g_upper_bias_a8e"] = (a8e["delta_g_upper_bias"] if a8e["available"]
                                         else None)

    h1 = {
        "rule": "Delta_G(F) = G(top) - G(linear); bootstrap CI lower bound > 0 and > "
                "epsilon(Delta_G random-vs-random null), with S CI > 0 at the top rung; "
                ">=1 factor confirms (prereg §6 H1)",
        "a8e_upper_bias": a8e,
        "per_factor": h1_rows,
        "confirmed_factors": [r["factor"] for r in h1_rows if r["confirmed"]],
        "confirmed_factors_fixed_0.05": [r["factor"] for r in h1_rows if r["confirmed_fixed_0.05"]],
        "confirmed": any(r["confirmed"] for r in h1_rows),
    }

    # H2 — heterogeneity: paired per-seed Delta_G differences, WITHIN-TYPE pairs
    # only (FIX 1: R^2 and norm-acc are never pooled). Holm family spans cells.
    h2_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if cell.factors[i].kind != cell.factors[j].kind:
                continue
            diff = dg[:, i] - dg[:, j]
            m, lo, hi = _boot_mean_ci(diff, n_boot=n_boot)
            h2_pairs.append({
                "pair": [names[i], names[j]],
                "kind": cell.factors[i].kind,
                "mean_diff": float(m),
                "ci": [float(lo), float(hi)],
                "ci_excludes_0": bool(lo > 0 or hi < 0),
                "p_raw": _wilcoxon_p(diff, "two-sided"),
            })
    for e, a in zip(h2_pairs, holm([e["p_raw"] for e in h2_pairs])):
        e["p_holm_cell"] = float(a)  # per-cell view; the primary family spans cells
    h2 = {
        "rule": "paired per-seed Delta_G difference for within-type pairs; Wilcoxon "
                "two-sided + bootstrap CI; Holm across pairs x condition cells (prereg §6 H2)",
        "pairs": h2_pairs,
    }

    # H3 — genuine invariance. PRIMARY (A10 §c): epsilon-invariant iff SUPPRESSED at
    # EVERY rung, i.e. the training-induced deficit D exceeds epsilon_D throughout.
    # A readout merely sitting AT the untrained floor did nothing to the factor and
    # is "recovered", not invariant — the case the frozen one-sided boolean pooled
    # in. That frozen boolean, its sub-cases and the §6 noise band are co-reported.
    h3_rows = []
    for fi, fac in enumerate(cell.factors):
        supp_rung = D["mean"][fi] > eps_d_used[fi]                  # A10 §c primary
        inv_point = G["mean"][fi] <= eps_used[fi]                   # frozen §4
        band = (G["lo"][fi] >= -eps_used[fi]) & (G["hi"][fi] <= eps_used[fi])
        supp = inv_point & (G["mean"][fi] < 0)
        all_inv = bool(inv_point.all())
        subcase = None
        if all_inv:
            subcase = ("suppressed" if supp.all()
                       else "noise_band" if band.all() else "mixed")
        h3_rows.append({
            "factor": fac.name,
            # --- A10 §c primary
            "eps_invariant": bool(supp_rung.all()),
            "suppressed_by_rung": [bool(b) for b in supp_rung],
            "deficit_by_rung": [float(x) for x in D["mean"][fi]],
            "deficit_ci_by_rung": [[float(lo), float(hi)]
                                   for lo, hi in zip(D["lo"][fi], D["hi"][fi])],
            "epsilon_d_by_rung": [float(x) for x in eps_d_used[fi]],
            # --- frozen §4 boolean, co-reported as sensitivity
            "invariant_all_rungs": all_inv,
            "invariant_all_rungs_fixed_0.05": bool((G["mean"][fi] <= EPS_FIXED).all()),
            "subcase": subcase,
            "invariant_by_rung": [bool(b) for b in inv_point],
            "noise_band_by_rung": [bool(b) for b in band],
            "null_saturated": bool(sat_flip[fi]),   # A6 gate input
        })
    # A6 (2026-07-28): the null-saturation gate. A null-saturated factor (random
    # floor near the ceiling of the normalized scale) satisfies G<=epsilon_G at every
    # rung TRIVIALLY, for want of headroom — indistinguishable from genuine invariance.
    # H3 asserts "genuine invariance", so such readouts are DROPPED from the confirmed
    # set (the A3 vacuity logic applied to saturation instead of non-identifiability)
    # and retained as a distinct, tagged diagnostic list. No other rule is touched.
    h3_confirmed = [r["factor"] for r in h3_rows
                    if r["eps_invariant"] and not r["null_saturated"]]
    h3_vacuous = [r["factor"] for r in h3_rows
                  if r["eps_invariant"] and r["null_saturated"]]
    h3_confirmed_frozen = [r["factor"] for r in h3_rows
                           if r["invariant_all_rungs"] and not r["null_saturated"]]
    h3_vacuous_frozen = [r["factor"] for r in h3_rows
                         if r["invariant_all_rungs"] and r["null_saturated"]]
    h3 = {
        "rule": "A10 §c PRIMARY: D(F,c) = R(random) - R(trained) > epsilon_D at EVERY "
                "rung for >=1 NON-null-saturated factor (A6 gates out saturated "
                "readouts — vacuous for want of floor headroom). The frozen one-sided "
                "rule (G <= epsilon_G at every rung), its suppressed sub-case and the "
                "§6 CI-in-band diagnostic are co-reported as sensitivities "
                "(prereg §6 H3 + A6 + A10 §c)",
        "per_factor": h3_rows,
        "confirmed_factors": h3_confirmed,
        "invariant_but_null_saturated": h3_vacuous,   # A6: retained, NOT confirmatory
        "confirmed": bool(h3_confirmed),
        "confirmed_factors_frozen": h3_confirmed_frozen,          # frozen sensitivity
        "invariant_but_null_saturated_frozen": h3_vacuous_frozen,
        "confirmed_frozen": bool(h3_confirmed_frozen),
    }

    # H4 — encoder-vs-projector: per-seed G(enc)-G(proj) = R(h)-R(projector)
    # (the shared random floor cancels), one-sided Wilcoxon on targeted factors.
    targeted = TARGETED_FACTORS.get(cell.condition)
    if targeted is None:
        targeted = tuple(names)
    targeted = [t for t in targeted if t in names]
    d_mean, d_lo, d_hi = _boot_mean_ci(d_h4, n_boot=n_boot)
    h4_tests = []
    for t in targeted:
        fi = names.index(t)
        for ri, rung in enumerate(cell.rungs):
            h4_tests.append({
                "factor": t,
                "rung": rung,
                "mean_diff": float(d_mean[fi, ri]),
                "ci": [float(d_lo[fi, ri]), float(d_hi[fi, ri])],
                "p_raw": _wilcoxon_p(d_h4[:, fi, ri], "greater"),
            })
    for e, a in zip(h4_tests, holm([e["p_raw"] for e in h4_tests])):
        e["p_holm_cell"] = float(a)
    h4 = {
        "rule": "paired G(encoder) - G(projector) > 0 for targeted factors, Wilcoxon "
                "one-sided; Holm across targeted factors x rungs x condition cells; "
                "strength-widening tested at assembly over >=2 strengths (prereg §6 H4)",
        "targeted_factors": targeted,
        "tests": h4_tests,
    }

    _apply_q16(report, h1, h2, h3, diag_names)   # drop non-identifiable readouts (A3)

    # A4 saturation-excluded flip lists, derived from the (q16-filtered) primary
    # lists so both exclusions compose; additive reporting only.
    sat_names = set(null_sat["saturated_factors_flip"])
    for key in ("flips_two_sided", "flips_primary", "flips_fixed_0.05"):
        fl = report[key]
        kept = [f for f in fl["flipped_factors"] if f not in sat_names]
        report[key + "_excl_null_saturated"] = {
            "n_flips": len(kept),
            "flipped_factors": kept,
            "excluded_null_saturated": sorted(set(fl["flipped_factors"]) & sat_names),
        }

    # Capacity-axis validity (Lee & Kondor 2026): "G rises with capacity" (H1) is
    # only well-posed if the ladder is a MONOTONE capacity axis. Report the frozen
    # capacity measure (trainable param count per rung, prereg A4c(3)) and verify it
    # is non-decreasing across rungs for every factor, so the x-axis is not asserted
    # but checked. Effective DOF remains a probe-build diagnostic (prereg §8 / A6).
    cap_rows, mono_all = [], True
    for fac in cell.factors:
        p = cell.rung_params.get(fac.name)
        if not p:
            continue
        nondec = all(b >= a for a, b in zip(p, p[1:]))
        mono_all = mono_all and nondec
        cap_rows.append({"factor": fac.name, "params_by_rung": [int(x) for x in p],
                         "monotone_nondecreasing": bool(nondec)})
    capacity_axis = {
        "measure": "trainable parameter count per (rung, factor) (prereg A4c(3))",
        "rungs": list(cell.rungs),
        "per_factor": cap_rows,
        "monotone_all_factors": bool(mono_all) if cap_rows else None,
        "note": "param-count monotonicity is the frozen capacity-axis check; the "
                "ladder's approximate function-class nesting is argued against Lee & "
                "Kondor 2026 in the methods; effective DOF is a co-reported diagnostic",
    }

    return {
        "cell": cell.name,
        "dataset": cell.dataset,
        "condition": cell.condition,
        "strength": cell.strength,
        "n_seeds": {"trained_used": int(n), "random": int(cell.random_stats.shape[0])},
        "epsilon_underpowered": bool(cell.random_stats.shape[0] < MIN_SEEDS),
        "warnings": list(cell.warnings),
        "factors": [{"name": f.name, "kind": f.kind} for f in cell.factors],
        "rungs": list(cell.rungs),
        "report": report,
        "levels": levels,
        "null_saturation": null_sat,
        "capacity_axis": capacity_axis,
        "flip_uncertainty": flip_unc,
        "epsilon_diagnostics": eps_diag,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "_h4_d": d_h4,
        "_trained_seeds": list(cell.trained_seeds),
        "_flip_draws": flip_draws,
    }


# --- study-level assembly (prereg §6 confirm/refute table) ---------------------

def _verdict_label_two_sided(case_lin: str, case_top: str) -> str:
    """PRIMARY verdict (A10 §c) from the two-sided deficit cases at both endpoints."""
    if case_lin == "suppressed":
        return ("suppressed_across_ladder" if case_top == "suppressed"
                else "linear_invariance_artifact")
    if case_top == "recovered":
        return "recovered_at_all_capacities"
    return f"other({case_lin}->{case_top})"


def _verdict_label(case_lin: str, case_top: str) -> str:
    """FROZEN one-sided verdict — co-reported sensitivity only since A10 §c."""
    if case_lin in _FROZEN_INVARIANT_CASES and case_top == "genuine":
        return "linear_invariance_artifact"
    if case_lin in _FROZEN_INVARIANT_CASES and case_top in _FROZEN_INVARIANT_CASES:
        return "invariant_across_ladder"
    if case_lin == "genuine" and case_top == "genuine":
        return "recovered_at_all_capacities"
    if "dead_zone" in (case_lin, case_top):
        return "inconclusive_probe_driven"
    return f"other({case_lin}->{case_top})"


def _headline_contrast(table: list[dict]) -> dict:
    """Prereg §0 headline: object hue under Color vs x/y position under Position."""
    def find(cond_prefix, factor):
        return next((t for t in table
                     if t["cell"].startswith(cond_prefix) and t["factor"] == factor), None)

    hue = find("color", "object_hue")
    pos = [t for t in (find("position", "pos_x"), find("position", "pos_y")) if t]
    return {
        "object_hue_color": hue["verdict"] if hue else "pending (color cell absent)",
        "position_crop": [t["verdict"] for t in pos] if pos else "pending (position cell absent)",
        # frozen one-sided reading alongside the A10 §c primary
        "object_hue_color_frozen": (hue["verdict_frozen"] if hue
                                    else "pending (color cell absent)"),
        "position_crop_frozen": ([t["verdict_frozen"] for t in pos] if pos
                                 else "pending (position cell absent)"),
        "complete": bool(hue and pos),
    }


def _h4_widening(results: list[dict], alpha: float) -> dict:
    """H4 widening-with-strength: paired per-seed d(strong) - d(weak) per condition."""
    tests = []
    by_cond: dict[str, dict[str, dict]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], {})[r["strength"]] = r
    for cond, cells in sorted(by_cond.items()):
        if "weak" not in cells or "strong" not in cells:
            continue
        rw, rs = cells["weak"], cells["strong"]
        dw, ds = rw.get("_h4_d"), rs.get("_h4_d")
        if dw is None or ds is None:
            continue
        sw, ss = rw["_trained_seeds"], rs["_trained_seeds"]
        common = [x for x in sw if x in ss]
        if len(common) < 5:
            continue
        diff = ds[[ss.index(x) for x in common]] - dw[[sw.index(x) for x in common]]
        names = [f["name"] for f in rs["factors"]]
        for t in rs["h4"]["targeted_factors"]:
            fi = names.index(t)
            for ri, rung in enumerate(rs["rungs"]):
                tests.append({
                    "condition": cond,
                    "factor": t,
                    "rung": rung,
                    "n_paired_seeds": len(common),
                    "mean_widening": float(diff[:, fi, ri].mean()),
                    "p_raw": _wilcoxon_p(diff[:, fi, ri], "greater"),
                })
    if not tests:
        return {"status": "not_testable", "tests": [],
                "note": "needs >=2 strengths of one condition with shared seeds"}
    for e, a in zip(tests, holm([e["p_raw"] for e in tests])):
        e["p_holm"] = float(a)
    sig = [e for e in tests if e["p_holm"] < alpha]
    return {
        "status": "confirmed" if sig else "refuted",
        "family": "targeted factor x rung x condition (Holm)",
        "significant": [[e["condition"], e["factor"], e["rung"]] for e in sig],
        "tests": tests,
    }


def assemble(results: list[dict], alpha: float = ALPHA) -> dict:
    """The prereg §6 confirm/refute table over whatever cells exist."""
    present = [r["cell"] for r in results]
    missing = [c for c in EXPECTED_CELLS if c not in present]
    provisional = bool(missing)

    notes = []
    for r in results:
        for w in r["warnings"]:
            notes.append(f"{r['cell']}: {w}")
        if r["epsilon_underpowered"]:
            notes.append(f"{r['cell']}: primary-epsilon verdicts are diagnostic-only "
                         f"(<{MIN_SEEDS} random seeds, D020)")
        a8e = r["h1"]["a8e_upper_bias"]
        if a8e["available"]:
            notes.append(f"{r['cell']}: Delta_G carries an A8 (e) upper bias of "
                         f"+{a8e['delta_g_upper_bias']:.4f} in the H1-confirming "
                         "direction (A14 g); quote it wherever Delta_G is quoted")
        else:
            notes.append(f"{r['cell']}: A14 (g) REQUIRES a numeric Delta_G upper bias "
                         f"and it is unavailable — {a8e['note']}")
        diag = r["epsilon_diagnostics"]
        if diag.get("watch_item_triggered"):
            cells = [(d["factor"], d["rung"]) for d in diag["verdict_disagreements"]]
            notes.append(f"{r['cell']}: Q13 WATCH-ITEM TRIGGERED — robust-null epsilon_G "
                         f"flips the invariance verdict at {cells}; a dated prereg "
                         "amendment (winsorized-null / MAD epsilon_G) must be weighed "
                         "before any headline claim rests on these cells")
        sat = r.get("null_saturation", {}).get("saturated_factors_flip", [])
        if sat:
            notes.append(f"{r['cell']}: null-saturated readouts {sat} (random floor >= "
                         f"{SATURATION_LEVEL} at a flip endpoint, A4) — quote the "
                         "saturation-excluded flip count alongside the primary")
    if provisional:
        notes.append(f"provisional: expected cells missing {missing}; refute verdicts are "
                     "not final until the full first-slice grid lands (D022)")

    # H1 — confirm rule is the CI conjunction; Wilcoxon is supporting evidence,
    # Holm-corrected across the factor x condition cell family (prereg §7).
    h1_rows = [dict(cell=r["cell"], **row) for r in results for row in r["h1"]["per_factor"]]
    conf_rows = [row for row in h1_rows if not row.get("diagnostic_only")]  # Q16 / A3
    for row, a in zip(conf_rows, holm([row["p_wilcoxon_greater"] for row in conf_rows])):
        row["p_wilcoxon_holm"] = float(a)
    h1_conf = [[r["cell"], f] for r in results for f in r["h1"]["confirmed_factors"]]
    h1 = {
        "statement": "G rises materially above the linear rung for >=1 factor",
        "family": "factor x condition cells (Holm on supporting Wilcoxon only; "
                  "the confirm rule is the CI conjunction of prereg §6 H1)",
        "confirmed_cells_factors": h1_conf,
        "confirmed_fixed_0.05": [[r["cell"], f] for r in results
                                 for f in r["h1"]["confirmed_factors_fixed_0.05"]],
        "per_factor": h1_rows,
        # A14 (g): the A8 (e) upper bias per cell, so no Delta_G is quoted without it.
        "a8e_upper_bias_by_cell": {r["cell"]: r["h1"]["a8e_upper_bias"] for r in results},
        "status": "confirmed" if h1_conf else "refuted",
    }

    # H2 — Holm across within-type pairs x condition cells.
    h2_rows = [dict(cell=r["cell"], **e) for r in results for e in r["h2"]["pairs"]]
    if h2_rows:
        for e, a in zip(h2_rows, holm([e["p_raw"] for e in h2_rows])):
            e["p_holm"] = float(a)
        h2_sig = [e for e in h2_rows if e["p_holm"] < alpha]
        h2_status = "confirmed" if h2_sig else "refuted"
    else:
        h2_sig, h2_status = [], "not_testable"
    h2 = {
        "statement": "the capacity effect Delta_G differs across within-type factor pairs",
        "family": "within-type factor pairs x condition cells (Holm)",
        "significant_pairs": [[e["cell"], *e["pair"]] for e in h2_sig],
        "pairs": h2_rows,
        "status": h2_status,
        "note": "R^2 and normalized-accuracy factors are never pooled (FIX 1); "
                "CI-excludes-0 co-reported per pair",
    }

    # H3 — existence across the ladder, per cell. A6 (2026-07-28): a null-saturated
    # readout's G<=eps is vacuous (no floor headroom), so it cannot source a
    # "genuine invariance" verdict — dropped from the confirmed set, kept as a tagged
    # diagnostic list. Composes with the Q16/A3 non-identifiability exclusion.
    def h3_rows_where(pred):
        return [[r["cell"], row["factor"]] for r in results
                for row in r["h3"]["per_factor"]
                if pred(row) and not row.get("diagnostic_only")]   # Q16 / A3

    h3_conf = [[r["cell"], row["factor"], row["subcase"]] for r in results
               for row in r["h3"]["per_factor"]
               if row["eps_invariant"] and not row.get("diagnostic_only")   # A10 §c, Q16
               and not row.get("null_saturated")]                           # A6
    h3_vacuous = h3_rows_where(lambda row: row["eps_invariant"] and row["null_saturated"])
    h3_conf_frozen = h3_rows_where(
        lambda row: row["invariant_all_rungs"] and not row["null_saturated"])
    h3_vacuous_frozen = h3_rows_where(
        lambda row: row["invariant_all_rungs"] and row["null_saturated"])
    h3 = {
        "statement": "some NON-null-saturated factor stays epsilon-invariant (SUPPRESSED, "
                     "D > epsilon_D) at every capacity",
        "confirmed_cells_factors": h3_conf,
        "invariant_but_null_saturated": h3_vacuous,   # A6: retained, non-confirmatory
        "status": "confirmed" if h3_conf else "refuted",
        # frozen one-sided sensitivity, co-reported per A10 §c
        "confirmed_cells_factors_frozen": h3_conf_frozen,
        "invariant_but_null_saturated_frozen": h3_vacuous_frozen,
        "status_frozen": "confirmed" if h3_conf_frozen else "refuted",
        "confirmed_fixed_0.05": [[r["cell"], row["factor"]] for r in results
                                 for row in r["h3"]["per_factor"]
                                 if row["invariant_all_rungs_fixed_0.05"]
                                 and not row.get("diagnostic_only")
                                 and not row.get("null_saturated")],
        "note": "A10 §c primary: epsilon-invariance requires a training-induced deficit "
                "above the random-vs-random null at every rung; a readout sitting AT the "
                "untrained floor is 'recovered', not invariant. A6 null-saturation gate "
                "applied unchanged; the frozen one-sided rule and its suppressed / "
                "noise-band sub-cases are co-reported",
    }
    if h3_vacuous:
        notes.append(f"H3 (A6): {len(h3_vacuous)} epsilon-invariant readout(s) "
                     f"{[f'{c}:{f}' for c, f in h3_vacuous]} are NULL-SATURATED and "
                     "EXCLUDED from the genuine-invariance verdict as vacuous (random "
                     "floor near ceiling; no headroom) — retained as diagnostics")
    if h3_vacuous_frozen:
        notes.append(f"H3 (A6, frozen sensitivity): {len(h3_vacuous_frozen)} readout(s) "
                     f"{[f'{c}:{f}' for c, f in h3_vacuous_frozen]} are invariant under "
                     "the frozen one-sided rule AND NULL-SATURATED — excluded from that "
                     "sensitivity's confirmed set as vacuous (random floor near ceiling, "
                     "no headroom for G to exceed epsilon_G)")
    frozen_only = [f"{c}:{f}" for c, f in h3_conf_frozen
                   if [c, f] not in [[cc, ff] for cc, ff, _ in h3_conf]]
    if frozen_only:
        notes.append(f"H3 (A10 §c): {len(frozen_only)} readout(s) {frozen_only} are "
                     "invariant under the FROZEN one-sided rule but not under the "
                     "two-sided primary — they sit at the untrained floor rather than "
                     "below it, so they are 'recovered', not suppressed. Both readings "
                     "are reported; the primary is the two-sided one")

    # H4 — sign component (Holm across cells) + widening component.
    h4_rows = [dict(cell=r["cell"], **t) for r in results for t in r["h4"]["tests"]]
    if h4_rows:
        for e, a in zip(h4_rows, holm([e["p_raw"] for e in h4_rows])):
            e["p_holm"] = float(a)
        h4_sig = [e for e in h4_rows if e["p_holm"] < alpha and e["ci"][0] > 0]
        sign_status = "confirmed" if h4_sig else "refuted"
    else:
        h4_sig, sign_status = [], "not_testable"
    widening = _h4_widening(results, alpha)
    if sign_status == "not_testable":
        h4_status = "not_testable"
    elif widening["status"] == "not_testable":
        h4_status = "partial"
    elif sign_status == "confirmed" and widening["status"] == "confirmed":
        h4_status = "confirmed"
    else:
        h4_status = "refuted"
    h4 = {
        "statement": "invariance concentrates in the projector (test of Cosentino et al. 2022)",
        "family": "targeted factor x rung x condition cells (Holm)",
        "sign_component": {"status": sign_status,
                           "significant": [[e["cell"], e["factor"], e["rung"]] for e in h4_sig],
                           "tests": h4_rows},
        "widening_component": widening,
        "status": h4_status,
    }
    if h4_status == "partial":
        h4["note"] = ("single-strength grid (Amendment A1): the sign component is "
                      "DESCRIPTIVE only; the prereg §6 widening rule is not evaluable, "
                      "so H4 is neither confirmed nor refuted")

    # Headline: verdict-stability flip count over (factor, condition) cells.
    # A4: the saturation-excluded variants are co-reported. A6 (2026-07-28): the
    # variant QUOTED IN CLAIMS is the null-saturation-excluded one — a flip on a
    # saturated readout cannot distinguish absence from lack of floor headroom, so
    # counting it would smuggle a measurement floor into the headline number. The
    # all-factor primary and the fixed-0.05 sensitivity remain co-reported.
    def flips(key):
        rows = [{"cell": r["cell"], "factor": f} for r in results
                for f in r["report"].get(key, {}).get("flipped_factors", [])]
        return {"n_flips": len(rows), "flips": rows}

    headline = {
        # A10 §c primary: suppressed at linear AND recovered at top
        "two_sided": flips("flips_two_sided"),
        "two_sided_excl_null_saturated": flips("flips_two_sided_excl_null_saturated"),
        # frozen one-sided sensitivities (A4/A6), co-reported forever
        "primary": flips("flips_primary"),
        "fixed_0.05": flips("flips_fixed_0.05"),
        "primary_excl_null_saturated": flips("flips_primary_excl_null_saturated"),
        "fixed_0.05_excl_null_saturated": flips("flips_fixed_0.05_excl_null_saturated"),
    }
    headline["headline_for_claims"] = {
        "variant": "two_sided_excl_null_saturated",
        "n_flips": headline["two_sided_excl_null_saturated"]["n_flips"],
        "note": "A10 §c: the flip count quoted in prose is the TWO-SIDED deficit flip "
                "(suppressed at the linear rung, recovered at the top), with the A6 "
                "null-saturation exclusion still applied. The frozen one-sided flip "
                "variants (primary epsilon_G and fixed 0.05, each all-factor and "
                "saturation-excluded) are co-reported as sensitivities. Quote them "
                "together, never one alone.",
    }
    n_ts = headline["two_sided_excl_null_saturated"]["n_flips"]
    n_frozen = headline["primary_excl_null_saturated"]["n_flips"]
    if n_ts != n_frozen:
        notes.append(f"headline flip count (A10 §c): two-sided primary = {n_ts}, frozen "
                     f"one-sided sensitivity = {n_frozen}; both are reported and any "
                     "headline sentence quotes them together")

    # Study-level flip uncertainty (A1 §c): cells resample independently, the
    # per-draw counts sum across cells.
    uncertainty = {}
    for key in ("two_sided", "two_sided_excl_null_saturated",
                "primary", "fixed_0.05",
                "primary_excl_null_saturated", "fixed_0.05_excl_null_saturated"):
        draws = [r.get("_flip_draws", {}).get(key) for r in results]
        if draws and all(d is not None for d in draws):
            total = np.sum(np.stack(draws), axis=0)
            lo, hi = np.percentile(total, [2.5, 97.5])
            uncertainty[key] = {
                "n_flips_mean": float(total.mean()),
                "n_flips_ci95": [float(lo), float(hi)],
                "note": "seed bootstrap at fixed epsilon; threshold uncertainty is "
                        "carried by the epsilon_G diagnostics",
            }
    headline["uncertainty"] = uncertainty

    # Genuine-vs-artifact verdict table per (condition, factor).
    table = []
    for r in results:
        flipped_ts = set(r["report"]["flips_two_sided"]["flipped_factors"])   # A10 §c
        flipped_p = set(r["report"]["flips_primary"]["flipped_factors"])
        flipped_f = set(r["report"]["flips_fixed_0.05"]["flipped_factors"])
        diag = set(r["report"].get("diagnostic_only_factors", []))   # Q16 / A3
        satset = set(r.get("null_saturation", {}).get("saturated_factors_flip", []))  # A4
        for fname, rows in r["report"]["table"].items():
            case_lin, case_top = rows[0]["case_two_sided"], rows[-1]["case_two_sided"]
            frozen_lin, frozen_top = rows[0]["case"], rows[-1]["case"]
            table.append({
                "cell": r["cell"],
                "factor": fname,
                "case_linear": case_lin,
                "case_top": case_top,
                "flip_two_sided": fname in flipped_ts,
                "verdict": _verdict_label_two_sided(case_lin, case_top),
                # frozen one-sided reading, co-reported per A10 §c
                "case_linear_frozen": frozen_lin,
                "case_top_frozen": frozen_top,
                "flip_primary": fname in flipped_p,
                "flip_fixed_0.05": fname in flipped_f,
                "verdict_frozen": _verdict_label(frozen_lin, frozen_top),
                "diagnostic_only": fname in diag,
                "null_saturated": fname in satset,
            })

    # Holm-family disclosure (defuses "the correction shrank as arms were dropped"):
    # the maximal pre-registered strong-slice family, which cells realized, and every
    # pre-verdict exclusion with its reason. Holm corrects over the REALIZED family
    # (tests actually run); this block shows the realization was by design, pre-verdict.
    holm_family = {
        "designed_strong_cells": list(DESIGNED_STRONG_CELLS),
        "realized_cells": present,
        "missing_expected_cells": missing,
        "pre_verdict_exclusions": PRE_VERDICT_EXCLUSIONS,
        "note": "Holm is applied over the realized factor x condition family; every "
                "cell/readout removed from it was removed pre-verdict on a shape-gate "
                "or construct/identifiability basis (logged above), never to lift power "
                "on survivors — a disclosed consequence of dropping untrustworthy "
                "cells, not a post-hoc power adjustment.",
    }

    return {
        "alpha": alpha,
        "cells": present,
        "missing_expected_cells": missing,
        "provisional": provisional,
        "holm_family": holm_family,
        "headline_flip_count": headline,
        "verdict_table": table,
        "headline_contrast": _headline_contrast(table),
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "notes": notes,
    }


# --- CLI ------------------------------------------------------------------------

_STACKS_VARIANT = re.compile(r"^stacks(?: \(\d+\))?\.npz$")


def discover_cells(root: str | Path) -> list[Path]:
    """Cell directories holding a canonical ``stacks.npz``.

    A download-renamed artifact ("stacks (6).npz") does not match the contract
    name, so a cell carrying only a renamed copy would be dropped SILENTLY —
    turning a documented exclusion (A5/A7) into a filename accident, which is
    exactly the appearance A6(c) exists to prevent. Raise instead: every exclusion
    must be declared in PRE_VERDICT_EXCLUSIONS, never produced by a glob miss.
    """
    root = Path(root)
    found = [p.parent for p in root.glob("*/stacks.npz")]
    stray = (
        {p.parent for p in root.glob("*/stacks*.npz")
         if p.name != "stacks.npz" and _STACKS_VARIANT.match(p.name)}
        - set(found)
    )
    stray = sorted(d for d in stray if d.name not in PRE_VERDICT_EXCLUSIONS)
    if stray:
        raise ValueError(
            "cell directories carry a non-canonical stacks file and would be dropped "
            f"silently: {[str(s) for s in stray]}. Rename to stacks.npz / meta.json, or "
            "move the directory out of the sweep root. Exclusions are declared in "
            "PRE_VERDICT_EXCLUSIONS, never produced by a glob miss."
        )
    # Excluded arms are skipped BY NAME, so the exclusion is a declared decision
    # rather than a side effect of which files happen to be present.
    return sorted(d for d in found if d.name not in PRE_VERDICT_EXCLUSIONS)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results/probes", help="directory holding the cell dirs")
    ap.add_argument("--cells", nargs="+", default=None, help="explicit cell dirs (default: discover)")
    ap.add_argument("--out", default=None, help="study table path (default <root>/hypotheses.json)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    dirs = [Path(c) for c in args.cells] if args.cells else discover_cells(args.root)
    results = []
    for d in dirs:
        try:
            cell = load_cell(d)
        except (OSError, ValueError, KeyError) as e:
            print(f"[hyp] SKIP {d}: {e}")
            continue
        res = analyze_cell(cell, n_boot=args.n_boot)
        for w in res["warnings"]:
            print(f"[hyp] {res['cell']}: WARNING {w}")
        out_cell = {k: v for k, v in res.items() if not k.startswith("_")}
        (d / "hypothesis_report.json").write_text(
            json.dumps(out_cell, indent=2, default=_json_default))
        print(f"[hyp] {res['cell']}: wrote {d / 'hypothesis_report.json'}")
        results.append(res)

    if not results:
        print("[hyp] no analyzable cells found — nothing assembled")
        return
    study = assemble(results)
    out = Path(args.out) if args.out else Path(args.root) / "hypotheses.json"
    out.write_text(json.dumps(study, indent=2, default=_json_default))

    hs = study["hypotheses"]
    print(f"[hyp] wrote {out} ({len(results)} cell(s); "
          f"provisional={study['provisional']}, missing={study['missing_expected_cells']})")
    for name in ("H1", "H2", "H3", "H4"):
        print(f"[hyp]   {name}: {hs[name]['status']}", end="")
        print(f" (frozen one-sided: {hs[name]['status_frozen']})"
              if "status_frozen" in hs[name] else "")
    hl = study["headline_flip_count"]
    print(f"[hyp]   flips (A10 §c two-sided PRIMARY): all={hl['two_sided']['n_flips']} "
          f"excl-null-saturated={hl['two_sided_excl_null_saturated']['n_flips']}")
    print(f"[hyp]   flips (frozen one-sided sensitivity): primary={hl['primary']['n_flips']} "
          f"fixed_0.05={hl['fixed_0.05']['n_flips']} | excl-null-saturated: "
          f"primary={hl['primary_excl_null_saturated']['n_flips']} "
          f"fixed_0.05={hl['fixed_0.05_excl_null_saturated']['n_flips']}")
    for key, u in hl.get("uncertainty", {}).items():
        print(f"[hyp]   flips[{key}] seed-bootstrap: mean={u['n_flips_mean']:.2f} "
              f"ci95={u['n_flips_ci95']}")
    for note in study["notes"]:
        print(f"[hyp]   note: {note}")


if __name__ == "__main__":
    _main()
