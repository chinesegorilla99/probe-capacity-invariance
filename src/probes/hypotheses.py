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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .instrument import (
    EPS_FIXED,
    RUNG_NAMES,
    _boot_mean_ci,
    build_report,
    epsilon_g,
    flip_bootstrap,
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

# Realized grid (prereg Amendment A1: the strong cross-section) — used only to
# mark the assembled table provisional while cells are missing.
EXPECTED_CELLS = ("color_strong", "position_strong", "control_strong",
                  "orientation_strong", "scale_strong")

# Q16 / Amendment A3 (2026-07-15): non-identifiable readouts. dSprites orientation
# is the full [0, 2*pi) circle and is unrecoverable for the symmetric shapes (square
# 90deg-periodic, ellipse 180deg-periodic), so no encoder can recover it and any
# G<=eps_G verdict on it is vacuous. Excluded from ALL confirmatory families (H1-H3)
# and the headline flip count; RETAINED per-factor as a labeled diagnostic. Keyed on
# (dataset, factor) so Shapes3D orientation (bounded arc, identifiable) is unaffected.
DIAGNOSTIC_ONLY_READOUTS = frozenset({("dsprites", "orientation")})


def _diagnostic_only_names(cell) -> set[str]:
    return {f.name for f in cell.factors
            if (cell.dataset, f.name) in DIAGNOSTIC_ONLY_READOUTS}


def _apply_q16(report: dict, h1: dict, h2: dict, h3: dict, diag_names: set[str]) -> None:
    """Drop non-identifiable diagnostic readouts from the confirmatory families and
    the flip count, tagging their per-factor rows; the per-factor table is retained."""
    if not diag_names:
        return
    report["diagnostic_only_factors"] = sorted(diag_names)
    for key in ("flips_primary", "flips_fixed_0.05"):
        fl = report[key]
        fl["flipped_factors"] = [f for f in fl["flipped_factors"] if f not in diag_names]
        fl["n_flips"] = len(fl["flipped_factors"])
    h1["confirmed_factors"] = [f for f in h1["confirmed_factors"] if f not in diag_names]
    h1["confirmed_factors_fixed_0.05"] = [f for f in h1["confirmed_factors_fixed_0.05"]
                                          if f not in diag_names]
    h1["confirmed"] = bool(h1["confirmed_factors"])
    h2["pairs"] = [p for p in h2["pairs"] if not (set(p["pair"]) & diag_names)]
    h3["confirmed_factors"] = [f for f in h3["confirmed_factors"] if f not in diag_names]
    h3["confirmed"] = bool(h3["confirmed_factors"])
    for rows in (h1["per_factor"], h3["per_factor"]):
        for row in rows:
            if row["factor"] in diag_names:
                row["diagnostic_only"] = True


_INVARIANT_CASES = {"invariant", "suppressed"}


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
    trained: np.ndarray       # [n, F, R] gate-passed trained-encoder h
    perm: np.ndarray          # [n, F, R] trained encoder, permuted labels
    projector: np.ndarray     # [n, F, R] trained-encoder projector features
    random_stats: np.ndarray  # [S_r, F, R]; first n rows are seed-paired to trained
    trained_seeds: list[int]
    random_seeds: list[int]
    warnings: list[str]


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
        trained=trained,
        perm=perm,
        projector=projector,
        random_stats=random_stats,
        trained_seeds=trained_seeds,
        random_seeds=random_seeds,
        warnings=warnings,
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
    conf = np.array([nm not in diag_names for nm in names], bool)  # confirmatory-factor mask

    # Per-seed paired quantities. All CIs below resample the same n-seed axis
    # with instrument's common bootstrap rng -> shared draws across gates.
    g = cell.trained - cell.random_stats[:n]     # per-seed G [n,F,R]
    dg = g[:, :, -1] - g[:, :, 0]                # per-seed capacity gap [n,F]
    d_h4 = cell.trained - cell.projector         # per-seed G(enc)-G(proj); floor cancels

    G = paired_gain(cell.trained, cell.random_stats, n_boot=n_boot)
    S = selectivity(cell.trained, cell.perm, n_boot=n_boot)
    eps = epsilon_g(cell.random_stats, n_boot=n_boot)
    eps_used = np.where(np.isnan(eps), EPS_FIXED, eps)

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

    # Flip-count seed-bootstrap uncertainty (A1 §c) at fixed eps thresholds;
    # raw draws kept under "_" for the study-level sum in assemble().
    flip_unc, flip_draws = {}, {}
    conf_factors = tuple(f for f in cell.factors if f.name not in diag_names)  # Q16 / A3
    for key, eps_arr in (("primary", eps_used),
                         ("fixed_0.05", np.full_like(eps_used, EPS_FIXED))):
        fb = flip_bootstrap(g[:, conf], eps_arr[conf], n_boot=n_boot, factors=conf_factors)
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
            "epsilon_delta_g": float(eps_dg[fi]),
            "s_top_ci_lo": s_top_lo,
            "p_wilcoxon_greater": _wilcoxon_p(dg[:, fi], "greater"),
            "g_non_decreasing": bool(np.all(np.diff(G["mean"][fi]) >= 0)),
            "confirmed": bool(dg_lo[fi] > 0 and dg_lo[fi] > eps_dg[fi] and s_top_lo > 0),
            "confirmed_fixed_0.05": bool(dg_lo[fi] > 0 and dg_lo[fi] > EPS_FIXED and s_top_lo > 0),
        })
    h1 = {
        "rule": "Delta_G(F) = G(top) - G(linear); bootstrap CI lower bound > 0 and > "
                "epsilon(Delta_G random-vs-random null), with S CI > 0 at the top rung; "
                ">=1 factor confirms (prereg §6 H1)",
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

    # H3 — genuine invariance: the §4 point boolean at every rung; suppressed
    # (G<0) and the §6 +-epsilon noise band reported distinctly.
    h3_rows = []
    for fi, fac in enumerate(cell.factors):
        inv_point = G["mean"][fi] <= eps_used[fi]
        band = (G["lo"][fi] >= -eps_used[fi]) & (G["hi"][fi] <= eps_used[fi])
        supp = inv_point & (G["mean"][fi] < 0)
        all_inv = bool(inv_point.all())
        subcase = None
        if all_inv:
            subcase = ("suppressed" if supp.all()
                       else "noise_band" if band.all() else "mixed")
        h3_rows.append({
            "factor": fac.name,
            "invariant_all_rungs": all_inv,
            "invariant_all_rungs_fixed_0.05": bool((G["mean"][fi] <= EPS_FIXED).all()),
            "subcase": subcase,
            "invariant_by_rung": [bool(b) for b in inv_point],
            "noise_band_by_rung": [bool(b) for b in band],
        })
    h3 = {
        "rule": "G(F,c) <= epsilon_G at EVERY rung for >=1 factor (§4 point boolean); "
                "suppressed sub-case reported distinctly; CI-in-band co-reported (prereg §6 H3)",
        "per_factor": h3_rows,
        "confirmed_factors": [r["factor"] for r in h3_rows if r["invariant_all_rungs"]],
        "confirmed": any(r["invariant_all_rungs"] for r in h3_rows),
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

def _verdict_label(case_lin: str, case_top: str) -> str:
    if case_lin in _INVARIANT_CASES and case_top == "genuine":
        return "linear_invariance_artifact"
    if case_lin in _INVARIANT_CASES and case_top in _INVARIANT_CASES:
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
        diag = r["epsilon_diagnostics"]
        if diag.get("watch_item_triggered"):
            cells = [(d["factor"], d["rung"]) for d in diag["verdict_disagreements"]]
            notes.append(f"{r['cell']}: Q13 WATCH-ITEM TRIGGERED — robust-null epsilon_G "
                         f"flips the invariance verdict at {cells}; a dated prereg "
                         "amendment (winsorized-null / MAD epsilon_G) must be weighed "
                         "before any headline claim rests on these cells")
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

    # H3 — existence across the ladder, per cell.
    h3_conf = [[r["cell"], row["factor"], row["subcase"]] for r in results
               for row in r["h3"]["per_factor"]
               if row["invariant_all_rungs"] and not row.get("diagnostic_only")]  # Q16 / A3
    h3 = {
        "statement": "some factor stays epsilon_G-invariant at every capacity",
        "confirmed_cells_factors": h3_conf,
        "confirmed_fixed_0.05": [[r["cell"], row["factor"]] for r in results
                                 for row in r["h3"]["per_factor"]
                                 if row["invariant_all_rungs_fixed_0.05"]
                                 and not row.get("diagnostic_only")],
        "status": "confirmed" if h3_conf else "refuted",
        "note": "suppressed (G<0 everywhere) and noise-band sub-cases reported distinctly",
    }

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
    def flips(key):
        rows = [{"cell": r["cell"], "factor": f} for r in results
                for f in r["report"][key]["flipped_factors"]]
        return {"n_flips": len(rows), "flips": rows}

    headline = {"primary": flips("flips_primary"), "fixed_0.05": flips("flips_fixed_0.05")}

    # Study-level flip uncertainty (A1 §c): cells resample independently, the
    # per-draw counts sum across cells.
    uncertainty = {}
    for key in ("primary", "fixed_0.05"):
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
        flipped_p = set(r["report"]["flips_primary"]["flipped_factors"])
        flipped_f = set(r["report"]["flips_fixed_0.05"]["flipped_factors"])
        diag = set(r["report"].get("diagnostic_only_factors", []))   # Q16 / A3
        for fname, rows in r["report"]["table"].items():
            case_lin, case_top = rows[0]["case"], rows[-1]["case"]
            table.append({
                "cell": r["cell"],
                "factor": fname,
                "case_linear": case_lin,
                "case_top": case_top,
                "flip_primary": fname in flipped_p,
                "flip_fixed_0.05": fname in flipped_f,
                "verdict": _verdict_label(case_lin, case_top),
                "diagnostic_only": fname in diag,
            })

    return {
        "alpha": alpha,
        "cells": present,
        "missing_expected_cells": missing,
        "provisional": provisional,
        "headline_flip_count": headline,
        "verdict_table": table,
        "headline_contrast": _headline_contrast(table),
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "notes": notes,
    }


# --- CLI ------------------------------------------------------------------------

def discover_cells(root: str | Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("*/stacks.npz"))


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
        print(f"[hyp]   {name}: {hs[name]['status']}")
    hl = study["headline_flip_count"]
    print(f"[hyp]   flips: primary={hl['primary']['n_flips']} "
          f"fixed_0.05={hl['fixed_0.05']['n_flips']}")
    for key, u in hl.get("uncertainty", {}).items():
        print(f"[hyp]   flips[{key}] seed-bootstrap: mean={u['n_flips_mean']:.2f} "
              f"ci95={u['n_flips_ci95']}")
    for note in study["notes"]:
        print(f"[hyp]   note: {note}")


if __name__ == "__main__":
    _main()
