"""Paper figures from the run_sweep contract (stacks.npz + meta.json per cell).

Consumes cells through the same loaders/statistics as the H1-H4 layer
(hypotheses.load_cell / analyze_cell / assemble + instrument's bootstrap), so
every plotted number matches the reported one. Works identically on real cells
(results/probes) and synthetic fixtures (results/_synthetic/<world>); any cell
whose meta.json carries "synthetic_world" stamps SYNTHETIC on every output.

Per root it renders (PNG + PDF + a CSV table twin per figure):

    g_ladder_<cell>   per-factor small multiples: G vs probe capacity, bootstrap
                      CI band, +-epsilon_G band, targeted factors highlighted
    h4_<cell>         targeted factors: recoverability of h vs projector vs the
                      random floor across the ladder (skipped when no targets)
    headline          the prereg 0 contrast: object hue (Color) vs x/y position
                      (Position), G across the ladder (needs both cells)
    verdicts          genuine-vs-artifact verdict matrix over (factor, cell)

    python -m src.eval.figures --root results/_synthetic/artifact \
        --out results/figures/_synthetic/artifact
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..probes.hypotheses import (
    TARGETED_FACTORS,
    analyze_cell,
    assemble,
    discover_cells,
    load_cell,
)
from ..probes.instrument import EPS_FIXED, _boot_mean_ci, epsilon_g, paired_gain

# Reference palette (validated): categorical slots + chart chrome, light mode.
BLUE, AQUA, YELLOW, ORANGE = "#2a78d6", "#1baf7a", "#eda100", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, CRITICAL = "#e1e0d9", "#c3c2b7", "#d03b3b"

VERDICT_STYLE = {  # verdict class -> (fill, short label)
    "invariant_across_ladder": (BLUE, "invariant"),
    "linear_invariance_artifact": (ORANGE, "artifact (flip)"),
    "recovered_at_all_capacities": (AQUA, "recovered"),
    "inconclusive_probe_driven": (GRID, "inconclusive"),
}
N_BOOT = 1000


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK2, "text.color": INK,
        "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 7,
    })


LINE = dict(lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.5, solid_capstyle="round")


def _stamp(fig, synthetic: bool, note: str = "") -> None:
    """Footer row: methods note bottom-left, SYNTHETIC banner bottom-right."""
    if note:
        fig.text(0.01, 0.002, note, ha="left", va="bottom", color=MUTED, fontsize=7)
    if synthetic:
        fig.text(0.99, 0.002, "SYNTHETIC — PIPELINE VALIDATION ONLY",
                 ha="right", va="bottom", color=CRITICAL, fontsize=8, fontweight="bold")


def _text_on(fill: str) -> str:
    """Ink or white in-fill text, whichever clears more WCAG contrast."""
    def lin(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    L = sum(w * lin(int(fill[i:i + 2], 16)) for w, i in
            ((0.2126, 1), (0.7152, 3), (0.0722, 5)))
    return "#ffffff" if 1.05 / (L + 0.05) >= 3.0 else INK


def _save(fig, out: Path, name: str, rows: list[dict]) -> None:
    """PNG + PDF + the CSV table twin (the accessibility/verification channel)."""
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    if rows:
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"[figures] wrote {out / name}.{{png,pdf,csv}}")


def _cell_stats(cell, n_boot: int = N_BOOT) -> dict:
    """G with CIs, epsilon_G, and seed-mean h/projector/floor levels per cell."""
    g = paired_gain(cell.trained, cell.random_stats, n_boot=n_boot)
    eps = epsilon_g(cell.random_stats, n_boot=n_boot)
    eps = np.where(np.isnan(eps), EPS_FIXED, eps)
    levels = {name: _boot_mean_ci(arr, n_boot=n_boot)
              for name, arr in (("h", cell.trained), ("projector", cell.projector),
                                ("random floor", cell.random_stats))}
    return {"G": g, "eps": eps, "levels": levels}


# --- figures ----------------------------------------------------------------------

def fig_g_ladder(cell, stats, out: Path, synthetic: bool) -> None:
    """Small multiples: one panel per factor, G vs rung with CI + epsilon band."""
    names = [f.name for f in cell.factors]
    targeted = set(TARGETED_FACTORS.get(cell.condition, ()))
    R = len(cell.rungs)
    x = np.arange(R)
    ncol = 3
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.15 * nrow + 0.4),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    G, eps, rows = stats["G"], stats["eps"], []
    for fi, (ax, fname) in enumerate(zip(axes, names)):
        hit = fname in targeted
        c = BLUE if hit else MUTED
        ax.fill_between(x, -eps[fi], eps[fi], color=GRID, alpha=0.65, lw=0)
        ax.axhline(0, color=BASELINE, lw=1)
        ax.fill_between(x, G["lo"][fi], G["hi"][fi], color=c, alpha=0.10, lw=0)
        ax.plot(x, G["mean"][fi], color=c, **LINE)
        ax.set_title(fname + (" (targeted)" if hit else ""),
                     color=INK if hit else INK2,
                     fontweight="bold" if hit else "normal")
        ax.set_xticks(x, cell.rungs)
        ax.tick_params(axis="x", rotation=20)
        for ri, rname in enumerate(cell.rungs):
            rows.append({"factor": fname, "rung": rname, "targeted": hit,
                         "G_mean": round(float(G["mean"][fi, ri]), 4),
                         "G_lo": round(float(G["lo"][fi, ri]), 4),
                         "G_hi": round(float(G["hi"][fi, ri]), 4),
                         "epsilon_G": round(float(eps[fi, ri]), 4)})
    for ax in axes[len(names):]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r * ncol].set_ylabel("encoder gain G")
    fig.suptitle(f"{cell.name} ({cell.dataset}) — G across the probe-capacity ladder, "
                 f"n={cell.trained.shape[0]} seeds", x=0.01, ha="left",
                 fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, synthetic,
           "gray band: ±ε_G (invariance threshold) · color shading: 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    _save(fig, out, f"g_ladder_{cell.name}", rows)


def fig_h4(cell, stats, out: Path, synthetic: bool) -> None:
    """Targeted factors: recoverability of h vs projector vs the random floor."""
    targeted = [t for t in TARGETED_FACTORS.get(cell.condition, ())
                if t in [f.name for f in cell.factors]]
    if not targeted:
        return
    names = [f.name for f in cell.factors]
    x = np.arange(len(cell.rungs))
    series = (("h", BLUE), ("projector", AQUA), ("random floor", MUTED))
    fig, axes = plt.subplots(1, len(targeted),
                             figsize=(2.9 * len(targeted), 2.5), sharey=True)
    axes, rows = np.atleast_1d(axes), []
    for ax, t in zip(axes, targeted):
        fi = names.index(t)
        for sname, c in series:
            m, lo, hi = stats["levels"][sname]
            ax.fill_between(x, lo[fi], hi[fi], color=c, alpha=0.10, lw=0)
            ax.plot(x, m[fi], color=c, label=sname, **LINE)
            for ri, rname in enumerate(cell.rungs):
                rows.append({"factor": t, "series": sname, "rung": rname,
                             "mean": round(float(m[fi, ri]), 4),
                             "lo": round(float(lo[fi, ri]), 4),
                             "hi": round(float(hi[fi, ri]), 4)})
        ax.set_title(t, color=INK, fontweight="bold")
        ax.set_xticks(x, cell.rungs)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("recoverability")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"{cell.name} — encoder h vs projector (H4), targeted factors",
                 x=0.01, ha="left", fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, synthetic, "shading: 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    _save(fig, out, f"h4_{cell.name}", rows)


def fig_headline(cells, all_stats, out: Path, synthetic: bool) -> None:
    """Prereg 0 contrast: object hue (Color) vs x/y position (Position)."""
    picks = []  # (label, cell, factor, color) in fixed slot order
    for cname, fname, c in (("color", "object_hue", BLUE),
                            ("position", "pos_x", AQUA), ("position", "pos_y", YELLOW)):
        cell = next((k for k in cells if k.condition == cname), None)
        if cell and fname in [f.name for f in cell.factors]:
            picks.append((f"{fname} ({cell.name})", cell, fname, c))
    if len(picks) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    rows = []
    for label, cell, fname, c in picks:
        fi = [f.name for f in cell.factors].index(fname)
        G = all_stats[cell.name]["G"]
        x = np.arange(len(cell.rungs))
        ax.fill_between(x, G["lo"][fi], G["hi"][fi], color=c, alpha=0.10, lw=0)
        # legend carries identity; end-labels would collide when curves coincide
        ax.plot(x, G["mean"][fi], color=c, label=label, **LINE)
        for ri, rname in enumerate(cell.rungs):
            rows.append({"series": label, "rung": rname,
                         "G_mean": round(float(G["mean"][fi, ri]), 4),
                         "G_lo": round(float(G["lo"][fi, ri]), 4),
                         "G_hi": round(float(G["hi"][fi, ri]), 4)})
    ax.axhline(0, color=BASELINE, lw=1)
    ax.set_xticks(np.arange(len(picks[0][1].rungs)), picks[0][1].rungs)
    ax.set_xlim(-0.3, len(picks[0][1].rungs) - 0.4)
    ax.set_ylabel("encoder gain G")
    ax.legend(loc="upper left")
    ax.set_title("Headline contrast — targeted-factor G across the ladder",
                 loc="left", fontsize=9.5, fontweight="bold", color=INK)
    _stamp(fig, synthetic, "shading: 95% bootstrap CI")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, "headline", rows)


def fig_verdicts(study, out: Path, synthetic: bool) -> None:
    """Verdict matrix over (factor row, cell column), flip carried in the label."""
    table = study["verdict_table"]
    cols = sorted({t["cell"] for t in table})
    rows_f = sorted({t["factor"] for t in table})
    by = {(t["factor"], t["cell"]): t for t in table}
    fig, ax = plt.subplots(figsize=(1.0 + 1.75 * len(cols), 0.8 + 0.42 * len(rows_f)))
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(rows_f))
    ax.invert_yaxis()
    ax.axis("off")
    csv_rows = []
    for yi, fname in enumerate(rows_f):
        ax.text(-0.08, yi + 0.5, fname, ha="right", va="center", color=INK2, fontsize=7.5)
        for xi, cname in enumerate(cols):
            t = by.get((fname, cname))
            if t is None:
                continue  # factor absent from this cell's dataset
            fill, label = VERDICT_STYLE.get(t["verdict"], (GRID, t["verdict"]))
            # 2px surface gap between fills; in-fill text picked by contrast
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=fill,
                                       edgecolor=SURFACE, lw=2))
            ax.text(xi + 0.5, yi + 0.5, label, ha="center", va="center",
                    fontsize=7, color=_text_on(fill))
            csv_rows.append({"factor": fname, "cell": cname, "verdict": t["verdict"],
                             "case_linear": t["case_linear"], "case_top": t["case_top"],
                             "flip_primary": t["flip_primary"]})
    for xi, cname in enumerate(cols):
        ax.text(xi + 0.5, -0.15, cname, ha="center", va="bottom",
                color=INK, fontsize=8, fontweight="bold")
    n_flips = study["headline_flip_count"]["primary"]["n_flips"]
    ax.set_title(f"Verdicts per (factor, cell) — {n_flips} primary flip(s)"
                 + (" · provisional grid" if study["provisional"] else ""),
                 loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=24)
    _stamp(fig, synthetic)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out, "verdicts", csv_rows)


# --- driver ---------------------------------------------------------------------

def render_root(root: str | Path, out: str | Path, n_boot: int = N_BOOT) -> list[str]:
    _style()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [load_cell(d) for d in discover_cells(root)]
    if not cells:
        print(f"[figures] no cells under {root} — nothing to render")
        return []
    import json
    synthetic = any("synthetic_world" in json.loads((c.path / "meta.json").read_text())
                    for c in cells)
    if synthetic:
        (out / "README.md").write_text(
            "# SYNTHETIC figures — pipeline validation only\n\nRendered from "
            "hand-set fixtures (src/probes/synthetic.py); not empirical results.\n")
    all_stats = {c.name: _cell_stats(c, n_boot) for c in cells}
    for c in cells:
        fig_g_ladder(c, all_stats[c.name], out, synthetic)
        fig_h4(c, all_stats[c.name], out, synthetic)
    fig_headline(cells, all_stats, out, synthetic)
    study = assemble([analyze_cell(c, n_boot=n_boot) for c in cells])
    fig_verdicts(study, out, synthetic)
    return sorted(p.name for p in out.glob("*.png"))


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="results/probes", help="directory holding cell dirs")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/figures/<root basename>)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    out = args.out or Path("results/figures") / Path(args.root).name
    made = render_root(args.root, out, n_boot=args.n_boot)
    print(f"[figures] {len(made)} figure(s) in {out}")


if __name__ == "__main__":
    _main()
