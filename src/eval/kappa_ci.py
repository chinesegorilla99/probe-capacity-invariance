"""Seed-bootstrap CIs for the A13(b) closure fraction kappa and for probe selectivity.

Read-only. Reproduces the 2026-08-24 numbers from results/probes_pinned/*/stacks.npz.
kappa(F) = 1 - D(top)/D(linear),  D = R(random encoder) - R(trained).
S(F,c)   = R(trained) - R(random labels).
"""
import json, pathlib

import numpy as np

B, RNG = 4000, np.random.default_rng(20260824)
OUT = {}

ROOT = pathlib.Path("results/probes_pinned")
CONFIRMATORY = ["color_strong", "control_strong", "position_strong"]
# color_weak (notebook 09) is exploratory and not preregistered. Ground is A1(a),
# which defers the weak strength axis and commits that "all Holm families span the
# realized cells only" -- not A7(d), which bars removal from the grid and says
# nothing about admission (see decision log D035). Scored only when present, and
# labelled in both the printout and the JSON so the status cannot be lost.
EXPLORATORY = ["color_weak"]

cells = CONFIRMATORY + [c for c in EXPLORATORY if (ROOT / c / "stacks.npz").exists()]

for cell in cells:
    exploratory = cell in EXPLORATORY
    meta = json.load(open(f"results/probes_pinned/{cell}/meta.json"))
    facs = [f["name"] for f in meta["factors"]]
    z = np.load(f"results/probes_pinned/{cell}/stacks.npz")
    tr, rn, pm = z["trained"], z["random"], z["perm"]      # (seed, factor, rung)
    D = rn - tr
    S = tr - pm
    n = tr.shape[0]
    rows = []
    for i, f in enumerate(facs):
        dl, dt = D[:, i, 0], D[:, i, -1]
        # kappa is only defined where there IS a linear-rung deficit to close.
        # A negative D(linear) flips the ratio's sign and makes the number unreadable.
        defined = dl.mean() > 0
        k_hat = 1.0 - dt.mean() / dl.mean()
        idx = RNG.integers(0, n, size=(B, n))                # paired seed resample
        num, den = dt[idx].mean(1), dl[idx].mean(1)
        kb = 1.0 - num / den
        kb = kb[np.isfinite(kb)]
        k_lo, k_hi = np.percentile(kb, [2.5, 97.5])
        s_top = S[:, i, -1]
        sb = s_top[idx].mean(1)
        s_lo, s_hi = np.percentile(sb, [2.5, 97.5])
        if not defined:
            k_hat, k_lo, k_hi = float("nan"), float("nan"), float("nan")
        rows.append(dict(factor=f, exploratory=exploratory, kappa_defined=bool(defined),
                         D_linear=float(dl.mean()), D_top=float(dt.mean()),
                         kappa=float(k_hat), kappa_ci95=[float(k_lo), float(k_hi)],
                         S_top=float(s_top.mean()), S_top_ci95=[float(s_lo), float(s_hi)],
                         R_trained_lin=float(tr[:, i, 0].mean()),
                         R_trained_top=float(tr[:, i, -1].mean()),
                         R_random_lin=float(rn[:, i, 0].mean()),
                         R_random_top=float(rn[:, i, -1].mean()),
                         R_perm_lin=float(pm[:, i, 0].mean()),
                         R_perm_top=float(pm[:, i, -1].mean())))
    OUT[cell] = rows
    tag = "  [EXPLORATORY -- not preregistered, no Holm family]" if exploratory else ""
    print(f"== {cell} (n={n} seeds, B={B}) =={tag}")
    for r in rows:
        k = (f"kappa={r['kappa']:+.4f} [{r['kappa_ci95'][0]:+.4f}, {r['kappa_ci95'][1]:+.4f}]"
             if r['kappa_defined'] else "kappa=UNDEFINED (D_linear <= 0)          ")
        print(f"  {r['factor']:12s} D_lin={r['D_linear']:+.4f} D_top={r['D_top']:+.4f} {k}  "
              f"S_top={r['S_top']:+.4f} [{r['S_top_ci95'][0]:+.4f}, {r['S_top_ci95'][1]:+.4f}]")
    print()

out = pathlib.Path("results/hypotheses/kappa_ci.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(OUT, indent=1))
print(f"wrote {out}")
