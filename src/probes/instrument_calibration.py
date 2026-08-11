"""Instrument calibration (prereg A8 §c/§e) — BLIND-SAFE, diagnostic only.

Two checks that must run BEFORE the confirmatory layer, and that are safe to run
because neither touches a targeted factor:

(A) NULL-HEADROOM (A8 §c). Both datasets are complete factorial grids split by
    random index, so probe-test is dense lattice interpolation and an untrained
    CNN plus a high-capacity probe already solves it. That saturates the
    random-encoder floor and leaves ``G`` no headroom. This sweeps probe-train
    size, gradient-step budget and the test split, and reports where the floor
    drops below the A4 saturation level so the study can pin a probe
    configuration with measurable range. Three test splits: ``interpolation``
    (the study's registered split), ``composition`` (unseen factor COMBINATIONS,
    the categorical-safe hard split) and ``extrapolation`` (unseen factor VALUES,
    skipped for a categorical anchor, where it scores 0 by construction).

(B) CAPACITY-AXIS VALIDITY (A8 §e). The ladder is a nested function class, so the
    residual risk on the capacity axis is OPTIMIZATION, not expressivity. The
    ladder's Adam linear rung is cross-checked against a closed-form / convex
    solver on identical features; a gap at the linear rung inflates every
    ``Delta_G``.

BLIND GUARD: only the categorical **shape** anchor (the §5 gate factor, orthogonal
to H1-H4) and the raw-pixel reference are scored. Targeted factors are never
touched — see ``_anchor_only``.

    python -m src.probes.instrument_calibration --dataset shapes3d \
        --probe-train-sizes 2000 5000 10000 40000 --probe-steps 1000 4000 \
        --partner-factor orientation --random-seed 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch.nn as nn

from ..data.registry import get_dataset
from ..data.splits import (
    make_combination_holdout_splits,
    make_splits,
    make_value_holdout_splits,
)
from ..encoders.augmentations import eval_transform
from ..encoders.random_encoder import build_random_encoder
from ..eval.extract import extract_features
from ..eval.metrics import linear_recoverability_categorical
from ..utils.config import load_config
from ..utils.device import pick_device
from .instrument import SATURATION_LEVEL
from .ladder import LADDER, fit_ladder

RUNG_NAMES = tuple(r.name for r in LADDER)


def _anchor_only(factors):
    """The categorical shape anchor — the ONLY readout this module may score."""
    cat = [f for f in factors if f.kind == "categorical"]
    if len(cat) != 1:
        raise ValueError(f"expected exactly one categorical anchor, got {[f.name for f in cat]}")
    return cat[0]


def _partner_factor(factors, anchor, name=None):
    """The factor whose values are withheld per anchor value in the compositional split."""
    if name:
        match = [f for f in factors if f.name == name]
        if not match:
            raise ValueError(f"unknown partner factor {name!r}; "
                             f"have {[f.name for f in factors]}")
        return match[0]
    return max((f for f in factors if f.name != anchor.name), key=lambda f: f.n_values)


def _subsample(H, y, n, seed=0):
    if not n or len(H) <= n:
        return H, y
    idx = np.random.default_rng(seed).choice(len(H), n, replace=False)
    return H[idx], y[idx]


def _feats(encoder, spec, idx, path, device, bs, nw, in_memory):
    ds = spec.cls(idx, transform=eval_transform(), path=path, return_label=True,
                  in_memory=in_memory)
    return extract_features(encoder, ds, device, bs, nw)


def run(args) -> dict:
    cfg = load_config(args.config)
    spec = get_dataset(args.dataset)
    device = pick_device(args.device)
    path = args.data_path or spec.default_path
    anchor = _anchor_only(spec.factors)
    base = make_splits(spec.n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])

    # Labels for the whole dataset drive the extrapolation split. Cheap: labels
    # only, never images.
    ds_all = spec.cls(np.arange(spec.n_total), transform=None, path=path, return_label=True)
    labels_all = np.asarray(ds_all.labels)
    del ds_all

    regimes = {"interpolation": base}
    partner = None
    if args.extrapolation:
        # Withholding whole VALUES of a class label leaves probe-test holding only
        # classes absent from probe-train, so accuracy is 0 by construction and the
        # cell reports the zero-accuracy floor whatever the encoder did (the first
        # run returned -1/3 in all 8 such cells). Compositional holdout replaces it.
        if anchor.kind == "categorical":
            print(f"[calib] SKIP extrapolation: '{anchor.name}' is categorical, so a "
                  f"value holdout scores 0 by construction — use --composition")
        else:
            regimes["extrapolation"] = make_value_holdout_splits(
                labels_all, anchor.index, base=base, n_holdout=args.n_holdout,
                split_seed=cfg["split"]["split_seed"])
    if args.composition:
        partner = _partner_factor(spec.factors, anchor, args.partner_factor)
        regimes["composition"] = make_combination_holdout_splits(
            labels_all, anchor.index, partner.index, base=base,
            n_holdout=args.n_holdout_partner, split_seed=cfg["split"]["split_seed"])
        print(f"[calib] composition regime: withholding {args.n_holdout_partner} "
              f"'{partner.name}' values per '{anchor.name}' value — "
              f"{len(regimes['composition']['probe_test'])} probe-test images")

    encoders = {"random": [build_random_encoder(s, device) for s in args.random_seed]}
    if args.pixel_reference:
        encoders["pixels"] = [nn.Flatten(start_dim=1).to(device)]   # A4 §b identity

    out: dict = {
        "kind": "instrument_calibration",
        "amendment": "A8 §c (headroom) + A8 §e (capacity axis)",
        "role": "diagnostic only — no decision rule, no confirmatory family",
        "blind_guard": f"scored on the '{anchor.name}' anchor only (§5 gate factor)",
        "dataset": spec.name,
        "rungs": list(RUNG_NAMES),
        "saturation_level": SATURATION_LEVEL,
        "regimes": list(regimes),
        "composition_partner": partner.name if partner else None,
        "gap_tolerance": args.gap_tolerance,
        "results": [],
    }

    for regime, splits in regimes.items():
        for size in args.probe_train_sizes:
            for role, models in encoders.items():
                # Features depend on (regime, role, seed) but not on the step budget,
                # so the budget sweep sits inside the seed loop and re-uses them.
                floors = {s: [] for s in args.probe_steps}
                lin_closed = []
                for si, model in enumerate(models):
                    f = {n: _feats(model, spec, splits[n], path, device,
                                   args.batch_size, args.num_workers, args.in_memory)
                         for n in ("probe_train", "probe_val", "probe_test")}
                    Htr, Ytr = _subsample(*f["probe_train"], size)
                    Hva, Yva = f["probe_val"]
                    Hte, Yte = f["probe_test"]
                    ytr, yva, yte = (np.rint(Y[:, anchor.index]).astype(int)
                                     for Y in (Ytr, Yva, Yte))

                    for steps in args.probe_steps:
                        rungs = fit_ladder(Htr, ytr, Hva, yva, Hte, yte, anchor.kind,
                                           anchor.chance, seed=si, device=device,
                                           epochs=args.epochs, steps=steps)
                        floors[steps].append([r["recoverability"] for r in rungs])
                    # A8 §e: same quantity, convex solver, identical features.
                    lin_closed.append(linear_recoverability_categorical(
                        Htr, ytr, Hva, yva, Hte, yte, anchor.chance)["recoverability"])
                    del f, Htr, Hva, Hte

                closed = float(np.mean(lin_closed))
                for steps in args.probe_steps:
                    arr = np.asarray(floors[steps])            # [S, R]
                    mean = arr.mean(0)
                    row = {
                        "regime": regime,
                        "probe_train_size": int(size),
                        "probe_steps": int(steps),
                        "encoder_role": role,
                        "n_seeds": int(arr.shape[0]),
                        "anchor_recoverability_mean": [float(x) for x in mean],
                        "anchor_recoverability_per_seed": arr.tolist(),
                        "top_rung_floor": float(mean[-1]),
                        "saturated_at_top": bool(mean[-1] >= SATURATION_LEVEL),
                        "saturated_at_linear": bool(mean[0] >= SATURATION_LEVEL),
                        "headroom_at_top": float(1.0 - mean[-1]),
                        "linear_rung_ladder_adam": float(mean[0]),
                        "linear_rung_closed_form": closed,
                        "linear_rung_gap": float(closed - mean[0]),
                        "capacity_axis_flag": bool(abs(closed - mean[0]) > args.gap_tolerance),
                    }
                    out["results"].append(row)
                    print(f"[calib] {regime:15s} n={size:<6d} steps={steps:<5d} {role:7s} "
                          f"top-floor={row['top_rung_floor']:.4f} "
                          f"headroom={row['headroom_at_top']:.4f} "
                          f"lin-gap={row['linear_rung_gap']:+.4f}")

    # A pin requires BOTH A8 §c (measurable range at the top rung) and A8 §e (a linear
    # rung that agrees with the convex solver). The first run's selector tested only §c
    # and sorted extrapolation first, so a regime that fails by construction — and is
    # therefore never saturated — won automatically.
    usable = [r for r in out["results"]
              if r["encoder_role"] == "random"
              and not r["saturated_at_top"]
              and not r["capacity_axis_flag"]]
    out["recommended_config"] = (
        min(usable, key=lambda r: (r["regime"] != "interpolation",
                                   -r["probe_train_size"], r["probe_steps"]))
        if usable else None
    )
    out["headroom_only"] = [                       # clears §c, still fails §e
        {k: r[k] for k in ("regime", "probe_train_size", "probe_steps",
                           "top_rung_floor", "linear_rung_gap")}
        for r in out["results"]
        if r["encoder_role"] == "random" and not r["saturated_at_top"]
        and r["capacity_axis_flag"]
    ]
    out["verdict"] = (
        "no swept configuration clears BOTH A8 §c (headroom at the top rung) and A8 §e "
        "(linear-rung agreement within tolerance); nothing may be pinned from this run"
        if not usable else
        f"pin probe_train={out['recommended_config']['probe_train_size']} "
        f"steps={out['recommended_config']['probe_steps']} under the "
        f"{out['recommended_config']['regime']} regime"
    )
    out_path = Path(args.out_root) / f"calibration_{spec.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[calib] {out['verdict']}\n[calib] wrote {out_path}")
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/probe/ladder.yaml")
    ap.add_argument("--dataset", default="shapes3d", choices=["shapes3d", "dsprites"])
    ap.add_argument("--probe-train-sizes", type=int, nargs="+",
                    default=[2000, 5000, 10000, 40000])
    ap.add_argument("--random-seed", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--probe-steps", type=int, nargs="+", default=[1000],
                    help="gradient-step budgets to fit each rung under (A9); several "
                         "values show whether the A8 §e gap closes with budget")
    ap.add_argument("--n-holdout", type=int, default=2,
                    help="factor values withheld from probe-train in the extrapolation regime")
    ap.add_argument("--extrapolation", action="store_true", default=True)
    ap.add_argument("--no-extrapolation", dest="extrapolation", action="store_false")
    ap.add_argument("--composition", action="store_true", default=True)
    ap.add_argument("--no-composition", dest="composition", action="store_false")
    ap.add_argument("--partner-factor", default=None,
                    help="factor withheld per anchor value in the compositional regime "
                         "(default: the highest-cardinality non-anchor factor)")
    ap.add_argument("--n-holdout-partner", type=int, default=3,
                    help="partner values withheld per anchor value")
    ap.add_argument("--pixel-reference", action="store_true", default=True)
    ap.add_argument("--no-pixel-reference", dest="pixel_reference", action="store_false")
    ap.add_argument("--gap-tolerance", type=float, default=0.02,
                    help="A8 §e: disclose a linear-rung solver disagreement above this")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--out-root", default="results/calibration")
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
