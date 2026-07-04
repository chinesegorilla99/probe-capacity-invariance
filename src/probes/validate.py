"""Phase-2 gate — validate the probe instrument on the control encoder.

Before trusting any G/S/epsilon_G number on the trained encoders, the instrument
must reproduce expected behavior on the controls (roadmap Phase-2 gate):

  1. Capacity is monotone   — probe params strictly increase across the ladder.
  2. Random-LABEL control collapses on held-out test — a probe cannot predict
     permuted labels it never saw, so R(random labels) ~ 0 at every rung. This is
     what makes selectivity S a meaningful gate.
  3. epsilon_G null is centered and tight — random-vs-random-encoder G is ~0 with a
     small positive band, so the init-noise threshold is well-formed.
  4. Signal positive control — the trained encoder reads the augmentation-preserved
     factor (shape) well above its own random-label floor (probe reads real signal).

This exercises the instrument end to end; the full >=10-seed G/S/flip sweep is Phase 3.

    python -m src.probes.validate --config configs/probe/ladder.yaml \
        --simclr results/encoders/standard_strong_seed0/backbone.pt \
                 results/encoders/standard_strong_seed1/backbone.pt \
        --random-seed 0 1 2 --subsample 8000 --epochs 60 \
        --out results/probes/phase2_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.shapes3d import DEFAULT_PATH, FACTORS, N_TOTAL, Shapes3D
from ..data.splits import make_splits
from ..encoders.augmentations import eval_transform
from ..encoders.random_encoder import build_random_encoder
from ..eval.extract import extract_features
from ..eval.quality_gate import load_backbone
from ..utils.config import load_config
from ..utils.device import pick_device
from .instrument import build_report, epsilon_g, stack_runs
from .ladder import LADDER, param_count

RANDLABEL_MAX = 0.10   # permuted-label recoverability must collapse on test
NULL_CENTER_MAX = 0.05  # |mean random-vs-random G| must be ~0
SIGNAL_MARGIN = 0.30   # trained shape recoverability over its random-label floor

SHAPE_IDX = next(i for i, f in enumerate(FACTORS) if f.kind == "categorical")


def _features(backbone, datasets, device, bs, nw):
    return {name: extract_features(backbone, ds, device, batch_size=bs, num_workers=nw)
            for name, ds in datasets.items()}


def _subsample(feats: dict, n: int, seed: int = 0) -> dict:
    """Cap probe_train size for validation speed (real sweep uses the full fixed split)."""
    if not n:
        return feats
    rng = np.random.default_rng(seed)
    H, Y = feats["probe_train"]
    if len(H) <= n:
        return feats
    idx = rng.choice(len(H), n, replace=False)
    return {**feats, "probe_train": (H[idx], Y[idx])}


def run(args) -> dict:
    cfg = load_config(args.config)
    device = pick_device(args.device)
    bs, nw = args.batch_size, args.num_workers
    n_total = cfg["data"].get("n_total", N_TOTAL)
    splits = make_splits(n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    path = cfg["data"].get("path", DEFAULT_PATH)
    datasets = {
        name: Shapes3D(splits[name], transform=eval_transform(), path=path, return_label=True,
                       in_memory=cfg["data"].get("in_memory", False))
        for name in ("probe_train", "probe_val", "probe_test")
    }
    pkw = dict(device=device, epochs=args.epochs)

    def feats_for(bb):
        return _subsample(_features(bb, datasets, device, bs, nw), args.subsample)

    print(f"[phase2] device={device} | trained={len(args.simclr)} random={len(args.random_seed)} "
          f"seeds | probe_train<= {args.subsample or 'full'}")

    # Trained-encoder runs (paired by index with random runs for G).
    trained_runs = [(feats_for(load_backbone(ck, device)), i) for i, ck in enumerate(args.simclr)]
    random_runs = [(feats_for(build_random_encoder(s, device)), s) for s in args.random_seed]

    trained_stack = stack_runs(trained_runs, **pkw)
    random_stack = stack_runs(random_runs, **pkw)
    perm_stack = stack_runs(trained_runs, permute=True, **pkw)  # random-LABEL control

    report = build_report(trained_stack, random_stack, trained_stack, perm_stack)

    # --- gate checks ---------------------------------------------------------
    params_mono = all(
        param_count(LADDER[i], 1) < param_count(LADDER[i + 1], 1) for i in range(len(LADDER) - 1)
    )
    randlabel_max = float(perm_stack.mean(0).max())          # worst permuted-label recoverability
    randlabel_ok = randlabel_max <= RANDLABEL_MAX
    eps = epsilon_g(random_stack)
    null_center = float(np.nanmax(np.abs((random_stack[:, None] - random_stack[None]).mean((0, 1)))))
    null_ok = bool(np.isnan(eps).all() or null_center <= NULL_CENTER_MAX)
    shape_signal = float(trained_stack.mean(0)[SHAPE_IDX].max() - perm_stack.mean(0)[SHAPE_IDX].max())
    signal_ok = shape_signal >= SIGNAL_MARGIN

    passed = bool(params_mono and randlabel_ok and null_ok and signal_ok)
    checks = {
        "params_monotone": params_mono,
        "randlabel_collapses": {"ok": randlabel_ok, "max_recov": randlabel_max, "thresh": RANDLABEL_MAX},
        "epsilon_null_centered": {"ok": null_ok, "abs_mean_null_G": null_center, "thresh": NULL_CENTER_MAX},
        "signal_positive_control": {"ok": signal_ok, "shape_gap": shape_signal, "thresh": SIGNAL_MARGIN},
        "passed": passed,
    }
    for k, v in checks.items():
        if k != "passed":
            print(f"[phase2] {k}: {'ok' if (v is True or v.get('ok')) else 'FAIL'}  {v if v is not True else ''}")
    print(f"[phase2] VERDICT: {'PASS ✅' if passed else 'FAIL ❌'}")

    out = {"device": str(device), "config": args.config, "checks": checks, "report": report}
    if args.out:
        p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2))
        print(f"[phase2] wrote {p}")
    return out


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--simclr", nargs="+", required=True, help="trained backbone.pt (>=1)")
    ap.add_argument("--random-seed", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--subsample", type=int, default=8000, help="cap probe_train (0 = full)")
    ap.add_argument("--out", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
