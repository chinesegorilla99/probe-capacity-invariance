"""Raw-pixel reference ladder (prereg Amendment A4 §b) — diagnostic only.

Fits the same four-rung ladder, under the identical protocol (standardization on
probe-train statistics, weight-decay grid on probe-val, fixed probe-train size,
probe-test-only reporting), on FLATTENED RAW PIXELS (identity encoder). This
quantifies what the input alone affords each probe family and contextualizes the
random-encoder floor (a random CNN is a function of pixels). It enters no
decision rule and no confirmatory family.

Seeds are probe seeds only ({0, 1, 2} per A4 — there is no encoder seed axis).
Writes results/probes/pixel_<dataset>.json.

    python -m src.probes.pixel_baseline --config configs/probe/ladder.yaml \
        --dataset shapes3d --seeds 0 1 2 --out-root results/probes
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
from ..eval.extract import extract_features
from ..utils.config import load_config
from ..utils.device import pick_device
from .instrument import ladder_recoverability
from .ladder import LADDER, param_count

RUNG_NAMES = tuple(r.name for r in LADDER)
A4_SEEDS = (0, 1, 2)


def run(args) -> dict:
    cfg = load_config(args.config)
    spec = get_dataset(args.dataset)
    device = pick_device(args.device)
    path = args.data_path or spec.default_path
    splits = make_splits(spec.n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    split_names = ("probe_train", "probe_val", "probe_test")

    identity = nn.Flatten(start_dim=1).to(device)  # the "encoder" is the input itself
    feats = {}
    for name in split_names:
        ds = spec.cls(splits[name], transform=eval_transform(), path=path,
                      return_label=True, in_memory=args.in_memory)
        feats[name] = extract_features(identity, ds, device, args.batch_size,
                                       args.num_workers)
        print(f"[pixel] {name}: {feats[name][0].shape}")
    in_dim = feats["probe_train"][0].shape[1]

    stacks = []
    for seed in args.seeds:
        recov, params, metrics = ladder_recoverability(
            feats, seed=seed, factors=spec.factors, device=device, epochs=args.epochs)
        stacks.append(recov)
        print(f"[pixel] seed {seed}: done")
    stack = np.stack(stacks)  # [S, F, R]

    out = {
        "kind": "pixel_baseline",
        "amendment": "A4",
        "role": "diagnostic only — no decision rule, no confirmatory family",
        "dataset": spec.name,
        "in_dim": int(in_dim),
        "rungs": list(RUNG_NAMES),
        "rung_params": {
            f.name: [param_count(r, round(1 / f.chance) if f.kind == "categorical" else 1,
                                 int(in_dim)) for r in LADDER]
            for f in spec.factors
        },
        "seeds": list(args.seeds),
        "probe_train_size": int(len(feats["probe_train"][0])),
        "epochs": args.epochs,
        "recoverability": {
            f.name: {
                "kind": f.kind,
                "per_seed": [[float(x) for x in stack[si, fi]] for si in range(len(args.seeds))],
                "mean": [float(x) for x in stack[:, fi].mean(0)],
            }
            for fi, f in enumerate(spec.factors)
        },
    }
    out_path = Path(args.out_root) / f"pixel_{spec.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[pixel] wrote {out_path}")
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/probe/ladder.yaml")
    ap.add_argument("--dataset", default="shapes3d", choices=["shapes3d", "dsprites"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(A4_SEEDS))
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100, help="probe training epochs")
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--out-root", default="results/probes")
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
