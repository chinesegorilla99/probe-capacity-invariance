"""Phase-3 sweep driver — produce the per-cell recoverability stacks (the contract).

For ONE cell (condition, strength, dataset) this fits the probe-capacity ladder
across every trained-encoder seed and the random-encoder floor, and writes the
artifact the H1-H4 statistics layer consumes:

    results/probes/<condition>_<strength>/
        stacks.npz   trained  [S_t, F, R]   trained-encoder recoverability (h)
                     random   [S_r, F, R]   random-encoder floor  (S_r >= 10)
                     perm     [S_t, F, R]   trained encoder, permuted labels (for S)
                     projector[S_t, F, R]   trained-encoder projector features (H4)
        meta.json    dataset / condition / strength, factor + rung metadata,
                     per-rung param counts, seed lists, per-encoder gate results

Axes: S = seeds, F = factors (dataset factor order), R = ladder rungs (linear ->
mlp_deep). The schema is STABLE — the stats session keys off it. Reuses
``instrument.stack_runs``; the H1-H4 tests live in a separate session.

Features are extracted and probed one encoder at a time so peak RAM stays at a
single encoder's features regardless of seed count.

    python -m src.probes.run_sweep --config configs/probe/ladder.yaml \
        --dataset shapes3d --condition color --strength strong \
        --encoders results/encoders/color_strong_seed*/backbone.pt \
        --random-seed 0 1 2 3 4 5 6 7 8 9 \
        --epochs 100 --num-workers 2 --out-root results/probes
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..data.registry import get_dataset
from ..data.splits import make_splits
from ..encoders.augmentations import eval_transform
from ..encoders.random_encoder import build_random_encoder
from ..eval.encoder_gate import gate_summary, per_seed_gate
from ..eval.extract import (
    extract_features,
    extract_projector_features,
    load_backbone_projector,
)
from ..utils.config import load_config
from ..utils.device import pick_device
from .instrument import stack_runs
from .ladder import LADDER, param_count

RUNG_NAMES = tuple(r.name for r in LADDER)


def _seed_from_path(path: str | Path) -> int:
    """Extract the encoder seed from its run directory name (``..._seed<N>``)."""
    m = re.search(r"seed(\d+)", Path(path).as_posix())
    if not m:
        raise ValueError(f"cannot parse seed from {path!r} (expected ..._seed<N>)")
    return int(m.group(1))


def _subsample(feats: dict, n: int, seed: int = 0) -> dict:
    if not n:
        return feats
    rng = np.random.default_rng(seed)
    H, Y = feats["probe_train"]
    if len(H) <= n:
        return feats
    idx = rng.choice(len(H), n, replace=False)
    return {**feats, "probe_train": (H[idx], Y[idx])}


def _out_dim(fac) -> int:
    return round(1.0 / fac.chance) if fac.kind == "categorical" else 1


def run(args) -> dict:
    cfg = load_config(args.config)
    spec = get_dataset(args.dataset)
    device = pick_device(args.device)
    bs, nw = args.batch_size, args.num_workers
    path = args.data_path or spec.default_path
    splits = make_splits(spec.n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    split_names = ("probe_train", "probe_val", "probe_test")
    datasets = {
        name: spec.cls(splits[name], transform=eval_transform(), path=path,
                       return_label=True, in_memory=args.in_memory)
        for name in split_names
    }
    pkw = dict(device=device, epochs=args.epochs, factors=spec.factors)

    def feats_for(backbone) -> dict:
        f = {n: extract_features(backbone, datasets[n], device, bs, nw) for n in split_names}
        return _subsample(f, args.subsample)

    def proj_feats_for(backbone, projector) -> dict:
        f = {n: extract_projector_features(backbone, projector, datasets[n], device, bs, nw)
             for n in split_names}
        return _subsample(f, args.subsample)

    enc_paths = sorted(args.encoders, key=_seed_from_path)
    print(f"[sweep] {args.condition}_{args.strength} on {spec.name} | device={device} | "
          f"{len(enc_paths)} trained, {len(args.random_seed)} random | "
          f"probe_train<={args.subsample or 'full'}")
    if len(enc_paths) < 10:
        print(f"[sweep] WARNING: only {len(enc_paths)} trained encoders (<10 = under-powered).")
    if len(args.random_seed) < 10:
        print(f"[sweep] WARNING: only {len(args.random_seed)} random seeds "
              f"(<10 = epsilon_G under-powered, prereg/D020).")

    # --- trained encoders: h, permuted-label, and projector stacks (one at a time) ---
    trained_rows, perm_rows, proj_rows, trained_seeds = [], [], [], []
    proj_out_dim = None
    for ck in enc_paths:
        seed = _seed_from_path(ck)
        backbone, projector = load_backbone_projector(ck, device)
        proj_out_dim = projector.out_dim
        fh = feats_for(backbone)
        trained_rows.append(stack_runs([(fh, seed)], **pkw))
        perm_rows.append(stack_runs([(fh, seed)], permute=True, **pkw))
        del fh
        fp = proj_feats_for(backbone, projector)
        proj_rows.append(stack_runs([(fp, seed)], **pkw))
        del fp, backbone, projector
        trained_seeds.append(seed)
        print(f"[sweep]   trained seed {seed}: probed h + projector")

    trained_stack = np.concatenate(trained_rows)
    perm_stack = np.concatenate(perm_rows)
    projector_stack = np.concatenate(proj_rows)

    # --- random-encoder floor (defines G / epsilon_G) ---
    random_rows = []
    for rs in args.random_seed:
        bb = build_random_encoder(rs, device)
        fr = feats_for(bb)
        random_rows.append(stack_runs([(fr, rs)], **pkw))
        del fr, bb
        print(f"[sweep]   random seed {rs}: probed h")
    random_stack = np.concatenate(random_rows)

    # --- per-encoder quality gate (prereg §5) from the linear-rung shape recov ---
    gates = per_seed_gate(trained_stack, random_stack, spec.factors)
    gate = gate_summary(gates)
    if not gate["all_passed"]:
        print(f"[sweep] QUALITY-GATE: {gate['n_passed']}/{gate['n_encoders']} passed — "
              f"failed seed idx {gate['failed_seed_indices']} (still written; excluded downstream).")
    else:
        print(f"[sweep] QUALITY-GATE: all {gate['n_encoders']} encoders passed.")

    # --- write the contract artifact ---
    out_dir = Path(args.out_root) / f"{args.condition}_{args.strength}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "stacks.npz",
        trained=trained_stack.astype(np.float32),
        random=random_stack.astype(np.float32),
        perm=perm_stack.astype(np.float32),
        projector=projector_stack.astype(np.float32),
    )
    meta = {
        "dataset": spec.name,
        "condition": args.condition,
        "strength": args.strength,
        "factors": [
            {"name": f.name, "kind": f.kind, "index": f.index,
             "n_values": f.n_values, "cyclic": f.cyclic}
            for f in spec.factors
        ],
        "rungs": list(RUNG_NAMES),
        "rung_params_h": {
            f.name: [param_count(r, _out_dim(f), 512) for r in LADDER] for f in spec.factors
        },
        "rung_params_projector": {
            f.name: [param_count(r, _out_dim(f), int(proj_out_dim)) for r in LADDER]
            for f in spec.factors
        },
        "projector_dim": int(proj_out_dim),
        "seeds": {"trained": trained_seeds, "random": list(args.random_seed)},
        "probe_train_size": int(len(datasets["probe_train"]) if not args.subsample
                                else args.subsample),
        "quality_gate": gate,
        "schema": {
            "stacks.npz": {
                "trained": ["S_trained", "F", "R"],
                "random": ["S_random", "F", "R"],
                "perm": ["S_trained", "F", "R"],
                "projector": ["S_trained", "F", "R"],
            },
            "axis_order": {"F": "meta.factors order", "R": "meta.rungs order (linear->mlp_deep)"},
            "recoverability": "norm-acc (categorical) | unclipped R^2 (continuous)",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[sweep] wrote {out_dir/'stacks.npz'} and {out_dir/'meta.json'} "
          f"(trained {trained_stack.shape}, random {random_stack.shape})")
    return {"out_dir": str(out_dir), "meta": meta}


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default="shapes3d", choices=["shapes3d", "dsprites"])
    ap.add_argument("--condition", required=True)
    ap.add_argument("--strength", required=True)
    ap.add_argument("--encoders", nargs="+", required=True, help="trained backbone.pt (>=10)")
    ap.add_argument("--random-seed", type=int, nargs="+",
                    default=list(range(10)), help=">=10 for a powered epsilon_G")
    ap.add_argument("--data-path", default=None, help="override dataset path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100, help="probe training epochs")
    ap.add_argument("--subsample", type=int, default=0, help="cap probe_train (0 = full/fixed)")
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--out-root", default="results/probes")
    run(ap.parse_args())


if __name__ == "__main__":
    _main()
