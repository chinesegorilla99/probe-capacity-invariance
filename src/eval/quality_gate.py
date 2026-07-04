"""Encoder-quality gate (prereg §5) — the Phase-1 trustworthiness check.

Freeze each encoder, fit a SINGLE linear probe per factor, and report normalized
recoverability for SimCLR vs the random-encoder floor vs the supervised-reference
ceiling. PASS requires the SimCLR encoder to (1) recover the augmentation-preserved
semantic factor (shape) well and above the untrained floor, and (2) show the
invariance signature — fall below the floor on the augmentation-targeted (hue)
factors — reproducibly across seeds. Required before any probing/metric work (Phase 2).

The floor is near-ceiling under a linear probe and the augmentation deliberately
suppresses some factors, so the gate does NOT ask SimCLR to beat the floor on
every factor. These are Phase-1 trustworthiness thresholds, deliberately distinct
from the frozen Phase-3 epsilon_G metric layer (not built here).

    python -m src.eval.quality_gate --config configs/run/pilot_mps.yaml \
        --simclr results/encoders/standard_strong_seed0/backbone.pt \
        --supervised results/encoders/supervised_seed0/backbone.pt \
        --random-seed 0 --out results/quality_gate/pilot.json

    # re-verdict a saved report with no feature extraction:
    python -m src.eval.quality_gate --from-report results/quality_gate/reference.json
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
from ..encoders.backbone import ResNet18CIFAR
from ..encoders.random_encoder import build_random_encoder
from ..utils.config import load_config
from ..utils.device import pick_device
from .extract import extract_features
from .metrics import (
    linear_recoverability_categorical,
    linear_recoverability_continuous,
)

# Gate thresholds (Phase-1 trustworthiness, NOT the frozen epsilon_G). The gate
# tests what a well-trained contrastive encoder should do under the standard
# augmentation, rather than assuming it beats the untrained floor on every factor
# (it cannot: a linear probe already saturates on random features, and the
# augmentation deliberately induces invariance to some factors).
SHAPE_ABS_MIN = 0.90         # recover the augmentation-preserved semantic factor well
STRUCTURE_MARGIN = 0.02      # ...and above the random-encoder floor
INVARIANCE_MARGIN = 0.02     # fall below the floor on augmentation-targeted factors
MIN_INVARIANT_FACTORS = 2    # ...on at least this many of them
REPRO_MAX_SPREAD = 0.05      # max shape-recoverability spread across seeds

# How the standard/strong augmentation (augmentations.py) maps onto factors:
#   ColorJitter + RandomGrayscale -> hue factors (invariance target)
#   no standard augmentation alters object shape -> shape is preserved
STRUCTURE_FACTOR = "shape"
COLOR_TARGETED = ("floor_hue", "wall_hue", "object_hue")

SHAPE_IDX = next(f.index for f in FACTORS if f.kind == "categorical")


def load_backbone(ckpt_path: str | Path, device: torch.device) -> ResNet18CIFAR:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet18CIFAR()
    model.load_state_dict(ckpt["backbone"])
    return model.to(device).eval()


def _features_for_splits(backbone, datasets, device, bs, nw):
    return {
        name: extract_features(backbone, ds, device, batch_size=bs, num_workers=nw)
        for name, ds in datasets.items()
    }


def evaluate_encoder(name: str, feats: dict) -> dict:
    """Per-factor normalized recoverability for one encoder's extracted features."""
    Htr, Ytr = feats["probe_train"]
    Hva, Yva = feats["probe_val"]
    Hte, Yte = feats["probe_test"]

    out: dict[str, dict] = {}
    for fac in FACTORS:
        if fac.kind == "categorical":
            ytr = np.rint(Ytr[:, fac.index]).astype(int)
            yva = np.rint(Yva[:, fac.index]).astype(int)
            yte = np.rint(Yte[:, fac.index]).astype(int)
            out[fac.name] = linear_recoverability_categorical(
                Htr, ytr, Hva, yva, Hte, yte, fac.chance
            )
        else:
            out[fac.name] = linear_recoverability_continuous(
                Htr, Ytr[:, fac.index], Hva, Yva[:, fac.index], Hte, Yte[:, fac.index]
            )
    return out


def evaluate_gate(simclr: dict, random: dict) -> dict:
    # Structure: recover the augmentation-preserved semantic factor well, and
    # above the untrained floor (a collapsed encoder could not).
    shape_s = simclr[STRUCTURE_FACTOR]["recoverability"]
    shape_r = random[STRUCTURE_FACTOR]["recoverability"]
    structure_ok = shape_s >= SHAPE_ABS_MIN and shape_s > shape_r + STRUCTURE_MARGIN

    # Invariance signature: fall below the untrained floor on the factors the
    # augmentation targets for invariance (positive evidence it worked).
    invariant = [
        f
        for f in COLOR_TARGETED
        if simclr[f]["recoverability"] < random[f]["recoverability"] - INVARIANCE_MARGIN
    ]
    invariance_ok = len(invariant) >= MIN_INVARIANT_FACTORS

    return {
        "passed": bool(structure_ok and invariance_ok),
        "structure_ok": bool(structure_ok),
        "structure_factor": STRUCTURE_FACTOR,
        "structure_recoverability": shape_s,
        "structure_random": shape_r,
        "invariance_ok": bool(invariance_ok),
        "invariant_factors": invariant,
        "n_invariant": len(invariant),
        "criteria": {
            "shape_abs_min": SHAPE_ABS_MIN,
            "structure_margin": STRUCTURE_MARGIN,
            "invariance_margin": INVARIANCE_MARGIN,
            "min_invariant_factors": MIN_INVARIANT_FACTORS,
        },
    }


def _print_verdict(gate: dict, repro: dict, passed: bool) -> None:
    print(
        f"\n[gate] structure: {gate['structure_factor']} recoverability (SimCLR) = "
        f"{gate['structure_recoverability']:+.3f} (random {gate['structure_random']:+.3f}, "
        f"need >= {SHAPE_ABS_MIN} and > random) -> {'ok' if gate['structure_ok'] else 'FAIL'}"
    )
    print(
        f"[gate] invariance: SimCLR below random on {gate['n_invariant']}/{len(COLOR_TARGETED)} "
        f"color factors {gate['invariant_factors']} (need >= {MIN_INVARIANT_FACTORS}) "
        f"-> {'ok' if gate['invariance_ok'] else 'FAIL'}"
    )
    if repro.get("n_seeds", 0) > 1:
        print(
            f"[gate] reproducibility: shape spread = {repro['shape_spread']:.3f} "
            f"(need <= {REPRO_MAX_SPREAD}) -> {'ok' if repro['repro_ok'] else 'FAIL'}"
        )
    print(f"[gate] VERDICT: {'PASS ✅' if passed else 'FAIL ❌'}")


def _print_table(table: dict, encoders: list[str]) -> None:
    width = 13
    header = "factor".ljust(width) + "type".ljust(7) + "".join(
        e.ljust(width) for e in encoders
    )
    print("\n" + header)
    print("-" * len(header))
    for fac in FACTORS:
        row = fac.name.ljust(width) + ("cat" if fac.kind == "categorical" else "cont").ljust(7)
        for e in encoders:
            v = table[e][fac.name]["recoverability"]
            row += f"{v:+.3f}".ljust(width)
        print(row)


def run_gate(args) -> dict:
    cfg = load_config(args.config)
    device = pick_device(args.device)
    bs, nw = args.batch_size, args.num_workers

    n_total = cfg["data"].get("n_total", N_TOTAL)
    splits = make_splits(n_total, cfg["split"]["sizes"], cfg["split"]["split_seed"])
    path = cfg["data"].get("path", DEFAULT_PATH)
    print(f"[gate] device={device} | building probe splits "
          f"(train={len(splits['probe_train'])}, val={len(splits['probe_val'])}, "
          f"test={len(splits['probe_test'])})")
    datasets = {
        name: Shapes3D(splits[name], transform=eval_transform(), path=path, return_label=True,
                       in_memory=cfg["data"].get("in_memory", False))
        for name in ("probe_train", "probe_val", "probe_test")
    }

    table: dict[str, dict] = {}

    # Random-encoder floor (defines G later); average over given seeds if multiple.
    rand_seed = args.random_seed[0]
    rand_bb = build_random_encoder(rand_seed, device)
    table["random"] = evaluate_encoder(
        "random", _features_for_splits(rand_bb, datasets, device, bs, nw)
    )

    # SimCLR (primary = first ckpt; extras used for the reproducibility check).
    simclr_per_seed = {}
    for i, ck in enumerate(args.simclr):
        bb = load_backbone(ck, device)
        res = evaluate_encoder(f"simclr{i}", _features_for_splits(bb, datasets, device, bs, nw))
        simclr_per_seed[str(ck)] = res
        if i == 0:
            table["simclr"] = res

    encoders = ["random", "simclr"]
    if args.supervised:
        sup_bb = load_backbone(args.supervised, device)
        table["supervised"] = evaluate_encoder(
            "supervised", _features_for_splits(sup_bb, datasets, device, bs, nw)
        )
        encoders.append("supervised")

    _print_table(table, encoders)

    gate = evaluate_gate(table["simclr"], table["random"])

    # Reproducibility across SimCLR seeds: spread of shape recoverability.
    shape_by_seed = {
        k: v["shape"]["recoverability"] for k, v in simclr_per_seed.items()
    }
    spread = (max(shape_by_seed.values()) - min(shape_by_seed.values())) if len(shape_by_seed) > 1 else 0.0
    repro_ok = spread <= REPRO_MAX_SPREAD
    repro = {
        "shape_recoverability_by_ckpt": shape_by_seed,
        "shape_spread": spread,
        "n_seeds": len(shape_by_seed),
        "repro_ok": bool(repro_ok),
    }

    passed = gate["passed"] and repro_ok
    _print_verdict(gate, repro, passed)

    report = {
        "device": str(device),
        "config": args.config,
        "table": table,
        "gate": {**gate, "repro_ok": bool(repro_ok), "passed_overall": bool(passed)},
        "reproducibility": repro,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[gate] wrote {out}")
    return report


def reverdict_from_report(path: str | Path) -> dict:
    """Recompute the verdict from a saved report — no feature extraction."""
    report = json.loads(Path(path).read_text())
    table = report["table"]
    gate = evaluate_gate(table["simclr"], table["random"])
    repro = dict(report.get("reproducibility", {}))
    spread = repro.get("shape_spread", 0.0)
    repro["repro_ok"] = spread <= REPRO_MAX_SPREAD
    passed = gate["passed"] and repro["repro_ok"]

    _print_table(table, [e for e in ("random", "simclr", "supervised") if e in table])
    _print_verdict(gate, repro, passed)
    return {"gate": {**gate, "repro_ok": repro["repro_ok"], "passed_overall": bool(passed)}}


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--simclr", nargs="+", default=None, help="one or more backbone.pt")
    ap.add_argument("--supervised", default=None)
    ap.add_argument("--random-seed", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-report", default=None,
                    help="recompute the verdict from a saved reference.json (no extraction)")
    args = ap.parse_args()
    if args.from_report:
        reverdict_from_report(args.from_report)
        return
    if not args.config or not args.simclr:
        ap.error("--config and --simclr are required unless --from-report is given")
    run_gate(args)


if __name__ == "__main__":
    _main()
