"""Run many SimCLR seeds concurrently across the available GPUs.

One arm is 12 independent single-GPU runs. ResNet18 at 64x64 with batch 512 leaves
a T4 badly underutilized, so packing several seeds per GPU raises aggregate
throughput without touching the recipe: each child is an ordinary
``train_simclr`` invocation with its own seed and its own CUDA device, and the
config it reads is unchanged.

Resume-aware: a seed whose ``backbone.pt`` already exists is skipped, so a
timed-out session continues where it stopped.

    python -m src.encoders.launch_parallel \
        --config configs/run/scale_recipe_control.yaml \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --per-gpu 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ..utils.config import apply_overrides, load_config

LOG_DIR = Path("results/encoders/_launch_logs")


def run_dir(cfg: dict, seed: int) -> Path:
    """Where train_simclr will write this seed (mirrors its run_id_from_cfg)."""
    aug = cfg["augmentation"]
    return (Path(cfg["output"]["dir"])
            / f"{aug['condition']}_{aug['strength']}_seed{seed}")


def visible_gpus() -> list[int]:
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _spawn(cfg_path: str, seed: int, gpu: int | None, overrides: list[str],
           cwd: Path) -> tuple[subprocess.Popen, Path]:
    env = dict(os.environ)
    if gpu is not None:
        # One visible device per child, so each child's "cuda" is its own GPU and
        # nothing needs to know about the others.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"seed{seed}.log"
    cmd = [sys.executable, "-m", "src.encoders.train_simclr", "--config", cfg_path,
           "--set", f"run.seed={seed}", *overrides]
    handle = log.open("w")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle,
                            stderr=subprocess.STDOUT)
    proc._log_handle = handle  # closed in the reaper below
    return proc, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--per-gpu", type=int, default=3,
                    help="concurrent runs per GPU (VRAM permitting)")
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="GPU ids; default every visible GPU, else CPU-serial")
    ap.add_argument("--set", nargs="*", default=[],
                    help="extra dotted overrides passed to every child")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cwd = Path.cwd()
    cfg = apply_overrides(load_config(args.config), args.set)
    gpus = args.gpus if args.gpus is not None else visible_gpus()
    slots = max(1, len(gpus) * args.per_gpu) if gpus else 1

    pending = [s for s in args.seeds if not (run_dir(cfg, s) / "backbone.pt").exists()]
    done = [s for s in args.seeds if s not in pending]
    print(f"[launch] {len(done)} seed(s) already complete: {done}")
    print(f"[launch] {len(pending)} to run: {pending}")
    print(f"[launch] {len(gpus) or 'no'} GPU(s) {gpus}, {args.per_gpu} per GPU "
          f"-> {slots} concurrent slot(s)")
    if args.dry_run or not pending:
        return

    queue = list(pending)
    running: dict[int, tuple[subprocess.Popen, Path, int | None, float]] = {}
    failed: list[int] = []
    t0 = time.time()

    while queue or running:
        while queue and len(running) < slots:
            seed = queue.pop(0)
            gpu = gpus[len(running) % len(gpus)] if gpus else None
            proc, log = _spawn(args.config, seed, gpu, args.set, cwd)
            running[seed] = (proc, log, gpu, time.time())
            print(f"[launch] seed {seed:2d} -> gpu {gpu} (pid {proc.pid})", flush=True)

        time.sleep(args.poll)
        for seed in list(running):
            proc, log, gpu, started = running[seed]
            if proc.poll() is None:
                continue
            proc._log_handle.close()
            mins = (time.time() - started) / 60
            if proc.returncode == 0:
                print(f"[launch] seed {seed:2d} DONE in {mins:.1f} min", flush=True)
            else:
                failed.append(seed)
                tail = "\n".join(log.read_text().splitlines()[-15:])
                print(f"[launch] seed {seed:2d} FAILED rc={proc.returncode} "
                      f"after {mins:.1f} min\n{tail}", flush=True)
            del running[seed]

    print(f"[launch] wall clock {(time.time() - t0) / 60:.1f} min")
    if failed:
        raise SystemExit(f"[launch] {len(failed)} seed(s) failed: {sorted(failed)}; "
                         f"logs in {LOG_DIR}")
    print("[launch] all seeds complete")


if __name__ == "__main__":
    main()
