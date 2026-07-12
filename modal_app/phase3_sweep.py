"""Phase 3 encoder sweep — Modal edition of notebooks/kaggle_phase3_sweep.ipynb.

Trains the 36-encoder grid (3 conditions x 12 seeds) on GPU workers, then probes
the frozen encoders into the per-cell stacks.npz the statistics layer consumes.
Persistent state (encoders, logs, probe outputs, dataset cache) lives on two
Modal Volumes so it survives across runs the way /kaggle/working persisted
across Kaggle sessions.

One-time setup:
    pip install modal
    modal setup                                    # authenticate this machine
    modal volume create phase3-data
    modal volume create phase3-results

Push local commits first — the image is built by git-cloning the public repo,
so uncommitted local changes are not visible inside the container.

Usage (from the repo root):
    modal run modal_app/phase3_sweep.py::verify_gpu
    modal run modal_app/phase3_sweep.py::download_data
    modal run modal_app/phase3_sweep.py::run_sweep
    modal run modal_app/phase3_sweep.py::progress
    modal run modal_app/phase3_sweep.py::quality_gate_check
    modal run modal_app/phase3_sweep.py::run_probes

To change GPU type or how many run in parallel, edit the ``gpu=`` and
``max_containers=`` arguments on ``train_cell`` / ``probe_sweep`` below —
Modal has no CUDA_VISIBLE_DEVICES pinning step like the Kaggle notebook;
each parallel worker gets its own container and its own GPU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO_URL = "https://github.com/chinesegorilla99/probe-capacity-invariance.git"
REPO_DIR = "/root/probe-capacity-invariance"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(f"git clone {REPO_URL} {REPO_DIR}")
    .run_commands(f"cd {REPO_DIR} && pip install -e . && pip install h5py")
)

app = modal.App("phase3-sweep", image=image)

data_vol = modal.Volume.from_name("phase3-data", create_if_missing=True)
results_vol = modal.Volume.from_name("phase3-results", create_if_missing=True)

VOLUMES = {
    f"{REPO_DIR}/data/raw": data_vol,
    f"{REPO_DIR}/results": results_vol,
}

SEEDS = list(range(12))
CONFIGS = [
    ("configs/run/color_strong.yaml", "color", "strong", "shapes3d", []),
    ("configs/run/control_strong.yaml", "control", "strong", "shapes3d", []),
    ("configs/run/position_strong.yaml", "position", "strong", "dsprites",
     ["--data-path", "data/raw/dsprites.npz"]),
]
TRAIN_JOBS = [(cfg, cond, strg, s) for (cfg, cond, strg, _ds, _extra) in CONFIGS for s in SEEDS]


def _run_id(cond: str, strg: str, seed: int) -> str:
    return f"{cond}_{strg}_seed{seed}"


@app.function(gpu="T4", timeout=120)
def verify_gpu():
    subprocess.run(["nvidia-smi"], check=True)


@app.function(cpu=2, memory=4096, volumes=VOLUMES, timeout=1800)
def download_data():
    subprocess.run(["python", "-m", "src.data.shapes3d", "--download", "--build-cache"],
                    cwd=REPO_DIR, check=True)
    subprocess.run(["python", "-m", "src.data.dsprites", "--download", "--build-cache"],
                    cwd=REPO_DIR, check=True)
    data_vol.commit()


# gpu / max_containers mirror the Kaggle "dual-T4" setup: 2 workers in parallel.
@app.function(gpu="L4", volumes=VOLUMES, timeout=6 * 3600, max_containers=2)
def train_cell(cfg: str, cond: str, strg: str, seed: int) -> tuple[str, str]:
    rid = _run_id(cond, strg, seed)
    backbone = Path(REPO_DIR) / "results" / "encoders" / rid / "backbone.pt"
    if backbone.exists():
        return rid, "skipped (done)"
    cmd = ["python", "-m", "src.encoders.train_simclr", "--config", cfg,
           "--set", f"run.seed={seed}", "run.num_workers=2", "run.device=cuda"]
    rc = subprocess.run(cmd, cwd=REPO_DIR).returncode
    results_vol.commit()
    return rid, ("ok" if rc == 0 else f"failed (exit {rc})")


@app.local_entrypoint()
def run_sweep():
    """Equivalent of notebook section 6 — trains every not-yet-done cell."""
    todo = [j for j in TRAIN_JOBS if not
            Path(f"results/encoders/{_run_id(j[1], j[2], j[3])}/backbone.pt").exists()]
    print(f"{len(TRAIN_JOBS)} cells total, {len(todo)} dispatched to Modal "
          f"(remainder already have a local backbone.pt — train_cell also "
          f"re-checks the volume and skips done cells server-side)")
    for rid, status in train_cell.starmap(TRAIN_JOBS):
        print(f"{rid}: {status}")


@app.function(volumes=VOLUMES, timeout=120)
def progress():
    """Equivalent of notebook section 8."""
    import json
    from collections import Counter

    EST_GPU_H_PER_CELL = 5.0
    enc = Path(REPO_DIR) / "results" / "encoders"
    rows = []
    for cfg, cond, strg, ds, extra in CONFIGS:
        for s in SEEDS:
            rid = _run_id(cond, strg, s)
            d = enc / rid
            if (d / "backbone.pt").exists():
                state = "done"
            elif (d / "last_ckpt.pt").exists():
                state = "partial"
            else:
                state = "todo"
            rows.append((rid, state))
    c = Counter(st for _, st in rows)
    print(f"done={c['done']}  partial={c['partial']}  todo={c['todo']}  (of {len(rows)})\n")
    for rid, state in rows:
        print(f"  {rid:28s} {state}")
    remaining = c["todo"] + 0.5 * c["partial"]
    print(f"\n~{remaining * EST_GPU_H_PER_CELL:.0f} GPU-h remaining")


@app.function(gpu="L4", volumes=VOLUMES, timeout=1800)
def quality_gate_check():
    """Equivalent of notebook section 9 — sanity-check the first two encoders."""
    import json as _json

    for rid in ["color_strong_seed0", "position_strong_seed0"]:
        mp = Path(REPO_DIR) / "results" / "encoders" / rid / "metrics.json"
        if not mp.exists():
            print(f"{rid}: not trained yet")
            continue
        m = _json.loads(mp.read_text())
        d = m.get("diagnostics", {})
        print(f"{rid}: loss={m['final_loss']:.4f} nan={m['nan_aborted']} epochs={m['epochs_run']} | "
              f"feat_std={d.get('feat_std'):.4f} eff_rank={d.get('eff_rank'):.1f} "
              f"align={d.get('alignment'):.3f} unif={d.get('uniformity'):.3f}")

    subprocess.run(
        ["python", "-m", "src.eval.quality_gate",
         "--config", "configs/run/color_strong.yaml",
         "--simclr", "results/encoders/color_strong_seed0/backbone.pt",
         "--random-seed", "0", "--out", "results/quality_gate/color_strong_seed0.json"],
        cwd=REPO_DIR, check=True,
    )
    results_vol.commit()


@app.function(gpu="L4", volumes=VOLUMES, timeout=6 * 3600, max_containers=1)
def probe_sweep(condition: str, strength: str, dataset: str, extra: list[str]):
    """Equivalent of one iteration of notebook section 10. Probing is
    feature-extraction + small-MLP fits — single GPU is enough, no need
    to parallelize this stage across containers."""
    import glob

    tag = f"{condition}_{strength}"
    out = Path(REPO_DIR) / "results" / "probes" / tag / "stacks.npz"
    if out.exists():
        print(f"{tag}: stacks.npz already present -> skip")
        return

    encs = sorted(glob.glob(f"{REPO_DIR}/results/encoders/{tag}_seed*/backbone.pt"))
    if len(encs) < len(SEEDS):
        print(f"{tag}: only {len(encs)}/{len(SEEDS)} encoders trained -> skip (train first)")
        return

    cmd = ["python", "-m", "src.probes.run_sweep",
           "--config", "configs/probe/ladder.yaml",
           "--dataset", dataset, "--condition", condition, "--strength", strength,
           "--encoders", *encs,
           "--random-seed", *[str(i) for i in range(len(SEEDS))],
           "--epochs", "100", "--num-workers", "2", *extra]
    subprocess.run(cmd, cwd=REPO_DIR, check=True)
    results_vol.commit()


@app.local_entrypoint()
def run_probes():
    """Equivalent of notebook section 10 — probes all three cells back-to-back."""
    for cfg, cond, strg, ds, extra in CONFIGS:
        probe_sweep.remote(cond, strg, ds, extra)
