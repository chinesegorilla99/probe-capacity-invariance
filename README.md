# Probe Capacity & Contrastive Invariance

**Working title:** *When "Invariant" Means "Linearly Inaccessible": A Controlled Study of Probe Capacity in Contrastive Learning* (provisional — final title is chosen from the results).

**One-sentence thesis.** When self-supervised contrastive models are described as "invariant" to some factor (color, position, rotation), that invariance is almost always *measured* with a linear probe — so a linear probe's failure to recover the factor may reflect the probe's weakness rather than the representation's content; this project tests, under fully controlled conditions, whether the invariance survives stronger probes or dissolves into a measurement artifact.

See [ResearchOverview.md](ResearchOverview.md) for the full design (source of truth) and [preregistration/prereg.md](preregistration/prereg.md) for the locked hypotheses.

## Status

Phase 1–3 in progress. SimCLR encoder training, the probe-capacity ladder, and the encoder-quality gate are implemented; metric and analysis code lands as phases complete. See [ResearchOverview.md](ResearchOverview.md) for the phase timeline.

## Layout

| Path | Purpose |
|---|---|
| `data/raw/` | Externally-sourced raw datasets (Shapes3D, dSprites). Git-ignored. |
| `data/synthetic/` | Generated/derived factor datasets & cached tensors. Git-ignored. |
| `src/encoders/` | SimCLR-style SSL encoder training (backbone `f` + projector `g`). |
| `src/probes/` | Capacity-ladder probes (linear → MLP rungs) over frozen reps. |
| `src/eval/` | Metrics: recoverability, selectivity, capacity gap, verdict stability. |
| `configs/` | YAML experiment configs (`run/`, `probe/`). |
| `results/` | Run outputs, tables, figures. Git-ignored. |
| `notebooks/` | Exploratory analysis / figures. |
| `preregistration/` | Frozen, dated prereg of H1–H4 + definitions. |
| `seeds.py` | Reproducibility / seeding convention placeholder (no logic yet). |

## Environment setup

Python **3.10–3.12** is recommended (PyTorch wheels may not yet exist for 3.13/3.14).

```bash
# Create and activate a venv with a 3.10–3.12 interpreter, e.g.:
python3.12 -m venv .venv
source .venv/bin/activate          # macOS/Linux

# Then install dependencies (run this yourself):
pip install --upgrade pip
pip install -e .
```

Dependencies (declared, not pinned, in [pyproject.toml](pyproject.toml)): torch, torchvision, numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, tqdm.

On Windows, create the venv with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\activate` instead of `source .venv/bin/activate`. For GPU training, install the CUDA build of PyTorch **before** `pip install -e .` (the default wheel is CPU-only), e.g.:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Pick the `cuXXX` tag that matches your driver from the [official selector](https://pytorch.org/get-started/locally/). Verify the GPU is visible before training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running the experiments

Run every command from the repo root with the venv active.

**1. Fetch datasets and build image caches** (one-time):

```bash
python -m src.data.shapes3d --download --build-cache
python -m src.data.dsprites --download --build-cache
```

**2. Train an encoder.** One experiment cell is `(condition, strength, seed)`; the seed controls both init and data order. A single seed:

```bash
python -m src.encoders.train_simclr --config configs/run/color_strong.yaml --set run.seed=0
```

A quick 1-epoch sanity run on a tiny subset (verify the config and device before committing a full cell):

```bash
python -m src.encoders.train_simclr --config configs/run/control_strong.yaml --set run.seed=0 run.epochs=1 data.subset=2000
```

Train ≥10 seeds per cell. All 10 seeds of a cell, sequentially:

```bash
# macOS/Linux / Git Bash
for s in 0 1 2 3 4 5 6 7 8 9; do \
  python -m src.encoders.train_simclr --config configs/run/control_strong.yaml --set run.seed=$s; done
```

```powershell
# Windows PowerShell
foreach ($s in 0..9) { python -m src.encoders.train_simclr --config configs/run/control_strong.yaml --set run.seed=$s }
```

Swap the config for `position_strong.yaml` (dSprites) or `color_strong.yaml` (Shapes3D). Outputs land in `results/encoders/<condition>_<strength>_seed<seed>/`, and a re-launched run resumes from the last-epoch checkpoint, so finished seeds skip instantly. Run configs live in `configs/run/` (`color_strong`, `control_strong`, `position_strong`, plus the `supervised` and reference baselines).

**3. Probe the trained encoders** into the analysis stacks:

```bash
python -m src.probes.run_sweep --config configs/probe/ladder.yaml \
  --dataset shapes3d --condition color --strength strong \
  --encoders results/encoders/color_strong_seed*/backbone.pt \
  --random-seed 0 1 2 3 4 5 6 7 8 9 --epochs 100
```

For the position arm use `--dataset dsprites --condition position --data-path data/raw/dsprites.npz`. The `--encoders` glob expands in the shell; on Windows PowerShell, list the `backbone.pt` paths explicitly or drive the grid from a notebook.

**Local CUDA notes.** The device is auto-selected (CUDA > MPS > CPU). On CUDA, AMP and `torch.compile` are enabled automatically; override with `--set run.amp=false run.compile=false`. On a consumer GPU you can cap board power before launching (e.g. `nvidia-smi -pl <watts>`) if the card is unstable under sustained load.

## Workflow / tooling

- **experiment-agent** — runs the code experiments; slots in **between the experiment phase and the writing phase** (see ResearchOverview Phase timeline).
