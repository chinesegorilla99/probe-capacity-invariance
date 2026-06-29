# Probe Capacity & Contrastive Invariance

**Working title:** *When "Invariant" Means "Linearly Inaccessible": A Controlled Study of Probe Capacity in Contrastive Learning* (provisional — final title is chosen from the results).

**One-sentence thesis.** When self-supervised contrastive models are described as "invariant" to some factor (color, position, rotation), that invariance is almost always *measured* with a linear probe — so a linear probe's failure to recover the factor may reflect the probe's weakness rather than the representation's content; this project tests, under fully controlled conditions, whether the invariance survives stronger probes or dissolves into a measurement artifact.

See [ResearchOverview.md](ResearchOverview.md) for the full design (source of truth) and [preregistration/prereg.md](preregistration/prereg.md) for the locked hypotheses.

## Status

Scaffolding only. **No research, experiment, training, probing, or metric code is implemented yet** — directories are intentionally empty (`# TODO (do not implement yet)`).

## Layout

| Path | Purpose |
|---|---|
| `data/raw/` | Externally-sourced raw datasets (Shapes3D, dSprites). Git-ignored. |
| `data/synthetic/` | Generated/derived factor datasets & cached tensors. Git-ignored. |
| `src/encoders/` | SimCLR-style SSL encoder training (backbone `f` + projector `g`). Empty. |
| `src/probes/` | Capacity-ladder probes (linear → MLP rungs) over frozen reps. Empty. |
| `src/eval/` | Metrics: recoverability, selectivity, capacity gap, verdict stability. Empty. |
| `configs/` | YAML experiment configs. Empty. |
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

## Workflow / tooling

- **experiment-agent** — runs the code experiments; slots in **between the experiment phase and the writing phase** (see ResearchOverview Phase timeline).
