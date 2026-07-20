# One command per regenerable artifact. Every number and figure in the paper
# must be reproducible from these targets plus the configs; verify artifact
# integrity with `make manifest-check`.

PY ?= .venv/bin/python

.PHONY: test hypotheses figures pixel-shapes3d pixel-dsprites manifest manifest-check

test:
	$(PY) -m unittest discover -s tests

# Confirmatory layer on the real cells (unblinding step — run only when the
# realized grid is complete): per-cell reports + assembled confirm/refute table.
hypotheses:
	$(PY) -m src.probes.hypotheses --root results/probes

figures:
	$(PY) -m src.eval.figures --root results/probes

# A4 raw-pixel reference ladder (diagnostic only).
pixel-shapes3d:
	$(PY) -m src.probes.pixel_baseline --dataset shapes3d

pixel-dsprites:
	$(PY) -m src.probes.pixel_baseline --dataset dsprites

manifest:
	$(PY) -m src.eval.manifest

manifest-check:
	$(PY) -m src.eval.manifest --check
