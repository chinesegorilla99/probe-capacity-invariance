"""Smoke test for the figure pipeline on the synthetic fixtures.

Renders every figure from an emitted world and checks the outputs exist with
their CSV table twins and the SYNTHETIC marker. Layout/aesthetics are reviewed
visually; this guards the plumbing.

Run:  python -m unittest tests.test_figures -v
"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from src.probes.synthetic import emit_world
from src.eval.figures import render_root

N_BOOT = 100  # smoke: enough to exercise every code path


class TestFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        root = emit_world("artifact", Path(cls.tmp), seed=0)
        cls.out = Path(cls.tmp) / "figs"
        cls.made = render_root(root, cls.out, n_boot=N_BOOT)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_all_figures_rendered(self):
        expected = {"g_ladder_color_strong.png", "g_ladder_control_strong.png",
                    "g_ladder_position_strong.png", "h4_color_strong.png",
                    "h4_position_strong.png", "headline.png", "verdicts.png"}
        self.assertEqual(set(self.made), expected)
        # control has no targeted factors -> no h4 figure
        self.assertNotIn("h4_control_strong.png", self.made)

    def test_every_figure_has_pdf_and_csv_twin(self):
        for name in self.made:
            stem = Path(name).stem
            self.assertTrue((self.out / f"{stem}.pdf").exists())
            with open(self.out / f"{stem}.csv") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)

    def test_synthetic_output_is_marked(self):
        readme = self.out / "README.md"
        self.assertTrue(readme.exists())
        self.assertIn("SYNTHETIC", readme.read_text())

    def test_verdict_csv_matches_ground_truth(self):
        with open(self.out / "verdicts.csv") as f:
            by = {(r["factor"], r["cell"]): r["verdict"] for r in csv.DictReader(f)}
        self.assertEqual(by[("object_hue", "color_strong")], "linear_invariance_artifact")
        self.assertEqual(by[("pos_x", "position_strong")], "linear_invariance_artifact")
        self.assertEqual(by[("shape", "control_strong")], "recovered_at_all_capacities")


if __name__ == "__main__":
    unittest.main()
