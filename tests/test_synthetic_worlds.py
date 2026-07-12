"""End-to-end two-world validation of the frozen H1-H4 decision procedure.

Emits the full first-slice grid under both hand-set ground truths (null /
artifact, src/probes/synthetic.py) at the sweep seed count (12, D025) and
asserts the UNCHANGED stats layer recovers each world's truth — including H2,
which is decidable at n=12 but not at n=10 (Q15; checked both ways here).
Complements tests/test_hypotheses.py (per-cell unit tests) at the
study-assembly level.

Run:  python -m unittest tests.test_synthetic_worlds -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.probes.synthetic import (
    ALPHA,
    N_SEEDS,
    WORLDS,
    emit_all,
    emit_world,
    h2_floor,
    validate_world,
)

N_BOOT = 400


class TestSyntheticWorlds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.roots = emit_all(cls.tmp, seed=0)  # sweep seed count (12, D025)
        cls.studies, cls.fails = {}, {}
        for w in WORLDS:
            cls.studies[w], cls.fails[w] = validate_world(w, cls.roots[w], n_boot=N_BOOT)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_fixtures_are_tagged_synthetic(self):
        metas = list(Path(self.tmp).glob("*/*/meta.json"))
        self.assertEqual(len(metas), 6)  # 3 cells x 2 worlds
        for m in metas:
            meta = json.loads(m.read_text())
            self.assertIn("synthetic_world", meta)
            self.assertEqual(len(meta["seeds"]["trained"]), N_SEEDS)
        self.assertTrue((Path(self.tmp) / "README.md").exists())

    def test_null_world_recovered(self):
        self.assertEqual(self.fails["null"], [])

    def test_artifact_world_recovered(self):
        self.assertEqual(self.fails["artifact"], [])

    def test_worlds_are_distinguished(self):
        # the same frozen procedure must give opposite answers per world
        h = {w: self.studies[w]["hypotheses"] for w in WORLDS}
        for hyp in ("H1", "H2", "H3"):
            self.assertNotEqual(h["null"][hyp]["status"], h["artifact"][hyp]["status"])
        flips = {w: self.studies[w]["headline_flip_count"]["primary"]["n_flips"]
                 for w in WORLDS}
        self.assertEqual(flips["null"], 0)
        self.assertEqual(flips["artifact"], 5)

    def test_h2_decidable_at_sweep_seed_count(self):
        # D025: at n=12 the true heterogeneity clears the assembled Holm family
        h2 = self.studies["artifact"]["hypotheses"]["H2"]
        self.assertEqual(h2["status"], "confirmed")
        self.assertLess(h2_floor(N_SEEDS, len(h2["pairs"])), ALPHA)
        # and stays clean under the null (no false positive)
        self.assertEqual(self.studies["null"]["hypotheses"]["H2"]["status"], "refuted")

    def test_h2_power_cliff_at_n10(self):
        # Q15: identical artifact ground truth, n=10 -> H2 structurally refuted
        tmp = tempfile.mkdtemp()
        try:
            root = emit_world("artifact", Path(tmp), n_t=10, n_r=10, seed=0)
            study, fails = validate_world("artifact", root, n_boot=N_BOOT, n_t=10)
            self.assertEqual(fails, [])  # expected() knows n=10 is undecidable
            pairs = study["hypotheses"]["H2"]["pairs"]
            self.assertEqual(study["hypotheses"]["H2"]["status"], "refuted")
            self.assertGreater(h2_floor(10, len(pairs)), ALPHA)
            self.assertGreater(min(e["p_holm"] for e in pairs), ALPHA)
            # the information is present — only the confirm rule can't reach it
            self.assertTrue(any(e["ci_excludes_0"] for e in pairs))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
