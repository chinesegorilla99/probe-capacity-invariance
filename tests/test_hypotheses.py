"""Unit tests for the H1-H4 statistics layer on synthetic stacks (run_sweep schema).

Synthetic cell design (Shapes3D factor set, base recoverability 0.5):
    floor_hue    null: trained rides the paired random rows -> G ~ 0 everywhere
    wall_hue     suppressed: G = -0.3 at every rung
    object_hue   capacity artifact: G = [-0.02, 0.15, 0.30, 0.40] across rungs
    scale        genuine flat: G = 0.5 at every rung
    shape        genuine flat categorical: G = 0.3 (no within-type partner tested vs R^2)
    orientation  null (like floor_hue)
Perm stack ~ 0 (unique-instance setting, D018) so S ~ R(real). The projector
stack sits `proj_gap` below trained h on the three hue factors (H4 signal).

Run:  python -m unittest tests.test_hypotheses -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.probes.hypotheses import (
    EXPECTED_CELLS,
    _wilcoxon_p,
    analyze_cell,
    assemble,
    epsilon_diagnostics,
    holm,
    load_cell,
)
from src.probes.instrument import paired_gain, selectivity

RUNGS = ["linear", "mlp_small", "mlp_large", "mlp_deep"]
FACTORS_META = [
    {"name": "floor_hue", "kind": "continuous", "index": 0, "n_values": 10, "cyclic": False},
    {"name": "wall_hue", "kind": "continuous", "index": 1, "n_values": 10, "cyclic": False},
    {"name": "object_hue", "kind": "continuous", "index": 2, "n_values": 10, "cyclic": False},
    {"name": "scale", "kind": "continuous", "index": 3, "n_values": 8, "cyclic": False},
    {"name": "shape", "kind": "categorical", "index": 4, "n_values": 4, "cyclic": False},
    {"name": "orientation", "kind": "continuous", "index": 5, "n_values": 15, "cyclic": True},
]
N_BOOT = 500  # plenty for stable CIs at these effect sizes; keeps tests fast


def make_stacks(rng, n_t=10, n_r=10, proj_gap=0.3):
    F, R = 6, 4
    base = 0.5
    random = base + rng.normal(0, 0.01, (n_r, F, R))
    g_target = np.zeros((F, R))
    g_target[1] = -0.3
    g_target[2] = [-0.02, 0.15, 0.30, 0.40]
    g_target[3] = 0.5
    g_target[4] = 0.3
    trained = np.empty((n_t, F, R))
    for f in (0, 5):  # null factors: per-seed G tightly ~0
        trained[:, f] = random[np.arange(n_t) % n_r, f] + rng.normal(0, 0.003, (n_t, R))
    for f in (1, 2, 3, 4):
        trained[:, f] = base + g_target[f] + rng.normal(0, 0.01, (n_t, R))
    perm = rng.normal(0, 0.01, (n_t, F, R))
    projector = trained.copy()
    for f in (0, 1, 2):
        projector[:, f] -= proj_gap
    return trained, random, perm, projector


def write_raw(tmp, *, condition, strength, trained, random, perm, projector,
              trained_seeds, random_seeds, failed=()):
    """Write one cell dir exactly matching the run_sweep contract."""
    d = Path(tmp) / f"{condition}_{strength}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "stacks.npz",
        trained=trained.astype(np.float32),
        random=random.astype(np.float32),
        perm=perm.astype(np.float32),
        projector=projector.astype(np.float32),
    )
    meta = {
        "dataset": "shapes3d",
        "condition": condition,
        "strength": strength,
        "factors": FACTORS_META,
        "rungs": RUNGS,
        "rung_params_h": {f["name"]: [513, 32897, 131585, 164353] for f in FACTORS_META},
        "projector_dim": 128,
        "seeds": {"trained": list(trained_seeds), "random": list(random_seeds)},
        "probe_train_size": 40000,
        "quality_gate": {
            "n_encoders": len(trained_seeds),
            "n_passed": len(trained_seeds) - len(failed),
            "failed_seed_indices": list(failed),
            "all_passed": not failed,
        },
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def write_cell(tmp, *, condition="color", strength="strong", n_t=10, n_r=10,
               failed=(), proj_gap=0.3, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    trained, random, perm, projector = make_stacks(rng, n_t, n_r, proj_gap)
    return write_raw(tmp, condition=condition, strength=strength, trained=trained,
                     random=random, perm=perm, projector=projector,
                     trained_seeds=range(n_t), random_seeds=range(n_r), failed=failed)


class TestHypotheses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cell_dir = write_cell(cls.tmp)
        cls.res = analyze_cell(load_cell(cls.cell_dir), n_boot=N_BOOT)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    # --- helpers ---------------------------------------------------------

    def test_holm_known_values(self):
        adj = holm([0.01, 0.04, 0.03, 0.005])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])

    def test_wilcoxon_zero_guard(self):
        self.assertEqual(_wilcoxon_p(np.zeros(10), "greater"), 1.0)

    def test_shared_bootstrap_g_and_s(self):
        # identical per-seed deltas + the common bootstrap rng => identical CIs,
        # i.e. the G and S gates resample the same seed draws
        rng = np.random.default_rng(7)
        t, o = rng.normal(size=(10, 6, 4)), rng.normal(size=(10, 6, 4))
        G, S = paired_gain(t, o, n_boot=N_BOOT), selectivity(t, o, n_boot=N_BOOT)
        for k in ("mean", "lo", "hi"):
            np.testing.assert_array_equal(G[k], S[k])

    # --- loading / schema --------------------------------------------------

    def test_gate_exclusion_and_pairing(self):
        d = write_cell(self.tmp, condition="gate", failed=(3,))
        cell = load_cell(d)
        self.assertEqual(cell.trained.shape[0], 9)
        self.assertEqual(cell.trained_seeds, [0, 1, 2, 4, 5, 6, 7, 8, 9])
        # paired random rows follow the surviving seed values; spare row appended
        random_full = np.asarray(np.load(d / "stacks.npz")["random"], float)
        expected = random_full[[0, 1, 2, 4, 5, 6, 7, 8, 9, 3]]
        np.testing.assert_allclose(cell.random_stats, expected)
        self.assertTrue(any("gate-failed" in w for w in cell.warnings))
        self.assertTrue(any("under-powered" in w for w in cell.warnings))

    def test_seed_value_alignment(self):
        # random seeds listed in reverse: row k belongs to seed 9-k; pairing must
        # reorder rows by seed VALUE, not position
        n, F, R = 10, 6, 4
        trained = np.zeros((n, F, R))
        random = np.stack([np.full((F, R), 9 - k, float) for k in range(n)])
        d = write_raw(self.tmp, condition="align", strength="strong",
                      trained=trained, random=random, perm=trained, projector=trained,
                      trained_seeds=range(10), random_seeds=list(reversed(range(10))))
        cell = load_cell(d)
        np.testing.assert_allclose(cell.random_stats[:, 0, 0], np.arange(10, dtype=float))

    def test_positional_fallback_on_disjoint_seeds(self):
        d = write_cell(self.tmp, condition="disjoint")
        meta = json.loads((d / "meta.json").read_text())
        meta["seeds"]["random"] = list(range(100, 110))
        (d / "meta.json").write_text(json.dumps(meta))
        cell = load_cell(d)
        self.assertTrue(any("positionally" in w for w in cell.warnings))

    def test_epsilon_underpowered_flag(self):
        d = write_cell(self.tmp, condition="fewrand", n_r=5)
        res = analyze_cell(load_cell(d), n_boot=N_BOOT)
        self.assertTrue(res["epsilon_underpowered"])
        self.assertTrue(any("diagnostic-only" in w for w in res["warnings"]))

    # --- per-cell H1-H4 -----------------------------------------------------

    def test_h1_capacity_dependence(self):
        self.assertEqual(self.res["h1"]["confirmed_factors"], ["object_hue"])
        self.assertEqual(self.res["h1"]["confirmed_factors_fixed_0.05"], ["object_hue"])
        row = next(r for r in self.res["h1"]["per_factor"] if r["factor"] == "object_hue")
        self.assertGreater(row["delta_g_ci"][0], row["epsilon_delta_g"])
        self.assertGreater(row["s_top_ci_lo"], 0)
        self.assertLess(row["p_wilcoxon_greater"], 0.01)

    def test_h2_within_type_only_and_heterogeneity(self):
        pairs = self.res["h2"]["pairs"]
        self.assertEqual(len(pairs), 10)  # C(5,2) continuous pairs; shape excluded
        for e in pairs:
            self.assertNotIn("shape", e["pair"])
        sig = {tuple(e["pair"]) for e in pairs if e["p_holm_cell"] < 0.05}
        expected_sig = {("floor_hue", "object_hue"), ("wall_hue", "object_hue"),
                        ("object_hue", "scale"), ("object_hue", "orientation")}
        self.assertEqual(sig, expected_sig)
        null_pair = next(e for e in pairs if e["pair"] == ["floor_hue", "wall_hue"])
        self.assertGreater(null_pair["p_holm_cell"], 0.05)
        self.assertLess(abs(null_pair["mean_diff"]), 0.05)

    def test_h3_invariance_and_subcases(self):
        by_factor = {r["factor"]: r for r in self.res["h3"]["per_factor"]}
        self.assertTrue(by_factor["floor_hue"]["invariant_all_rungs"])
        self.assertTrue(by_factor["wall_hue"]["invariant_all_rungs"])
        self.assertTrue(by_factor["orientation"]["invariant_all_rungs"])
        self.assertEqual(by_factor["wall_hue"]["subcase"], "suppressed")
        self.assertNotEqual(by_factor["floor_hue"]["subcase"], "suppressed")
        for f in ("object_hue", "scale", "shape"):
            self.assertFalse(by_factor[f]["invariant_all_rungs"])
        self.assertTrue(self.res["h3"]["confirmed"])

    def test_h4_encoder_vs_projector(self):
        self.assertEqual(self.res["h4"]["targeted_factors"],
                         ["floor_hue", "wall_hue", "object_hue"])
        tests = self.res["h4"]["tests"]
        self.assertEqual(len(tests), 12)  # 3 targeted factors x 4 rungs
        for t in tests:
            self.assertLess(t["p_holm_cell"], 0.05)
            self.assertGreater(t["ci"][0], 0)

    def test_flip_count(self):
        rep = self.res["report"]
        self.assertEqual(rep["flips_primary"]["flipped_factors"], ["object_hue"])
        self.assertEqual(rep["flips_fixed_0.05"]["flipped_factors"], ["object_hue"])

    def test_flip_uncertainty(self):
        # A1 §c: seed-bootstrap at fixed eps. object_hue flips in essentially
        # every draw; under the tight primary eps the null factors hug the band,
        # so an occasional draw adds a boundary flip (the fragility this report
        # exists to expose) — the far-from-boundary fixed-0.05 eps stays exact.
        for key in ("primary", "fixed_0.05"):
            fu = self.res["flip_uncertainty"][key]
            self.assertGreater(fu["per_factor_flip_fraction"]["object_hue"], 0.9)
            self.assertLess(fu["per_factor_flip_fraction"]["scale"], 0.1)
            self.assertGreaterEqual(fu["n_flips_ci95"][0], 1.0)
            self.assertLessEqual(fu["n_flips_ci95"][1], 2.0)
            self.assertAlmostEqual(fu["n_flips_mean"], 1.0, delta=0.3)
        self.assertEqual(self.res["flip_uncertainty"]["fixed_0.05"]["n_flips_ci95"],
                         [1.0, 1.0])
        self.assertEqual(set(self.res["_flip_draws"]),
                         {"primary", "fixed_0.05",
                          "primary_excl_null_saturated", "fixed_0.05_excl_null_saturated"})

    def test_null_saturation_absent_at_low_floor(self):
        # A4: floors sit at ~0.5 here, so nothing is saturated and the
        # saturation-excluded flip lists equal the primary ones
        ns = self.res["null_saturation"]
        self.assertEqual(ns["level"], 0.90)
        self.assertEqual(ns["saturated_factors_flip"], [])
        rep = self.res["report"]
        for key in ("flips_primary", "flips_fixed_0.05"):
            self.assertEqual(rep[key + "_excl_null_saturated"]["flipped_factors"],
                             rep[key]["flipped_factors"])
            self.assertEqual(rep[key + "_excl_null_saturated"]["excluded_null_saturated"], [])

    def test_levels_reported(self):
        # A1 §c: absolute R levels co-reported; trained mean must match the stack
        lv = self.res["levels"]
        self.assertEqual(set(lv), {"trained", "random_floor", "projector"})
        cell = load_cell(self.cell_dir)
        np.testing.assert_allclose(lv["trained"]["scale"]["mean"],
                                   cell.trained.mean(0)[3], rtol=1e-6)
        np.testing.assert_allclose(lv["random_floor"]["scale"]["mean"],
                                   cell.random_stats.mean(0)[3], rtol=1e-6)
        for r in range(4):  # floor ~BASE=0.5, CI brackets the mean
            row = lv["random_floor"]["floor_hue"]
            self.assertLess(row["lo"][r], row["mean"][r])
            self.assertGreater(row["hi"][r], row["mean"][r])

    # --- epsilon diagnostics (Q13 watch-item) --------------------------------

    def test_epsilon_watch_item(self):
        rng = np.random.default_rng(1)
        random = 0.5 + rng.normal(0, 0.01, (10, 2, 1))
        random[0, 0, 0] = -1.5  # one wild null seed inflates the linear-rung tail
        from src.probes.instrument import epsilon_g
        eps = epsilon_g(random, n_boot=N_BOOT)
        diag = epsilon_diagnostics(random, eps, np.zeros((2, 1)), ["a", "b"], ["linear"],
                                   n_boot=N_BOOT)
        eps_primary = diag["epsilon_primary"]["a"][0]
        eps_mad = diag["epsilon_mad"]["a"][0]
        self.assertLess(eps_mad, eps_primary / 2)  # robust null ignores the tail
        g_mid = np.array([[(eps_primary + eps_mad) / 2], [0.0]])
        diag2 = epsilon_diagnostics(random, eps, g_mid, ["a", "b"], ["linear"],
                                    n_boot=N_BOOT)
        self.assertTrue(diag2["watch_item_triggered"])
        self.assertEqual(diag2["verdict_disagreements"][0]["factor"], "a")
        # untouched cell: no disagreement
        self.assertTrue(all(d["factor"] == "a" for d in diag2["verdict_disagreements"]))

    # --- study-level assembly -------------------------------------------------

    def test_assemble_single_cell(self):
        study = assemble([self.res])
        self.assertEqual(study["missing_expected_cells"],
                         [c for c in EXPECTED_CELLS if c != "color_strong"])
        self.assertTrue(study["provisional"])
        hs = study["hypotheses"]
        self.assertEqual(hs["H1"]["status"], "confirmed")
        self.assertEqual(hs["H2"]["status"], "confirmed")
        self.assertEqual(hs["H3"]["status"], "confirmed")
        self.assertEqual(hs["H4"]["status"], "partial")  # widening needs 2 strengths
        self.assertEqual(hs["H4"]["sign_component"]["status"], "confirmed")
        self.assertEqual(study["headline_flip_count"]["primary"]["n_flips"], 1)
        row = next(t for t in study["verdict_table"]
                   if t["factor"] == "object_hue" and t["cell"] == "color_strong")
        self.assertEqual(row["verdict"], "linear_invariance_artifact")
        self.assertFalse(study["headline_contrast"]["complete"])
        # A1: descriptive H4 explicitly labeled; study-level flip uncertainty present
        self.assertIn("DESCRIPTIVE", hs["H4"]["note"])
        self.assertEqual(study["headline_flip_count"]["uncertainty"]["fixed_0.05"]
                         ["n_flips_ci95"], [1.0, 1.0])
        unc = study["headline_flip_count"]["uncertainty"]["primary"]
        self.assertGreaterEqual(unc["n_flips_ci95"][0], 1.0)
        self.assertLessEqual(unc["n_flips_ci95"][1], 2.0)
        json.dumps(study)  # JSON-native without a default converter

    def test_null_saturation_excludes_saturated_flip(self):
        # A4: lift object_hue's random floor to ~0.95 (>= 0.90 at both flip
        # endpoints) while keeping its G pattern; the factor still flips in the
        # primary count but is dropped from the saturation-excluded variant
        tmp2 = tempfile.mkdtemp()
        try:
            rng = np.random.default_rng(4)
            trained, random, perm, projector = make_stacks(rng)
            random[:, 2, :] += 0.45                       # floor 0.5 -> ~0.95
            trained[:, 2, :] += 0.45                      # keep per-seed G unchanged
            d = write_raw(tmp2, condition="sat", strength="strong",
                          trained=trained, random=random, perm=perm, projector=projector,
                          trained_seeds=range(10), random_seeds=range(10))
            res = analyze_cell(load_cell(d), n_boot=N_BOOT)

            ns = res["null_saturation"]
            self.assertEqual(ns["saturated_factors_flip"], ["object_hue"])
            self.assertTrue(all(ns["saturated_by_rung"]["object_hue"]))
            self.assertFalse(any(ns["saturated_by_rung"]["floor_hue"]))

            rep = res["report"]
            self.assertEqual(rep["flips_primary"]["flipped_factors"], ["object_hue"])
            excl = rep["flips_primary_excl_null_saturated"]
            self.assertEqual(excl["flipped_factors"], [])
            self.assertEqual(excl["excluded_null_saturated"], ["object_hue"])
            self.assertEqual(
                res["flip_uncertainty"]["fixed_0.05_excl_null_saturated"]["n_flips_ci95"],
                [0.0, 0.0])
            self.assertNotIn(
                "object_hue",
                res["flip_uncertainty"]["primary_excl_null_saturated"]
                ["per_factor_flip_fraction"])

            study = assemble([res])
            hl = study["headline_flip_count"]
            self.assertEqual(hl["primary"]["n_flips"], 1)
            self.assertEqual(hl["primary_excl_null_saturated"]["n_flips"], 0)
            row = next(t for t in study["verdict_table"]
                       if t["factor"] == "object_hue" and t["cell"] == "sat_strong")
            self.assertTrue(row["null_saturated"])
            self.assertTrue(any("null-saturated" in n for n in study["notes"]))
            json.dumps(study)
        finally:
            shutil.rmtree(tmp2)

    def test_assemble_widening_two_strengths(self):
        tmp2 = tempfile.mkdtemp()
        try:
            weak = analyze_cell(load_cell(
                write_cell(tmp2, strength="weak", proj_gap=0.1, rng_seed=2)), n_boot=N_BOOT)
            strong = analyze_cell(load_cell(
                write_cell(tmp2, strength="strong", proj_gap=0.3, rng_seed=3)), n_boot=N_BOOT)
            study = assemble([weak, strong])
            wid = study["hypotheses"]["H4"]["widening_component"]
            self.assertEqual(wid["status"], "confirmed")
            self.assertEqual(study["hypotheses"]["H4"]["status"], "confirmed")
        finally:
            shutil.rmtree(tmp2)


if __name__ == "__main__":
    unittest.main()
