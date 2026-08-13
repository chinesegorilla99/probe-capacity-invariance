"""Unit tests for the counterfactual displacement spectrum (newplan §5.2/§5.4).

Every fixture is a hand-built factorial grid or a synthetic field with a KNOWN
answer, so each of the six technical requirements is checked against ground truth
rather than against the implementation's own output. No dataset, no encoder, no
training.

Run:  python -m unittest tests.test_displacement -v
"""

import argparse
import unittest

import numpy as np

from src.probes.displacement import (
    run,
    analyze_role,
    bootstrap_stats,
    epsilon_m,
    grid_codes,
    grid_index,
    sample_contexts,
    spectrum_stats,
    sweep_rows,
    whitener,
)

N_BOOT = 200


def toy_grid(radices=(3, 4, 2)):
    """Complete factorial grid, shuffled so nothing depends on row order."""
    mesh = np.meshgrid(*[np.arange(r) for r in radices], indexing="ij")
    labels = np.stack([m.ravel() for m in mesh], axis=1).astype(float)
    return labels[np.random.default_rng(0).permutation(labels.shape[0])]


class TestGrid(unittest.TestCase):
    def test_sweep_rows_hold_context_fixed(self):
        # The whole instrument rests on this: a swept row differs from its context in
        # the swept factor ONLY. Checked against the label codes directly.
        labels = toy_grid()
        codes, values = grid_codes(labels)
        lookup = grid_index(codes, values)
        col = 1
        ctx = sample_contexts(values, col, 16)
        rows = sweep_rows(ctx, lookup, values, col)
        self.assertEqual(rows.shape, (16, len(values[col])))
        for i in range(16):
            block = codes[rows[i]]
            for c in range(labels.shape[1]):
                if c == col:
                    np.testing.assert_array_equal(block[:, c], np.arange(len(values[c])))
                else:
                    self.assertEqual(len(set(block[:, c])), 1)   # context held fixed
                    self.assertEqual(block[0, c], ctx[i, c])

    def test_incomplete_grid_is_refused(self):
        # Exact counterfactual pairs require completeness; a hole must raise rather
        # than silently address the wrong row.
        codes, values = grid_codes(toy_grid()[:-1])
        with self.assertRaises(ValueError):
            grid_index(codes, values)

    def test_constant_column_is_harmless(self):
        # dSprites carries a constant colour column; radix 1 must drop out.
        labels = np.concatenate([np.zeros((toy_grid().shape[0], 1)), toy_grid()], axis=1)
        codes, values = grid_codes(labels)
        self.assertEqual(len(values[0]), 1)
        grid_index(codes, values)   # must not raise

    def test_contexts_are_identical_across_calls(self):
        # Requirement 6: the same context set for every factor, encoder and role.
        _, values = grid_codes(toy_grid())
        a = sample_contexts(values, 1, 32)
        np.testing.assert_array_equal(a, sample_contexts(values, 1, 32))
        self.assertFalse(np.array_equal(a, sample_contexts(values, 0, 32)))


class TestWhitening(unittest.TestCase):
    def test_whitening_makes_the_statistic_affine_invariant(self):
        # Requirement 4: unwhitened, the spectrum is only ORTHOGONALLY invariant.
        # Re-parameterize the embedding by an arbitrary invertible map; every
        # reported statistic must be unchanged.
        rng = np.random.default_rng(1)
        d = 6
        Htr = rng.normal(size=(400, d))
        Hs = rng.normal(size=(24, 5, d))
        A = rng.normal(size=(d, d))
        while abs(np.linalg.det(A)) < 1e-3:
            A = rng.normal(size=(d, d))

        base = analyze_role({"f": Hs}, whitener(Htr), n_boot=N_BOOT)["f"]
        mapped = analyze_role({"f": Hs @ A.T}, whitener(Htr @ A.T), n_boot=N_BOOT)["f"]
        for key in ("m", "rho", "r_eff"):
            self.assertAlmostEqual(base[key], mapped[key], places=5, msg=key)

    def test_underdetermined_whitening_is_refused(self):
        # With fewer probe-train samples than dimensions the covariance is singular and
        # the floored directions dominate ||Delta||, so m_F would report the floor
        # rather than the encoder. That must raise, not return a plausible number.
        rng = np.random.default_rng(11)
        with self.assertRaises(ValueError):
            whitener(rng.normal(size=(20, 64)))

    def test_null_direction_is_floored_not_amplified(self):
        # A rank-deficient embedding must not blow the displacement norm up.
        rng = np.random.default_rng(2)
        H = rng.normal(size=(200, 4))
        H[:, 3] = 0.0                      # a dead coordinate
        W = whitener(H)
        self.assertTrue(np.isfinite(W).all())
        self.assertLess(abs(W[3, 3]), 1e3)


class TestSpectrum(unittest.TestCase):
    def test_single_direction_field_gives_rho_one(self):
        # Ground truth: all energy on one direction -> rho = 1, r_eff = 1.
        rng = np.random.default_rng(3)
        d = 8
        u = np.zeros(d); u[2] = 1.0
        amp = rng.uniform(0.5, 2.0, (40, 4, 1))
        Hs = np.cumsum(np.concatenate([np.zeros((40, 1, d)), amp * u], axis=1), axis=1)
        st = analyze_role({"f": Hs}, np.eye(d), n_boot=N_BOOT)["f"]
        self.assertGreater(st["rho"], 0.999)
        self.assertLess(st["r_eff"], 1.01)
        self.assertGreater(st["ci"]["rho"][0], 0.99)

    def test_isotropic_field_spreads_across_directions(self):
        # Ground truth: isotropic displacements in d dims -> rho ~ 1/d, r_eff ~ d.
        rng = np.random.default_rng(4)
        d = 8
        Hs = np.cumsum(rng.normal(size=(300, 5, d)), axis=1)
        st = analyze_role({"f": Hs}, np.eye(d), n_boot=N_BOOT)["f"]
        self.assertLess(st["rho"], 0.30)
        self.assertGreater(st["r_eff"], d * 0.6)

    def test_held_out_rho_is_not_the_optimistic_in_sample_rho(self):
        # Requirement 2: sigma_1^2 is biased upward in sample. On an isotropic field
        # the true rho is 1/d, and the in-sample estimate must overstate it.
        rng = np.random.default_rng(5)
        d, n = 12, 30
        M = rng.normal(size=(n, d))
        st = spectrum_stats(M[: n // 2], M[n // 2:])
        self.assertGreater(st["rho_in_sample"], st["rho"])
        self.assertGreater(st["rho_in_sample"], 1.0 / d)

    def test_bootstrap_resamples_contexts_not_displacements(self):
        # Contexts are the independent unit. With every displacement inside a context
        # identical, resampling contexts must still vary the statistic, while a
        # single context leaves it degenerate — which row-level resampling would not.
        rng = np.random.default_rng(6)
        d = 5
        per_ctx = rng.normal(size=(20, 1, d))
        fit = rng.normal(size=(50, d))
        wide = bootstrap_stats(fit, np.repeat(per_ctx, 4, axis=1), n_boot=N_BOOT)
        single = bootstrap_stats(fit, np.repeat(per_ctx[:1], 4, axis=1), n_boot=N_BOOT)
        self.assertGreater(wide["m"][1] - wide["m"][0], 0.0)
        self.assertAlmostEqual(single["m"][1] - single["m"][0], 0.0, places=9)


class TestMonotonicity(unittest.TestCase):
    def test_monotone_and_folded_single_direction_fields_are_separated(self):
        # Requirement 1: rho_F ~ 1 is NECESSARY, not sufficient. Both fixtures live
        # on one direction, so both score rho ~ 1; only the monotone one is readable
        # by a linear probe, and the check must tell them apart.
        d = 6
        u = np.zeros(d); u[0] = 1.0
        rng = np.random.default_rng(7)
        ramp = np.linspace(0, 1, 5)                   # monotone in the value index
        tent = np.array([0.0, 0.5, 1.0, 0.5, 0.0])    # single direction, folded
        mk = lambda prof: (prof[None, :, None] * u[None, None, :] * np.ones((30, 1, 1))
                           + rng.normal(0, 1e-4, (30, 5, d)))
        mono = analyze_role({"f": mk(ramp)}, np.eye(d), n_boot=N_BOOT)["f"]
        fold = analyze_role({"f": mk(tent)}, np.eye(d), n_boot=N_BOOT)["f"]

        self.assertGreater(mono["rho"], 0.99)
        self.assertGreater(fold["rho"], 0.99)          # both single-direction
        self.assertGreater(mono["monotonicity"]["monotone_fraction"], 0.99)
        self.assertLess(fold["monotonicity"]["monotone_fraction"], 0.05)
        self.assertGreater(mono["monotonicity"]["mean_abs_spearman"], 0.99)
        self.assertLess(fold["monotonicity"]["mean_abs_spearman"], 0.6)


class TestEpsilonM(unittest.TestCase):
    def test_epsilon_m_is_the_null_quantile_and_needs_two_seeds(self):
        # Requirement 3: same estimator as epsilon_G (A8 a) / epsilon_D (A10 c).
        rng = np.random.default_rng(8)
        m = 1.0 + rng.normal(0, 0.05, 12)
        i, j = np.triu_indices(12, k=1)
        pool = np.concatenate([m[i] - m[j], m[j] - m[i]])
        self.assertAlmostEqual(epsilon_m(m), float(np.percentile(pool, 97.5)))
        self.assertTrue(np.isnan(epsilon_m([1.0])))

    def test_epsilon_m_does_not_shrink_with_control_sample_size(self):
        # The A8 (a) failure mode must not be reintroduced: a null QUANTILE converges,
        # it does not collapse as control seeds are added.
        vals = []
        for s in (12, 24, 48):
            rng = np.random.default_rng(9)
            vals.append(epsilon_m(1.0 + rng.normal(0, 0.05, s)))
        for v in vals:
            self.assertGreater(v, 0.08)
            self.assertLess(v, 0.22)
        self.assertGreater(vals[-1], 0.8 * vals[0])

    def test_destruction_certificate_needs_a_deficit_beyond_the_null(self):
        # The documented reading: destroyed iff m(random) - m(trained) > epsilon_m.
        # A collapsed encoder (m = 0) certifies; one that moves like an untrained one
        # does not.
        rng = np.random.default_rng(10)
        null = 1.0 + rng.normal(0, 0.05, 12)
        eps, m_rand = epsilon_m(null), float(np.mean(null))
        self.assertGreater(m_rand - 0.0, eps)          # collapsed encoder: destroyed
        self.assertLess(m_rand - m_rand * 0.99, eps)   # moves like untrained: not


class TestBlindGuard(unittest.TestCase):
    """The confirmatory blind: trained-encoder targeted-factor values need A11 named."""

    def test_guard_refuses_a_trained_checkpoint_without_an_amendment(self):
        args = argparse.Namespace(trained_encoders=["results/encoders/x/backbone.pt"],
                                  amendment=None, config=None, dataset="shapes3d")
        with self.assertRaises(SystemExit) as cm:
            run(args)
        self.assertIn("BLIND GUARD", str(cm.exception))

    def test_guard_fires_before_any_other_work(self):
        # A guard that trips only after the config, device and dataset resolve can be
        # pre-empted by an unrelated failure. With every other field absent it must
        # still be the blind message that comes back, not an AttributeError.
        args = argparse.Namespace(trained_encoders=["x/backbone.pt"], amendment=None)
        with self.assertRaises(SystemExit) as cm:
            run(args)
        self.assertIn("BLIND GUARD", str(cm.exception))

    def test_random_and_pixel_roles_need_no_flag(self):
        # A4 precedent: random-encoder and pixel-reference values are disclosable, so
        # the guard must not block the blind-safe path.
        args = argparse.Namespace(trained_encoders=None, amendment=None)
        with self.assertRaises(Exception) as cm:
            run(args)
        self.assertNotIsInstance(cm.exception, SystemExit)


if __name__ == "__main__":
    unittest.main()
