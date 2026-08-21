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
    calibrate_m_star,
    certificate,
    context_scale,
    epsilon_m,
    grid_codes,
    grid_index,
    sample_contexts,
    spectrum_stats,
    shared_direction_fraction,
    sweep_rows,
    displacement_scale,
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
    def test_whitening_is_exactly_invariant_under_an_orthogonal_reparameterization(self):
        # An orthogonal map leaves the covariance spectrum alone, so the A15 (d) rank cap
        # retains the same k and the whitened subspace is the rotated one. Exact.
        rng = np.random.default_rng(1)
        d = 6
        Htr, Hs = rng.normal(size=(400, d)), rng.normal(size=(24, 5, d))
        Q = np.linalg.qr(rng.normal(size=(d, d)))[0]
        base = analyze_role({"f": Hs}, whitener(Htr), n_boot=N_BOOT)["f"]
        rot = analyze_role({"f": Hs @ Q.T}, whitener(Htr @ Q.T), n_boot=N_BOOT)["f"]
        for key in ("m", "rho", "r_eff", "m_total", "m_shared", "m_ctx", "d_prime"):
            self.assertAlmostEqual(base[key], rot[key], places=5, msg=key)

    def test_rank_capped_whitening_trades_exact_affine_invariance_for_conditioning(self):
        # A15 (d), disclosed rather than asserted away, and it lands on the CERTIFIED
        # quantity: truncating at a variance fraction is not exactly affine invariant,
        # because an invertible map moves which directions clear the cap. Measured at
        # d = 64: 17% at 0.95, 13% at 0.99, 6.7% at 0.999, 0.3% at full rank. That is why
        # the var_fraction sweep bounds every certificate verdict instead of garnishing it.
        rng = np.random.default_rng(1)
        d = 64
        Htr, Hs = rng.normal(size=(400, d)), rng.normal(size=(24, 5, d))

        def deviation(var_fraction):
            devs = []
            for _ in range(5):
                A = rng.normal(size=(d, d))
                base = analyze_role({"f": Hs}, whitener(Htr, var_fraction), n_boot=N_BOOT)["f"]
                mapped = analyze_role({"f": Hs @ A.T}, whitener(Htr @ A.T, var_fraction),
                                      n_boot=N_BOOT)["f"]
                devs.append(abs(base["m_total"] - mapped["m_total"]) / base["m_total"])
            return float(np.mean(devs))

        capped, tighter, full = deviation(0.99), deviation(0.999), deviation(1.0)
        self.assertGreater(capped, 0.05)        # the cost is real and must be reported
        self.assertLess(tighter, capped)        # and it is monotone in the cap
        self.assertLess(full, 0.01)             # localizing the cause in the truncation

    def test_underdetermined_whitening_is_refused(self):
        # With fewer probe-train samples than dimensions the covariance is singular and
        # the floored directions dominate ||Delta||, so m_F would report the floor
        # rather than the encoder. That must raise, not return a plausible number.
        rng = np.random.default_rng(11)
        with self.assertRaises(ValueError):
            whitener(rng.normal(size=(20, 64)))

    def test_null_direction_is_projected_out_not_amplified(self):
        # A15 (d): a dead coordinate leaves the subspace instead of being floored and
        # retained. Flooring at 1e-6 kept it and multiplied it by up to 1e3, and did so
        # by an amount set by each role's own collapse -- across exactly the roles the
        # certificate compares.
        rng = np.random.default_rng(2)
        H = rng.normal(size=(200, 4))
        H[:, 3] = 0.0                      # a dead coordinate
        W, info = whitener(H, return_info=True)
        self.assertTrue(np.isfinite(W).all())
        self.assertLessEqual(info["rank"], 3)
        self.assertEqual(W.shape, (info["rank"], 4))
        self.assertLess(np.linalg.norm(W @ np.eye(4)[3]), 1e-8)   # dead axis contributes nothing


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


class TestDestructionScale(unittest.TestCase):
    """A12: the certificate needs an ABSOLUTE scale. A11 (b)'s deficit against the random
    encoder certified reduction, not destruction — these fixtures pin the difference.

    After whitening the embedding has unit variance in every direction, so a whitened
    displacement is measured in standard deviations of the representation itself. A probe
    picks its own direction, so that, not a variance share over all dimensions, is what
    bounds it.
    """

    @staticmethod
    def _field(scale, d=32, n_ctx=200, n_steps=4):
        u = np.zeros(d); u[0] = 1.0
        return np.ones((n_ctx, n_steps, 1)) * u[None, None, :] * scale

    def test_halved_but_decodable_factor_is_not_certified_destroyed(self):
        # The case that defeats a RELATIVE certificate: half the untrained displacement,
        # yet a pure consistent ramp a linear probe recovers exactly. In whitened units
        # it still moves half a standard deviation per step — nowhere near destroyed.
        full = displacement_scale(self._field(1.0))
        half = displacement_scale(self._field(0.5))
        self.assertAlmostEqual(half["m_total"], 0.5, places=6)
        self.assertAlmostEqual(half["m_shared"], 0.5, places=6)
        self.assertGreater(half["m_total"], 0.1, "halved-but-decodable read as destroyed")
        self.assertLess(half["m_total"], full["m_total"])   # reduced, reported as reduced

    def test_collapsed_encoder_is_certified_destroyed(self):
        # h independent of the factor -> the pair lands on the same point. With
        # independent generative factors this is the airtight case.
        st = displacement_scale(self._field(0.0))
        self.assertLess(st["m_total"], 1e-9)
        self.assertLess(st["m_shared"], 1e-9)

    def test_context_dependent_movement_is_not_linearly_readable(self):
        # Real movement (m_total ~ 1) whose direction flips per context, so the shared
        # component vanishes: present, but a single fixed readout cannot see it. This is
        # the regime that needs probe capacity, and rho_F alone cannot detect it.
        d, n_ctx = 32, 200
        u = np.zeros(d); u[0] = 1.0
        sign = np.where(np.arange(n_ctx) % 2 == 0, 1.0, -1.0)[:, None, None]
        field = sign * u[None, None, :] * np.ones((n_ctx, 4, 1))
        st = displacement_scale(field)
        self.assertAlmostEqual(st["m_total"], 1.0, places=6)
        self.assertLess(st["m_shared"], 1e-9)
        self.assertLess(shared_direction_fraction(field), 0.05)

    def test_consistent_direction_field_is_fully_shared(self):
        field = self._field(1.0)
        self.assertAlmostEqual(displacement_scale(field)["m_shared"], 1.0, places=6)
        self.assertGreater(shared_direction_fraction(field), 0.99)


class TestCertificate(unittest.TestCase):
    """A15 (b): the eta / eta_star certificate that replaces 'm_total << 1'."""

    @staticmethod
    def _fixture(amplitude, d=64, n_ctx=128, n_vals=6, seed=4):
        """A12 (a)'s family h = z + a*v*u: known Bayes R2 = a^2/(a^2+1)."""
        rng = np.random.default_rng(seed)
        u = np.zeros(d); u[0] = 1.0
        vals = np.arange(n_vals, dtype=float)
        vals = (vals - vals.mean()) / vals.std()
        return rng.normal(size=(n_ctx, 1, d)) + amplitude * vals[None, :, None] * u

    def test_m_total_is_a_width_free_discriminability_index(self):
        # Why the certificate keys on m_total and NOT on a ratio to a context reference:
        # whitening fixes per-direction variance at 1 and a matched readout over an
        # r-dimensional displacement gains exactly sqrt(r), so the same decodability gives
        # the same m_total at any embedding width. Dividing by m_ctx ~ sqrt(2k) would put
        # a rank dependence back in, which is what makes the calibration transferable.
        m_by_d, ratio_by_d = [], []
        for d in (32, 128, 512):
            H = self._fixture(1.0, d=d, seed=5)
            sc = displacement_scale(np.diff(H, axis=1))
            m_by_d.append(sc["m_total"])
            ratio_by_d.append(sc["m_total"] / context_scale(H, np.eye(d)))
        self.assertLess(max(m_by_d) / min(m_by_d), 1.05)              # width-free
        self.assertGreater(max(ratio_by_d) / min(ratio_by_d), 3.0)    # the ratio is not

    def test_a_decodable_ramp_is_not_certified_destroyed_and_a_collapsed_one_is(self):
        cal = calibrate_m_star(dims=64, n_ctx=128, seed=6)
        self.assertTrue(cal["separable"], "fixture classes overlap: no m_star")
        m_star = cal["m_star"]
        for amp, expected in ((10.0, "moves"), (3.0, "moves"),
                              (0.0, "eps_destroyed"), (1e-3, "eps_destroyed")):
            H = self._fixture(amp)
            sc = displacement_scale(np.diff(H, axis=1))
            v = certificate(sc["m_total"], sc["m_shared"],
                            context_scale(H, np.eye(H.shape[-1])), H.shape[-1], m_star)
            self.assertEqual(v["verdict"], expected, f"amplitude {amp}")

    def test_the_unadjudicated_band_is_reported_not_silently_split(self):
        # Between "known decodable" and "known absent" the fixture says nothing, and the
        # calibration must expose that band as the resolution the certificate declines to
        # adjudicate, rather than letting the midpoint pretend to decide it.
        cal = calibrate_m_star(dims=64, n_ctx=128, seed=6)
        lo, hi = cal["unadjudicated_band"]
        self.assertLess(lo, cal["m_star"])
        self.assertGreater(hi, cal["m_star"])
        self.assertGreater(hi / max(lo, 1e-12), 2.0)

    def test_no_verdict_without_a_calibration(self):
        # The A15 (b) discipline: an unset threshold must never silently become a chosen
        # one. Absent m_star the artifact records 'uncalibrated', not 'eps_destroyed'.
        v = certificate(0.0, 0.0, 10.0, rank=8, m_star=None)
        self.assertEqual(v["verdict"], "uncalibrated")
        self.assertIsNone(v["m_star"])

    def test_calibration_reports_overlap_instead_of_picking_a_number(self):
        # If the known-decodable and known-absent classes overlap there is no separating
        # threshold, and the honest output is None rather than a chosen constant.
        cal = calibrate_m_star(dims=64, n_ctx=128, amplitudes=(0.5,), seed=7)
        self.assertIsNone(cal["m_star"])
        self.assertFalse(cal["separable"])

    def test_whitening_check_flags_a_subspace_that_is_not_unit_variance(self):
        # m_total reads as a discriminability index only if whitening actually delivered
        # unit variance per direction. m_ctx must land near sqrt(2k); a shortfall means it
        # did not, and the reading fails with it.
        rng = np.random.default_rng(9)
        k = 16
        H = rng.normal(size=(64, 5, k))
        good = certificate(1.0, 1.0, context_scale(H, np.eye(k)), rank=k)
        self.assertAlmostEqual(good["whitening_check"], 1.0, delta=0.15)
        bad = certificate(1.0, 1.0, context_scale(0.1 * H, np.eye(k)), rank=k)
        self.assertLess(bad["whitening_check"], 0.2)

    def test_context_reference_uses_a_derangement_and_holds_the_factor_fixed(self):
        # m_ctx must pair DIFFERENT contexts at the SAME factor value. Self-pairing would
        # return 0 and make every eta infinite.
        rng = np.random.default_rng(8)
        H = rng.normal(size=(64, 5, 16))
        m_ctx = context_scale(H, np.eye(16))
        self.assertGreater(m_ctx, 1.0)
        self.assertAlmostEqual(m_ctx, np.sqrt(2 * 16), delta=1.5)   # ~sqrt(2k) when decorrelated


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
