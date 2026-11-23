"""Stability checks at fixed analytical inventory, including reacting solutes."""

import unittest

import numpy as np
from particula_aiomfac import AIOMFACModel, MixtureSpec
from particula_aiomfac._equilibrium import ActivityDomainError


COMPONENTS = {
    "water": {16: 1},
    "alcohol": {141: 2, 151: 1, 153: 1},
    "salt": {204: 2, 261: 1},
}


class LiquidPhaseTests(unittest.TestCase):
    def test_two_liquids_are_invariant_to_order_and_extensive_scale(self):
        feed = np.array([0.4, 0.42, 0.18])
        model = AIOMFACModel.prepare(MixtureSpec(COMPONENTS))
        split = model.phase_equilibrium(feed, 303.15, 0)
        alpha, beta = split.alpha_masses, split.beta_masses
        np.testing.assert_allclose(alpha + beta, feed, atol=1e-15)
        self.assertGreater(alpha[0] / alpha.sum(), beta[0] / beta.sum())
        for field in ("neutral_activities", "ion_activity_products"):
            np.testing.assert_allclose(
                getattr(split.alpha, field),
                getattr(split.beta, field),
                rtol=1e-7,
            )
        for liquid in (split.alpha, split.beta):
            np.testing.assert_allclose(
                liquid.ion_molalities @ model.charge, 0.0, atol=1e-12
            )
        self.assertGreaterEqual(split.minimum_tpd, -1e-7)
        rev = AIOMFACModel.prepare(
            MixtureSpec(dict(reversed(COMPONENTS.items())))
        )
        other = rev.phase_equilibrium(feed[::-1] * 1e-18, 303.15, 2)
        np.testing.assert_allclose(
            other.alpha_masses[::-1] * 1e18, alpha, rtol=1e-6
        )
        np.testing.assert_allclose(
            other.beta_masses[::-1] * 1e18, beta, rtol=1e-6
        )

    def test_one_liquid_at_high_water_and_zero_salt(self):
        model = AIOMFACModel.prepare(MixtureSpec(COMPONENTS))
        for feed in (
            [0.9, 0.07, 0.03],
            [0.9, 0.1, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1e-12, 1e-12],
        ):
            out = model.phase_equilibrium(feed, 303.15, 0)
            self.assertIsNone(out.beta)
            np.testing.assert_array_equal(out.alpha_masses, feed)
            np.testing.assert_allclose(
                out.alpha.neutral_activities,
                model.evaluate(feed, 303.15).neutral_activities,
            )

    def test_phase_speciation_and_closed_carbonate_boundary(self):
        sulfate = AIOMFACModel.prepare(
            MixtureSpec({"water": {16: 1}, "salt": {204: 1, 248: 1}})
        )
        out = sulfate.phase_equilibrium([0.99, 0.01], 298.15, 0)
        self.assertIsNone(out.beta)
        self.assertEqual(out.alpha.solver_status, 1)
        self.assertLess(
            max(out.alpha.diagnostics["equilibrium_residuals"]), 1e-8
        )
        carbonate = AIOMFACModel.prepare(
            MixtureSpec({"water": {16: 1}, "salt": {203: 1, 250: 1}})
        )
        direct = carbonate.evaluate([0.99, 0.01], 298.15, diagnostics=True)
        self.assertEqual(direct.solver_status, 1)
        with self.assertRaisesRegex(ActivityDomainError, "Gibbs-consistent"):
            carbonate.phase_equilibrium([0.99, 0.01], 298.15, 0)

    def test_two_salts_can_share_a_homogeneous_liquid(self):
        model = AIOMFACModel.prepare(
            MixtureSpec(
                {
                    "water": {16: 1},
                    "NaCl": {202: 1, 242: 1},
                    "K2SO4": {203: 2, 261: 1},
                }
            )
        )
        self.assertIsNone(
            model.phase_equilibrium([0.9, 0.05, 0.05], 298.15, 0).beta
        )
