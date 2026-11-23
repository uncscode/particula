"""Water vapor, crystal and independently mass-bearing aqueous ions."""

import os
import unittest

import numpy as np
import pytest

pytest.importorskip("particula")
from particula_aiomfac import MixtureSpec  # noqa: E402
from particula_aiomfac.adapter import AIOMFACActivity  # noqa: E402


def _model(extra=None, minerals=None):
    recipe = {"NaCl": {202: 1, 242: 1}, "water": {16: 1}, **(extra or {})}
    return AIOMFACActivity(
        MixtureSpec(recipe),
        dict(enumerate(recipe)),
        phase_mode="stable",
        solids=minerals or {"NaCl": "halite"},
    )


def _solve(model, totals, rh=None, execution="cpu", **kwargs):
    return model.solve_partitioning(
        inventories=totals,
        temperature=298.15,
        relative_humidity=rh,
        saturation_concentrations={"water": 100},
        execution=execution,
        **kwargs,
    )


class TestCrystalWater(unittest.TestCase):
    """Exercise crystallization, coexistence and deliquescence under both reservoirs."""

    def check_branches(self, execution):
        model = _model()
        wet = _solve(model, [100, 200], 0.85, execution)
        aqueous = wet.solid_liquid
        self.assertIs(wet.liquids, aqueous.liquids)
        np.testing.assert_allclose(aqueous.solid_masses, [0], atol=1e-10)
        self.assertEqual(aqueous.ion_ids, (202, 242))
        self.assertAlmostEqual(aqueous.ion_masses.sum(), 100)
        expected_water = wet.water_exchanged + 200
        closed = _solve(model, [100, expected_water], execution=execution)
        self.assertAlmostEqual(closed.relative_humidity, 0.85, places=9)
        self.assertEqual(closed.water_exchanged, 0)
        np.testing.assert_allclose(
            closed.gas_concentrations, [0, 85], rtol=1e-9
        )
        np.testing.assert_allclose(
            closed.solid_liquid.neutral_masses,
            aqueous.neutral_masses,
            rtol=1e-9,
        )
        self.assertAlmostEqual(
            closed.solid_liquid.neutral_masses.sum()
            + closed.gas_concentrations.sum(),
            expected_water,
        )
        # Independent pure-salt check, Greenspan (1977), saturated halite brine.
        plateau = _solve(model, [100, 100], execution=execution)
        self.assertAlmostEqual(plateau.relative_humidity, 0.7529, delta=0.004)
        brine = plateau.solid_liquid
        self.assertGreater(brine.solid_masses[0], 0)
        self.assertLess(brine.solid_masses[0], 100)
        self.assertAlmostEqual(
            brine.ion_masses.sum() + brine.solid_masses.sum(), 100
        )
        self.assertAlmostEqual(
            brine.neutral_masses.sum() + plateau.gas_concentrations.sum(), 100
        )
        self.assertLess(brine.complementarity_error, 1e-8)
        self.assertLess(plateau.vapor_residual, 1e-8)
        with self.assertRaisesRegex(ValueError, "no unique brine amount"):
            _solve(model, [100, 20], plateau.relative_humidity, execution)
        dry = _solve(model, [100, 20], 0.60, execution)
        self.assertIsNone(dry.liquids)
        self.assertEqual(dry.water_exchanged, 40)
        self.assertAlmostEqual(dry.solid_liquid.solid_masses.sum(), 100)
        self.assertEqual(dry.solid_liquid.ion_masses.sum(), 0)
        closed_dry = _solve(model, [100, 60], execution=execution)
        self.assertIsNone(closed_dry.liquids)
        self.assertEqual(closed_dry.relative_humidity, dry.relative_humidity)
        np.testing.assert_allclose(closed_dry.gas_concentrations, [0, 60])

    def test_cpu_water_and_salt_balances(self):
        self.check_branches("cpu")

    @unittest.skipUnless(
        os.getenv("AIOMFAC_TEST_CUDA") == "1", "requires NVIDIA device"
    )
    def test_cuda_water_and_salt_balances(self):
        self.check_branches("cuda")

    def test_mineral_selection_and_dry_boundaries(self):
        model = _model({"KCl": {203: 1, 242: 1}})
        with self.assertRaisesRegex(ValueError, "fully assigned"):
            _solve(model, [100, 100, 50], 0.65)
        no_crystal = _solve(model, [100, 100, 0], 0.70, solids={})
        self.assertIsNotNone(no_crystal.liquids)
        self.assertIsNone(no_crystal.solid_liquid)
        no_seed = _solve(model, [0, 60, 0])
        self.assertIsNone(no_seed.liquids)
        self.assertEqual(no_seed.solid_liquid.solid_masses.sum(), 0)
        self.assertEqual(no_seed.relative_humidity, 0.60)


class TestMixedCrystalReservoir(unittest.TestCase):
    """Compare eutonic brines and test coupled semivolatile uptake."""

    def check_mixed(self, execution):
        model = _model(
            {"KCl": {203: 1, 242: 1}}, {"NaCl": "halite", "KCl": "sylvite"}
        )
        plateau = _solve(model, [30, 90, 50], execution=execution)
        dry = _solve(model, [30, 60, 50], execution=execution)
        self.assertIsNone(dry.liquids)
        np.testing.assert_allclose(dry.solid_liquid.solid_masses, [30, 50])
        # Li et al. observed 72.3 ± 0.5% MDRH in NaCl/KCl mixed particles.
        # Our ±1% model-comparison tolerance exceeds the reported uncertainty.
        # https://www.sciencedirect.com/science/article/pii/S1352231013007644
        self.assertAlmostEqual(plateau.relative_humidity, 0.723, delta=0.01)
        self.assertTrue(np.all(plateau.solid_liquid.solid_masses > 0))
        np.testing.assert_allclose(
            plateau.solid_liquid.log_saturation, 0, atol=1e-7
        )
        other = _solve(model, [65, 85, 15], execution=execution)
        self.assertAlmostEqual(
            other.relative_humidity, plateau.relative_humidity, places=7
        )
        np.testing.assert_allclose(
            other.liquids.alpha.ion_molalities,
            plateau.liquids.alpha.ion_molalities,
            rtol=5e-7,
        )
        partial = _solve(model, [30, 0, 50], 0.76, execution)
        self.assertEqual(partial.solid_liquid.solid_masses[0], 0)
        self.assertGreater(partial.solid_liquid.solid_masses[1], 0)
        closed = _solve(
            model, [30, partial.water_exchanged, 50], execution=execution
        )
        self.assertAlmostEqual(closed.relative_humidity, 0.76, places=7)
        np.testing.assert_allclose(
            closed.solid_liquid.solid_masses,
            partial.solid_liquid.solid_masses,
            atol=1e-5,
        )

    def check_vapor(self, execution):
        model = _model({"IPA": {141: 2, 151: 1, 153: 1}})

        def solve(totals, rh=None):
            return model.solve_partitioning(
                inventories=totals,
                temperature=298.15,
                saturation_concentrations={"IPA": 100, "water": 100},
                relative_humidity=rh,
                execution=execution,
            )

        dry = solve([50, 25, 3], 0.60)
        np.testing.assert_allclose(dry.gas_concentrations, [0, 60, 3])
        self.assertIsNone(dry.liquids)
        # Alcohol induces condensation below the binary halite DRH.
        wet = solve([50, 25, 80], 0.70)
        self.assertGreater(wet.solid_liquid.solid_masses[0], 0)
        self.assertLess(wet.solid_liquid.solid_masses[0], 50)
        water = model.prepared.neutrals.index("water")
        ipa = model.prepared.neutrals.index("IPA")
        np.testing.assert_allclose(
            wet.liquids.alpha.neutral_activities[[water, ipa]],
            wet.gas_concentrations[[1, 2]] / 100,
            atol=1e-8,
        )
        self.assertLess(wet.gas_concentrations[2], 80)
        closed = solve([50, 25 + wet.water_exchanged, 80])
        np.testing.assert_allclose(
            closed.gas_concentrations, wet.gas_concentrations, atol=1e-7
        )
        np.testing.assert_allclose(
            closed.solid_liquid.neutral_masses.sum(axis=0)
            + closed.gas_concentrations[[1, 2]],
            [25 + wet.water_exchanged, 80],
            rtol=1e-10,
        )
        self.assertAlmostEqual(
            closed.solid_liquid.solid_masses.sum()
            + closed.solid_liquid.ion_masses.sum(),
            50,
        )

    def test_cpu_eutonic(self):
        self.check_mixed("cpu")

    def test_cpu_organic_vapor(self):
        self.check_vapor("cpu")

    @unittest.skipUnless(
        os.getenv("AIOMFAC_TEST_CUDA") == "1", "requires NVIDIA device"
    )
    def test_cuda_eutonic_and_organic_vapor(self):
        self.check_mixed("cuda")
        self.check_vapor("cuda")

    def test_small_liquid_at_saturated_brine_boundary(self):
        model = _model({"IPA": {141: 2, 151: 1, 153: 1}})
        feed = np.array([15.85285599, 47.4187595, 2.5812405])
        phases = model.phase_equilibrium(feed, 298.15)
        self.assertIsNotNone(phases.beta)
        self.assertLess(phases.beta_masses.sum() / sum(feed), 0.001)
        self.assertLess(phases.chemical_potential_error, 1e-7)
        np.testing.assert_allclose(
            phases.alpha.neutral_activities,
            phases.beta.neutral_activities,
            rtol=1e-7,
        )
