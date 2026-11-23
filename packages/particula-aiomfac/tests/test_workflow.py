"""Native pressure, target mapping, occupancy and step atomicity."""

import importlib.util
import json
import runpy
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("particula"), "requires combined install"
)
class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = runpy.run_path(
            str(
                Path(__file__).resolve().parents[1] / "examples/closed_water.py"
            )
        )

    def test_permuted_pressure_and_flux(self):
        from particula.gas.properties.mean_free_path import (
            get_molecule_mean_free_path,
        )
        from particula.util.constants import GAS_CONSTANT

        for permuted in (False, True):
            proc, particle, gas, source, water, expected = self.demo["setup"](
                1.02, permuted
            )
            radius = particle.radii[0, 0]
            mw = gas.molar_mass[source]
            temperature = self.demo["TEMPERATURE"]
            mfp = get_molecule_mean_free_path(
                molar_mass=mw, temperature=temperature, pressure=101325.0
            )
            kn = mfp / radius
            # Fuchs-Sutugin correction for accommodation equal to one.
            transition = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
            pressure = (
                gas.concentration[0, source] * GAS_CONSTANT * temperature / mw
            )
            rate = (
                4
                * np.pi
                * radius
                * 2.5e-5
                * transition
                * (pressure - expected)
                * mw
                / (GAS_CONSTANT * temperature)
            )
            actual = proc.mass_transfer_rate(
                particle, gas, temperature, 101325.0
            )
            np.testing.assert_allclose(
                actual[0, water], rate, rtol=1e-9, atol=0
            )
            if permuted:
                self.assertNotEqual(actual[0, water], actual[1, water])
                np.testing.assert_array_equal(actual[2], 0)
        for ratio in (0.98, 1.02):
            self.demo["trajectory"](ratio, permuted=True)

    def test_two_reference_compositions_share_one_native_pressure_call(self):
        from particula.util.constants import GAS_CONSTANT

        proc, particles, gas, source, water, _ = self.demo["setup"](1.02, True)
        salt_only = particles.masses[0, 1]
        salt_only[:] = 0
        salt_only[[water, 0, 1]] = np.array([0.98, 0.02, 0.2]) * 5e-15
        radius = np.maximum(particles.radii[0], 1e-10)
        _, surface = proc._native_pressure_terms(
            particles, gas, 298.0, radius, [source], [water]
        )
        water_activities = np.array(
            [
                json.loads((self.demo["REF"] / "expected" / name).read_text())[
                    "neutrals"
                ]["water"]["activity"]
                for name in ("mixed.json", "absent-salt.json")
            ]
        )
        kelvin = np.exp(
            2
            * self.demo["SIGMA"]
            * gas.molar_mass[source]
            / (GAS_CONSTANT * 298.0 * self.demo["DENSITY"] * radius[:2])
        )
        np.testing.assert_allclose(
            surface[:2, water],
            water_activities * self.demo["PURE_PRESSURE"] * kelvin,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_failure_is_atomic(self):
        proc, particle, gas, source, water, _ = self.demo["setup"](0.98, True)
        particle.masses[0, 1, :] = 0
        particle.masses[0, 1, 1] = 5e-15  # inert-only active row
        before = particle.masses.copy(), gas.concentration.copy()
        with self.assertRaisesRegex(ValueError, "batch index"):
            proc.step(particle, gas, 298, 101325.0, time_step=1e-4)
        np.testing.assert_array_equal(before[0], particle.masses)
        np.testing.assert_array_equal(before[1], gas.concentration)

    def test_provider_reports_native_slot_after_skipping_vacancies(self):
        proc, particle, _, _, water, _ = self.demo["setup"](0.98, True)
        masses = particle.masses[0].copy()
        masses[2, 1] = 1e-15  # inert-only occupied slot
        provider = proc.activity_strategy
        active = np.array([False, True, True])
        with self.assertRaisesRegex(
            provider.trial_failure_types, r"batch index \(2,\)"
        ):
            provider.neutral_activities(
                masses, np.array([289.0, 298.0, 301.0]), [water], active
            )

    def test_sulfate_refresh_and_failed_step(self):
        from unittest.mock import patch

        from particula_aiomfac import ConvergenceError

        proc, p, gas, source, water, _ = self.demo["setup"](
            1.02, case_id="sulfate"
        )
        dry = p.masses[..., 1].copy()

        def evaluate():
            return proc.activity_strategy.prepared.evaluate(p.masses[0], 298.15)

        before = evaluate()
        np.testing.assert_allclose(
            before.neutral_activities[0, water],
            0.9602314418143285,
            atol=1e-10,
            rtol=0,
        )
        water_budget = (
            gas.concentration[0, source]
            + p.concentration[0] @ p.masses[0, :, water]
        )
        proc.step(p, gas, 298.15, 101325.0, time_step=1e-4)
        after = evaluate()
        self.assertGreater(
            after.neutral_activities[0, water],
            before.neutral_activities[0, water],
        )
        self.assertLess(
            after.ion_molalities[0, after.ion_ids.index(261)],
            before.ion_molalities[0, before.ion_ids.index(261)],
        )
        np.testing.assert_array_equal(p.masses[..., 1], dry)
        np.testing.assert_allclose(
            gas.concentration[0, source]
            + p.concentration[0] @ p.masses[0, :, water],
            water_budget,
            rtol=1e-12,
        )
        initial = p.masses.copy(), gas.concentration.copy()
        with patch(
            "particula_aiomfac.model.solve",
            side_effect=ConvergenceError("forced exhaustion"),
        ):
            with self.assertRaisesRegex(ConvergenceError, "batch index"):
                proc.step(p, gas, 298.15, 101325.0, time_step=1e-4)
        np.testing.assert_array_equal(p.masses, initial[0])
        np.testing.assert_array_equal(gas.concentration, initial[1])

    def test_layout_admission(self):
        from particula.dynamics.condensation.species_mapping import (
            CondensationSpeciesMap,
        )
        from particula_aiomfac import MixtureSpec
        from particula_aiomfac.adapter import AIOMFACActivity

        spec = MixtureSpec({"water": {16: 1}, "salt": {202: 1, 242: 1}})
        for mapping, inert in [
            ({0: "water"}, ()),
            ({0: "water", 1: "salt"}, (1,)),
            ({0: "water", 2: "salt"}, ()),
        ]:
            with self.assertRaises(ValueError):
                AIOMFACActivity(spec, mapping, inert)
        proc, p, gas, source, water, _ = self.demo["setup"](1.02, True)
        for lane in (0, 1):
            proc.species_map = CondensationSpeciesMap(
                tuple(gas.name), ((source, lane),)
            )
            with self.assertRaisesRegex(ValueError, "neutral target"):
                proc.mass_transfer_rate(p, gas, 298.0, 101325.0)

    def test_prepared_gpu_binds_direct_provider(self):
        from particula.execution.prepared_condensation import _activity_plan

        process, particle, gas, source, water, _ = self.demo["setup"](1.02)
        mode, *_ = _activity_plan(
            process.activity_strategy, particle, gas, ((source, water),), ()
        )
        self.assertEqual(mode, 3)
