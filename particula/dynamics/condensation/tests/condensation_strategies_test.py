"""Test module for the condensation strategies."""

# pylint: disable=R0801
# pylint: disable=protected-access

import unittest

import numpy as np

import particula as par  # new – we will build real objects
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
)


# pylint: disable=too-many-instance-attributes
class TestCondensationIsothermal(unittest.TestCase):
    """Test class for the CondensationIsothermal strategy."""

    def setUp(self):
        """Set up condensation strategy and test aerosol system."""
        self.molar_mass = 0.018  # kg/mol for water
        self.diffusion_coefficient = 2e-5  # m^2/s
        self.accommodation_coefficient = 1.0
        self.strategy = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        self.temperature = 298.15  # K
        self.pressure = 101325  # Pa
        self.time_step = 1.0  # s

        # ---------- vapor-pressure strategies ----------
        vp_water = par.gas.VaporPressureFactory().get_strategy("water_buck")
        vp_constant = par.gas.VaporPressureFactory().get_strategy(
            "constant", {"vapor_pressure": 1e-24, "vapor_pressure_units": "Pa"}
        )

        # ---------- gas species ----------
        mm_water = 18.015e-3  # kg/mol
        mm_core = 132.14e-3
        water_sat = vp_water.saturation_concentration(
            mm_water, self.temperature
        )
        water_conc = water_sat * 1.02  # 2 % supersaturation

        self.gas_species = (
            par.gas.GasSpeciesBuilder()
            .set_name(["H2O", "NH4HSO4"])
            .set_molar_mass(np.array([mm_water, mm_core]), "kg/mol")
            .set_vapor_pressure_strategy([vp_water, vp_constant])
            .set_concentration(np.array([water_conc, 1e-30]), "kg/m^3")
            .set_partitioning(True)
            .build()
        )

        # ---------- particle (one particle for speed) ----------
        density_core = 1.77e3
        r = np.array([100e-9])  # 100 nm core radius
        mass_core = 4 / 3 * np.pi * r**3 * density_core
        mass_spec = np.column_stack([mass_core * 0, mass_core])  # [water, core]
        densities = np.array([1e3, density_core])

        activity = (
            par.particles.ActivityKappaParameterBuilder()
            .set_density(densities, "kg/m^3")
            .set_kappa(np.array([0.0, 0.61]))
            .set_molar_mass(np.array([mm_water, mm_core]), "kg/mol")
            .set_water_index(0)
            .build()
        )
        surface = (
            par.particles.SurfaceStrategyVolumeBuilder()
            .set_density(densities, "kg/m^3")
            .set_surface_tension(np.array([0.072, 0.092]), "N/m")
            .build()
        )
        self.particle = (
            par.particles.ResolvedParticleMassRepresentationBuilder()
            .set_distribution_strategy(
                par.particles.ParticleResolvedSpeciatedMass()
            )
            .set_activity_strategy(activity)
            .set_surface_strategy(surface)
            .set_mass(mass_spec, "kg")
            .set_density(densities, "kg/m^3")
            .set_charge(0)
            .set_volume(1e-6, "m^3")  # 1 cm³ parcel
            .build()
        )

    def test_mean_free_path(self):
        """Test the mean free path call."""
        result = self.strategy.mean_free_path(
            temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_knudsen_number(self):
        """Test the Knudsen number call."""
        radius = 1e-9  # m
        result = self.strategy.knudsen_number(
            radius=radius, temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_first_order_mass_transport(self):
        """Test the first order mass transport call."""
        radius = 1e-9  # m
        result = self.strategy.first_order_mass_transport(
            particle_radius=radius,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsNotNone(result)

    def test_fill_zero_radius(self):
        """_fill_zero_radius changes zeros to max radius."""
        radii = np.array([0.0, 1e-9, 2e-9])
        filled = self.strategy._fill_zero_radius(radii.copy())
        # zero should become the maximum original non-zero radius (2 nm)
        self.assertTrue(np.all(filled != 0.0))
        self.assertEqual(filled[0], np.max(radii))

    def test_fill_zero_radius_all_zeros_warns(self):
        """_fill_zero_radius with all zeros should warn."""
        radii = np.array([0.0, 0.0, 0.0])
        with self.assertWarns(RuntimeWarning):
            self.strategy._fill_zero_radius(radii.copy())

    def test_rate_respects_skip_indices(self):
        """rate() must zero the chosen indices."""
        # Skip the condensing water (index 0) to make the effect obvious
        strategy_skip = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[0],
        )

        rates_skip = strategy_skip.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        # Without skipping we should get non-zero water condensation
        strategy_noskip = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )
        rates_noskip = strategy_noskip.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        # Dimensionality: index 0 must be zero when skipped
        self.assertTrue(np.allclose(rates_skip[..., 0], 0.0))

        # For the un-skipped case the values must not be exactly zero
        # (tiny positive/negative numbers are allowed)
        self.assertTrue(np.any(rates_noskip[..., 0] != 0.0))

    def test_step_skip_preserves_skipped_species(self):
        """step() must not change masses/concentrations of skipped indices."""
        skip_idx = 1  # core species
        strategy_skip = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[skip_idx],
        )

        initial_particle_mass = self.particle.get_species_mass().copy()
        initial_gas_conc = self.gas_species.get_concentration().copy()

        particle_new, gas_new = strategy_skip.step(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=0.2,  # short step
        )

        final_particle_mass = particle_new.get_species_mass()
        final_gas_conc = gas_new.get_concentration()

        # water (index 0) should have transferred mass
        self.assertGreater(
            final_particle_mass[..., 0].sum(),
            initial_particle_mass[..., 0].sum(),
        )
        self.assertLess(final_gas_conc[0], initial_gas_conc[0])

        # core (index 1) should be unchanged
        np.testing.assert_allclose(
            final_particle_mass[..., skip_idx],
            initial_particle_mass[..., skip_idx],
        )
        self.assertAlmostEqual(
            final_gas_conc[skip_idx], initial_gas_conc[skip_idx]
        )

    def test_apply_skip_partitioning_direct(self):
        """_apply_skip_partitioning zeroes selected indices on arrays."""
        strategy = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[0, 2],
        )

        array_1d = np.arange(4.0)
        returned_1d = strategy._apply_skip_partitioning(array_1d)
        self.assertIs(returned_1d, array_1d)
        np.testing.assert_array_equal(array_1d, np.array([0.0, 1.0, 0.0, 3.0]))

        array_2d = np.tile(np.arange(4.0), (2, 1))
        returned_2d = strategy._apply_skip_partitioning(array_2d)
        self.assertIs(returned_2d, array_2d)
        expected_2d = np.tile(np.array([0.0, 1.0, 0.0, 3.0]), (2, 1))
        np.testing.assert_array_equal(array_2d, expected_2d)


class TestCondensationIsothermalStaggered(unittest.TestCase):
    """Test class for the CondensationIsothermalStaggered strategy."""

    def test_defaults(self):
        """Defaults stored correctly for staggered strategy."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        self.assertEqual(strategy.theta_mode, "half")
        self.assertEqual(strategy.num_batches, 1)
        self.assertTrue(strategy.shuffle_each_step)
        self.assertIsNone(strategy.random_state)

    def test_theta_mode_random_stores_seed(self):
        """Random theta mode stores provided random state."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=42
        )
        self.assertEqual(strategy.theta_mode, "random")
        self.assertEqual(strategy.random_state, 42)

    def test_theta_mode_batch_stores_batches(self):
        """Batch theta mode stores batch count."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="batch", num_batches=4
        )
        self.assertEqual(strategy.theta_mode, "batch")
        self.assertEqual(strategy.num_batches, 4)

    def test_invalid_theta_mode_raises(self):
        """Invalid theta_mode raises ValueError."""
        with self.assertRaises(ValueError):
            CondensationIsothermalStaggered(
                molar_mass=0.018, theta_mode="unsupported"
            )

    def test_num_batches_less_than_one_raises(self):
        """num_batches below one raises ValueError."""
        with self.assertRaises(ValueError):
            CondensationIsothermalStaggered(molar_mass=0.018, num_batches=0)

    def test_stub_methods_raise_not_implemented(self):
        """Stub methods should raise NotImplementedError until implemented."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        with self.assertRaises(NotImplementedError):
            strategy.mass_transfer_rate(None, None, 298.15, 101325)
        with self.assertRaises(NotImplementedError):
            strategy.rate(None, None, 298.15, 101325)
        with self.assertRaises(NotImplementedError):
            strategy.step(None, None, 298.15, 101325, 1.0)
