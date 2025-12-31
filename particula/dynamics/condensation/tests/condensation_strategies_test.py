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

    def test_get_theta_values_half_mode_returns_half(self):
        """Half mode returns 0.5 with correct shape and dtype."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="half"
        )
        theta = strategy._get_theta_values(5)
        np.testing.assert_array_equal(theta, np.full(5, 0.5))
        self.assertEqual(theta.shape, (5,))
        self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_random_mode_range_and_shape(self):
        """Random mode stays in [0, 1] with correct shape and dtype."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=123
        )
        theta = strategy._get_theta_values(100)
        self.assertTrue(np.all(theta >= 0.0))
        self.assertTrue(np.all(theta <= 1.0))
        self.assertEqual(theta.shape, (100,))
        self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_random_mode_reproducible(self):
        """Random mode reproducible with same integer seed."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=42
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=42
        )
        theta_a = strategy_a._get_theta_values(50)
        theta_b = strategy_b._get_theta_values(50)
        np.testing.assert_array_equal(theta_a, theta_b)

    def test_get_theta_values_random_mode_different_seed(self):
        """Different seeds produce different theta arrays."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=1
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=2
        )
        theta_a = strategy_a._get_theta_values(20)
        theta_b = strategy_b._get_theta_values(20)
        self.assertFalse(np.allclose(theta_a, theta_b))

    def test_get_theta_values_random_mode_accepts_random_state(self):
        """Random mode supports np.random.RandomState."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.RandomState(7),
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.RandomState(7),
        )
        theta_a = strategy_a._get_theta_values(15)
        theta_b = strategy_b._get_theta_values(15)
        np.testing.assert_array_equal(theta_a, theta_b)

    def test_get_theta_values_random_mode_accepts_generator(self):
        """Random mode supports np.random.Generator."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.default_rng(9),
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            theta_mode="random",
            random_state=np.random.default_rng(9),
        )
        theta_a = strategy_a._get_theta_values(12)
        theta_b = strategy_b._get_theta_values(12)
        np.testing.assert_array_equal(theta_a, theta_b)

    def test_get_theta_values_batch_mode_returns_ones(self):
        """Batch mode returns ones with correct shape and dtype."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="batch"
        )
        theta = strategy._get_theta_values(8)
        np.testing.assert_array_equal(theta, np.ones(8))
        self.assertEqual(theta.shape, (8,))
        self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_single_particle_all_modes(self):
        """Single particle handled for half, random, and batch modes."""
        strategy_half = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="half"
        )
        strategy_random = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="random", random_state=5
        )
        strategy_batch = CondensationIsothermalStaggered(
            molar_mass=0.018, theta_mode="batch"
        )

        theta_half = strategy_half._get_theta_values(1)
        theta_random = strategy_random._get_theta_values(1)
        theta_batch = strategy_batch._get_theta_values(1)

        np.testing.assert_array_equal(theta_half, np.array([0.5]))
        self.assertEqual(theta_half.shape, (1,))
        self.assertTrue(0.0 <= theta_random[0] <= 1.0)
        self.assertEqual(theta_random.shape, (1,))
        np.testing.assert_array_equal(theta_batch, np.array([1.0]))
        self.assertEqual(theta_batch.shape, (1,))

    def test_get_theta_values_zero_particles_returns_empty(self):
        """Zero particles returns empty float64 array."""
        for mode in ("half", "random", "batch"):
            strategy = CondensationIsothermalStaggered(
                molar_mass=0.018, theta_mode=mode, random_state=3
            )
            theta = strategy._get_theta_values(0)
            self.assertEqual(theta.shape, (0,))
            self.assertEqual(theta.dtype, np.float64)

    def test_get_theta_values_invalid_theta_mode_raises_value_error(self):
        """Mutating theta_mode to unsupported value triggers ValueError."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        strategy.theta_mode = "unsupported"
        with self.assertRaises(ValueError):
            strategy._get_theta_values(3)

    def test_make_batches_correct_count(self):
        """_make_batches creates requested batch count when possible."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=4, shuffle_each_step=False
        )
        batches = strategy._make_batches(100)
        self.assertEqual(len(batches), 4)
        self.assertEqual(sum(len(batch) for batch in batches), 100)

    def test_make_batches_no_shuffle_when_disabled(self):
        """Shuffling disabled preserves the original order."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = strategy._make_batches(10)
        np.testing.assert_array_equal(batches[0], np.arange(10))

    def test_make_batches_shuffles_when_enabled(self):
        """Shuffling enabled permutes indices."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=1,
            shuffle_each_step=True,
            random_state=0,
        )
        batches = strategy._make_batches(20)
        self.assertFalse(np.array_equal(batches[0], np.arange(20)))

    def test_make_batches_preserves_all_indices(self):
        """All indices appear exactly once across batches."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=5,
            shuffle_each_step=True,
            random_state=123,
        )
        batches = strategy._make_batches(57)
        all_indices = np.concatenate(batches)
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(57))

    def test_make_batches_reproducible(self):
        """Same seed yields identical batches."""
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=3,
            shuffle_each_step=True,
            random_state=21,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=3,
            shuffle_each_step=True,
            random_state=21,
        )
        batches_a = strategy_a._make_batches(20)
        batches_b = strategy_b._make_batches(20)
        for batch_a, batch_b in zip(batches_a, batches_b):
            np.testing.assert_array_equal(batch_a, batch_b)

    def test_make_batches_single_batch(self):
        """num_batches=1 returns a single full batch."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = strategy._make_batches(7)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 7)

    def test_make_batches_more_batches_than_particles(self):
        """Excess batches clip to particle count."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=10, shuffle_each_step=False
        )
        batches = strategy._make_batches(3)
        self.assertEqual(len(batches), 3)
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(3))
        self.assertTrue(all(len(batch) == 1 for batch in batches))

    def test_make_batches_single_particle(self):
        """Single particle returns one-element batch."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=4, shuffle_each_step=True
        )
        batches = strategy._make_batches(1)
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0], np.array([0], dtype=np.intp))

    def test_make_batches_zero_particles_returns_empty(self):
        """Zero particles returns empty list."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        batches = strategy._make_batches(0)
        self.assertEqual(batches, [])

    def test_make_batches_random_state_shuffle_path(self):
        """Uses RandomState shuffle path when provided."""
        random_state_a = np.random.RandomState(5)
        random_state_b = np.random.RandomState(5)
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=random_state_a,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=random_state_b,
        )
        batches_a = strategy_a._make_batches(6)
        batches_b = strategy_b._make_batches(6)
        self.assertFalse(
            np.array_equal(np.concatenate(batches_a), np.arange(6))
        )
        for batch_a, batch_b in zip(batches_a, batches_b):
            np.testing.assert_array_equal(batch_a, batch_b)

    def test_make_batches_generator_shuffle_path(self):
        """Uses Generator shuffle path when provided."""
        generator_a = np.random.default_rng(7)
        generator_b = np.random.default_rng(7)
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=generator_a,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=0.018,
            num_batches=2,
            shuffle_each_step=True,
            random_state=generator_b,
        )
        batches_a = strategy_a._make_batches(6)
        batches_b = strategy_b._make_batches(6)
        self.assertFalse(
            np.array_equal(np.concatenate(batches_a), np.arange(6))
        )
        for batch_a, batch_b in zip(batches_a, batches_b):
            np.testing.assert_array_equal(batch_a, batch_b)
