"""Test module for the condensation strategies."""

# pylint: disable=R0801
# pylint: disable=protected-access
import copy
import logging
import logging.handlers
import types
import unittest

import numpy as np
import particula as par  # new – we will build real objects
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    _partial_pressure_from_strategy,
    _pure_vapor_pressure_from_strategy,
    _require_matching_types,
    _require_single_box,
    _unwrap_gas,
    _unwrap_particle,
)
from particula.gas.gas_data import GasData, from_species
from particula.particles.particle_data import ParticleData, from_representation


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
        self.activity_strategy = activity
        self.surface_strategy = surface
        self.vapor_pressure_strategy = (
            self.gas_species.pure_vapor_pressure_strategy
        )

    def _make_data_strategy(self) -> CondensationIsothermal:
        """Return CondensationIsothermal configured for data-only inputs."""
        return CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
        )

    def _make_data_inputs(self) -> tuple[ParticleData, GasData]:
        """Return ParticleData and GasData versions of fixtures."""
        return (
            from_representation(self.particle),
            from_species(self.gas_species),
        )

    def test_mean_free_path(self):
        """Test the mean free path call."""
        result = self.strategy.mean_free_path(
            temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_unwrap_helpers_accept_legacy_and_data(self):
        """Unwrap helpers return data and legacy flags for valid inputs."""
        particle_data = from_representation(self.particle)
        gas_data = from_species(self.gas_species)

        particle_unwrapped, particle_is_legacy = _unwrap_particle(self.particle)
        gas_unwrapped, gas_is_legacy = _unwrap_gas(self.gas_species)
        self.assertIsInstance(particle_unwrapped, ParticleData)
        self.assertIsInstance(gas_unwrapped, GasData)
        self.assertTrue(particle_is_legacy)
        self.assertTrue(gas_is_legacy)

        particle_unwrapped, particle_is_legacy = _unwrap_particle(particle_data)
        gas_unwrapped, gas_is_legacy = _unwrap_gas(gas_data)
        self.assertIsInstance(particle_unwrapped, ParticleData)
        self.assertIsInstance(gas_unwrapped, GasData)
        self.assertFalse(particle_is_legacy)
        self.assertFalse(gas_is_legacy)

    def test_unwrap_helpers_invalid_type_raises(self):
        """Unwrap helpers raise TypeError for unsupported types."""
        with self.assertRaises(TypeError):
            _unwrap_particle("not a particle")
        with self.assertRaises(TypeError):
            _unwrap_gas(123)

    def test_require_matching_types_raises_on_mismatch(self):
        """require_matching_types rejects mixed legacy/data inputs."""
        with self.assertRaises(TypeError):
            _require_matching_types(True, False)

    def test_require_single_box_raises_for_multi_box(self):
        """require_single_box rejects multi-box inputs."""
        with self.assertRaises(ValueError):
            _require_single_box(2, "ParticleData")

    def test_vapor_pressure_helpers_handle_sequence_and_single(self):
        """Vapor-pressure helpers accept sequences and single strategies."""
        temperature = self.temperature
        strategy_sequence = self.vapor_pressure_strategy
        strategy_single = strategy_sequence[0]

        pure_sequence = _pure_vapor_pressure_from_strategy(
            strategy_sequence, temperature
        )
        pure_single = _pure_vapor_pressure_from_strategy(
            strategy_single, temperature
        )
        self.assertEqual(pure_sequence.shape[0], len(strategy_sequence))
        self.assertEqual(pure_single.shape, ())

        gas_data = from_species(self.gas_species)
        concentration = gas_data.concentration[0]
        molar_mass = gas_data.molar_mass
        partial_sequence = _partial_pressure_from_strategy(
            strategy_sequence,
            concentration=concentration,
            molar_mass=molar_mass,
            temperature=temperature,
        )
        partial_single = _partial_pressure_from_strategy(
            strategy_single,
            concentration=concentration[0],
            molar_mass=molar_mass[0],
            temperature=temperature,
        )
        self.assertEqual(partial_sequence.shape[0], len(strategy_sequence))
        self.assertIsInstance(partial_single, np.ndarray)
        self.assertEqual(partial_single.shape, ())

    def test_data_only_requires_strategy_configuration(self):
        """Data-only inputs require strategies on the condensation strategy."""
        particle_data, gas_data = self._make_data_inputs()
        strategy = CondensationIsothermal(molar_mass=self.molar_mass)
        with self.assertRaises(TypeError):
            strategy.calculate_pressure_delta(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                radius=particle_data.radii[0],
            )

    def test_data_only_missing_vapor_pressure_strategy_raises(self):
        """GasData requires a vapor_pressure_strategy on the strategy."""
        particle_data, gas_data = self._make_data_inputs()
        strategy = CondensationIsothermal(
            molar_mass=self.molar_mass,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
        )
        with self.assertRaises(TypeError):
            strategy.calculate_pressure_delta(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                radius=particle_data.radii[0],
            )

    def test_rate_rejects_mixed_legacy_and_data_inputs(self):
        """rate() raises TypeError when legacy/data inputs are mixed."""
        particle_data, _ = self._make_data_inputs()
        with self.assertRaises(TypeError):
            self.strategy.rate(
                particle=particle_data,
                gas_species=self.gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
            )

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

    def test_isothermal_step_with_particle_data_gas_data(self):
        """step() supports ParticleData and GasData inputs."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        particle_new, gas_new = strategy.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(particle_new, ParticleData)
        self.assertIsInstance(gas_new, GasData)
        self.assertEqual(particle_new.masses.shape, particle_data.masses.shape)

    def test_isothermal_step_returns_matching_types(self):
        """step() returns matching types for legacy and data-only inputs."""
        particle_legacy, gas_legacy = self.strategy.step(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(
            particle_legacy, par.particles.ParticleRepresentation
        )
        self.assertIsInstance(gas_legacy, par.gas.GasSpecies)

        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        particle_new, gas_new = strategy.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(particle_new, ParticleData)
        self.assertIsInstance(gas_new, GasData)

    def test_isothermal_step_numerical_parity(self):
        """step() data-only path matches legacy path numerically."""
        # Use the same strategy (with all strategies set) for both
        # paths so the only difference is input types.
        strategy = self._make_data_strategy()

        # Run legacy path
        particle_legacy = copy.deepcopy(self.particle)
        gas_legacy = copy.deepcopy(self.gas_species)
        particle_legacy, gas_legacy = strategy.step(
            particle=particle_legacy,
            gas_species=gas_legacy,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        legacy_mass = particle_legacy.get_species_mass()
        legacy_gas = gas_legacy.get_concentration()

        # Run data-only path
        particle_data, gas_data = self._make_data_inputs()
        particle_data_new, gas_data_new = strategy.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        data_mass = particle_data_new.masses[0]
        data_gas = gas_data_new.concentration[0]

        np.testing.assert_allclose(
            data_mass,
            legacy_mass,
            rtol=1e-10,
            err_msg="Particle mass diverges between legacy and data paths",
        )
        np.testing.assert_allclose(
            data_gas,
            legacy_gas,
            rtol=1e-10,
            err_msg="Gas concentration diverges between legacy and data paths",
        )

    def test_isothermal_rate_numerical_parity(self):
        """rate() data-only path matches legacy path numerically."""
        # Use the same strategy for both paths
        strategy = self._make_data_strategy()
        legacy_rate = strategy.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        particle_data, gas_data = self._make_data_inputs()
        data_rate = strategy.rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        np.testing.assert_allclose(
            data_rate,
            legacy_rate,
            rtol=1e-10,
            err_msg="rate() diverges between legacy and data paths",
        )

    def test_isothermal_rate_with_particle_data(self):
        """rate() supports ParticleData inputs."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        rate = strategy.rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(rate, np.ndarray)
        self.assertEqual(rate.shape, particle_data.masses[0].shape)

    def test_isothermal_mass_transfer_rate_with_particle_data(self):
        """mass_transfer_rate() supports ParticleData inputs."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        mass_rate = strategy.mass_transfer_rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(mass_rate, np.ndarray)
        self.assertEqual(mass_rate.shape, particle_data.masses[0].shape)

    def test_calculate_pressure_delta_with_particle_data(self):
        """calculate_pressure_delta() works with data containers."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        radius = particle_data.radii[0]
        pressure_delta = strategy.calculate_pressure_delta(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            radius=radius,
        )
        self.assertIsInstance(pressure_delta, np.ndarray)
        self.assertEqual(pressure_delta.shape[0], radius.shape[0])
        self.assertTrue(np.all(np.isfinite(pressure_delta)))


class TestCondensationIsothermalStaggered(unittest.TestCase):
    """Test class for the CondensationIsothermalStaggered strategy."""

    def setUp(self):
        """Reuse isothermal fixtures for staggered tests."""
        base = TestCondensationIsothermal()
        base.setUp()
        self.molar_mass = base.molar_mass
        self.temperature = base.temperature
        self.pressure = base.pressure
        self.time_step = 0.1
        self.particle = base.particle
        self.gas_species = base.gas_species
        self.activity_strategy = base.activity_strategy
        self.surface_strategy = base.surface_strategy
        self.vapor_pressure_strategy = base.vapor_pressure_strategy

    def _make_data_strategy(self) -> CondensationIsothermalStaggered:
        """Return staggered strategy configured for data-only inputs."""
        return CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
        )

    def _make_data_inputs(self) -> tuple[ParticleData, GasData]:
        """Return ParticleData and GasData versions of fixtures."""
        return (
            from_representation(self.particle),
            from_species(self.gas_species),
        )

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

    def test_mass_transfer_rate_returns_array(self):
        """mass_transfer_rate returns finite array with expected shape."""
        strategy = CondensationIsothermalStaggered(molar_mass=self.molar_mass)
        mass_rate = strategy.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(mass_rate, np.ndarray)
        self.assertEqual(
            mass_rate.shape, self.particle.get_species_mass().shape
        )
        self.assertTrue(np.all(np.isfinite(mass_rate)))

    def test_rate_returns_array_and_respects_skip(self):
        """Rate returns array and zeros skipped indices."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, skip_partitioning_indices=[0]
        )
        rate = strategy.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(rate, np.ndarray)
        self.assertEqual(rate.shape, self.particle.get_species_mass().shape)
        np.testing.assert_array_equal(rate[..., 0], np.zeros_like(rate[..., 0]))
        self.assertTrue(np.all(np.isfinite(rate)))

    def test_rate_api_accepts_positional_arguments(self):
        """Rate supports positional call matching base signature."""
        strategy = CondensationIsothermalStaggered(molar_mass=self.molar_mass)
        rate_positional = strategy.rate(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
        )
        self.assertIsInstance(rate_positional, np.ndarray)
        self.assertEqual(
            rate_positional.shape, self.particle.get_species_mass().shape
        )

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
        for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
            np.testing.assert_array_equal(batch_a, batch_b)

    def test_make_batches_single_batch(self):
        """num_batches=1 returns a single full batch."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = strategy._make_batches(7)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 7)

    def test_validate_num_batches_zero_raises_value_error(self):
        """_validate_num_batches raises when requested batches are zero."""
        strategy = CondensationIsothermalStaggered(molar_mass=0.018)
        strategy.num_batches = 0
        with self.assertRaisesRegex(
            ValueError, r"num_batches.*(>=|at least)\s*1"
        ):
            strategy._make_batches(5)

    def test_validate_num_batches_negative_raises_value_error(self):
        """_validate_num_batches raises when requested batches are negative."""
        with self.assertRaisesRegex(
            ValueError, r"num_batches.*(>=|at least)\s*1"
        ):
            CondensationIsothermalStaggered(molar_mass=0.018, num_batches=-1)

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
        for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
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
        for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
            np.testing.assert_array_equal(batch_a, batch_b)

    def _make_empty_particle(self):
        """Create an empty particle representation matching fixtures."""
        n_species = self.particle.get_species_mass().shape[1]
        empty_mass = np.empty((0, n_species))
        empty_concentration = np.empty((0,), dtype=np.float64)
        empty_charge = np.empty((0,), dtype=np.float64)
        return self.particle.__class__(
            strategy=self.particle.strategy,
            activity=self.particle.activity,
            surface=self.particle.surface,
            distribution=empty_mass,
            density=self.particle.density,
            concentration=empty_concentration,
            charge=empty_charge,
            volume=self.particle.volume,
        )

    def _make_particle_with_masses(
        self, mass_distribution: np.ndarray, concentration: float
    ):
        """Create a particle with supplied masses and concentration."""
        charge = np.zeros(mass_distribution.shape[0], dtype=np.float64)
        concentrations = np.full(
            mass_distribution.shape[0], concentration, dtype=np.float64
        )
        return self.particle.__class__(
            strategy=self.particle.strategy,
            activity=self.particle.activity,
            surface=self.particle.surface,
            distribution=mass_distribution,
            density=self.particle.density,
            concentration=concentrations,
            charge=charge,
            volume=self.particle.volume,
        )

    def _make_three_particle_state(self):
        """Return particle and gas copies with three distinct particles."""
        base_mass = self.particle.get_species_mass()[0]
        mass_distribution = np.vstack(
            [
                base_mass,
                base_mass * 1.1,
                base_mass * 0.9,
            ]
        )
        particle = self._make_particle_with_masses(
            mass_distribution=mass_distribution,
            concentration=float(self.particle.concentration[0]),
        )
        gas_species = copy.deepcopy(self.gas_species)
        return particle, gas_species

    def test_calculate_single_particle_transfer_default_transport(self):
        """Helper returns finite mass change with internal transport."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        mass_change = strategy._calculate_single_particle_transfer(
            particle=self.particle,
            particle_index=0,
            gas_species=self.gas_species,
            gas_concentration=self.gas_species.get_concentration().copy(),
            temperature=self.temperature,
            pressure=self.pressure,
            dt_local=0.05,
            radii=None,
            first_order_mass_transport=None,
        )
        species_count = self.particle.get_species_mass().shape[1]
        self.assertEqual(mass_change.shape, (species_count,))
        self.assertTrue(np.all(np.isfinite(mass_change)))

    def test_calculate_single_particle_transfer_scalar_transport(self):
        """Helper handles scalar transport coefficient with finite mass."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        scalar_transport = float(
            strategy.first_order_mass_transport(
                particle_radius=self.particle.get_radius()[0],
                temperature=self.temperature,
                pressure=self.pressure,
            )
        )
        mass_change = strategy._calculate_single_particle_transfer(
            particle=self.particle,
            particle_index=0,
            gas_species=self.gas_species,
            gas_concentration=self.gas_species.get_concentration().copy(),
            temperature=self.temperature,
            pressure=self.pressure,
            dt_local=0.05,
            radii=self.particle.get_radius(),
            first_order_mass_transport=scalar_transport,
        )
        species_count = self.particle.get_species_mass().shape[1]
        self.assertEqual(mass_change.shape, (species_count,))
        self.assertTrue(np.all(np.isfinite(mass_change)))

    def test_step_half_mode_produces_valid_output(self):
        """Step with theta_mode='half' returns updated particle and gas."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsNotNone(particle_new)
        self.assertIsNotNone(gas_new)

    def test_staggered_step_with_particle_data_gas_data(self):
        """Staggered step supports ParticleData and GasData inputs."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        particle_new, gas_new = strategy.step(
            particle_data,
            gas_data,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(particle_new, ParticleData)
        self.assertIsInstance(gas_new, GasData)
        self.assertEqual(particle_new.masses.shape, particle_data.masses.shape)

    def test_staggered_step_returns_matching_types(self):
        """Staggered step returns matching types for legacy and data inputs."""
        particle_legacy, gas_legacy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass
        ).step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(
            particle_legacy, par.particles.ParticleRepresentation
        )
        self.assertIsInstance(gas_legacy, par.gas.GasSpecies)

        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        particle_new, gas_new = strategy.step(
            particle_data,
            gas_data,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(particle_new, ParticleData)
        self.assertIsInstance(gas_new, GasData)

    def test_staggered_step_numerical_parity(self):
        """Staggered step() data-only path matches legacy numerically."""
        # Use the same strategy for both paths so the only
        # difference is input type (facade vs data container).
        strategy = self._make_data_strategy()

        particle_legacy = copy.deepcopy(self.particle)
        gas_legacy = copy.deepcopy(self.gas_species)
        particle_legacy, gas_legacy = strategy.step(
            particle=particle_legacy,
            gas_species=gas_legacy,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        legacy_mass = particle_legacy.get_species_mass()
        legacy_gas = gas_legacy.get_concentration()

        particle_data, gas_data = self._make_data_inputs()
        particle_data_new, gas_data_new = strategy.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        data_mass = particle_data_new.masses[0]
        data_gas = gas_data_new.concentration[0]

        np.testing.assert_allclose(
            data_mass,
            legacy_mass,
            rtol=1e-10,
            err_msg=(
                "Staggered particle mass diverges between legacy and data paths"
            ),
        )
        np.testing.assert_allclose(
            data_gas,
            legacy_gas,
            rtol=1e-10,
            err_msg=(
                "Staggered gas concentration diverges between legacy "
                "and data paths"
            ),
        )

    def test_staggered_rate_numerical_parity(self):
        """Staggered rate() data-only path matches legacy numerically."""
        # Use the same strategy for both paths
        strategy = self._make_data_strategy()
        legacy_rate = strategy.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        particle_data, gas_data = self._make_data_inputs()
        data_rate = strategy.rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        np.testing.assert_allclose(
            data_rate,
            legacy_rate,
            rtol=1e-10,
            err_msg="Staggered rate() diverges between legacy and data paths",
        )

    def test_staggered_rate_with_particle_data(self):
        """rate() supports ParticleData inputs for staggered strategy."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        rate = strategy.rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(rate, np.ndarray)
        self.assertEqual(rate.shape, particle_data.masses[0].shape)
        self.assertTrue(np.all(np.isfinite(rate)))

    def test_staggered_mass_transfer_rate_with_particle_data(self):
        """mass_transfer_rate() supports ParticleData inputs for staggered."""
        strategy = self._make_data_strategy()
        particle_data, gas_data = self._make_data_inputs()
        mass_rate = strategy.mass_transfer_rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(mass_rate, np.ndarray)
        self.assertEqual(mass_rate.shape, particle_data.masses[0].shape)
        self.assertTrue(np.all(np.isfinite(mass_rate)))

    def test_step_random_mode_produces_valid_output(self):
        """Step with theta_mode='random' returns updated particle and gas."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="random",
            random_state=42,
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsNotNone(particle_new)
        self.assertIsNotNone(gas_new)

    def test_step_batch_mode_produces_valid_output(self):
        """Step with theta_mode='batch' returns updated particle and gas."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="batch",
            num_batches=2,
            shuffle_each_step=False,
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsNotNone(particle_new)
        self.assertIsNotNone(gas_new)

    def test_step_non_negative_masses_and_gas(self):
        """Step keeps particle masses and gas concentrations non-negative."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertTrue(np.all(particle_new.get_species_mass() >= 0.0))
        self.assertTrue(np.all(gas_new.get_concentration() >= 0.0))

    def test_step_basic_mass_conservation(self):
        """Total mass approximately conserved for small time step."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        initial_total = (
            self.particle.get_mass().sum()
            + self.gas_species.get_concentration().sum()
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=0.01,
        )
        final_total = (
            particle_new.get_mass().sum() + gas_new.get_concentration().sum()
        )
        np.testing.assert_allclose(initial_total, final_total, rtol=1e-10)

    def test_step_respects_skip_partitioning_indices(self):
        """Skip indices remain unchanged after step."""
        skip_idx = 0
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            skip_partitioning_indices=[skip_idx],
        )
        initial_mass = self.particle.get_species_mass().copy()
        initial_gas = self.gas_species.get_concentration().copy()
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        np.testing.assert_allclose(
            particle_new.get_species_mass()[..., skip_idx],
            initial_mass[..., skip_idx],
        )
        np.testing.assert_allclose(
            gas_new.get_concentration()[skip_idx], initial_gas[skip_idx]
        )

    def test_step_positional_api_matches_signature(self):
        """Step accepts positional args compatible with base signature."""
        strategy = CondensationIsothermalStaggered(molar_mass=self.molar_mass)
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            self.time_step,
        )
        self.assertIsNotNone(particle_new)
        self.assertIsNotNone(gas_new)
        self.assertEqual(
            particle_new.get_species_mass().shape,
            self.particle.get_species_mass().shape,
        )

    def test_step_zero_time_step_returns_inputs(self):
        """time_step=0 returns unchanged particle and gas."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        initial_mass = self.particle.get_species_mass().copy()
        initial_gas = self.gas_species.get_concentration().copy()
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=0.0,
        )
        np.testing.assert_allclose(
            particle_new.get_species_mass(), initial_mass
        )
        np.testing.assert_allclose(gas_new.get_concentration(), initial_gas)

    def test_step_empty_particles_noop(self):
        """Zero-particle input returns unchanged gas and empty particle."""
        empty_particle = self._make_empty_particle()
        strategy = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass, theta_mode="half"
        )
        particle_new, gas_new = strategy.step(
            empty_particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertEqual(particle_new.get_species_mass().shape[0], 0)
        np.testing.assert_allclose(
            gas_new.get_concentration(), self.gas_species.get_concentration()
        )

    def test_step_updates_gas_between_batches(self):
        """Multi-batch step differs from single batch due to gas updates."""
        self._make_three_particle_state()
        particle_multi, gas_multi = self._make_three_particle_state()

        strategy_multi = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=2,
            shuffle_each_step=False,
        )

        gas_inputs: list[np.ndarray] = []

        def fake_calc(self, *, gas_concentration, **kwargs):
            gas_inputs.append(np.array(gas_concentration, copy=True))
            return np.array([gas_concentration[0] * 0.05, 0.0])

        strategy_multi._calculate_single_particle_transfer = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_multi
        )

        strategy_multi.step(
            particle_multi,
            gas_multi,
            self.temperature,
            self.pressure,
            time_step=1.0,
        )

        # second batch sees reduced gas compared to first batch
        self.assertGreater(len(gas_inputs), 2)
        self.assertLess(gas_inputs[2][0], gas_inputs[0][0])
        self.assertTrue(np.all(np.isfinite(gas_inputs[-1])))
        self.assertTrue(np.all(gas_inputs[-1] >= 0.0))

    def test_step_num_batches_changes_outcome(self):
        """Different num_batches yields different final states."""
        particle_one, gas_one = self._make_three_particle_state()
        particle_many, gas_many = self._make_three_particle_state()

        strategy_one = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=1,
            shuffle_each_step=False,
        )
        strategy_many = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=False,
        )

        def fake_calc(self, *, gas_concentration, **kwargs):
            # Scale mass change by current gas to make batching observable.
            return np.array([gas_concentration[0] * 0.1, 0.0])

        strategy_one._calculate_single_particle_transfer = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_one
        )
        strategy_many._calculate_single_particle_transfer = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_many
        )

        particle_one, gas_one = strategy_one.step(
            particle_one,
            gas_one,
            self.temperature,
            self.pressure,
            time_step=1.0,
        )
        particle_many, gas_many = strategy_many.step(
            particle_many,
            gas_many,
            self.temperature,
            self.pressure,
            time_step=1.0,
        )

        gas_delta = np.linalg.norm(
            gas_one.get_concentration() - gas_many.get_concentration()
        )
        particle_delta = np.linalg.norm(
            particle_one.get_species_mass() - particle_many.get_species_mass()
        )
        self.assertGreater(gas_delta, 0.0)
        self.assertGreater(particle_delta, 0.0)
        self.assertTrue(np.all(np.isfinite(particle_many.get_species_mass())))
        self.assertTrue(np.all(particle_many.get_species_mass() >= 0.0))

    def test_step_batch_order_affects_result(self):
        """Different batch ordering changes outcomes (Gauss-Seidel property)."""
        particle_ordered, gas_ordered = self._make_three_particle_state()
        particle_shuffled, gas_shuffled = self._make_three_particle_state()

        strategy_ordered = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=False,
        )
        strategy_shuffled = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=True,
            random_state=11,
        )

        def fake_calc(self, *, gas_concentration, particle_index, **kwargs):
            # Include particle_index so order changes accumulation.
            return np.array(
                [gas_concentration[0] * 0.05 + particle_index * 1e-11, 0.0]
            )

        strategy_ordered._calculate_single_particle_transfer = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_ordered
        )
        strategy_shuffled._calculate_single_particle_transfer = (
            types.MethodType(  # type: ignore[attr-defined]
                fake_calc, strategy_shuffled
            )
        )

        particle_ordered, gas_ordered = strategy_ordered.step(
            particle_ordered,
            gas_ordered,
            self.temperature,
            self.pressure,
            time_step=1.0,
        )
        particle_shuffled, gas_shuffled = strategy_shuffled.step(
            particle_shuffled,
            gas_shuffled,
            self.temperature,
            self.pressure,
            time_step=1.0,
        )

        gas_delta = np.linalg.norm(
            gas_ordered.get_concentration() - gas_shuffled.get_concentration()
        )
        particle_delta = np.linalg.norm(
            particle_ordered.get_species_mass()
            - particle_shuffled.get_species_mass()
        )
        self.assertGreater(gas_delta, 0.0)
        self.assertGreater(particle_delta, 0.0)

    def test_step_num_batches_exceeds_particle_count_clips(self):
        """Excess batches clip to particle count without changing outcome."""
        particle_high, gas_high = self._make_three_particle_state()
        particle_exact, gas_exact = self._make_three_particle_state()

        strategy_high = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=10,
            shuffle_each_step=False,
        )
        strategy_exact = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=False,
        )

        particle_high, gas_high = strategy_high.step(
            particle_high,
            gas_high,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        particle_exact, gas_exact = strategy_exact.step(
            particle_exact,
            gas_exact,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )

        np.testing.assert_allclose(
            gas_high.get_concentration(), gas_exact.get_concentration()
        )
        np.testing.assert_allclose(
            particle_high.get_species_mass(), particle_exact.get_species_mass()
        )
        self.assertTrue(np.all(gas_high.get_concentration() >= 0.0))

    def test_step_batch_with_theta_modes(self):
        """Batch mode works with half, random, and batch theta modes."""
        for mode in ("half", "batch"):
            particle, gas_species = self._make_three_particle_state()
            strategy = CondensationIsothermalStaggered(
                molar_mass=self.molar_mass,
                theta_mode=mode,
                num_batches=3,
                shuffle_each_step=False,
            )
            particle_new, gas_new = strategy.step(
                particle,
                gas_species,
                self.temperature,
                self.pressure,
                time_step=self.time_step,
            )
            self.assertTrue(np.all(particle_new.get_species_mass() >= 0.0))
            self.assertTrue(np.all(np.isfinite(gas_new.get_concentration())))

        particle_a, gas_a = self._make_three_particle_state()
        particle_b, gas_b = self._make_three_particle_state()
        strategy_a = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="random",
            num_batches=3,
            shuffle_each_step=False,
            random_state=7,
        )
        strategy_b = CondensationIsothermalStaggered(
            molar_mass=self.molar_mass,
            theta_mode="random",
            num_batches=3,
            shuffle_each_step=False,
            random_state=7,
        )
        particle_a, gas_a = strategy_a.step(
            particle_a,
            gas_a,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        particle_b, gas_b = strategy_b.step(
            particle_b,
            gas_b,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )

        np.testing.assert_allclose(
            gas_a.get_concentration(), gas_b.get_concentration()
        )
        np.testing.assert_allclose(
            particle_a.get_species_mass(), particle_b.get_species_mass()
        )

    def test_num_batches_exceeds_particles_clips_and_logs_info(self):
        """Clips to particle count when batches exceed size and logs info."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1000, shuffle_each_step=False
        )
        with self.assertLogs("particula", level=logging.INFO) as cm:
            batches = strategy._make_batches(5)
        self.assertEqual(len(batches), 5)
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(5))
        # Verify clipping log message is present.
        self.assertTrue(
            any(
                "Clipping num_batches" in record.message
                for record in cm.records
            )
        )

    def _assert_no_clipping_log(self, batches_func, *args, **kwargs):
        """Helper to verify no clipping log is emitted during batch creation.

        Args:
            batches_func: Function to call that creates batches.
            *args: Positional arguments to pass to batches_func.
            **kwargs: Keyword arguments to pass to batches_func.

        Returns:
            The result from batches_func.
        """
        handler = logging.handlers.MemoryHandler(capacity=100)
        particula_logger = logging.getLogger("particula")
        particula_logger.addHandler(handler)
        original_level = particula_logger.level
        particula_logger.setLevel(logging.DEBUG)
        try:
            result = batches_func(*args, **kwargs)
            handler.flush()
            # Check that no clipping log was emitted.
            self.assertFalse(
                any(
                    "Clipping num_batches" in record.getMessage()
                    for record in handler.buffer
                )
            )
            return result
        finally:
            particula_logger.removeHandler(handler)
            particula_logger.setLevel(original_level)

    def test_num_batches_equals_particles_no_log(self):
        """Equal batches and particles should not clip or log."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=5, shuffle_each_step=False
        )
        batches = self._assert_no_clipping_log(strategy._make_batches, 5)
        self.assertEqual(len(batches), 5)
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(5))

    def test_num_batches_one_creates_single_batch(self):
        """Single batch should include all particles and avoid logging."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=1, shuffle_each_step=False
        )
        batches = self._assert_no_clipping_log(strategy._make_batches, 12)
        self.assertEqual(len(batches), 1)
        np.testing.assert_array_equal(batches[0], np.arange(12))

    def test_zero_particles_returns_empty_batches_no_log(self):
        """Zero particles returns empty list without logging."""
        strategy = CondensationIsothermalStaggered(
            molar_mass=0.018, num_batches=3, shuffle_each_step=False
        )
        batches = self._assert_no_clipping_log(strategy._make_batches, 0)
        self.assertEqual(batches, [])
