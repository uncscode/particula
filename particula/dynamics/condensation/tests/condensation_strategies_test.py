"""Test module for the condensation strategies."""

# pylint: disable=R0801
# pylint: disable=protected-access
import copy
import types
import unittest

import numpy as np
import particula as par  # new – we will build real objects
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    _normalize_first_order_mass_transport,
    _partial_pressure_from_strategy,
    _pure_vapor_pressure_from_strategy,
    _require_matching_types,
    _require_single_box,
    _unwrap_gas,
    _unwrap_particle,
    _validate_time_step,
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

    def test_validate_time_step_allows_zero_and_rejects_invalid(self):
        """_validate_time_step accepts zero and rejects invalid values."""
        _validate_time_step(0.0)
        with self.assertRaisesRegex(ValueError, "time_step must be finite"):
            _validate_time_step(-0.5)

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

    def test_mass_transfer_rate_rejects_multi_box_data_inputs(self):
        """mass_transfer_rate() preserves the public single-box guard."""
        strategy = self._make_data_strategy()

        particle_data = from_representation(self.particle, n_boxes=2)
        gas_data = from_species(self.gas_species, n_boxes=2)

        with self.assertRaisesRegex(
            ValueError,
            "ParticleData must have n_boxes=1 for condensation strategies, "
            "got n_boxes=2.",
        ):
            strategy.mass_transfer_rate(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
            )

    def test_step_rejects_multi_box_data_inputs(self):
        """step() preserves the public single-box guard for data inputs."""
        strategy = self._make_data_strategy()

        particle_data = from_representation(self.particle, n_boxes=2)
        gas_data = from_species(self.gas_species, n_boxes=2)

        with self.assertRaisesRegex(
            ValueError,
            "ParticleData must have n_boxes=1 for condensation strategies, "
            "got n_boxes=2.",
        ):
            strategy.step(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=self.time_step,
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

    def test_normalize_first_order_mass_transport_2d(self):
        """Normalize transport shape when pressure delta is 2D."""
        pressure_delta = np.ones((3, 2))
        mass_transport = np.array([1.0, 2.0, 3.0])

        normalized = _normalize_first_order_mass_transport(
            mass_transport, pressure_delta
        )

        self.assertEqual(normalized.shape, (3, 1))

    def test_step_rejects_invalid_time_step(self):
        """step() rejects non-finite or negative timesteps."""
        with self.assertRaisesRegex(ValueError, "time_step must be finite"):
            self.strategy.step(
                particle=self.particle,
                gas_species=self.gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=-1.0,
            )

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

    def test_apply_skip_partitioning_rejects_non_integer_indices(self):
        """_apply_skip_partitioning rejects non-integer indices."""
        strategy = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[0.5],
        )

        with self.assertRaises(TypeError):
            strategy._apply_skip_partitioning(np.arange(3.0))

    def test_apply_skip_partitioning_out_of_bounds_raises(self):
        """_apply_skip_partitioning rejects out-of-range indices."""
        strategy = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[3],
        )

        with self.assertRaises(IndexError):
            strategy._apply_skip_partitioning(np.arange(3.0))

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

    def test_staggered_step_rejects_multi_box_data_inputs(self):
        """Staggered step rejects unsupported multi-box data inputs."""
        strategy = self._make_data_strategy()
        particle_data = from_representation(self.particle, n_boxes=2)
        gas_data = from_species(self.gas_species, n_boxes=2)

        with self.assertRaisesRegex(
            ValueError,
            "ParticleData must have n_boxes=1 for condensation strategies, "
            "got n_boxes=2.",
        ):
            strategy.step(
                particle_data,
                gas_data,
                self.temperature,
                self.pressure,
                time_step=self.time_step,
            )

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
