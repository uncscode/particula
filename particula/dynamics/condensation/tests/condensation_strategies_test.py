"""Test module for the condensation strategies."""

# pylint: disable=R0801
# pylint: disable=protected-access
import copy
import types
import unittest
import warnings

import numpy as np
import particula as par
import pytest
from particula.dynamics.condensation._native_step import (
    validate_native_time_step,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    _partial_pressure_from_strategy,
    _pure_vapor_pressure_from_strategy,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas.gas_data import GasData
from particula.particles.particle_data import ParticleData


def _repeat_particle(particle, n_boxes):
    """Construct multi-box native storage for admission rejection checks."""
    return ParticleData(
        masses=np.repeat(particle.masses, n_boxes, axis=0),
        concentration=np.repeat(particle.concentration, n_boxes, axis=0),
        charge=np.repeat(particle.charge, n_boxes, axis=0),
        density=particle.density.copy(),
        volume=np.repeat(particle.volume, n_boxes),
        distribution_type=particle.distribution_type,
    )


def _repeat_gas(gas, n_boxes):
    """Construct matching full gas storage without a facade conversion."""
    return GasData(
        name=gas.name.copy(),
        molar_mass=gas.molar_mass.copy(),
        partitioning=gas.partitioning.copy(),
        concentration=np.repeat(gas.concentration, n_boxes, axis=0),
    )


# pylint: disable=too-many-instance-attributes
@pytest.mark.usefixtures("forbid_legacy_facades")
class TestCondensationIsothermal(unittest.TestCase):
    """Test class for the CondensationIsothermal strategy."""

    def setUp(self):
        """Set up condensation strategy and test aerosol system."""
        self.molar_mass = 0.018  # kg/mol for water
        self.diffusion_coefficient = 2e-5  # m^2/s
        self.accommodation_coefficient = 1.0

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
            par.gas.GasDataBuilder()
            .set_names(["H2O", "NH4HSO4"])
            .set_molar_mass(np.array([mm_water, mm_core]), "kg/mol")
            .set_concentration(np.array([water_conc, 1e-30]), "kg/m^3")
            .set_partitioning([True, True])
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
        self.particle = par.particles.create_resolved_particle_data(
            masses=mass_spec, density=densities, volume=1e-6
        )
        self.activity_strategy = activity
        self.surface_strategy = surface
        self.vapor_pressure_strategy = [vp_water, vp_constant]
        self.molar_mass = self.gas_species.molar_mass.copy()
        self.species_map = CondensationSpeciesMap(
            tuple(self.gas_species.name), ((0, 0), (1, 1))
        )
        self.strategy = self._make_data_strategy()

    def _make_data_strategy(self) -> CondensationIsothermal:
        """Return CondensationIsothermal configured for data-only inputs."""
        return CondensationIsothermal(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
        )

    def _make_data_inputs(self) -> tuple[ParticleData, GasData]:
        """Return independent copies of native particle and gas fixtures."""
        return (
            copy.deepcopy(self.particle),
            copy.deepcopy(self.gas_species),
        )

    def test_validate_time_step_allows_zero_and_rejects_invalid(self):
        """_validate_time_step accepts zero and rejects invalid values."""
        validate_native_time_step(0.0)
        with self.assertRaisesRegex(
            ValueError, "time_step must be a finite nonnegative scalar"
        ):
            validate_native_time_step(-0.5)

    def test_mean_free_path(self):
        """Test the mean free path call."""
        result = self.strategy.mean_free_path(
            temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_native_step_preserves_input_types_and_identity(self):
        """Supported native inputs are retained without wrapping or copies."""
        particle, gas = self._make_data_inputs()
        result = self.strategy.step(
            particle, gas, self.temperature, self.pressure, 0.0
        )
        self.assertIsInstance(result[0], ParticleData)
        self.assertIsInstance(result[1], GasData)
        self.assertIs(result[0], particle)
        self.assertIs(result[1], gas)

    def test_native_rate_rejects_invalid_container_types(self):
        """Unsupported particle/gas inputs cannot reach the rate calculation."""
        for particle, gas in (
            (object(), self.gas_species),
            (self.particle, object()),
        ):
            with self.subTest(particle=type(particle), gas=type(gas)):
                with self.assertRaises(TypeError):
                    self.strategy.rate(
                        particle, gas, self.temperature, self.pressure
                    )

    def test_native_step_rejects_invalid_container_types(self):
        """A native particle cannot be stepped with an obsolete gas-like object."""
        before = self.particle.masses.copy()
        with self.assertRaises(TypeError):
            self.strategy.step(
                self.particle, object(), self.temperature, self.pressure, 1.0
            )
        np.testing.assert_array_equal(self.particle.masses, before)

    def test_native_pressure_rejects_multi_box(self):
        """Pressure evaluation enforces the public single-box contract."""
        with self.assertRaisesRegex(ValueError, "n_boxes=1"):
            self.strategy.calculate_pressure_delta(
                _repeat_particle(self.particle, 2),
                _repeat_gas(self.gas_species, 2),
                self.temperature,
                self.particle.radii[0],
            )

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

        gas_data = copy.deepcopy(self.gas_species)
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

    def test_rate_rejects_untagged_population(self):
        """Native process configuration cannot infer missing distribution meaning."""
        particle, gas = self._make_data_inputs()
        particle.distribution_type = None
        with self.assertRaisesRegex(ValueError, "distribution_type"):
            self.strategy.rate(particle, gas, self.temperature, self.pressure)

    def test_mass_transfer_rate_rejects_multi_box_data_inputs(self):
        """mass_transfer_rate() preserves the public single-box guard."""
        strategy = self._make_data_strategy()

        particle_data = _repeat_particle(self.particle, 2)
        gas_data = _repeat_gas(self.gas_species, 2)

        with self.assertRaisesRegex(
            ValueError,
            "ParticleData must have n_boxes=1 for condensation",
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

        particle_data = _repeat_particle(self.particle, 2)
        gas_data = _repeat_gas(self.gas_species, 2)

        with self.assertRaisesRegex(
            ValueError,
            "ParticleData must have n_boxes=1 for condensation",
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
        result = CondensationIsothermal(molar_mass=0.018).knudsen_number(
            radius=radius, temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_first_order_mass_transport(self):
        """Test the first order mass transport call."""
        radius = 1e-9  # m
        result = CondensationIsothermal(
            molar_mass=0.018
        ).first_order_mass_transport(
            particle_radius=radius,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsNotNone(result)

    def test_native_transport_preserves_particle_species_axes(self):
        """Three particle radii and two species cannot broadcast ambiguously."""
        particle = par.particles.create_resolved_particle_data(
            masses=self.particle.masses[0] * np.array([[1.0], [2.0], [3.0]]),
            density=self.particle.density,
            volume=self.particle.volume[0],
        )
        rates = self.strategy.mass_transfer_rate(
            particle, self.gas_species, self.temperature, self.pressure
        )
        self.assertEqual(rates.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(rates)))
        self.assertTrue(np.all(rates[:, 0] != 0))

    def test_step_rejects_invalid_time_step(self):
        """step() rejects non-finite or negative timesteps."""
        with self.assertRaisesRegex(
            ValueError, "time_step must be a finite nonnegative scalar"
        ):
            self.strategy.step(
                particle=self.particle,
                gas_species=self.gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=-1.0,
            )

    def test_inactive_zero_radius_has_zero_rate(self):
        """Free resolved slots have finite zero rates beside an active slot."""
        particle = par.particles.create_resolved_particle_data(
            masses=np.vstack([np.zeros(2), self.particle.masses[0, 0]]),
            density=self.particle.density,
            volume=self.particle.volume[0],
            concentration=[0.0, 1.0],
        )
        rate = self.strategy.rate(
            particle, self.gas_species, self.temperature, self.pressure
        )
        self.assertTrue(np.all(np.isfinite(rate)))
        np.testing.assert_array_equal(rate[0], 0.0)
        self.assertTrue(np.any(rate[1] != 0.0))

    def test_all_inactive_zero_radii_are_safe_without_warning(self):
        """All-free native storage needs no legacy one-metre-radius warning."""
        particle = par.particles.create_resolved_particle_data(
            masses=np.zeros((3, 2)),
            density=self.particle.density,
            volume=self.particle.volume[0],
            concentration=np.zeros(3),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rate = self.strategy.rate(
                particle, self.gas_species, self.temperature, self.pressure
            )
        np.testing.assert_array_equal(rate, np.zeros((3, 2)))

    def test_rate_respects_skip_indices(self):
        """rate() must zero the chosen indices."""
        # Skip the condensing water (index 0) to make the effect obvious
        strategy_skip = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
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
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
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
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[skip_idx],
        )

        initial_particle_mass = self.particle.masses[0].copy()
        initial_gas_conc = self.gas_species.concentration[0].copy()

        particle_new, gas_new = strategy_skip.step(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=0.2,  # short step
        )

        final_particle_mass = particle_new.masses[0]
        final_gas_conc = gas_new.concentration[0]

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

    def test_skip_partitioning_masks_native_rates_without_mutating_inputs(self):
        """Skipped gas lanes zero their mapped rate while other lanes retain it."""
        baseline = self.strategy.rate(
            self.particle, self.gas_species, self.temperature, self.pressure
        )
        masses = self.particle.masses.copy()
        gas = self.gas_species.concentration.copy()
        self.strategy.skip_partitioning_indices = [0]
        masked = self.strategy.rate(
            self.particle, self.gas_species, self.temperature, self.pressure
        )
        np.testing.assert_array_equal(masked[:, 0], 0)
        np.testing.assert_array_equal(masked[:, 1], baseline[:, 1])
        np.testing.assert_array_equal(self.particle.masses, masses)
        np.testing.assert_array_equal(self.gas_species.concentration, gas)

    def test_native_skip_rejects_non_integer_indices(self):
        """Native map admission rejects non-integer skip indices."""
        strategy = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[0.5],
        )

        with self.assertRaisesRegex(ValueError, "skip indices"):
            strategy.rate(
                self.particle, self.gas_species, self.temperature, self.pressure
            )

    def test_native_skip_out_of_bounds_raises(self):
        """Native map admission rejects out-of-range skip indices."""
        strategy = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[3],
        )

        with self.assertRaisesRegex(ValueError, "skip indices"):
            strategy.rate(
                self.particle, self.gas_species, self.temperature, self.pressure
            )

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
        """step() retains native types for held and copied containers."""
        particle_result, gas_result = self.strategy.step(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(particle_result, ParticleData)
        self.assertIsInstance(gas_result, GasData)

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


@pytest.mark.usefixtures("forbid_legacy_facades")
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
        self.species_map = base.species_map

    def _make_data_strategy(self) -> CondensationIsothermalStaggered:
        """Return staggered strategy configured for data-only inputs."""
        return CondensationIsothermalStaggered(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
        )

    def _make_data_inputs(self) -> tuple[ParticleData, GasData]:
        """Return independent copies of native particle and gas fixtures."""
        return (
            copy.deepcopy(self.particle),
            copy.deepcopy(self.gas_species),
        )

    def test_mass_transfer_rate_returns_array(self):
        """mass_transfer_rate returns finite array with expected shape."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )
        mass_rate = strategy.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(mass_rate, np.ndarray)
        self.assertEqual(mass_rate.shape, self.particle.masses[0].shape)
        self.assertTrue(np.all(np.isfinite(mass_rate)))

    def test_rate_returns_array_and_respects_skip(self):
        """Rate returns array and zeros skipped indices."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            skip_partitioning_indices=[0],
        )
        rate = strategy.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(rate, np.ndarray)
        self.assertEqual(rate.shape, self.particle.masses[0].shape)
        np.testing.assert_array_equal(rate[..., 0], np.zeros_like(rate[..., 0]))
        self.assertTrue(np.all(np.isfinite(rate)))

    def test_rate_api_accepts_positional_arguments(self):
        """Rate supports positional call matching base signature."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )
        rate_positional = strategy.rate(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
        )
        self.assertIsInstance(rate_positional, np.ndarray)
        self.assertEqual(rate_positional.shape, self.particle.masses[0].shape)

    def _make_empty_particle(self):
        """Create an empty tagged resolved population."""
        return par.particles.create_resolved_particle_data(
            masses=np.empty((0, self.particle.n_species)),
            density=self.particle.density,
            volume=self.particle.volume[0],
        )

    def _make_particle_with_masses(self, mass_distribution, concentration):
        """Create resolved masses with explicit represented counts."""
        return par.particles.create_resolved_particle_data(
            masses=mass_distribution,
            density=self.particle.density,
            volume=self.particle.volume[0],
            concentration=np.full(mass_distribution.shape[0], concentration),
        )

    def _make_three_particle_state(self):
        """Return particle and gas copies with three distinct particles."""
        base_mass = self.particle.masses[0][0]
        mass_distribution = np.vstack(
            [
                base_mass,
                base_mass * 1.1,
                base_mass * 0.9,
            ]
        )
        particle = self._make_particle_with_masses(
            mass_distribution=mass_distribution,
            concentration=float(self.particle.concentration[0, 0]),
        )
        gas_species = copy.deepcopy(self.gas_species)
        return particle, gas_species

    def test_native_single_particle_rate_default_transport(self):
        """Single-particle native transport produces finite full-species rates."""
        strategy = self._make_data_strategy()
        mass_rate = strategy.mass_transfer_rate(
            self.particle, self.gas_species, self.temperature, self.pressure
        )
        self.assertEqual(mass_rate.shape, (1, self.particle.n_species))
        self.assertTrue(np.all(np.isfinite(mass_rate)))

    def test_native_single_particle_scalar_transport_matches_full_gas(self):
        """Scalar transport broadcasts identically to explicit full-gas lanes."""
        strategy = self._make_data_strategy()
        scalar = strategy.mass_transfer_rate(
            self.particle, self.gas_species, self.temperature, self.pressure
        )
        strategy.diffusion_coefficient = np.full(
            2, strategy.diffusion_coefficient
        )
        strategy.accommodation_coefficient = np.full(
            2, strategy.accommodation_coefficient
        )
        expanded = strategy.mass_transfer_rate(
            self.particle, self.gas_species, self.temperature, self.pressure
        )
        self.assertEqual(expanded.shape, (1, self.particle.n_species))
        self.assertTrue(np.all(np.isfinite(expanded)))
        np.testing.assert_allclose(expanded, scalar, rtol=1e-15, atol=0)

    def test_step_half_mode_produces_valid_output(self):
        """Step with theta_mode='half' returns updated particle and gas."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
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
        particle_data = _repeat_particle(self.particle, 2)
        gas_data = _repeat_gas(self.gas_species, 2)

        with self.assertRaisesRegex(
            ValueError,
            "ParticleData must have n_boxes=1 for condensation",
        ):
            strategy.step(
                particle_data,
                gas_data,
                self.temperature,
                self.pressure,
                time_step=self.time_step,
            )

    def test_staggered_step_returns_matching_types(self):
        """Staggered stepping retains native types for held and copied data."""
        particle_result, gas_result = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        ).step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertIsInstance(particle_result, ParticleData)
        self.assertIsInstance(gas_result, GasData)

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
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
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
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
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
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertTrue(np.all(particle_new.masses[0] >= 0.0))
        self.assertTrue(np.all(gas_new.concentration[0] >= 0.0))

    def test_step_basic_mass_conservation(self):
        """Total mass approximately conserved for small time step."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
        )
        initial_total = (
            self.particle.species_mass_density.sum()
            + self.gas_species.concentration[0].sum()
        )
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=0.01,
        )
        final_total = (
            particle_new.species_mass_density.sum()
            + gas_new.concentration[0].sum()
        )
        np.testing.assert_allclose(initial_total, final_total, rtol=1e-10)

    def test_step_respects_skip_partitioning_indices(self):
        """Skip indices remain unchanged after step."""
        skip_idx = 0
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            skip_partitioning_indices=[skip_idx],
        )
        initial_mass = self.particle.masses[0].copy()
        initial_gas = self.gas_species.concentration[0].copy()
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        np.testing.assert_allclose(
            particle_new.masses[0][..., skip_idx],
            initial_mass[..., skip_idx],
        )
        np.testing.assert_allclose(
            gas_new.concentration[0][skip_idx], initial_gas[skip_idx]
        )

    def test_step_positional_api_matches_signature(self):
        """Step accepts positional args compatible with base signature."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )
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
            particle_new.masses[0].shape,
            self.particle.masses[0].shape,
        )

    def test_step_zero_time_step_returns_inputs(self):
        """time_step=0 returns unchanged particle and gas."""
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
        )
        initial_mass = self.particle.masses[0].copy()
        initial_gas = self.gas_species.concentration[0].copy()
        particle_new, gas_new = strategy.step(
            self.particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=0.0,
        )
        np.testing.assert_allclose(particle_new.masses[0], initial_mass)
        np.testing.assert_allclose(gas_new.concentration[0], initial_gas)

    def test_step_empty_particles_noop(self):
        """Zero-particle input returns unchanged gas and empty particle."""
        empty_particle = self._make_empty_particle()
        gas_before = self.gas_species.concentration.copy()
        strategy = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
        )
        particle_new, gas_new = strategy.step(
            empty_particle,
            self.gas_species,
            self.temperature,
            self.pressure,
            time_step=self.time_step,
        )
        self.assertEqual(particle_new.masses[0].shape[0], 0)
        np.testing.assert_allclose(gas_new.concentration, gas_before)

    def test_step_updates_gas_between_batches(self):
        """Multi-batch step differs from single batch due to gas updates."""
        self._make_three_particle_state()
        particle_multi, gas_multi = self._make_three_particle_state()

        strategy_multi = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=2,
            shuffle_each_step=False,
        )

        gas_inputs: list[np.ndarray] = []

        def fake_calc(self, material, gas, *args):
            gas_inputs.append(gas.concentration[0].copy())
            rates = np.zeros_like(material.masses[0])
            rates[:, 0] = gas.concentration[0, 0] * material.volume[0] * 0.05
            return rates

        strategy_multi._native_rate_from_parameters = types.MethodType(  # type: ignore[attr-defined]
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
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=1,
            shuffle_each_step=False,
        )
        strategy_many = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=False,
        )

        def fake_calc(self, material, gas, *args):
            rates = np.zeros_like(material.masses[0])
            rates[:, 0] = gas.concentration[0, 0] * material.volume[0] * 0.1
            return rates

        strategy_one._native_rate_from_parameters = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_one
        )
        strategy_many._native_rate_from_parameters = types.MethodType(  # type: ignore[attr-defined]
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
            gas_one.concentration[0] - gas_many.concentration[0]
        )
        particle_delta = np.linalg.norm(
            particle_one.masses[0] - particle_many.masses[0]
        )
        self.assertGreater(gas_delta, 0.0)
        self.assertGreater(particle_delta, 0.0)
        self.assertTrue(np.all(np.isfinite(particle_many.masses[0])))
        self.assertTrue(np.all(particle_many.masses[0] >= 0.0))

    def test_step_batch_order_affects_result(self):
        """Different batch ordering changes outcomes (Gauss-Seidel property)."""
        particle_ordered, gas_ordered = self._make_three_particle_state()
        particle_shuffled, gas_shuffled = self._make_three_particle_state()

        strategy_ordered = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=False,
        )
        strategy_shuffled = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=3,
            shuffle_each_step=True,
            random_state=11,
        )

        def fake_calc(self, material, gas, *args):
            # Frozen core mass distinguishes particles across reordered batches.
            rates = np.zeros_like(material.masses[0])
            rates[:, 0] = (
                gas.concentration[0, 0] * material.volume[0] * 0.05
                + material.masses[0, :, 1] * 0.1
            )
            return rates

        strategy_ordered._native_rate_from_parameters = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_ordered
        )
        strategy_shuffled._native_rate_from_parameters = types.MethodType(  # type: ignore[attr-defined]
            fake_calc, strategy_shuffled
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
            gas_ordered.concentration[0] - gas_shuffled.concentration[0]
        )
        particle_delta = np.linalg.norm(
            particle_ordered.masses[0] - particle_shuffled.masses[0]
        )
        self.assertGreater(gas_delta, 0.0)
        self.assertGreater(particle_delta, 0.0)

    def test_step_num_batches_exceeds_particle_count_clips(self):
        """Excess batches clip to particle count without changing outcome."""
        particle_high, gas_high = self._make_three_particle_state()
        particle_exact, gas_exact = self._make_three_particle_state()

        strategy_high = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="half",
            num_batches=10,
            shuffle_each_step=False,
        )
        strategy_exact = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
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
            gas_high.concentration[0], gas_exact.concentration[0]
        )
        np.testing.assert_allclose(
            particle_high.masses[0], particle_exact.masses[0]
        )
        self.assertTrue(np.all(gas_high.concentration[0] >= 0.0))

    def test_step_batch_with_theta_modes(self):
        """Batch mode works with half, random, and batch theta modes."""
        for mode in ("half", "batch"):
            particle, gas_species = self._make_three_particle_state()
            strategy = CondensationIsothermalStaggered(
                activity_strategy=self.activity_strategy,
                surface_strategy=self.surface_strategy,
                vapor_pressure_strategy=self.vapor_pressure_strategy,
                species_map=self.species_map,
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
            self.assertTrue(np.all(particle_new.masses[0] >= 0.0))
            self.assertTrue(np.all(np.isfinite(gas_new.concentration[0])))

        particle_a, gas_a = self._make_three_particle_state()
        particle_b, gas_b = self._make_three_particle_state()
        strategy_a = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            theta_mode="random",
            num_batches=3,
            shuffle_each_step=False,
            random_state=7,
        )
        strategy_b = CondensationIsothermalStaggered(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
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
            gas_a.concentration[0], gas_b.concentration[0]
        )
        np.testing.assert_allclose(particle_a.masses[0], particle_b.masses[0])


@pytest.mark.parametrize(
    "strategy_type",
    [
        CondensationIsothermal,
        CondensationIsothermalStaggered,
        par.dynamics.CondensationLatentHeat,
    ],
)
def test_empty_multispecies_native_configuration(
    strategy_type, forbid_legacy_facades
):
    """Empty material avoids Kelvin broadcasting without bypassing gas checks."""
    fixture = TestCondensationIsothermal()
    fixture.setUp()
    particle = par.particles.create_resolved_particle_data(
        masses=np.empty((0, 2)), density=fixture.particle.density, volume=1e-6
    )
    gas = fixture.gas_species
    strategy = strategy_type(
        molar_mass=gas.molar_mass,
        activity_strategy=fixture.activity_strategy,
        surface_strategy=fixture.surface_strategy,
        vapor_pressure_strategy=fixture.vapor_pressure_strategy,
        species_map=fixture.species_map,
    )
    before = gas.concentration.copy()
    rate = strategy.mass_transfer_rate(
        particle, gas, fixture.temperature, fixture.pressure
    )
    assert rate.shape == (0, 2)
    result = strategy.step(
        particle, gas, fixture.temperature, fixture.pressure, 1.0
    )
    assert result[0] is particle and result[1] is gas
    np.testing.assert_array_equal(gas.concentration, before)
    strategy.vapor_pressure_strategy = par.gas.ConstantVaporPressureStrategy(
        np.nan
    )
    with pytest.raises(ValueError, match="vapor pressure must be finite"):
        strategy.step(particle, gas, fixture.temperature, fixture.pressure, 1.0)
    np.testing.assert_array_equal(gas.concentration, before)


@pytest.mark.parametrize(
    "strategy_type",
    [
        CondensationIsothermal,
        CondensationIsothermalStaggered,
        par.dynamics.CondensationLatentHeat,
    ],
)
@pytest.mark.parametrize(
    "method", ["calculate_pressure_delta", "mass_transfer_rate", "rate", "step"]
)
@pytest.mark.parametrize("fault", ["untagged", "particle_type", "gas_type"])
def test_native_only_entry_points_reject_without_mutation(
    strategy_type, method, fault, forbid_legacy_facades
):
    """Every public entry rejects unsupported carriers before state changes."""
    fixture = TestCondensationIsothermal()
    fixture.setUp()
    particle, gas = fixture._make_data_inputs()
    strategy = strategy_type(
        molar_mass=gas.molar_mass,
        activity_strategy=fixture.activity_strategy,
        surface_strategy=fixture.surface_strategy,
        vapor_pressure_strategy=fixture.vapor_pressure_strategy,
        species_map=fixture.species_map,
    )
    original_particle, original_gas = particle, gas
    masses = particle.masses.copy()
    concentration = particle.concentration.copy()
    gas_concentration = gas.concentration.copy()
    if fault == "untagged":
        particle.distribution_type = None
    elif fault == "particle_type":
        particle = object()
    else:
        gas = object()
    args = [particle, gas, fixture.temperature]
    if method == "calculate_pressure_delta":
        args.append(fixture.particle.radii[0])
    else:
        args.append(fixture.pressure)
        if method == "step":
            args.append(1.0)
    expected = ValueError if fault == "untagged" else TypeError
    match = (
        "distribution_type"
        if fault == "untagged"
        else "ParticleData and GasData"
    )
    with pytest.raises(expected, match=match):
        getattr(strategy, method)(*args)
    np.testing.assert_array_equal(original_particle.masses, masses)
    np.testing.assert_array_equal(
        original_particle.concentration, concentration
    )
    np.testing.assert_array_equal(original_gas.concentration, gas_concentration)
