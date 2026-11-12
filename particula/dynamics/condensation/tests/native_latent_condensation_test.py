"""Native resolved latent-heat condensation regression contracts."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import particula as par
import pytest
from particula.dynamics.condensation import (
    CondensationIsothermal,
    CondensationLatentHeat,
)
from particula.dynamics.condensation.mass_transfer import get_mass_transfer
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas.gas_data import GasData
from particula.gas.latent_heat_strategies import ConstantLatentHeat
from particula.particles.particle_data import ParticleData


@pytest.mark.usefixtures("forbid_legacy_facades")
class TestCondensationLatentHeat(unittest.TestCase):
    """Test class for the CondensationLatentHeat strategy."""

    def setUp(self):
        """Set up latent heat condensation strategy fixtures."""
        self.diffusion_coefficient = 2e-5
        self.accommodation_coefficient = 1.0
        self.temperature = 298.15
        self.pressure = 101325

        vp_water = par.gas.VaporPressureFactory().get_strategy("water_buck")
        vp_constant = par.gas.VaporPressureFactory().get_strategy(
            "constant", {"vapor_pressure": 1e-24, "vapor_pressure_units": "Pa"}
        )

        mm_water = 18.015e-3
        mm_core = 132.14e-3
        water_sat = vp_water.saturation_concentration(
            mm_water, self.temperature
        )
        water_conc = water_sat * 1.02

        self.gas_species = (
            par.gas.GasDataBuilder()
            .set_names(["H2O", "NH4HSO4"])
            .set_molar_mass(np.array([mm_water, mm_core]), "kg/mol")
            .set_concentration(np.array([water_conc, 1e-30]), "kg/m^3")
            .set_partitioning([True, True])
            .build()
        )

        density_core = 1.77e3
        r = np.array([100e-9])
        mass_core = 4 / 3 * np.pi * r**3 * density_core
        mass_spec = np.column_stack([mass_core * 0, mass_core])
        densities = np.array([1e3, density_core])

        self.activity_strategy = (
            par.particles.ActivityKappaParameterBuilder()
            .set_density(densities, "kg/m^3")
            .set_kappa(np.array([0.0, 0.61]))
            .set_molar_mass(np.array([mm_water, mm_core]), "kg/mol")
            .set_water_index(0)
            .build()
        )
        self.surface_strategy = (
            par.particles.SurfaceStrategyVolumeBuilder()
            .set_density(densities, "kg/m^3")
            .set_surface_tension(np.array([0.072, 0.092]), "N/m")
            .build()
        )
        self.particle = par.particles.create_resolved_particle_data(
            masses=mass_spec, density=densities, volume=1e-6
        )

        self.molar_mass = self.gas_species.molar_mass.copy()
        self.vapor_pressure_strategy = [vp_water, vp_constant]
        self.species_map = CondensationSpeciesMap(
            tuple(self.gas_species.name), ((0, 0), (1, 1))
        )
        self.data_strategy = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
        )

    def _make_data_inputs(self) -> tuple[ParticleData, GasData]:
        """Return independent copies of the native fixtures."""
        return (
            copy.deepcopy(self.particle),
            copy.deepcopy(self.gas_species),
        )

    def _build_single_species_fixture(
        self, n_particles: int = 1
    ) -> tuple[
        GasData,
        ParticleData,
        par.particles.ActivityKappaParameter,
        par.particles.SurfaceStrategyVolume,
    ]:
        """Build single-species particle/gas fixtures for shape tests."""
        vp_water = par.gas.VaporPressureFactory().get_strategy("water_buck")
        mm_water = 18.015e-3
        water_sat = vp_water.saturation_concentration(
            mm_water, self.temperature
        )
        water_conc = water_sat * 1.02

        gas_species = (
            par.gas.GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass(np.array([mm_water]), "kg/mol")
            .set_concentration(np.array([water_conc]), "kg/m^3")
            .set_partitioning([True])
            .build()
        )

        density = np.array([1e3])
        r = np.full(n_particles, 100e-9)
        mass = 4 / 3 * np.pi * r**3 * density[0]
        mass_spec = mass[:, np.newaxis]

        activity = (
            par.particles.ActivityKappaParameterBuilder()
            .set_density(density, "kg/m^3")
            .set_kappa(np.array([0.0]))
            .set_molar_mass(np.array([mm_water]), "kg/mol")
            .set_water_index(0)
            .build()
        )
        surface = (
            par.particles.SurfaceStrategyVolumeBuilder()
            .set_density(density, "kg/m^3")
            .set_surface_tension(np.array([0.072]), "N/m")
            .build()
        )
        particle = par.particles.create_resolved_particle_data(
            masses=mass_spec, density=density, volume=1e-6
        )

        self.molar_mass = gas_species.molar_mass.copy()
        self.activity_strategy = activity
        self.surface_strategy = surface
        self.vapor_pressure_strategy = [vp_water]
        self.species_map = CondensationSpeciesMap(
            tuple(gas_species.name), ((0, 0),)
        )
        return gas_species, particle, activity, surface

    def test_instantiation_with_strategy(self):
        """Latent heat strategy is used when provided."""
        strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.018,
            latent_heat_strategy=strategy,
        )

        self.assertIs(cond._latent_heat_strategy, strategy)

    def test_instantiation_with_constant_fallback(self):
        """Constant latent heat fallback wraps in ConstantLatentHeat."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.018,
            latent_heat=2.26e6,
        )

        self.assertIsInstance(cond._latent_heat_strategy, ConstantLatentHeat)
        assert cond._latent_heat_strategy is not None
        self.assertEqual(cond._latent_heat_strategy.latent_heat_ref, 2.26e6)

    def test_instantiation_isothermal_fallback(self):
        """Default latent heat uses isothermal fallback without warnings."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.018,
        )

        self.assertIsNone(cond._latent_heat_strategy)

    def test_resolution_priority_strategy_over_constant(self):
        """Strategy takes precedence over constant latent heat."""
        strategy = ConstantLatentHeat(latent_heat_ref=1.0e6)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.018,
            latent_heat_strategy=strategy,
            latent_heat=2.0e6,
        )

        self.assertIs(cond._latent_heat_strategy, strategy)

    def test_resolution_with_array_latent_heat_logs_and_falls_back(self):
        """Array latent heat logs warning and falls back to isothermal."""
        with self.assertLogs("particula", level="WARNING") as cm:
            cond = CondensationLatentHeat(
                activity_strategy=self.activity_strategy,
                surface_strategy=self.surface_strategy,
                vapor_pressure_strategy=self.vapor_pressure_strategy,
                species_map=self.species_map,
                molar_mass=0.018,
                latent_heat=np.array([2.26e6, 1.5e6]),
            )

        self.assertIsNone(cond._latent_heat_strategy)
        self.assertTrue(
            any(
                "Array-like latent_heat" in record.getMessage()
                for record in cm.records
            )
        )

    def test_resolution_with_negative_latent_heat_logs_and_falls_back(self):
        """Negative latent heat logs warning and falls back to isothermal."""
        with self.assertLogs("particula", level="WARNING") as cm:
            cond = CondensationLatentHeat(
                activity_strategy=self.activity_strategy,
                surface_strategy=self.surface_strategy,
                vapor_pressure_strategy=self.vapor_pressure_strategy,
                species_map=self.species_map,
                molar_mass=0.018,
                latent_heat=-1.0,
            )

        self.assertIsNone(cond._latent_heat_strategy)
        self.assertTrue(
            any(
                "Negative latent_heat" in record.getMessage()
                for record in cm.records
            )
        )

    def test_last_latent_heat_energy_initialized(self):
        """Diagnostic latent heat energy starts at zero."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.018,
        )

        self.assertEqual(cond.last_latent_heat_energy, 0.0)

    def test_base_class_params_forwarded(self):
        """Base parameters are forwarded to the parent class."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.02,
            diffusion_coefficient=1.5e-5,
            accommodation_coefficient=0.7,
            update_gases=False,
        )

        self.assertEqual(cond.molar_mass, 0.02)
        self.assertEqual(cond.diffusion_coefficient, 1.5e-5)
        self.assertEqual(cond.accommodation_coefficient, 0.7)
        self.assertFalse(cond.update_gases)

    def test_mass_transfer_rate_isothermal_parity(self):
        """Latent heat strategy falls back to isothermal rates."""
        latent_cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )
        iso_cond = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        latent_rate = latent_cond.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        iso_rate = iso_cond.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        np.testing.assert_allclose(latent_rate, iso_rate, rtol=1e-15)

    def test_rate_isothermal_parity(self):
        """rate() matches isothermal strategy when latent heat is absent."""
        latent_cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )
        iso_cond = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        latent_rate = latent_cond.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        iso_rate = iso_cond.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        np.testing.assert_allclose(latent_rate, iso_rate, rtol=1e-15)

    def test_mass_transfer_rate_reduced_with_latent_heat(self):
        """Latent heat correction reduces mass transfer magnitude."""
        latent_cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )
        iso_cond = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        latent_rate = latent_cond.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        iso_rate = iso_cond.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        self.assertTrue(np.all(np.abs(latent_rate) <= np.abs(iso_rate) + 1e-30))

    def test_mass_transfer_rate_single_species(self):
        """Single-species output retains particle and species axes."""
        gas_species, particle, _, _ = self._build_single_species_fixture()
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )

        rate = cond.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        self.assertEqual(rate.shape, (1, 1))

    def test_get_vapor_pressure_surface_single_species(self):
        """Surface pressure retains particle/species axes for single species."""
        gas_species, particle, _, _ = self._build_single_species_fixture(
            n_particles=3
        )
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )

        vapor_pressure_surface = cond._native_pressure_terms(
            particle=particle,
            gas=gas_species,
            temperature=self.temperature,
            radius=particle.radii[0],
            sources=np.arange(gas_species.n_species, dtype=np.intp),
            targets=np.arange(particle.n_species, dtype=np.intp),
        )[1]

        self.assertEqual(vapor_pressure_surface.shape, (3, 1))
        self.assertTrue(np.all(np.isfinite(vapor_pressure_surface)))

    def test_mass_transfer_rate_multi_species(self):
        """Multi-species output preserves particle/species shape."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )

        rate = cond.mass_transfer_rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        self.assertEqual(rate.shape, (1, 2))

    def test_get_vapor_pressure_surface_multi_species(self):
        """Surface vapor pressure preserves particle/species shape."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )

        vapor_pressure_surface = cond._native_pressure_terms(
            particle=self.particle,
            gas=self.gas_species,
            temperature=self.temperature,
            radius=self.particle.radii[0],
            sources=np.arange(self.gas_species.n_species, dtype=np.intp),
            targets=np.arange(self.particle.n_species, dtype=np.intp),
        )[1]

        self.assertEqual(vapor_pressure_surface.shape, (1, 2))
        self.assertTrue(np.all(np.isfinite(vapor_pressure_surface)))

    def test_rate_scales_by_concentration(self):
        """Condensation rate scales linearly with concentration."""
        particle_data, gas_data = self._make_data_inputs()

        base_rate = self.data_strategy.rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        particle_data_double = copy.deepcopy(particle_data)
        particle_data_double.concentration[0] = (
            particle_data_double.concentration[0] * 2.0
        )
        double_rate = self.data_strategy.rate(
            particle=particle_data_double,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        np.testing.assert_allclose(double_rate, base_rate * 2.0, rtol=1e-15)

    def test_rate_respects_skip_partitioning(self):
        """Skip-partitioning indices zero out selected species."""
        particle_data, gas_data = self._make_data_inputs()
        strategy = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            skip_partitioning_indices=[1],
        )

        rates = strategy.rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        self.assertTrue(np.all(rates[:, 1] == 0.0))

    def test_rate_skip_partitioning_out_of_bounds_raises(self):
        """Out-of-bounds skip indices raise an error."""
        particle_data, gas_data = self._make_data_inputs()
        strategy = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            skip_partitioning_indices=[5],
        )

        with self.assertRaisesRegex(ValueError, "skip indices"):
            strategy.rate(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
            )

    def test_nonfinite_pressure_delta_sanitized(self):
        """Non-finite pressure deltas are sanitized to zero."""
        gas_species, particle, _, _ = self._build_single_species_fixture(
            n_particles=3
        )
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )

        with patch.object(
            CondensationLatentHeat,
            "_native_pressure_terms",
            return_value=(
                np.array([[np.nan], [np.inf], [-np.inf]]),
                np.ones((3, 1)),
            ),
        ) as pressure_terms:
            rates = cond.mass_transfer_rate(
                particle=particle,
                gas_species=gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
            )
        pressure_terms.assert_called_once()
        self.assertTrue(np.all(np.isfinite(rates)))
        np.testing.assert_array_equal(rates, np.zeros((3, 1)))

    def test_nonfinite_pressure_delta_matches_isothermal(self):
        """Latent heat sanitization mirrors isothermal behavior."""
        gas_species, particle, _, _ = self._build_single_species_fixture(
            n_particles=3
        )
        latent_cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )
        iso_cond = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )
        nonfinite = np.array([np.nan, np.inf, -np.inf])

        with (
            patch.object(
                CondensationLatentHeat,
                "_native_pressure_terms",
                return_value=(nonfinite[:, None], np.ones((3, 1))),
            ),
            patch.object(
                CondensationIsothermal,
                "_native_pressure_terms",
                return_value=(nonfinite[:, None], np.ones((3, 1))),
            ),
        ):
            latent_rates = latent_cond.mass_transfer_rate(
                particle=particle,
                gas_species=gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
            )
            iso_rates = iso_cond.mass_transfer_rate(
                particle=particle,
                gas_species=gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
            )

        np.testing.assert_allclose(latent_rates, iso_rates, rtol=1e-14)

    def test_nonfinite_latent_heat_raises(self):
        """Non-finite latent heat raises an error."""
        gas_species, particle, _, _ = self._build_single_species_fixture()

        class NonFiniteLatentHeat:
            """Latent heat strategy returning non-finite value."""

            def latent_heat(self, temperature: float) -> float:
                return float("nan")

        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            latent_heat_strategy=NonFiniteLatentHeat(),
        )

        with self.assertRaises(ValueError):
            cond.mass_transfer_rate(
                particle=particle,
                gas_species=gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
            )

    def test_nonfinite_thermal_conductivity_raises(self):
        """Non-finite thermal conductivity raises an error."""
        gas_species, particle, _, _ = self._build_single_species_fixture()
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        with patch(
            "particula.dynamics.condensation.condensation_strategies."
            "get_thermal_conductivity",
            return_value=np.nan,
        ):
            with self.assertRaises(ValueError):
                cond.mass_transfer_rate(
                    particle=particle,
                    gas_species=gas_species,
                    temperature=self.temperature,
                    pressure=self.pressure,
                )

    def test_nonfinite_vapor_pressure_surface_raises(self):
        """Non-finite surface vapor pressure raises an error."""
        gas_species, particle, _, surface = self._build_single_species_fixture()
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        with patch.object(
            surface,
            "kelvin_term",
            return_value=np.array([np.nan]),
        ):
            with self.assertRaises(ValueError):
                cond.mass_transfer_rate(
                    particle=particle,
                    gas_species=gas_species,
                    temperature=self.temperature,
                    pressure=self.pressure,
                )

    def test_inherited_methods_work(self):
        """Base class helpers remain available."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=0.018,
        )

        mean_free_path = cond.mean_free_path(
            temperature=298.15, pressure=101325
        )
        knudsen_number = cond.knudsen_number(
            radius=1e-7, temperature=298.15, pressure=101325
        )

        self.assertGreater(mean_free_path, 0.0)
        self.assertGreater(knudsen_number, 0.0)

    def test_step_mass_conservation_single_species(self):
        """Single-species step conserves total mass."""
        gas_species, particle, _, _ = self._build_single_species_fixture()
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_mass = (
            particle.species_mass_density.sum()
            + gas_species.concentration[0].sum()
        )
        particle_new, gas_new = cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_mass = (
            particle_new.species_mass_density.sum()
            + gas_new.concentration[0].sum()
        )

        relative_error = abs(final_mass - initial_mass) / initial_mass
        self.assertLess(relative_error, 1e-12)

    def test_step_mass_conservation_multi_species(self):
        """Multi-species step conserves total mass."""
        particle = copy.deepcopy(self.particle)
        gas_species = copy.deepcopy(self.gas_species)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_mass = (
            particle.species_mass_density.sum()
            + gas_species.concentration[0].sum()
        )
        particle_new, gas_new = cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_mass = (
            particle_new.species_mass_density.sum()
            + gas_new.concentration[0].sum()
        )

        relative_error = abs(final_mass - initial_mass) / initial_mass
        self.assertLess(relative_error, 1e-12)

    def test_step_mass_conservation_data_only(self):
        """Data-only step conserves total mass."""
        particle_data, gas_data = self._make_data_inputs()
        cond = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        initial_mass = np.sum(
            particle_data.masses[0]
            * particle_data.concentration[0][:, np.newaxis]
            / particle_data.volume[0]
        ) + np.sum(gas_data.concentration[0])
        particle_new, gas_new = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        final_mass = np.sum(
            particle_new.masses[0]
            * particle_new.concentration[0][:, np.newaxis]
            / particle_new.volume[0]
        ) + np.sum(gas_new.concentration[0])

        relative_error = abs(final_mass - initial_mass) / initial_mass
        self.assertLess(relative_error, 1e-12)

    def test_step_rejects_nonpositive_volume(self):
        """Step rejects nonpositive volumes for norm_conc."""
        particle_data, gas_data = self._make_data_inputs()
        particle_data.volume[0] = 0.0

        with self.assertRaisesRegex(
            ValueError, "particle-resolved volume must be positive"
        ):
            self.data_strategy.step(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=1.0,
            )

    def test_step_rejects_invalid_concentration_for_norm_conc(self):
        """Step rejects non-finite or negative concentrations for norm_conc."""
        particle_data, gas_data = self._make_data_inputs()

        for bad_value in (-1.0, np.nan):
            with self.subTest(bad_value=bad_value):
                particle_copy = copy.deepcopy(particle_data)
                particle_copy.concentration[0][0] = bad_value

                with self.assertRaisesRegex(
                    ValueError, "particle data must have finite physical values"
                ):
                    self.data_strategy.step(
                        particle=particle_copy,
                        gas_species=gas_data,
                        temperature=self.temperature,
                        pressure=self.pressure,
                        time_step=1.0,
                    )

    def test_step_energy_tracking_matches_dm_times_l(self):
        """Latent heat energy matches dm × L for a step."""
        particle = copy.deepcopy(self.particle)
        gas_species = copy.deepcopy(self.gas_species)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        particle_data = particle
        gas_data = gas_species
        mass_rate = cond.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        if isinstance(mass_rate, (int, float)):
            mass_rate_array = np.array([mass_rate])
        else:
            mass_rate_array = mass_rate
        # Native mass rates already exclude skipped full-gas map lanes.
        norm_conc = particle_data.concentration[0] / particle_data.volume[0]
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate_array,
            time_step=1.0,
            gas_mass=np.asarray(gas_data.concentration[0], dtype=np.float64),
            particle_mass=particle_data.masses[0],
            particle_concentration=norm_conc,
        )

        cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        latent_heat = cond._latent_heat_strategy.latent_heat(self.temperature)

        np.testing.assert_allclose(
            cond.last_latent_heat_energy,
            np.sum(mass_transfer * particle_data.volume[0] * latent_heat),
            rtol=1e-14,
        )

    def test_step_energy_tracking_update_gases_false(self):
        """Energy tracking works when gas updates are disabled."""
        gas_species, particle, _, _ = self._build_single_species_fixture()
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
            update_gases=False,
        )

        particle_data = particle
        gas_data = gas_species
        mass_rate = cond.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        if isinstance(mass_rate, (int, float)):
            mass_rate_array = np.array([mass_rate])
        else:
            mass_rate_array = mass_rate
        # Native mass rates already exclude skipped full-gas map lanes.
        norm_conc = particle_data.concentration[0] / particle_data.volume[0]
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate_array,
            time_step=1.0,
            gas_mass=np.asarray(gas_data.concentration[0], dtype=np.float64),
            particle_mass=particle_data.masses[0],
            particle_concentration=norm_conc,
        )
        species_mass = particle_data.masses[0]
        species_count = 1 if species_mass.ndim == 1 else species_mass.shape[1]
        if mass_transfer.ndim == 1:
            mass_transfer = mass_transfer.reshape(-1, species_count)
        elif mass_transfer.shape[1] > species_count:
            mass_transfer = mass_transfer[:, :species_count]
        elif mass_transfer.shape[1] < species_count:
            mass_transfer = np.broadcast_to(
                mass_transfer, (mass_transfer.shape[0], species_count)
            )
        initial_conc = gas_species.concentration[0].copy()

        cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        np.testing.assert_allclose(
            gas_species.concentration[0], initial_conc, rtol=1e-15
        )
        latent_heat = cond._latent_heat_strategy.latent_heat(self.temperature)
        np.testing.assert_allclose(
            cond.last_latent_heat_energy,
            np.sum(mass_transfer * particle_data.volume[0] * latent_heat),
            rtol=1e-14,
        )

    def test_step_energy_tracking_data_inputs(self):
        """Latent heat energy tracks on data container inputs."""
        particle_data, gas_data = self._make_data_inputs()
        cond = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        mass_rate = cond.mass_transfer_rate(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        if isinstance(mass_rate, (int, float)):
            mass_rate_array = np.array([mass_rate])
        else:
            mass_rate_array = mass_rate
        # Native mass rates already exclude skipped full-gas map lanes.
        norm_conc = particle_data.concentration[0] / particle_data.volume[0]
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate_array,
            time_step=1.0,
            gas_mass=np.asarray(gas_data.concentration[0], dtype=np.float64),
            particle_mass=particle_data.masses[0],
            particle_concentration=norm_conc,
        )
        species_mass = particle_data.masses[0]
        species_count = 1 if species_mass.ndim == 1 else species_mass.shape[1]
        if mass_transfer.ndim == 1:
            mass_transfer = mass_transfer.reshape(-1, species_count)
        elif mass_transfer.shape[1] > species_count:
            mass_transfer = mass_transfer[:, :species_count]
        elif mass_transfer.shape[1] < species_count:
            mass_transfer = np.broadcast_to(
                mass_transfer, (mass_transfer.shape[0], species_count)
            )

        cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        latent_heat = cond._latent_heat_strategy.latent_heat(self.temperature)
        np.testing.assert_allclose(
            cond.last_latent_heat_energy,
            np.sum(mass_transfer * particle_data.volume[0] * latent_heat),
            rtol=1e-14,
        )

    def test_step_energy_tracking_skip_partitioning(self):
        """Skip-partitioning affects latent heat bookkeeping."""
        particle = copy.deepcopy(self.particle)
        gas_species = copy.deepcopy(self.gas_species)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
            skip_partitioning_indices=[1],
        )

        particle_data = particle
        gas_data = gas_species
        mass_rate = cond.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        if isinstance(mass_rate, (int, float)):
            mass_rate_array = np.array([mass_rate])
        else:
            mass_rate_array = mass_rate
        # Native mass rates already exclude skipped full-gas map lanes.
        norm_conc = particle_data.concentration[0] / particle_data.volume[0]
        mass_transfer = get_mass_transfer(
            mass_rate=mass_rate_array,
            time_step=1.0,
            gas_mass=np.asarray(gas_data.concentration[0], dtype=np.float64),
            particle_mass=particle_data.masses[0],
            particle_concentration=norm_conc,
        )
        species_mass = particle_data.masses[0]
        species_count = 1 if species_mass.ndim == 1 else species_mass.shape[1]
        if mass_transfer.ndim == 1:
            mass_transfer = mass_transfer.reshape(-1, species_count)
        elif mass_transfer.shape[1] > species_count:
            mass_transfer = mass_transfer[:, :species_count]
        elif mass_transfer.shape[1] < species_count:
            mass_transfer = np.broadcast_to(
                mass_transfer, (mass_transfer.shape[0], species_count)
            )

        cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        latent_heat = cond._latent_heat_strategy.latent_heat(self.temperature)
        np.testing.assert_allclose(
            cond.last_latent_heat_energy,
            np.sum(mass_transfer * particle_data.volume[0] * latent_heat),
            rtol=1e-14,
        )

    def test_step_energy_sign_convention(self):
        """Condensation is positive energy; evaporation is negative."""
        gas_species, particle, _, _ = self._build_single_species_fixture()
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )

        cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        self.assertGreater(cond.last_latent_heat_energy, 0.0)

        mm_water = 18.015e-3
        gas_sub = (
            par.gas.GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass(np.array([mm_water]), "kg/mol")
            .set_concentration(np.array([1e-30]), "kg/m^3")
            .set_partitioning([True])
            .build()
        )
        particle_evap = copy.deepcopy(particle)
        cond.step(
            particle=particle_evap,
            gas_species=gas_sub,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        self.assertLess(cond.last_latent_heat_energy, 0.0)

    def test_step_isothermal_parity(self):
        """Latent heat fallback matches isothermal step outputs."""
        particle_latent = copy.deepcopy(self.particle)
        gas_latent = copy.deepcopy(self.gas_species)
        particle_iso = copy.deepcopy(self.particle)
        gas_iso = copy.deepcopy(self.gas_species)
        latent_cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )
        iso_cond = CondensationIsothermal(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        particle_out, gas_out = latent_cond.step(
            particle=particle_latent,
            gas_species=gas_latent,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        particle_iso_out, gas_iso_out = iso_cond.step(
            particle=particle_iso,
            gas_species=gas_iso,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        np.testing.assert_allclose(
            particle_out.total_mass[0],
            particle_iso_out.total_mass[0],
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            gas_out.concentration[0],
            gas_iso_out.concentration[0],
            rtol=1e-15,
        )
        self.assertEqual(latent_cond.last_latent_heat_energy, 0.0)

    def test_step_retains_native_objects(self):
        """Native stepping returns the original held containers."""
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
        )

        particle, gas = self._make_data_inputs()
        particle_new, gas_new = cond.step(
            particle=particle,
            gas_species=gas,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        self.assertIsInstance(particle_new, ParticleData)
        self.assertIsInstance(gas_new, GasData)
        self.assertIs(particle_new, particle)
        self.assertIs(gas_new, gas)

    def test_step_returns_correct_types_data(self):
        """Data-only inputs return data types."""
        particle_data, gas_data = self._make_data_inputs()
        cond = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
        )

        particle_new, gas_new = cond.step(
            particle=particle_data,
            gas_species=gas_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        self.assertIsInstance(particle_new, ParticleData)
        self.assertIsInstance(gas_new, GasData)

    def test_step_updates_gas_when_enabled(self):
        """Gas concentrations decrease by mass transfer when enabled."""
        particle = copy.deepcopy(self.particle)
        gas_species = copy.deepcopy(self.gas_species)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
            update_gases=True,
        )

        particle_data = particle
        gas_data = gas_species
        mass_rate = cond.mass_transfer_rate(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        mass_rate_array = (
            np.array([mass_rate])
            if isinstance(mass_rate, (int, float))
            else mass_rate
        )
        # Native mass rates already exclude skipped full-gas map lanes.
        norm_conc = particle_data.concentration[0] / particle_data.volume[0]
        expected_mass_transfer = get_mass_transfer(
            mass_rate=mass_rate_array,
            time_step=1.0,
            gas_mass=np.asarray(gas_data.concentration[0], dtype=np.float64),
            particle_mass=particle_data.masses[0],
            particle_concentration=norm_conc,
        )
        # Exchange only representable increments, including tiny core changes.
        expected_masses = (
            particle_data.masses[0]
            + expected_mass_transfer / norm_conc[:, None]
        )
        applied_density = (
            expected_masses - particle_data.masses[0]
        ) * norm_conc[:, None]
        expected_gas = gas_species.concentration[0] - applied_density.sum(
            axis=0
        )

        _, gas_new = cond.step(
            particle=particle,
            gas_species=gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )

        np.testing.assert_allclose(
            gas_new.concentration[0], expected_gas, rtol=1e-15
        )

    def test_step_clamps_gas_concentration_nonnegative(self):
        """Gas concentrations are clamped to nonnegative values."""
        particle_data, gas_data = self._make_data_inputs()
        gas_data.concentration[0] = np.array([1e-12, 2e-12])
        cond = CondensationLatentHeat(
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            latent_heat_strategy=ConstantLatentHeat(latent_heat_ref=2.26e6),
        )
        mass_transfer = np.full_like(particle_data.masses[0], 1.0)

        with patch.object(
            cond, "mass_transfer_rate", return_value=mass_transfer
        ) as proposal:
            _, gas_new = cond.step(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=1.0,
            )

        proposal.assert_called_once()
        np.testing.assert_allclose(gas_new.concentration[0], 0.0, rtol=1e-15)

    def test_small_negative_concentration_rejects_without_mutation(self):
        """Native execution rejects even tiny negative population counts."""
        particle, gas = self._make_data_inputs()
        particle.concentration[0, 0] = -1.0e-13
        masses_before = particle.masses.copy()
        gas_before = gas.concentration.copy()
        with self.assertRaisesRegex(ValueError, "finite physical values"):
            self.data_strategy.step(
                particle, gas, self.temperature, self.pressure, 1.0
            )
        np.testing.assert_array_equal(particle.masses, masses_before)
        np.testing.assert_array_equal(gas.concentration, gas_before)
        self.assertEqual(particle.concentration[0, 0], -1.0e-13)

    def test_step_rejects_invalid_time_step(self):
        """step() rejects non-finite or negative timesteps."""
        particle_data, gas_data = self._make_data_inputs()
        with self.assertRaisesRegex(
            ValueError, "time_step must be a finite nonnegative scalar"
        ):
            self.data_strategy.step(
                particle=particle_data,
                gas_species=gas_data,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=-0.5,
            )

    def test_get_vapor_pressure_surface_rejects_nonfinite_kelvin(self):
        """Non-finite Kelvin terms raise a clear error."""
        particle = copy.deepcopy(self.particle)
        gas_species = copy.deepcopy(self.gas_species)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        with patch.object(
            self.surface_strategy,
            "kelvin_term",
            return_value=np.array([[np.nan, np.nan]]),
        ):
            with self.assertRaisesRegex(
                ValueError, "kelvin_term must be finite"
            ):
                cond._native_pressure_terms(
                    particle=particle,
                    gas=gas_species,
                    temperature=self.temperature,
                    radius=particle.radii[0],
                    sources=np.arange(gas_species.n_species, dtype=np.intp),
                    targets=np.arange(particle.n_species, dtype=np.intp),
                )[1]

    def test_step_dynamic_viscosity_pass_through(self):
        """Dynamic viscosity override is forwarded to mass_transfer_rate."""
        particle = copy.deepcopy(self.particle)
        gas_species = copy.deepcopy(self.gas_species)
        cond = CondensationLatentHeat(
            activity_strategy=self.activity_strategy,
            surface_strategy=self.surface_strategy,
            vapor_pressure_strategy=self.vapor_pressure_strategy,
            species_map=self.species_map,
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )
        override = 1.23e-5

        with patch.object(
            cond, "mass_transfer_rate", wraps=cond.mass_transfer_rate
        ) as spy:
            cond.step(
                particle=particle,
                gas_species=gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=1.0,
                dynamic_viscosity=override,
            )

        self.assertEqual(spy.call_args.args[4], override)

    def test_step_mass_condensation_runnable_compatibility(self):
        """MassCondensation retains native containers with/without latent heat."""
        from particula.dynamics.condensation.species_mapping import (
            CondensationSpeciesMap,
        )

        core_mass = 4 / 3 * np.pi * (100e-9) ** 3 * 1770.0
        for latent_heat in (ConstantLatentHeat(2.26e6), None):
            particle = par.particles.create_resolved_particle_data(
                masses=[[0.0, core_mass]], density=[1000.0, 1770.0], volume=1e-6
            )
            saturation = self.vapor_pressure_strategy[
                0
            ].saturation_concentration(18.015e-3, self.temperature)
            gas = (
                par.gas.GasDataBuilder()
                .set_names(["H2O", "NH4HSO4"])
                .set_molar_mass([18.015e-3, 132.14e-3])
                .set_concentration([float(saturation) * 1.02, 1e-30])
                .set_partitioning([True, True])
                .build()
            )
            aerosol = par.Aerosol(
                particles=particle,
                gas=gas,
                environment=par.gas.EnvironmentData(
                    [self.temperature], [self.pressure], [[1.02, 0.0]]
                ),
            )
            strategy = CondensationLatentHeat(
                molar_mass=gas.molar_mass.copy(),
                diffusion_coefficient=self.diffusion_coefficient,
                accommodation_coefficient=self.accommodation_coefficient,
                latent_heat_strategy=latent_heat,
                activity_strategy=self.activity_strategy,
                surface_strategy=self.surface_strategy,
                vapor_pressure_strategy=self.vapor_pressure_strategy,
                species_map=CondensationSpeciesMap(
                    ("H2O", "NH4HSO4"), ((0, 0), (1, 1))
                ),
            )
            result = par.dynamics.MassCondensation(strategy).execute(
                aerosol, 1.0
            )
            self.assertIsInstance(result.particles, ParticleData)
            self.assertIsInstance(result.gas, GasData)
            self.assertIs(result, aerosol)
            self.assertIs(result.particles, particle)
            self.assertIs(result.gas, gas)
