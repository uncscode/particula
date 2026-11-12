"""Integration test for isothermal condensation on particle-resolved aerosol."""

import unittest

import numpy as np
import numpy.testing as npt
import particula as par
import pytest
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import EnvironmentData, GasDataBuilder


@pytest.fixture(autouse=True)
def forbid_facades(forbid_legacy_facades):
    """Require native execution without temporary facade construction."""


class TestCondensationParticleResolved(unittest.TestCase):
    """Verify that mass is transferred from gas to particles."""

    def setUp(self):
        """Set up aerosol system with gas and particles for testing."""
        # ---------- vapor-pressure strategies ----------
        self.vp_water = par.gas.VaporPressureFactory().get_strategy(
            "water_buck"
        )
        self.vp_constant = par.gas.VaporPressureFactory().get_strategy(
            "constant", {"vapor_pressure": 1e-24, "vapor_pressure_units": "Pa"}
        )

        # ---------- gas species (partitioning + gas-only) ----------
        molar_mass_water = 18.015e-3  # kg/mol
        molar_mass_am_sulf = 132.14e-3  # kg/mol
        water_sat = self.vp_water.saturation_concentration(
            molar_mass=molar_mass_water, temperature=298.15
        )
        # 2 % supersaturation
        water_conc = water_sat * 1.02

        self.gas = (
            GasDataBuilder()
            .set_names(["H2O", "NH4HSO4", "N2"])
            .set_molar_mass([molar_mass_water, molar_mass_am_sulf, 0.028])
            .set_concentration([water_conc, 1e-30, 0.79])
            .set_partitioning([True, True, False])
            .build()
        )
        self.environment = EnvironmentData(
            [298.15], [101325.0], [[1.02, 0.0, 0.0]]
        )

        # ---------- particle distribution ----------
        density_core = 1.77e3  # kg/m^3
        radii = par.particles.get_lognormal_sample_distribution(
            mode=np.array([100e-9, 400e-9]),
            geometric_standard_deviation=np.array([1.3, 1.4]),
            number_of_particles=np.array([1, 0.5]),
            number_of_samples=2_000,  # keep integration test fast
        )
        mass_core = 4 / 3 * np.pi * radii**3 * density_core
        mass_speciation = np.column_stack([mass_core * 0, mass_core])
        densities = np.array([1.0e3, density_core])

        activity = (
            par.particles.ActivityKappaParameterBuilder()
            .set_density(densities, "kg/m^3")
            .set_kappa(np.array([0.0, 0.61]))
            .set_molar_mass(
                np.array([molar_mass_water, molar_mass_am_sulf]), "kg/mol"
            )
            .set_water_index(0)
            .build()
        )
        surface = (
            par.particles.SurfaceStrategyVolumeBuilder()
            .set_density(densities, "kg/m^3")
            .set_surface_tension(np.array([0.072, 0.092]), "N/m")
            .build()
        )
        self.aerosol = par.Aerosol(
            gas=self.gas,
            environment=self.environment,
            particles=par.particles.create_resolved_particle_data(
                masses=mass_speciation,
                density=densities,
                volume=1e-6,
            ),
        )

        # ---------- condensation process ----------
        self.condensation = par.dynamics.MassCondensation(
            condensation_strategy=par.dynamics.CondensationIsothermal(
                molar_mass=self.gas.molar_mass.copy(),
                activity_strategy=activity,
                surface_strategy=surface,
                vapor_pressure_strategy=[
                    self.vp_water,
                    self.vp_constant,
                    self.vp_constant,
                ],
                species_map=CondensationSpeciesMap(
                    ("H2O", "NH4HSO4", "N2"), ((0, 0), (1, 1))
                ),
                diffusion_coefficient=2e-5,
                accommodation_coefficient=1,
                update_gases=True,
            )
        )

    # ------------------------------------------------------------------
    # actual tests
    # ------------------------------------------------------------------

    def test_condensation_transfers_mass(self):
        """Particle mass should increase while gas-phase water decreases."""
        initial_species_mass = self.aerosol.particles.masses[0].copy()
        particle_concentration = (
            self.aerosol.particles.concentration[0]
            / self.aerosol.particles.volume[0]
        ).copy()
        initial_particle_inventory = np.sum(
            initial_species_mass * particle_concentration[:, None], axis=0
        )
        initial_partitioning_gas = self.aerosol.gas.concentration[0, :2].copy()
        initial_nitrogen = self.aerosol.gas.concentration[0, 2]
        initial_gas_water = initial_partitioning_gas[0]

        aerosol = self.aerosol
        for _ in range(5):
            aerosol = self.condensation.execute(aerosol, 0.1, 1)

        final_species_mass = aerosol.particles.masses[0].copy()
        final_particle_inventory = np.sum(
            final_species_mass * particle_concentration[:, None], axis=0
        )
        final_partitioning_gas = aerosol.gas.concentration[0, :2]
        final_nitrogen = aerosol.gas.concentration[0, 2]
        final_gas_water = final_partitioning_gas[0]

        self.assertGreater(
            final_particle_inventory[0], initial_particle_inventory[0]
        )
        self.assertLess(final_gas_water, initial_gas_water)
        assert np.all(np.isfinite(initial_particle_inventory))
        assert np.all(np.isfinite(final_particle_inventory))
        assert np.all(np.isfinite(initial_partitioning_gas))
        assert np.all(np.isfinite(final_partitioning_gas))
        assert np.all(initial_particle_inventory >= 0.0)
        assert np.all(final_particle_inventory >= 0.0)
        assert np.all(initial_partitioning_gas >= 0.0)
        assert np.all(final_partitioning_gas >= 0.0)
        npt.assert_allclose(
            final_particle_inventory + final_partitioning_gas,
            initial_particle_inventory + initial_partitioning_gas,
            rtol=1e-12,
            atol=1e-30,
        )
        assert np.isfinite(final_nitrogen)
        assert final_nitrogen >= 0.0
        npt.assert_allclose(
            final_nitrogen, initial_nitrogen, rtol=0.0, atol=0.0
        )
