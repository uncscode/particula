"""Integration test for isothermal condensation on particle-resolved aerosol."""

import unittest

import numpy as np

import particula as par


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

        self.gas_partitioning = (
            par.gas.GasSpeciesBuilder()
            .set_name(["H2O", "NH4HSO4"])
            .set_molar_mass(
                np.array([molar_mass_water, molar_mass_am_sulf]), "kg/mol"
            )
            .set_vapor_pressure_strategy([self.vp_water, self.vp_constant])
            .set_concentration(np.array([water_conc, 1e-30]), "kg/m^3")
            .set_partitioning(True)
            .build()
        )

        self.gas_inert = (
            par.gas.GasSpeciesBuilder()
            .set_name("N2")
            .set_molar_mass(0.028, "kg/mol")
            .set_vapor_pressure_strategy(self.vp_constant)
            .set_concentration(0.79, "kg/m^3")
            .set_partitioning(False)
            .build()
        )
        # ---------- atmosphere ----------
        self.atmosphere = (
            par.gas.AtmosphereBuilder()
            .set_temperature(298.15, "K")
            .set_pressure(101325, "Pa")
            .set_more_partitioning_species(self.gas_partitioning)
            .set_more_gas_only_species(self.gas_inert)
            .build()
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
            atmosphere=self.atmosphere,
            particles=(
                par.particles.ResolvedParticleMassRepresentationBuilder()
                .set_distribution_strategy(
                    par.particles.ParticleResolvedSpeciatedMass()
                )
                .set_activity_strategy(activity)
                .set_surface_strategy(surface)
                .set_mass(mass_speciation, "kg")
                .set_density(densities, "kg/m^3")
                .set_charge(0)
                .set_volume(1e-6, "m^3")  # 1 cm³ air parcel
                .build()
            ),
        )

        # ---------- condensation process ----------
        self.condensation = par.dynamics.MassCondensation(
            condensation_strategy=par.dynamics.CondensationIsothermal(
                molar_mass=np.array([molar_mass_water, molar_mass_am_sulf]),
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
        initial_particle_mass = self.aerosol.particles.get_mass_concentration()
        initial_gas_water = (
            self.aerosol.atmosphere.partitioning_species.get_concentration()[
                0
            ].item()
        )

        aerosol = self.aerosol
        for _ in range(5):
            aerosol = self.condensation.execute(aerosol, 0.1, 1)

        final_particle_mass = aerosol.particles.get_mass_concentration()
        final_gas_water = (
            aerosol.atmosphere.partitioning_species.get_concentration()[
                0
            ].item()
        )

        # assertions
        self.assertGreater(final_particle_mass, initial_particle_mass)
        self.assertLess(final_gas_water, initial_gas_water)
        # mass conservation (water + core species)
        total_initial = initial_particle_mass + initial_gas_water
        total_final = final_particle_mass + final_gas_water
        self.assertAlmostEqual(total_initial, total_final, delta=1e-9)
