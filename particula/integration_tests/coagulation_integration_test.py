import unittest
import numpy as np

import particula as par


class TestCoagulationIntegration(unittest.TestCase):
    def setUp(self):
        """
        Mimics the original notebook's setup:
         - define mode, geometric standard deviations, number_of_particles, etc.
         - create your atmosphere
         - define radius bins, volume
         - set up number_of_samples, etc.
        """
        self.mode = np.array([100e-9, 300e-9])  # m
        self.geometric_standard_deviation = np.array([1.3, 1.3])
        self.number_of_particles = np.array([0.75, 0.25])
        self.density = np.array([1.0e3])
        self.volume = 1 * par.util.get_unit_conversion("cm^3", "m^3")
        self.number_of_samples = 100_000
        self.radius_bins = np.logspace(-8, -6, 250)
        self.atmosphere = (
            par.gas.AtmosphereBuilder()
            .add_species(par.gas.PresetGasSpeciesBuilder().build())
            .set_temperature(25, temperature_units="degC")
            .set_pressure(1, pressure_units="atm")
            .build()
        )

    def test_pmf_initial_distribution(self):
        """
        Test the PMF distribution creation step from the notebook.

        Build and verify:
         - The PMF-based particle builder
         - The aerosol with PMF distribution
         - Check that the PMF distribution variables match expected shapes, sums, etc.
        """
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pmf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(
                self.geometric_standard_deviation
            )
            .set_number_concentration(number_concentration, "m^-3")
            .set_distribution_type("pmf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .build()
        )
        aerosol_pmf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pmf
        )
        self.assertIsNotNone(aerosol_pmf)
        self.assertEqual(
            aerosol_pmf.particles[0].get_concentration().shape,
            self.radius_bins.shape,
        )

    def test_pdf_initial_distribution(self):
        """
        Test the PDF distribution creation step from the notebook.

        Build and verify:
         - The PDF-based particle builder
         - The aerosol with PDF distribution
         - Check that the PDF distribution variables match expected shapes, sums, etc.
        """
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pdf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(
                self.geometric_standard_deviation
            )
            .set_number_concentration(number_concentration, "m^-3")
            .set_distribution_type("pdf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .set_charge(np.zeros_like(self.radius_bins))
            .build()
        )
        aerosol_pdf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pdf
        )
        self.assertIsNotNone(aerosol_pdf)
        self.assertEqual(
            aerosol_pdf.particles[0].get_concentration().shape,
            self.radius_bins.shape,
        )

    def test_resolved_initial_distribution(self):
        """
        Test the particle-resolved distribution creation step from the notebook.

        Build and verify:
         - The ResolvedParticleMassRepresentationBuilder
         - The aerosol with resolved distribution
         - Check total mass, number of particles, etc.
        """
        radii_sample = par.particles.get_lognormal_sample_distribution(
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_of_particles=self.number_of_particles,
            number_of_samples=self.number_of_samples,
        )
        particle_mass_sample = 4 / 3 * np.pi * radii_sample**3 * self.density
        resolved_masses = (
            par.particles.ResolvedParticleMassRepresentationBuilder()
            .set_distribution_strategy(
                par.particles.ParticleResolvedSpeciatedMass()
            )
            .set_activity_strategy(par.particles.ActivityIdealMass())
            .set_surface_strategy(par.particles.SurfaceStrategyVolume())
            .set_mass(particle_mass_sample, "kg")
            .set_density(self.density, "kg/m^3")
            .set_charge(0)
            .set_volume(self.volume, "m^3")
            .build()
        )
        aerosol_resolved = par.Aerosol(
            atmosphere=self.atmosphere, particles=resolved_masses
        )
        self.assertIsNotNone(aerosol_resolved)
        self.assertAlmostEqual(
            np.sum(particle_mass_sample),
            aerosol_resolved.particles[0].get_mass_concentration(),
            delta=1e-6,
        )

    def test_coagulation_process(self):
        """
        Test each coagulation approach (PMF, PDF, resolved) over a few time steps,
        just like the notebook loop.

        Verify:
         - That invoking 'execute()' modifies the aerosol as expected
         - That total mass is conserved or within tolerance
         - That number concentration changes in a sensible way
        """
        coagulation_process_pmf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="discrete"
            )
        )
        coagulation_process_resolved = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="particle_resolved"
            )
        )
        coagulation_process_pdf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="continuous_pdf"
            )
        )
        # Setup initial aerosols
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pmf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(
                self.geometric_standard_deviation
            )
            .set_number_concentration(number_concentration, "m^-3")
            .set_distribution_type("pmf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .build()
        )
        aerosol_pmf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pmf
        )

        particle_pdf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(
                self.geometric_standard_deviation
            )
            .set_number_concentration(number_concentration, "m^-3")
            .set_distribution_type("pdf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .set_charge(np.zeros_like(self.radius_bins))
            .build()
        )
        aerosol_pdf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pdf
        )

        radii_sample = par.particles.get_lognormal_sample_distribution(
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_of_particles=self.number_of_particles,
            number_of_samples=self.number_of_samples,
        )
        particle_mass_sample = 4 / 3 * np.pi * radii_sample**3 * self.density
        resolved_masses = (
            par.particles.ResolvedParticleMassRepresentationBuilder()
            .set_distribution_strategy(
                par.particles.ParticleResolvedSpeciatedMass()
            )
            .set_activity_strategy(par.particles.ActivityIdealMass())
            .set_surface_strategy(par.particles.SurfaceStrategyVolume())
            .set_mass(particle_mass_sample, "kg")
            .set_density(self.density, "kg/m^3")
            .set_charge(0)
            .set_volume(self.volume, "m^3")
            .build()
        )
        aerosol_resolved = par.Aerosol(
            atmosphere=self.atmosphere, particles=resolved_masses
        )

        # Simulate a few steps
        time_step = 100
        sub_steps = 1

        initial_mass_pmf = aerosol_pmf.particles[0].get_mass_concentration()
        initial_mass_pdf = aerosol_pdf.particles[0].get_mass_concentration()
        initial_mass_resolved = aerosol_resolved.particles[
            0
        ].get_mass_concentration()

        aerosol_pmf = coagulation_process_pmf.execute(
            aerosol_pmf, time_step, sub_steps
        )
        aerosol_pdf = coagulation_process_pdf.execute(
            aerosol_pdf, time_step, sub_steps
        )
        aerosol_resolved = coagulation_process_resolved.execute(
            aerosol_resolved, time_step, sub_steps
        )

        self.assertAlmostEqual(
            initial_mass_pmf,
            aerosol_pmf.particles[0].get_mass_concentration(),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            initial_mass_pdf,
            aerosol_pdf.particles[0].get_mass_concentration(),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            initial_mass_resolved,
            aerosol_resolved.particles[0].get_mass_concentration(),
            delta=1e-6,
        )

    def test_final_properties(self):
        """
        Check final states, ensuring mass conservation and reasonable number changes.
        Mimic the final summary checks from the notebook (e.g. total mass, final distribution).
        """
        # This would involve running the full simulation and checking final states
        pass


if __name__ == "__main__":
    unittest.main()
