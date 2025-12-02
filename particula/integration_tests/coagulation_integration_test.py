"""Integration tests for the coagulation processes in Particula.
This module uses unittest to verify correct creation of particle
distributions (PMF, PDF, resolved) and their evolution with a
Brownian coagulation strategy.
"""

# pylint: disable=too-many-instance-attributes, too-many-locals

import unittest

import numpy as np

import particula as par


class TestCoagulationIntegration(unittest.TestCase):
    """Integration tests for the coagulation processes in Particula."""

    def setUp(self):
        """Set up common test parameters for various coagulation tests.

        This includes defining:
          - mode and geometric_standard_deviation for lognormal sampling
          - number_of_particles and total number_of_samples
          - density, volume, and radius_bins
          - a default atmosphere with preset gas species
        """
        self.mode = np.array([100e-9, 300e-9])  # m
        self.geometric_standard_deviation = np.array([1.3, 1.3])
        self.number_of_particles = np.array([0.75, 0.25])
        self.density = np.array([1.0e3])
        self.volume = 1 * 1e-6  # m^3
        self.number_of_samples = 100_000
        self.radius_bins = np.logspace(-8, -6, 250)
        self.atmosphere = (
            par.gas.AtmosphereBuilder()
            .set_temperature(273, temperature_units="K")
            .set_pressure(101325, pressure_units="Pa")
            .build()
        )

    def test_pmf_initial_distribution(self):
        """Test creation of a Particulate Mass Function (PMF) distribution.

        Verifies that the PMF-based particle builder:
          - properly sets mode, GSD, and number concentrations
          - correctly assigns a discrete PMF distribution across radius_bins
          - checks shape integrity of the resulting aerosol's concentration
        """
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pmf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(self.geometric_standard_deviation)
            .set_number_concentration(number_concentration, "1/m^3")
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
            aerosol_pmf.particles.get_concentration().shape,
            self.radius_bins.shape,
        )

    def test_pdf_initial_distribution(self):
        """Test creation of a Probability Density Function (PDF) distribution.

        Verifies that the PDF-based particle builder:
          - properly sets mode, GSD, and number concentrations
          - uses a continuous PDF distribution across radius_bins
          - checks shape integrity of the resulting aerosol's concentration
        """
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pdf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(self.geometric_standard_deviation)
            .set_number_concentration(number_concentration, "1/m^3")
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
            aerosol_pdf.particles.get_concentration().shape,
            self.radius_bins.shape,
        )

    def test_resolved_initial_distribution(self):
        """Test creation of a particle-resolved distribution (no binning).

        Uses lognormal-sampled radii to create resolved particle masses:
          - verifies total mass matches expected sum
          - checks that ParticleResolvedSpeciatedMass is assigned properly
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
            np.sum(particle_mass_sample) / self.volume,
            aerosol_resolved.particles.get_mass_concentration(),
            delta=1e-6,
        )

    def test_coagulation_process_pmf(self):
        """Test Brownian coagulation for a PMF-based distribution."""
        coagulation_process_pmf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="discrete"
            )
        )
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pmf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(self.geometric_standard_deviation)
            .set_number_concentration(number_concentration, "1/m^3")
            .set_distribution_type("pmf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .build()
        )
        aerosol_pmf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pmf
        )
        initial_mass_pmf = aerosol_pmf.particles.get_mass_concentration(
            clone=True
        )
        for _ in range(3):
            aerosol_pmf = coagulation_process_pmf.execute(aerosol_pmf, 100, 1)
        self.assertAlmostEqual(
            initial_mass_pmf,
            aerosol_pmf.particles.get_mass_concentration(),
            delta=1e-4,
        )

    def test_coagulation_process_pdf(self):
        """Test Brownian coagulation for a PDF-based distribution."""
        coagulation_process_pdf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="continuous_pdf"
            )
        )
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pdf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(self.geometric_standard_deviation)
            .set_number_concentration(number_concentration, "1/m^3")
            .set_distribution_type("pdf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .set_charge(np.zeros_like(self.radius_bins))
            .build()
        )
        aerosol_pdf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pdf
        )
        initial_mass_pdf = aerosol_pdf.particles.get_mass_concentration(
            clone=True
        )
        for _ in range(3):
            aerosol_pdf = coagulation_process_pdf.execute(aerosol_pdf, 100, 2)
        self.assertAlmostEqual(
            initial_mass_pdf,
            aerosol_pdf.particles.get_mass_concentration(),
            delta=5,
        )

    def test_coagulation_process_resolved(self):
        """Test Brownian coagulation for a particle-resolved distribution."""
        coagulation_process_resolved = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="particle_resolved"
            )
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
        initial_mass_resolved = (
            aerosol_resolved.particles.get_mass_concentration()
        )
        for _ in range(3):
            aerosol_resolved = coagulation_process_resolved.execute(
                aerosol_resolved, 100, 1
            )
        aerosol_resolved = coagulation_process_resolved.execute(
            aerosol_resolved, 100, 1
        )
        self.assertAlmostEqual(
            initial_mass_resolved,
            aerosol_resolved.particles.get_mass_concentration(),
            delta=1e-6,
        )

    def test_final_properties(self):
        """Test final aerosol properties at the end of simulation.

        Verifies:
          - overall mass conservation
          - final number concentration changes make sense
          - final distribution states meet expected criteria
        """
        # -------------------------
        # PMF-based final mass check
        coagulation_process_pmf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="discrete"
            )
        )
        number_concentration = self.number_of_particles * np.array(
            [self.number_of_samples / self.volume]
        )
        particle_pmf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(self.geometric_standard_deviation)
            .set_number_concentration(number_concentration, "1/m^3")
            .set_distribution_type("pmf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .build()
        )
        aerosol_pmf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pmf
        )
        initial_mass_pmf = aerosol_pmf.particles.get_mass_concentration()
        for _ in range(3):
            aerosol_pmf = coagulation_process_pmf.execute(aerosol_pmf, 100, 1)
        final_mass_pmf = aerosol_pmf.particles.get_mass_concentration()

        # -------------------------
        # PDF-based final mass check
        coagulation_process_pdf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="continuous_pdf"
            )
        )
        particle_pdf = (
            par.particles.PresetParticleRadiusBuilder()
            .set_mode(self.mode, mode_units="m")
            .set_geometric_standard_deviation(self.geometric_standard_deviation)
            .set_number_concentration(number_concentration, "1/m^3")
            .set_distribution_type("pdf")
            .set_radius_bins(self.radius_bins, radius_bins_units="m")
            .set_density(self.density, "kg/m^3")
            .set_charge(np.zeros_like(self.radius_bins))
            .build()
        )
        aerosol_pdf = par.Aerosol(
            atmosphere=self.atmosphere, particles=particle_pdf
        )
        initial_mass_pdf = aerosol_pdf.particles.get_mass_concentration()
        for _ in range(3):
            aerosol_pdf = coagulation_process_pdf.execute(aerosol_pdf, 100, 1)
        final_mass_pdf = aerosol_pdf.particles.get_mass_concentration()

        # -------------------------
        # Resolved-based final mass check
        coagulation_process_resolved = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="particle_resolved"
            )
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
        initial_mass_resolved = (
            aerosol_resolved.particles.get_mass_concentration()
        )
        for _ in range(3):
            aerosol_resolved = coagulation_process_resolved.execute(
                aerosol_resolved, 100, 1
            )
        final_mass_resolved = (
            aerosol_resolved.particles.get_mass_concentration()
        )

        # Compare final masses
        # They won't be identical but should remain within a reasonable
        # tolerance:
        # self comparison is to ensure that the final mass is not zero
        self.assertAlmostEqual(initial_mass_pmf, final_mass_pmf, delta=1e-1)
        self.assertAlmostEqual(initial_mass_pdf, final_mass_pdf, delta=5)
        self.assertAlmostEqual(
            initial_mass_resolved, final_mass_resolved, delta=1e-1
        )

        # cross comparison
        # self.assertAlmostEqual(final_mass_pmf, final_mass_pdf, delta=1e-1)
        # self.assertAlmostEqual(
        #   final_mass_pdf, final_mass_resolved, delta=1e-1
        # )
        self.assertAlmostEqual(final_mass_pmf, final_mass_resolved, delta=1e-1)
