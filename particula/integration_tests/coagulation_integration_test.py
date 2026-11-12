"""Integration tests for the coagulation processes in Particula.
This module uses unittest to verify correct creation of particle
distributions (PMF, PDF, resolved) and their evolution with a
Brownian coagulation strategy.
"""

# pylint: disable=too-many-instance-attributes, too-many-locals

import unittest

import numpy as np
import particula as par
import pytest
from particula.gas import EnvironmentData, GasDataBuilder


@pytest.fixture(autouse=True)
def forbid_facades(forbid_legacy_facades):
    """Guard actual native integration against obsolete construction."""


def _mass_density(particle):
    """Independent population-aware mass concentration in kg/m^3."""
    product = particle.masses[0].sum(axis=1) * particle.concentration[0]
    if particle.distribution_type == "continuous_pdf":
        return np.sum(
            (product[1:] + product[:-1]) * np.diff(particle.radius_grid) / 2
        )
    if particle.distribution_type == "particle_resolved":
        return product.sum() / particle.volume[0]
    return product.sum()


class TestCoagulationIntegration(unittest.TestCase):
    """Integration tests for the coagulation processes in Particula."""

    def setUp(self):
        """Set up common test parameters for various coagulation tests.

        This includes defining:
          - mode and geometric_standard_deviation for lognormal sampling
          - number_of_particles and total number_of_samples
          - density, volume, and radius_bins
          - explicit environment and independent inert gas data
        """
        self.mode = np.array([100e-9, 300e-9])  # m
        self.geometric_standard_deviation = np.array([1.3, 1.3])
        self.number_of_particles = np.array([0.75, 0.25])
        self.density = np.array([1.0e3])
        self.volume = 1 * 1e-6  # m^3
        self.number_of_samples = 100_000
        self.radius_bins = np.logspace(-8, -6, 250)
        self.environment = EnvironmentData([273.0], [101325.0], [[0.0]])
        self.gas = (
            GasDataBuilder()
            .set_names(["inert"])
            .set_molar_mass([0.028])
            .set_concentration([0.0])
            .set_partitioning([False])
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
        particle_pmf = par.particles.create_lognormal_binned_particle_data(
            distribution_type="discrete",
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_concentration=number_concentration,
            radius_bins=self.radius_bins,
            density=self.density,
        )
        aerosol_pmf = par.Aerosol(
            gas=self.gas, environment=self.environment, particles=particle_pmf
        )
        self.assertIsNotNone(aerosol_pmf)
        self.assertEqual(
            aerosol_pmf.particles.concentration[0].shape,
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
        particle_pdf = par.particles.create_lognormal_binned_particle_data(
            distribution_type="continuous_pdf",
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_concentration=number_concentration,
            radius_bins=self.radius_bins,
            density=self.density,
        )
        aerosol_pdf = par.Aerosol(
            gas=self.gas, environment=self.environment, particles=particle_pdf
        )
        self.assertIsNotNone(aerosol_pdf)
        self.assertEqual(
            aerosol_pdf.particles.concentration[0].shape,
            self.radius_bins.shape,
        )

    def test_resolved_initial_distribution(self):
        """Test creation of a particle-resolved distribution (no binning).

        Uses lognormal-sampled radii to create resolved particle masses:
          - verifies total mass matches expected sum
          - retains resolved count and volume semantics
        """
        radii_sample = par.particles.get_lognormal_sample_distribution(
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_of_particles=self.number_of_particles,
            number_of_samples=self.number_of_samples,
        )
        particle_mass_sample = 4 / 3 * np.pi * radii_sample**3 * self.density
        resolved_masses = par.particles.create_resolved_particle_data(
            masses=particle_mass_sample,
            density=self.density,
            volume=self.volume,
        )
        aerosol_resolved = par.Aerosol(
            gas=self.gas,
            environment=self.environment,
            particles=resolved_masses,
        )
        self.assertIsNotNone(aerosol_resolved)
        self.assertAlmostEqual(
            np.sum(particle_mass_sample) / self.volume,
            _mass_density(aerosol_resolved.particles),
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
        particle_pmf = par.particles.create_lognormal_binned_particle_data(
            distribution_type="discrete",
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_concentration=number_concentration,
            radius_bins=self.radius_bins,
            density=self.density,
        )
        aerosol_pmf = par.Aerosol(
            gas=self.gas, environment=self.environment, particles=particle_pmf
        )
        initial_mass_pmf = _mass_density(aerosol_pmf.particles)
        for _ in range(3):
            aerosol_pmf = coagulation_process_pmf.execute(aerosol_pmf, 100, 1)
        self.assertAlmostEqual(
            initial_mass_pmf,
            _mass_density(aerosol_pmf.particles),
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
        particle_pdf = par.particles.create_lognormal_binned_particle_data(
            distribution_type="continuous_pdf",
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_concentration=number_concentration,
            radius_bins=self.radius_bins,
            density=self.density,
        )
        aerosol_pdf = par.Aerosol(
            gas=self.gas, environment=self.environment, particles=particle_pdf
        )
        initial_mass_pdf = _mass_density(aerosol_pdf.particles)
        for _ in range(3):
            aerosol_pdf = coagulation_process_pdf.execute(aerosol_pdf, 100, 2)
        self.assertAlmostEqual(
            initial_mass_pdf,
            _mass_density(aerosol_pdf.particles),
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
        resolved_masses = par.particles.create_resolved_particle_data(
            masses=particle_mass_sample,
            density=self.density,
            volume=self.volume,
        )
        aerosol_resolved = par.Aerosol(
            gas=self.gas,
            environment=self.environment,
            particles=resolved_masses,
        )
        initial_mass_resolved = _mass_density(aerosol_resolved.particles)
        for _ in range(3):
            aerosol_resolved = coagulation_process_resolved.execute(
                aerosol_resolved, 100, 1
            )
        aerosol_resolved = coagulation_process_resolved.execute(
            aerosol_resolved, 100, 1
        )
        self.assertAlmostEqual(
            initial_mass_resolved,
            _mass_density(aerosol_resolved.particles),
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
        particle_pmf = par.particles.create_lognormal_binned_particle_data(
            distribution_type="discrete",
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_concentration=number_concentration,
            radius_bins=self.radius_bins,
            density=self.density,
        )
        aerosol_pmf = par.Aerosol(
            gas=self.gas, environment=self.environment, particles=particle_pmf
        )
        initial_mass_pmf = _mass_density(aerosol_pmf.particles)
        for _ in range(3):
            aerosol_pmf = coagulation_process_pmf.execute(aerosol_pmf, 100, 1)
        final_mass_pmf = _mass_density(aerosol_pmf.particles)

        # -------------------------
        # PDF-based final mass check
        coagulation_process_pdf = par.dynamics.Coagulation(
            coagulation_strategy=par.dynamics.BrownianCoagulationStrategy(
                distribution_type="continuous_pdf"
            )
        )
        particle_pdf = par.particles.create_lognormal_binned_particle_data(
            distribution_type="continuous_pdf",
            mode=self.mode,
            geometric_standard_deviation=self.geometric_standard_deviation,
            number_concentration=number_concentration,
            radius_bins=self.radius_bins,
            density=self.density,
        )
        aerosol_pdf = par.Aerosol(
            gas=self.gas, environment=self.environment, particles=particle_pdf
        )
        initial_mass_pdf = _mass_density(aerosol_pdf.particles)
        for _ in range(3):
            aerosol_pdf = coagulation_process_pdf.execute(aerosol_pdf, 100, 1)
        final_mass_pdf = _mass_density(aerosol_pdf.particles)

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
        resolved_masses = par.particles.create_resolved_particle_data(
            masses=particle_mass_sample,
            density=self.density,
            volume=self.volume,
        )
        aerosol_resolved = par.Aerosol(
            gas=self.gas,
            environment=self.environment,
            particles=resolved_masses,
        )
        initial_mass_resolved = _mass_density(aerosol_resolved.particles)
        for _ in range(3):
            aerosol_resolved = coagulation_process_resolved.execute(
                aerosol_resolved, 100, 1
            )
        final_mass_resolved = _mass_density(aerosol_resolved.particles)

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
