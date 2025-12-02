"""Tests for BrownianCoagulationStrategy."""

# pylint: disable=duplicate-code, too-many-instance-attributes

import unittest

import numpy as np

from particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy import (  # noqa: E501
    BrownianCoagulationStrategy,
)
from particula.particles import (
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
)


class TestBrownianCoagulationStrategy(unittest.TestCase):
    """Test suite for the BrownianCoagulationStrategy class."""

    def setUp(self):
        """Set up particle representations and strategies for testing."""
        # Setup a particle representation for testing
        self.particle = PresetParticleRadiusBuilder().build()
        # Create strategies for both distribution types
        self.strategy_discrete = BrownianCoagulationStrategy(
            distribution_type="discrete"
        )
        self.strategy_continuous_pdf = BrownianCoagulationStrategy(
            distribution_type="continuous_pdf"
        )
        self.particle_resolved = (
            PresetResolvedParticleMassBuilder().set_volume(1e-6, "m^3").build()
        )
        self.strategy_particle_resolved = BrownianCoagulationStrategy(
            distribution_type="particle_resolved"
        )
        self.temperature = 298.15  # Kelvin
        self.pressure = 101325  # Pascal

    def test_kernel_discrete(self):
        """Test the kernel calculation for discrete distribution."""
        # Test the kernel calculation for discrete distribution
        kernel = self.strategy_discrete.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(kernel, np.ndarray)

    def test_step_discrete(self):
        """Test the step method for discrete distribution."""
        # Test the step method for discrete distribution
        initial_concentration = self.particle.get_concentration().copy()
        self.strategy_discrete.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        updated_concentration = self.particle.get_concentration()
        self.assertFalse(
            np.array_equal(initial_concentration, updated_concentration)
        )

    def test_step_particle_resolved(self):
        """Test the kernel calculation for particle_resolved distribution."""
        # Test the kernel calculation for particle_resolved distribution
        old_concentration = self.particle_resolved.get_total_concentration()
        self.strategy_particle_resolved.step(
            particle=self.particle_resolved,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1000,
        )
        new_concentration = self.particle_resolved.get_total_concentration()
        self.assertNotEqual(old_concentration, new_concentration)

    def test_kernel_continuous_pdf(self):
        """Test the kernel calculation for continuous_pdf distribution."""
        # Test the kernel calculation for continuous_pdf distribution
        kernel = self.strategy_continuous_pdf.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(kernel, np.ndarray)

    def test_step_continuous_pdf(self):
        """Test the step method for continuous_pdf distribution."""
        # Test the step method for continuous_pdf distribution
        initial_concentration = self.particle.get_concentration().copy()
        self.strategy_continuous_pdf.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        updated_concentration = self.particle.get_concentration()
        self.assertFalse(
            np.array_equal(initial_concentration, updated_concentration)
        )
