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
from particula.particles.particle_data import ParticleData


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
        self.particle_data = ParticleData(
            masses=np.array(
                [
                    [
                        [1e-18, 1e-18],
                        [2e-18, 2e-18],
                        [3e-18, 3e-18],
                        [4e-18, 4e-18],
                    ]
                ]
            ),
            concentration=np.array([[1e6, 2e6, 1.5e6, 0.8e6]]),
            charge=np.array([[0.0, 1.0, -1.0, 0.0]]),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([1.0]),
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

    def test_brownian_kernel_with_particle_data(self):
        """Test kernel calculation with ParticleData inputs."""
        kernel = self.strategy_discrete.kernel(
            particle=self.particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        kernel = np.asarray(kernel)
        self.assertIsInstance(kernel, np.ndarray)
        particle_count = self.particle_data.radii[0].size
        self.assertEqual(np.shape(kernel), (particle_count, particle_count))

    def test_brownian_step_with_particle_data(self):
        """Test coagulation step with ParticleData inputs."""
        particle_data = self.particle_data.copy()
        initial_concentration = particle_data.concentration.copy()
        self.strategy_discrete.step(
            particle=particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        self.assertFalse(
            np.array_equal(initial_concentration, particle_data.concentration)
        )

    def test_brownian_step_returns_matching_types(self):
        """Test step returns ParticleData when given ParticleData."""
        particle_data = self.particle_data.copy()
        result = self.strategy_discrete.step(
            particle=particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        self.assertIsInstance(result, ParticleData)

    def test_loss_rate_with_particle_data(self):
        """Test loss rate calculation with ParticleData inputs."""
        kernel = self.strategy_discrete.kernel(
            particle=self.particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        loss_rate = self.strategy_discrete.loss_rate(
            particle=self.particle_data,
            kernel=np.atleast_1d(kernel),
        )
        self.assertEqual(
            np.shape(loss_rate), (self.particle_data.radii[0].size,)
        )

    def test_gain_rate_with_particle_data(self):
        """Test gain rate calculation with ParticleData inputs."""
        kernel = self.strategy_discrete.kernel(
            particle=self.particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        gain_rate = self.strategy_discrete.gain_rate(
            particle=self.particle_data,
            kernel=np.atleast_1d(kernel),
        )
        self.assertEqual(
            np.shape(gain_rate), (self.particle_data.radii[0].size,)
        )

    def test_net_rate_with_particle_data(self):
        """Test net rate equals gain minus loss for ParticleData inputs."""
        kernel = self.strategy_discrete.kernel(
            particle=self.particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        kernel_arr = np.atleast_1d(kernel)
        loss_rate = self.strategy_discrete.loss_rate(
            particle=self.particle_data,
            kernel=kernel_arr,
        )
        gain_rate = self.strategy_discrete.gain_rate(
            particle=self.particle_data,
            kernel=kernel_arr,
        )
        net_rate = self.strategy_discrete.net_rate(
            particle=self.particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertTrue(np.allclose(net_rate, gain_rate - loss_rate))

    def test_diffusive_knudsen_with_particle_data(self):
        """Test diffusive Knudsen number calculation with ParticleData."""
        knudsen_number = self.strategy_discrete.diffusive_knudsen(
            particle=self.particle_data,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        particle_count = self.particle_data.radii[0].size
        knudsen_shape = np.shape(knudsen_number)
        if len(knudsen_shape) == 2:
            self.assertEqual(knudsen_shape, (particle_count, particle_count))
        else:
            self.assertEqual(knudsen_shape, (particle_count,))
