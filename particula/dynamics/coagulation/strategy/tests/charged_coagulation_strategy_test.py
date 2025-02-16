"""
Unit tests for the ChargedCoagulationStrategy class.

This module contains tests for the ChargedCoagulationStrategy class, which
implements the charged particle coagulation strategy. The tests cover both
discrete and continuous_pdf distribution types.
"""

import unittest
import numpy as np
from particula.dynamics.coagulation.strategy.charged_coagulation_strategy import ChargedCoagulationStrategy
from particula.dynamics.coagulation.charged_kernel_strategy import HardSphereKernelStrategy
from particula.particles import PresetParticleRadiusBuilder

class TestChargedCoagulationStrategy(unittest.TestCase):
    """
    Test suite for the ChargedCoagulationStrategy class.
    """
    def setUp(self):
        """
        Set up the test environment.

        Initializes a particle representation and creates instances of
        ChargedCoagulationStrategy for both discrete and continuous_pdf
        distribution types.
        """
        # Setup a particle representation for testing
        self.particle = PresetParticleRadiusBuilder().build()
        self.temperature = 298.15  # Kelvin
        self.pressure = 101325  # Pascal

        # Create a kernel strategy instance
        self.kernel_strategy = HardSphereKernelStrategy()

        # Create strategies for both distribution types
        self.strategy_discrete = ChargedCoagulationStrategy(
            distribution_type="discrete",
            kernel_strategy=self.kernel_strategy
        )
        self.strategy_continuous_pdf = ChargedCoagulationStrategy(
            distribution_type="continuous_pdf",
            kernel_strategy=self.kernel_strategy
        )

    def test_kernel_discrete(self):
        """
        Test the kernel calculation for discrete distribution.

        Verifies that the kernel method returns an ndarray for the discrete
        distribution type.
        """
        # Test the kernel calculation for discrete distribution
        kernel = self.strategy_discrete.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(kernel, np.ndarray)

    def test_step_discrete(self):
        """
        Test the step method for discrete distribution.

        Ensures that the step method updates the particle concentration for
        the discrete distribution type.
        """
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

    def test_kernel_continuous_pdf(self):
        """
        Test the kernel calculation for continuous_pdf distribution.

        Verifies that the kernel method returns an ndarray for the
        continuous_pdf distribution type.
        """
        # Test the kernel calculation for continuous_pdf distribution
        kernel = self.strategy_continuous_pdf.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsInstance(kernel, np.ndarray)

    def test_step_continuous_pdf(self):
        """
        Test the step method for continuous_pdf distribution.

        Ensures that the step method updates the particle concentration for
        the continuous_pdf distribution type.
        """
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
