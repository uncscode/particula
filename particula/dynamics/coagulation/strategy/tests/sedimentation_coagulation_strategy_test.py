"""
Unit tests for the SedimentationCoagulationStrategy class.

This module contains tests for the SedimentationCoagulationStrategy class,
which implements the sedimentation coagulation strategy. The tests cover
both discrete and continuous_pdf distribution types.
"""

import unittest
import numpy as np
from particula.dynamics.coagulation.strategy.sedimentation_coagulation_strategy import SedimentationCoagulationStrategy
from particula.particles import PresetParticleRadiusBuilder


class TestSedimentationCoagulationStrategy(unittest.TestCase):
    """
    Test suite for the SedimentationCoagulationStrategy class.
    """

    def setUp(self):
        """
        Set up the test environment.

        Initializes a particle representation and creates instances of
        SedimentationCoagulationStrategy for both discrete and continuous_pdf
        distribution types.
        """
        self.particle = PresetParticleRadiusBuilder().build()
        self.temperature = 298.15  # Kelvin
        self.pressure = 101325  # Pascal
        self.turbulent_kinetic_energy = 0.1  # m^2/s^2
        self.fluid_density = 1.225  # kg/m^3

        # Create strategies for both distribution types
        self.strategy_discrete = SedimentationCoagulationStrategy(
            distribution_type="discrete",
            turbulent_kinetic_energy=self.turbulent_kinetic_energy,
            fluid_density=self.fluid_density
        )
        self.strategy_continuous_pdf = SedimentationCoagulationStrategy(
            distribution_type="continuous_pdf",
            turbulent_kinetic_energy=self.turbulent_kinetic_energy,
            fluid_density=self.fluid_density
        )

    def test_kernel_discrete(self):
        """
        Test the kernel calculation for discrete distribution.

        Verifies that the kernel method returns an ndarray for the discrete
        distribution type.
        """
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
