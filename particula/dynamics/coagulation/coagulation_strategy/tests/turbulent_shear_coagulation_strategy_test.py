"""Unit tests for the TurbulentShearCoagulationStrategy class.

This module contains tests for the TurbulentShearCoagulationStrategy class,
which implements the turbulent shear coagulation strategy. The tests cover
both discrete and continuous_pdf distribution types.
"""

# pylint: disable=duplicate-code, too-many-instance-attributes

import unittest

import numpy as np

from particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy import (  # noqa: E501
    TurbulentShearCoagulationStrategy,
)

# pylint: disable=line-too-long
from particula.particles import (
    PresetParticleRadiusBuilder,
    PresetResolvedParticleMassBuilder,
)


class TestTurbulentShearCoagulationStrategy(unittest.TestCase):
    """Test suite for the TurbulentShearCoagulationStrategy class."""

    def setUp(self):
        """Set up the test environment.

        Initializes a particle representation and creates instances of
        TurbulentShearCoagulationStrategy for discrete, continuous_pdf, and
        particle_resolved distribution types.
        """
        self.particle = PresetParticleRadiusBuilder().build()
        self.temperature = 298.15  # Kelvin
        self.pressure = 101325  # Pascal
        self.turbulent_dissipation = 1  # m^2/s^2
        self.fluid_density = 1.225  # kg/m^3

        # Create strategies for all distribution types
        self.strategy_discrete = TurbulentShearCoagulationStrategy(
            distribution_type="discrete",
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )
        self.strategy_continuous_pdf = TurbulentShearCoagulationStrategy(
            distribution_type="continuous_pdf",
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )

        self.particle_resolved = (
            PresetResolvedParticleMassBuilder().set_volume(1e-6, "m^3").build()
        )
        self.strategy_particle_resolved = TurbulentShearCoagulationStrategy(
            distribution_type="particle_resolved",
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )

    def test_kernel_discrete(self):
        """Test the kernel calculation for discrete distribution.

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
        """Test the step method for discrete distribution.

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
        """Test the kernel calculation for continuous_pdf distribution.

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
        """Test the step method for continuous_pdf distribution.

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

    def test_step_particle_resolved(self):
        """Test the step method for particle_resolved distribution."""
        old_concentration = self.particle_resolved.get_total_concentration()
        self.strategy_particle_resolved.step(
            particle=self.particle_resolved,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1000,
        )
        new_concentration = self.particle_resolved.get_total_concentration()
        self.assertNotEqual(old_concentration, new_concentration)
