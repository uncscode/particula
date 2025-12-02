"""Unit tests for the CombineCoagulationStrategy class.

This module contains tests for the CombineCoagulationStrategy class,
which combines multiple coagulation strategies into one. The tests cover
both discrete and continuous_pdf distribution types.
"""

# pylint: disable=duplicate-code

import unittest

import numpy as np

from particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy import (  # noqa: E501
    BrownianCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy import (  # noqa: E501
    CombineCoagulationStrategy,
)
from particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy import (  # noqa: E501
    TurbulentShearCoagulationStrategy,
)

# pylint: disable=line-too-long
from particula.particles import PresetParticleRadiusBuilder


# pylint: disable=too-many-instance-attributes
class TestCombineCoagulationStrategy(unittest.TestCase):
    """Test suite for the CombineCoagulationStrategy class."""

    def setUp(self):
        """Set up the test environment.

        Initializes a particle representation and creates instances of
        CombineCoagulationStrategy for both discrete and continuous_pdf
        distribution types.
        """
        self.particle = PresetParticleRadiusBuilder().build()
        self.temperature = 298.15  # Kelvin
        self.pressure = 101325  # Pascal
        self.turbulent_dissipation = 0.1  # m^2/s^2
        self.fluid_density = 1.225  # kg/m^3

        # Create individual strategies
        self.turbulent_shear_strategy = TurbulentShearCoagulationStrategy(
            distribution_type="discrete",
            turbulent_dissipation=self.turbulent_dissipation,
            fluid_density=self.fluid_density,
        )
        self.brownian_strategy = BrownianCoagulationStrategy(
            distribution_type="discrete"
        )

        # Combine strategies
        self.combined_strategy = CombineCoagulationStrategy(
            strategies=[self.turbulent_shear_strategy, self.brownian_strategy]
        )

    def test_kernel_combination(self):
        """Test the kernel combination.

        Verifies that the combined kernel is the sum of the individual kernels.
        """
        kernel_combined = self.combined_strategy.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        kernel_turbulent_shear = self.turbulent_shear_strategy.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        kernel_brownian = self.brownian_strategy.kernel(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
        )

        np.testing.assert_array_almost_equal(
            kernel_combined, kernel_turbulent_shear + kernel_brownian
        )

    def test_step_combination(self):
        """Test the step method for combined strategy.

        Ensures that the step method updates the particle concentration.
        """
        initial_concentration = self.particle.get_concentration().copy()
        self.combined_strategy.step(
            particle=self.particle,
            temperature=self.temperature,
            pressure=self.pressure,
            time_step=1.0,
        )
        updated_concentration = self.particle.get_concentration()
        self.assertFalse(
            np.array_equal(initial_concentration, updated_concentration)
        )
