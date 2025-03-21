"""
Test module for the condensation strategies.
"""

import unittest
from unittest.mock import MagicMock
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.particles.representation import ParticleRepresentation
from particula.gas.species import GasSpecies


# pylint: disable=too-many-instance-attributes
class TestCondensationIsothermal(unittest.TestCase):
    """
    Test class for the CondensationIsothermal strategy.
    """

    def setUp(self):
        self.molar_mass = 0.018  # kg/mol for water
        self.diffusion_coefficient = 2e-5  # m^2/s
        self.accommodation_coefficient = 1.0
        self.strategy = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
        )

        self.particle = MagicMock(spec=ParticleRepresentation)
        self.gas_species = MagicMock(spec=GasSpecies)
        self.temperature = 298.15  # K
        self.pressure = 101325  # Pa
        self.time_step = 1.0  # s

    def test_mean_free_path(self):
        """Test the mean free path call."""
        result = self.strategy.mean_free_path(
            temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_knudsen_number(self):
        """Test the Knudsen number call"""
        radius = 1e-9  # m
        result = self.strategy.knudsen_number(
            radius=radius, temperature=self.temperature, pressure=self.pressure
        )
        self.assertIsNotNone(result)

    def test_first_order_mass_transport(self):
        """Test the first order mass transport call."""
        radius = 1e-9  # m
        result = self.strategy.first_order_mass_transport(
            particle_radius=radius,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        self.assertIsNotNone(result)
