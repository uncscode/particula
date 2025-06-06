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

import numpy as np                       # new
from unittest.mock import patch          # new


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

        self.particle.concentration = 1.0        # new – scalar concentration
        self.particle.get_radius.return_value = np.array([1e-9])  # new – default radius
        self.particle.get_species_mass.return_value = np.array([1.0])  # new dummy mass
        self.particle.surface.kelvin_term.return_value = np.array([1.0])  # new

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

    def test_fill_zero_radius(self):
        """_fill_zero_radius changes zeros to max radius."""
        radii = np.array([0.0, 1e-9, 2e-9])
        filled = self.strategy._fill_zero_radius(radii.copy())
        # zero should become the maximum original non-zero radius (2 nm)
        self.assertTrue(np.all(filled != 0.0))
        self.assertEqual(filled[0], np.max(radii))

    def test_rate_respects_skip_indices(self):
        """Chosen indices must be zeroed in the returned rate."""
        strategy_skip = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[1],
        )
        # fake mass-transfer values (3 “species”)
        strategy_skip.mass_transfer_rate = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        rates = strategy_skip.rate(
            particle=self.particle,
            gas_species=self.gas_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        np.testing.assert_array_equal(rates, np.array([1.0, 0.0, 3.0]))

    def test_step_zeroes_mass_rate_before_transfer(self):
        """step() must pass a zeroed mass_rate to get_mass_transfer."""
        strategy_skip = CondensationIsothermal(
            molar_mass=self.molar_mass,
            diffusion_coefficient=self.diffusion_coefficient,
            accommodation_coefficient=self.accommodation_coefficient,
            skip_partitioning_indices=[1],
        )
        # return non-zero values so we can see the zeroing
        strategy_skip.mass_transfer_rate = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))

        with patch(
            "particula.dynamics.condensation.condensation_strategies.get_mass_transfer",
            autospec=True,
        ) as mocked_get_mass_transfer:
            mocked_get_mass_transfer.return_value = np.array([0.1, 0.0, 0.3])

            strategy_skip.step(
                particle=self.particle,
                gas_species=self.gas_species,
                temperature=self.temperature,
                pressure=self.pressure,
                time_step=self.time_step,
            )

            # extract the mass_rate argument passed into get_mass_transfer
            passed_mass_rate = mocked_get_mass_transfer.call_args.kwargs["mass_rate"]
            np.testing.assert_array_equal(passed_mass_rate, np.array([1.0, 0.0, 3.0]))
