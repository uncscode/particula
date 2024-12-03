"""Test the species factory module, does build the gas species object."""

import unittest
from particula.gas.species_factories import GasSpeciesFactory
from particula.gas.species_builders import (
    GasSpeciesBuilder,
    PresetGasSpeciesBuilder,
)
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


class TestGasSpeciesFactory(unittest.TestCase):
    """Test the GasSpeciesFactory class."""
    def setUp(self):
        """Set up the test case."""
        self.factory = GasSpeciesFactory()

    def test_get_builders(self):
        """Test getting the builders."""
        builders = self.factory.get_builders()
        self.assertIsInstance(builders["gas_species"], GasSpeciesBuilder)
        self.assertIsInstance(
            builders["preset_gas_species"], PresetGasSpeciesBuilder
        )

    def test_get_strategy_gas_species(self):
        """Test getting a gas species strategy."""
        parameters = {
            "name": "Oxygen",
            "molar_mass": 0.032,
            "vapor_pressure_strategy": ConstantVaporPressureStrategy(50),
            "condensable": False,
            "concentration": 1.2,
        }
        gas_species = self.factory.get_strategy("gas_species", parameters)
        self.assertIsInstance(gas_species, GasSpecies)

    def test_get_strategy_preset_gas_species(self):
        """Test getting a preset gas species strategy."""
        parameters = {}
        preset_gas_species = self.factory.get_strategy(
            "preset_gas_species", parameters
        )
        self.assertIsInstance(preset_gas_species, GasSpecies)
        self.assertEqual(preset_gas_species.mass, 28.97)
        self.assertEqual(preset_gas_species.viscosity, 1.81e-5)
        self.assertEqual(preset_gas_species.temperature, 293.15)

    def test_get_strategy_invalid(self):
        """Test getting an invalid strategy."""
        with self.assertRaises(ValueError):
            self.factory.get_strategy("invalid_type", {})
