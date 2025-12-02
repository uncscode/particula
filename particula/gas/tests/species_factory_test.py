"""Test the species factory module, does build the gas species object."""

import unittest

from particula.gas.species import GasSpecies
from particula.gas.species_builders import (
    GasSpeciesBuilder,
    PresetGasSpeciesBuilder,
)
from particula.gas.species_factories import GasSpeciesFactory
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
            "molar_mass_units": "kg/mol",
            "vapor_pressure_strategy": ConstantVaporPressureStrategy(50),
            "partitioning": False,
            "concentration": 1.2,
            "concentration_units": "kg/m^3",
        }
        gas_species = self.factory.get_strategy("gas_species", parameters)
        self.assertIsInstance(gas_species, GasSpecies)
        self.assertEqual(gas_species.name, parameters["name"])
        self.assertEqual(gas_species.molar_mass, parameters["molar_mass"])
        self.assertEqual(
            gas_species.pure_vapor_pressure_strategy,
            parameters["vapor_pressure_strategy"],
        )
        self.assertEqual(gas_species.partitioning, parameters["partitioning"])
        self.assertEqual(gas_species.concentration, parameters["concentration"])
        self.assertIsInstance(gas_species, GasSpecies)

    def test_get_strategy_preset_gas_species(self):
        """Test getting a preset gas species strategy."""
        parameters = {}
        preset_gas_species = self.factory.get_strategy(
            "preset_gas_species", parameters
        )
        self.assertIsInstance(preset_gas_species, GasSpecies)
        self.assertEqual(preset_gas_species.molar_mass, 0.1)
        self.assertIsInstance(
            preset_gas_species.pure_vapor_pressure_strategy,
            ConstantVaporPressureStrategy,
        )
        self.assertEqual(preset_gas_species.partitioning, False)
        self.assertEqual(preset_gas_species.concentration, 1.0)

    def test_get_strategy_invalid(self):
        """Test getting an invalid strategy."""
        with self.assertRaises(ValueError):
            self.factory.get_strategy("invalid_type", {})
