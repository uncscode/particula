"""Testing Gas class"""

import pytest
import numpy as np
from particula.next.gas.species import Gas, GasSpecies


def test_gas_species_init():
    """Test the initialization of a GasSpecies object."""
    species = GasSpecies(
        name="Oxygen",
        mass=32.0,
        vapor_pressure=213.0,
        condensable=True)
    assert species.name == "Oxygen"
    assert species.mass == 32.0
    assert species.vapor_pressure == 213.0
    assert species.condensable is True


def test_gas_species_mass():
    """Test the get_mass method of GasSpecies."""
    species = GasSpecies(name="Oxygen", mass=32.0)
    assert species.get_mass() == np.float64(32.0)


def test_gas_species_condensable():
    """Test the is_condensable method of GasSpecies."""
    species = GasSpecies(name="Oxygen", mass=32.0, condensable=True)
    assert species.is_condensable() is True


def test_gas_init():
    """Test the initialization of a Gas object with default values."""
    gas = Gas()
    assert gas.temperature == 298.15
    assert gas.total_pressure == 101325
    assert len(gas.components) == 0


def test_gas_add_species():
    """Test adding a species to a Gas object."""
    gas = Gas()
    gas.add_species("Oxygen", 32.0, 213.0, True)
    assert len(gas.components) == 1
    assert gas.components[0].name == "Oxygen"


def test_gas_remove_species():
    """Test removing a species from a Gas object."""
    gas = Gas()
    gas.add_species("Oxygen", 32.0)
    gas.add_species("Nitrogen", 28.0)
    gas.remove_species("Oxygen")
    assert len(gas.components) == 1
    assert gas.components[0].name == "Nitrogen"


def test_gas_get_mass():
    """Test getting the mass of all species in a Gas object."""
    gas = Gas()
    gas.add_species("Oxygen", 32.0)
    gas.add_species("Nitrogen", 28.0)
    expected_masses = np.array([32.0, 28.0], dtype=np.float64)
    np.testing.assert_array_equal(gas.get_mass(), expected_masses)


def test_gas_get_mass_condensable():
    """Test getting the mass of all condensable species in a Gas object."""
    gas = Gas()
    gas.add_species("Oxygen", 32.0, condensable=False)
    gas.add_species("Water", 18.0, condensable=True)
    expected_masses = np.array([18.0], dtype=np.float64)
    np.testing.assert_array_equal(gas.get_mass_condensable(), expected_masses)


def test_gas_get_mass_nonexistent_species():
    """Test getting the mass of a nonexistent species raises a ValueError."""
    gas = Gas()
    with pytest.raises(ValueError):
        gas.get_mass(name="Helium")


def test_gas_get_mass_condensable_nonexistent_species():
    """Test getting the mass of a nonexistent condensable species raises a
    ValueError."""
    gas = Gas()
    with pytest.raises(ValueError):
        gas.get_mass_condensable(name="Helium")
