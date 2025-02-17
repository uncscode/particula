"""Test the gas species builder"""

# pylint: disable=##R0801

import numpy as np
import pytest
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_gas_species_builder_single_species():
    """Test building a single gas species with the GasSpeciesBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(vapor_pressure=101325)
    name = "Oxygen"
    molar_mass = 0.032  # kg/mol
    condensable = False
    concentration = 1.2  # kg/m^3

    gas_species = GasSpecies(
        name=name,
        molar_mass=molar_mass,
        vapor_pressure_strategy=vapor_pressure_strategy,
        condensable=condensable,
        concentration=concentration,
    )

    assert gas_species.name == name
    assert gas_species.get_molar_mass() == molar_mass
    assert gas_species.get_condensable() is condensable
    assert gas_species.get_concentration() == concentration
    assert gas_species.get_pure_vapor_pressure(298) == 101325


def test_gas_species_builder_array_species():
    """Test building an array of gas species with the GasSpeciesBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325])
    )
    names = np.array(["Oxygen", "Nitrogen"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    condensables = np.array([False, False])
    concentrations = np.array([1.2, 0.8])  # kg/m^3

    # Assuming the builder and GasSpecies class can handle array inputs
    # directly.
    gas_species = GasSpecies(
        name=names,
        molar_mass=molar_masses,
        vapor_pressure_strategy=vapor_pressure_strategy,
        condensable=condensables,
        concentration=concentrations,
    )

    assert np.array_equal(gas_species.name, names)
    assert np.array_equal(gas_species.get_molar_mass(), molar_masses)
    assert np.array_equal(gas_species.get_condensable(), condensables)
    assert np.array_equal(gas_species.get_concentration(), concentrations)
    # This assumes get_pure_vapor_pressure can handle and return arrays
    # correctly.
    assert np.array_equal(
        gas_species.get_pure_vapor_pressure(np.array([298, 300])),
        np.array([101325, 101325]),
    )


def test_gas_species_negative_molar_mass():
    """Test gas species with negative molar mass."""
    with pytest.raises(ValueError):
        GasSpecies(
            name="NegativeMass",
            molar_mass=-1,
            condensable=False,
            concentration=1e-10,
        )

    # array input
    with pytest.raises(ValueError):
        GasSpecies(
            name="NegativeMass",
            molar_mass=np.array([-1, 1]),
            condensable=False,
            concentration=np.array([1e-10, 1e-10]),
        )


def test_gas_species_zero_molar_mass():
    """Test gas species with zero molar mass."""
    with pytest.raises(ValueError):
        GasSpecies(
            name="ZeroMass",
            molar_mass=0,
            condensable=False,
            concentration=1e-10,
        )

    # array input
    with pytest.raises(ValueError):
        GasSpecies(
            name="ZeroMass",
            molar_mass=np.array([0, 1]),
            condensable=False,
            concentration=np.array([1e-10, 1e-10]),
        )


def test_gas_species_negative_concentration():
    """Test gas species with negative concentration."""
    with pytest.warns(UserWarning):
        gas_species = GasSpecies(
            name="NegativeConcentration",
            molar_mass=1e10,
            condensable=False,
            concentration=-1,
        )
        assert gas_species.get_concentration() == 0.0

    # array input
    with pytest.warns(UserWarning):
        gas_species = GasSpecies(
            name="NegativeConcentration",
            molar_mass=np.array([1, 1]),
            condensable=False,
            concentration=np.array([-1, 1]),
        )
        assert np.array_equal(gas_species.get_concentration(), [0.0, 1.0])


def test_gas_species_zero_concentration():
    """Test gas species with zero concentration."""
    zero_concentration_species = GasSpecies(
        name="ZeroConcentration",
        molar_mass=1e10,
        condensable=False,
        concentration=0,
    )

    assert zero_concentration_species.get_molar_mass() == 1e10
    assert zero_concentration_species.get_concentration() == 0

    # array input
    zero_concentration_species = GasSpecies(
        name="ZeroConcentration",
        molar_mass=np.array([1, 1]),
        condensable=False,
        concentration=np.array([0, 1]),
    )

    assert np.array_equal(zero_concentration_species.get_molar_mass(), [1, 1])
    assert np.array_equal(zero_concentration_species.get_concentration(), [0, 1])


def test_gas_species_get_methods():
    """Test the get methods of GasSpecies."""
    gas_species = GasSpecies(
        name="TestSpecies",
        molar_mass=0.044,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1000),
        condensable=True,
        concentration=0.5,
    )
    assert gas_species.get_name() == "TestSpecies"
    assert gas_species.get_molar_mass() == 0.044
    assert gas_species.get_condensable() is True
    assert gas_species.get_concentration() == 0.5


def test_gas_species_set_concentration():
    """Test the set_concentration method of GasSpecies."""
    gas_species = GasSpecies(
        name="TestSpecies",
        molar_mass=0.044,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1000),
        condensable=True,
        concentration=0.5,
    )
    gas_species.set_concentration(1.0)
    assert gas_species.get_concentration() == 1.0
    with pytest.warns(UserWarning):
        gas_species.set_concentration(-1.0)
    assert gas_species.get_concentration() == 0.0


def test_gas_species_add_concentration():
    """Test the add_concentration method of GasSpecies."""
    gas_species = GasSpecies(
        name="TestSpecies",
        molar_mass=0.044,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1000),
        condensable=True,
        concentration=0.5,
    )
    gas_species.add_concentration(0.5)
    assert gas_species.get_concentration() == 1.0
    with pytest.warns(UserWarning):
        gas_species.add_concentration(-3.0)
    assert gas_species.get_concentration() == 0.0


def test_gas_species_array_operations():
    """Test array operations for concentration in GasSpecies."""
    gas_species = GasSpecies(
        name="TestSpecies",
        molar_mass=0.044,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1000),
        condensable=True,
        concentration=0.5,
    )
    gas_species.set_concentration(np.array([0.5, 1.0]))
    assert np.array_equal(gas_species.get_concentration(), [0.5, 1.0])
    gas_species.add_concentration(np.array([0.5, -1.0]))
    assert np.array_equal(gas_species.get_concentration(), [1.0, 0.0])
