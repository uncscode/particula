"""Test the gas species builder."""

# pylint: disable=##R0801

import numpy as np
import pytest

from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


def test_gas_species_builder_single_species():
    """Test building a single gas species with the GasSpeciesBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=101325
    )
    name = "Oxygen"
    molar_mass = 0.032  # kg/mol
    partitioning = False
    concentration = 1.2  # kg/m^3

    gas_species = GasSpecies(
        name=name,
        molar_mass=molar_mass,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=partitioning,
        concentration=concentration,
    )

    assert gas_species.name == name
    assert gas_species.get_molar_mass() == molar_mass
    assert gas_species.get_partitioning() is partitioning
    assert gas_species.get_concentration() == concentration
    assert gas_species.get_pure_vapor_pressure(298) == 101325


def test_gas_species_builder_array_species():
    """Test building an array of gas species with the GasSpeciesBuilder."""
    vapor_pressure_strategy = ConstantVaporPressureStrategy(
        vapor_pressure=np.array([101325, 101325])
    )
    names = np.array(["Oxygen", "Nitrogen"])
    molar_masses = np.array([0.032, 0.028])  # kg/mol
    partitioning = False
    concentrations = np.array([1.2, 0.8])  # kg/m^3

    # Assuming the builder and GasSpecies class can handle array inputs
    # directly.
    gas_species = GasSpecies(
        name=names,
        molar_mass=molar_masses,
        vapor_pressure_strategy=vapor_pressure_strategy,
        partitioning=partitioning,
        concentration=concentrations,
    )

    assert np.array_equal(gas_species.name, names)
    assert np.array_equal(gas_species.get_molar_mass(), molar_masses)
    assert gas_species.get_partitioning() is partitioning
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
            partitioning=False,
            concentration=1e-10,
        )

    # array input
    with pytest.raises(ValueError):
        GasSpecies(
            name="NegativeMass",
            molar_mass=np.array([-1, 1]),
            partitioning=False,
            concentration=np.array([1e-10, 1e-10]),
        )


def test_gas_species_zero_molar_mass():
    """Test gas species with zero molar mass."""
    with pytest.raises(ValueError):
        GasSpecies(
            name="ZeroMass",
            molar_mass=0,
            partitioning=False,
            concentration=1e-10,
        )

    # array input
    with pytest.raises(ValueError):
        GasSpecies(
            name="ZeroMass",
            molar_mass=np.array([0, 1]),
            partitioning=False,
            concentration=np.array([1e-10, 1e-10]),
        )


def test_gas_species_negative_concentration():
    """Test gas species with negative concentration."""
    with pytest.warns(UserWarning):
        gas_species = GasSpecies(
            name="NegativeConcentration",
            molar_mass=1e10,
            partitioning=False,
            concentration=-1,
        )
        assert gas_species.get_concentration() == 0.0

    # array input
    with pytest.warns(UserWarning):
        gas_species = GasSpecies(
            name="NegativeConcentration",
            molar_mass=np.array([1, 1]),
            partitioning=False,
            concentration=np.array([-1, 1]),
        )
        assert np.array_equal(gas_species.get_concentration(), [0.0, 1.0])


def test_gas_species_zero_concentration():
    """Test gas species with zero concentration."""
    zero_concentration_species = GasSpecies(
        name="ZeroConcentration",
        molar_mass=1e10,
        partitioning=False,
        concentration=0,
    )

    assert zero_concentration_species.get_molar_mass() == 1e10
    assert zero_concentration_species.get_concentration() == 0

    # array input
    zero_concentration_species = GasSpecies(
        name="ZeroConcentration",
        molar_mass=np.array([1, 1]),
        partitioning=False,
        concentration=np.array([0, 1]),
    )

    assert np.array_equal(zero_concentration_species.get_molar_mass(), [1, 1])
    assert np.array_equal(
        zero_concentration_species.get_concentration(), [0, 1]
    )


def test_gas_species_get_methods():
    """Test the get methods of GasSpecies."""
    gas_species = GasSpecies(
        name="TestSpecies",
        molar_mass=0.044,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1000),
        partitioning=True,
        concentration=0.5,
    )
    assert gas_species.get_name() == "TestSpecies"
    assert gas_species.get_molar_mass() == 0.044
    assert gas_species.get_partitioning() is True
    assert gas_species.get_concentration() == 0.5


def test_gas_species_set_concentration():
    """Test the set_concentration method of GasSpecies."""
    gas_species = GasSpecies(
        name="TestSpecies",
        molar_mass=0.044,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1000),
        partitioning=True,
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
        partitioning=True,
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
        partitioning=True,
        concentration=0.5,
    )
    gas_species.set_concentration(np.array([0.5, 1.0]))
    assert np.array_equal(gas_species.get_concentration(), [0.5, 1.0])
    gas_species.add_concentration(np.array([0.5, -1.0]))
    assert np.array_equal(gas_species.get_concentration(), [1.0, 0.0])


# ---------------------------------------------------------------------------
# Operator add and iadd tests
# ---------------------------------------------------------------------------


def _simple_species(name: str, partitioning: bool = True) -> GasSpecies:
    """Utility constructor to avoid repetition."""
    return GasSpecies(
        name=name,
        molar_mass=0.01,  # kg / mol
        vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
        partitioning=partitioning,
        concentration=1.0,
    )


def test_gas_species_append():
    """The append method should mutate *self* and concatenate attributes."""
    s1 = _simple_species("A")
    s2 = _simple_species("B")

    s1.append(s2)

    assert np.array_equal(s1.get_name(), np.array(["A", "B"]))
    assert len(s1) == 2
    # concentration and molar-mass also doubled in length
    assert s1.get_concentration().shape == (2,)
    assert s1.get_molar_mass().shape == (2,)


def test_gas_species_iadd():
    """Using += should call append and mutate self."""
    s1 = _simple_species("A")
    s2 = _simple_species("B")

    s1 += s2  # in-place

    assert np.array_equal(s1.get_name(), np.array(["A", "B"]))
    assert len(s1) == 2


def test_gas_species_add():
    """Using + should return a NEW object and keep the originals unchanged."""
    s1 = _simple_species("A")
    s2 = _simple_species("B")

    merged = s1 + s2

    # originals unchanged
    assert s1.get_name() == "A"
    assert s2.get_name() == "B"

    # merged contains both
    assert np.array_equal(merged.get_name(), np.array(["A", "B"]))
    assert len(merged) == 2
    # ensure it is a different object
    assert merged is not s1 and merged is not s2


def test_gas_species_partitioning_mismatch_errors():
    """Append / += / + must raise if partitioning flags differ."""
    s_true = _simple_species("A", partitioning=True)
    s_false = _simple_species("B", partitioning=False)

    # append
    with pytest.raises(ValueError):
        s_true.append(s_false)

    # iadd
    with pytest.raises(ValueError):
        s_true += s_false  # type: ignore[operator]

    # add
    with pytest.raises(ValueError):
        _ = s_true + s_false
