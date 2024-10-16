"""Test thermal conductivity property functions."""

import pytest
import numpy as np
from particula.gas.properties import get_thermal_conductivity


def test_thermal_conductivity_normal():
    """Test the thermal_conductivity function with a
    normal temperature value."""
    temperature = 300  # in Kelvin
    expected = 1e-3 * (4.39 + 0.071 * 300)
    assert (
        pytest.approx(get_thermal_conductivity(temperature)) == expected
    ), "Failed at normal temperature"


def test_thermal_conductivity_array():
    """Test the thermal_conductivity function with an array of temperatures."""
    temperatures = np.array([250, 300, 350])
    expected = 1e-3 * (4.39 + 0.071 * temperatures)
    result = get_thermal_conductivity(temperatures)
    assert np.allclose(result, expected), "Failed with temperature array"


def test_thermal_conductivity_below_absolute_zero():
    """Test for error handling with temperature below absolute zero."""
    with pytest.raises(ValueError):
        get_thermal_conductivity(-1)


def test_thermal_conductivity_edge_case_zero():
    """Test the thermal_conductivity function at absolute zero."""
    temperature = 0
    expected = 1e-3 * (4.39 + 0.071 * 0)
    assert (
        pytest.approx(get_thermal_conductivity(temperature)) == expected
    ), "Failed at absolute zero"
