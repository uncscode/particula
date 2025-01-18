"""Test Kolmogorov Time"""

import numpy as np
import pytest
from particula.gas.properties.kolmogorov_module import get_kolmogorov_time


def test_kolmogorov_time_float():
    """
    Test kolmogorov_time with float inputs.
    """
    assert np.isclose(get_kolmogorov_time(1.0, 1.0), 1.0)
    assert np.isclose(get_kolmogorov_time(0.01, 0.01), 1.0)
    assert np.isclose(get_kolmogorov_time(0.1, 0.01), 3.1622776601683795)


def test_kolmogorov_time_array():
    """
    Test kolmogorov_time with numpy array inputs.
    """
    kinematic_viscosity = np.array([1.0, 0.01, 0.1])
    turbulent_dissipation = np.array([1.0, 0.01, 0.01])
    expected = np.array([1.0, 1.0, 3.1622776601683795])
    result = get_kolmogorov_time(kinematic_viscosity, turbulent_dissipation)
    assert np.allclose(result, expected)


def test_kolmogorov_time_input_range():
    """
    Test kolmogorov_time with invalid input ranges.
    """
    with pytest.raises(ValueError):
        get_kolmogorov_time(-1.0, 1.0)
    with pytest.raises(ValueError):
        get_kolmogorov_time(1.0, -1.0)
    with pytest.raises(ValueError):
        get_kolmogorov_time(-1.0, -1.0)
