"""
Test Special Functions
"""

import numpy as np
from particula.next.particles.properties.special_functions import (
    debye_function,
)


def test_debye_function_single_float():
    """Test the Debye function for a single float value."""
    result = debye_function(1.0)
    assert np.isclose(result, 0.7765038970390566)

    result = debye_function(1.0, n=2)
    assert np.isclose(result, 0.7078773477535959)


def test_debye_function_numpy_array():
    """Test the Debye function for a numpy array."""
    input_array = np.array([1.0, 2.0, 3.0])
    expected_output = np.array([0.7765039, 0.60594683, 0.47943507])
    result = debye_function(input_array)
    assert np.allclose(result, expected_output)


def test_debye_function_with_integration_points():
    """Test Debye function with lower integration points."""
    result = debye_function(1.0, integration_points=100)
    assert np.isclose(result, 0.7674304601560753)

    result = debye_function(1.0, integration_points=100, n=2)
    assert np.isclose(result, 0.7077640915847944)
