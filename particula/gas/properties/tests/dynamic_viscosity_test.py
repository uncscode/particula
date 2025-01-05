"""Test dynamic viscosity property functions."""

import pytest
from particula.gas.properties import get_dynamic_viscosity


def test_dynamic_viscosity_normal_conditions():
    """Test dynamic viscosity under normal conditions."""
    assert (
        pytest.approx(get_dynamic_viscosity(300), 1e-5)
        == 1.8459162511975804e-5
    ), "Failed under normal conditions"


def test_dynamic_viscosity_high_temperature():
    """Test dynamic viscosity at high temperature."""
    assert (
        pytest.approx(get_dynamic_viscosity(1000), 1e-5)
        == 4.1520063611410934e-05
    ), "Failed at high temperature"


def test_dynamic_viscosity_low_temperature():
    """Test dynamic viscosity at low temperature."""
    assert (
        pytest.approx(get_dynamic_viscosity(250), 1e-5) == 1.599052394e-5
    ), "Failed at low temperature"


def test_dynamic_viscosity_reference_values():
    """Test dynamic viscosity with reference values."""
    assert (
        pytest.approx(get_dynamic_viscosity(300, 1.85e-5, 300), 1e-5)
        == 1.85e-5
    ), "Failed with reference values"


def test_dynamic_viscosity_zero_temperature():
    """Test for error handling with zero temperature."""
    with pytest.raises(ValueError):
        get_dynamic_viscosity(0)


def test_dynamic_viscosity_negative_temperature():
    """Test for error handling with negative temperature."""
    with pytest.raises(ValueError):
        get_dynamic_viscosity(-10)
