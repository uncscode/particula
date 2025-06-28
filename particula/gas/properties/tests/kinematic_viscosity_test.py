"""Test module for the kinematic_viscosity module."""

import pytest

from particula.gas.properties.kinematic_viscosity import (
    get_kinematic_viscosity,
    get_kinematic_viscosity_via_system_state,
)


def test_kinematic_viscosity():
    """Test kinematic viscosity calculation."""
    dynamic_viscosity = 1.85e-5  # Pa*s
    fluid_density = 1.2  # kg/m^3
    expected_kinematic_viscosity = dynamic_viscosity / fluid_density
    assert (
        pytest.approx(
            get_kinematic_viscosity(dynamic_viscosity, fluid_density), 1e-5
        )
        == expected_kinematic_viscosity
    )


def test_kinematic_viscosity_via_system_state():
    """Test kinematic viscosity calculation via system state."""
    temperature = 300  # K
    fluid_density = 1.2  # kg/m^3
    expected_kinematic_viscosity = get_kinematic_viscosity_via_system_state(
        temperature, fluid_density
    )
    assert (
        pytest.approx(expected_kinematic_viscosity, 1e-5)
        == expected_kinematic_viscosity
    )


def test_kinematic_viscosity_zero_temperature():
    """Test for error handling with zero temperature."""
    with pytest.raises(ValueError):
        get_kinematic_viscosity_via_system_state(0, 1.2)


def test_kinematic_viscosity_negative_temperature():
    """Test for error handling with negative temperature."""
    with pytest.raises(ValueError):
        get_kinematic_viscosity_via_system_state(-10, 1.2)
