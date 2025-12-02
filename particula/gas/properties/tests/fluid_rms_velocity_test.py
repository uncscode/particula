"""Test file for fluid_rms_velocity.py."""

import numpy as np
import pytest

from particula.gas.properties.fluid_rms_velocity import (
    get_fluid_rms_velocity,
)  # Update 'your_module' with the actual module name


def test_get_fluid_rms_velocity_scalar():
    """Test get_fluid_rms_velocity with scalar inputs."""
    re_lambda = 100.0
    kinematic_viscosity = 1.5e-5  # m²/s
    turbulent_dissipation = 1e-3  # m²/s³

    expected = (
        re_lambda**0.5 * (kinematic_viscosity * turbulent_dissipation) ** 0.25
    ) / (15**0.25)
    result = get_fluid_rms_velocity(
        re_lambda, kinematic_viscosity, turbulent_dissipation
    )

    assert np.isclose(result, expected, atol=1e-10), (
        f"Expected {expected}, got {result}"
    )


def test_get_fluid_rms_velocity_array():
    """Test get_fluid_rms_velocity with NumPy array inputs."""
    re_lambda = np.array([100.0, 200.0])
    kinematic_viscosity = np.array([1.5e-5, 1.2e-5])
    turbulent_dissipation = np.array([1e-3, 2e-3])

    expected = (
        re_lambda**0.5 * (kinematic_viscosity * turbulent_dissipation) ** 0.25
    ) / (15**0.25)
    result = get_fluid_rms_velocity(
        re_lambda, kinematic_viscosity, turbulent_dissipation
    )

    assert np.allclose(result, expected, atol=1e-10), (
        f"Expected {expected}, got {result}"
    )


def test_get_fluid_rms_velocity_invalid_values():
    """Test that get_fluid_rms_velocity raises errors for invalid inputs."""
    with pytest.raises(ValueError):
        get_fluid_rms_velocity(-10.0, 1.5e-5, 1e-3)  # Negative Reynolds number

    with pytest.raises(ValueError):
        get_fluid_rms_velocity(
            100.0, -1.5e-5, 1e-3
        )  # Negative kinematic viscosity

    with pytest.raises(ValueError):
        get_fluid_rms_velocity(
            100.0, 1.5e-5, -1e-3
        )  # Negative turbulent dissipation


def test_get_fluid_rms_velocity_edge_case():
    """Test get_fluid_rms_velocity with very small values close to machine
    precision.
    """
    re_lambda = 1.0
    kinematic_viscosity = 1e-10
    turbulent_dissipation = 1e-10

    expected = (
        re_lambda**0.5 * (kinematic_viscosity * turbulent_dissipation) ** 0.25
    ) / (15**0.25)
    result = get_fluid_rms_velocity(
        re_lambda, kinematic_viscosity, turbulent_dissipation
    )

    assert np.isclose(result, expected, atol=1e-10), (
        f"Expected {expected}, got {result}"
    )
