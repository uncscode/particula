"""Test Diffusive Knudsen number calculation."""

import pytest
import numpy as np
from particula.particles.properties import diffusive_knudsen_number


def test_diffusive_knudsen_number_scalar():
    """Test the diffusive Knudsen number calculation with scalar inputs."""
    radius = 100e-9  # meters
    mass_particle = 1e-24  # kg
    friction_factor = 1.0  # dimensionless
    coulomb_potential_ratio = 0.5  # dimensionless
    temperature = 298.15  # K

    result = diffusive_knudsen_number(
        radius,
        mass_particle,
        friction_factor,
        coulomb_potential_ratio,
        temperature,
    )

    expected_result = 5.35520625418024e-16
    assert np.isclose(result, expected_result), "Test failed for scalar inputs"


def test_diffusive_knudsen_number_array():
    """Test the diffusive Knudsen number calculation with array inputs."""
    radius = np.array([100e-9, 1e-6])
    mass_particle = np.array([1e-24, 2e-20])
    friction_factor = np.array([1.0, 1.5])
    coulomb_potential_ratio = np.array([0.5, 0])  # different charges
    temperature = 298.15

    result = diffusive_knudsen_number(
        radius,
        mass_particle,
        friction_factor,
        coulomb_potential_ratio,
        temperature,
    )

    expected_result = np.array(
        [[5.35520625e-16, 9.72085031e-17], [1.14745697e-16, 4.27728106e-15]]
    )
    assert np.allclose(result, expected_result), "Test failed for array inputs"


def test_with_zero_coulomb_potential_ratio():
    """Test the diffusive Knudsen number calculation with zero Coulomb
    potential ratio."""
    radius = 0.1
    mass_particle = 1e-18
    friction_factor = 1.0
    coulomb_potential_ratio = 0.0  # no charges
    temperature = 298.15

    result = diffusive_knudsen_number(
        radius,
        mass_particle,
        friction_factor,
        coulomb_potential_ratio,
        temperature,
    )
    expected_result = 4.53674166858771e-19
    assert np.isclose(result, expected_result)


def test_invalid_inputs():
    """Test the diffusive Knudsen number calculation with invalid inputs."""
    with pytest.raises(TypeError):
        diffusive_knudsen_number(
            "invalid",  # invalid radius
            1e-18,  # valid mass
            1.0,  # valid friction factor
            0.5,  # valid coulomb potential ratio
            298.15,  # valid temperature
        )
