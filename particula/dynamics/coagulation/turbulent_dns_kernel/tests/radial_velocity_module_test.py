"""Test the radial relative velocity formulations."""

import numpy as np
import pytest

from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (  # noqa: E501
    get_radial_relative_velocity_ao2008,
    get_radial_relative_velocity_dz2002,
)


def test_get_radial_relative_velocity_dz2002():
    """Test get_radial_relative_velocity_dz2002 with a small array input."""
    velocity_dispersion = 0.1  # m/s
    particle_inertia_time = np.array([0.02, 0.03, 0.05])  # s

    expected_shape = (3, 3)
    result = get_radial_relative_velocity_dz2002(
        velocity_dispersion, particle_inertia_time
    )

    assert result.shape == expected_shape, (
        f"Expected shape {expected_shape}, but got {result.shape}"
    )
    assert np.all(result >= 0), "Expected all values to be non-negative"


def test_get_radial_relative_velocity_ao2008():
    """Now expect NotImplementedError for valid inputs."""
    velocity_dispersion = 0.1  # m/s
    particle_inertia_time = np.array([0.02, 0.03, 0.05])  # s

    with pytest.raises(NotImplementedError):
        get_radial_relative_velocity_ao2008(
            velocity_dispersion, particle_inertia_time
        )


def test_invalid_inputs():
    """Ensure validation errors are raised for invalid inputs."""
    velocity_dispersion = 0.1  # m/s
    particle_inertia_time = np.array([0.02, 0.03, 0.05])  # s

    with pytest.raises(ValueError):
        get_radial_relative_velocity_dz2002(
            -velocity_dispersion, particle_inertia_time
        )  # Negative dispersion

    with pytest.raises(ValueError):
        get_radial_relative_velocity_dz2002(
            velocity_dispersion, -particle_inertia_time
        )  # Negative inertia

    with pytest.raises(ValueError):
        get_radial_relative_velocity_ao2008(
            -velocity_dispersion, particle_inertia_time
        )  # Negative dispersion

    with pytest.raises(ValueError):
        get_radial_relative_velocity_ao2008(
            velocity_dispersion, -particle_inertia_time
        )  # Negative inertia


def test_edge_cases():
    """Test the function with extreme values such as zero inertia and very large
    inertia.
    """
    velocity_dispersion = 0.1  # m/s
    particle_inertia_time = np.array(
        [0.01, 1e-6, 10.0]
    )  # Small and large inertia values

    wr_dz2002 = get_radial_relative_velocity_dz2002(
        velocity_dispersion, particle_inertia_time
    )

    # Ensure no negative values
    assert np.all(wr_dz2002 >= 0), "Expected all values to be non-negative"

    # Ensure numerical stability for large inertia values
    assert np.isfinite(wr_dz2002).all(), "Expected all values to be finite"

    with pytest.raises(NotImplementedError):
        get_radial_relative_velocity_ao2008(
            velocity_dispersion, particle_inertia_time
        )
