"""Test Diffusive Knudsen number calculation."""

import numpy as np
import pytest
from particula.particles.properties.diffusive_knudsen_module import (
    get_diffusive_knudsen_number,
)


def test_diffusive_knudsen_number_scalar():
    """Test the diffusive Knudsen number calculation with scalar inputs."""
    radius = 100e-9  # meters
    particle_mass = 1e-24  # kg
    friction_factor = 1.0  # dimensionless
    coulomb_potential_ratio = 0.5  # dimensionless
    temperature = 298.15  # K

    result = get_diffusive_knudsen_number(
        radius,
        particle_mass,
        friction_factor,
        coulomb_potential_ratio,
        temperature,
    )

    expected_result = (
        np.sqrt(1.380649e-23 * temperature * particle_mass / 2)
        / (friction_factor / 2 * 2 * radius)
        * (0.5 / -np.expm1(-0.5))
        / 1.5
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-14, atol=0)


def test_diffusive_knudsen_number_array():
    """Test the diffusive Knudsen number calculation with array inputs."""
    radius = np.array([100e-9, 1e-6])
    particle_mass = np.array([1e-24, 2e-20])
    friction_factor = np.array([1.0, 1.5])
    coulomb_potential_ratio = np.array([0.5, 0])  # different charges
    temperature = 298.15

    result = get_diffusive_knudsen_number(
        radius,
        particle_mass,
        friction_factor,
        coulomb_potential_ratio,
        temperature,
    )

    reduced_mass = (
        particle_mass[:, None]
        * particle_mass
        / (particle_mass[:, None] + particle_mass)
    )
    reduced_friction = (
        friction_factor[:, None]
        * friction_factor
        / (friction_factor[:, None] + friction_factor)
    )
    expected_result = (
        np.sqrt(1.380649e-23 * temperature * reduced_mass)
        / (reduced_friction * (radius[:, None] + radius))
        * np.array([(0.5 / -np.expm1(-0.5)) / 1.5, 1.0])
    )
    np.testing.assert_allclose(result, expected_result, rtol=1e-14, atol=0)


def test_with_zero_coulomb_potential_ratio():
    """Test the diffusive Knudsen number calculation with zero Coulomb
    potential ratio.
    """
    radius = 0.1
    particle_mass = 1e-18
    friction_factor = 1.0
    coulomb_potential_ratio = 0.0  # no charges
    temperature = 298.15

    result = get_diffusive_knudsen_number(
        radius,
        particle_mass,
        friction_factor,
        coulomb_potential_ratio,
        temperature,
    )
    expected_result = 4.53674166858771e-19
    assert np.isclose(result, expected_result)


def test_invalid_inputs():
    """Test the diffusive Knudsen number calculation with invalid inputs."""
    with pytest.raises(TypeError):
        get_diffusive_knudsen_number(
            "invalid",  # invalid radius
            1e-18,  # valid mass
            1.0,  # valid friction factor
            0.5,  # valid coulomb potential ratio
            298.15,  # valid temperature
        )


def test_near_zero_kinetic_enhancement_guard() -> None:
    """Ensure extreme negative potentials yield finite near-zero values."""
    result = get_diffusive_knudsen_number(
        particle_radius=100e-9,
        particle_mass=1e-24,
        friction_factor=1.0,
        coulomb_potential_ratio=-200.0,
        temperature=298.15,
    )

    assert np.isfinite(result)
    assert result <= 1e-30


@pytest.mark.parametrize("potential", [-3.0, -0.5, 0.0, 0.5, 3.0])
@pytest.mark.parametrize("friction", [1e-8, 1e-27])
def test_charged_physical_limits(potential, friction):
    """Dimensional rates approach diffusion flux and ballistic collision flux."""
    from particula.dynamics.coagulation.charged_dimensionless_kernel import (
        get_dimensional_kernel,
        get_hard_sphere_kernel,
    )
    from particula.util.constants import BOLTZMANN_CONSTANT

    r, m, t = 50e-9, 1e-19, 298.15
    a, mr, fr = 2 * r, m / 2, friction / 2
    kinetic = 1 + potential if potential >= 0 else np.exp(potential)
    continuum = 1.0 if potential == 0 else potential / -np.expm1(-potential)
    kn = get_diffusive_knudsen_number(r, m, friction, potential, t)
    rate = get_dimensional_kernel(
        get_hard_sphere_kernel(kn), potential, a, mr, fr
    )
    expected = (
        4 * np.pi * a * BOLTZMANN_CONSTANT * t / fr * continuum
        if friction == 1e-8
        else np.pi
        * a
        * a
        * np.sqrt(8 * BOLTZMANN_CONSTANT * t / (np.pi * mr))
        * kinetic
    )
    np.testing.assert_allclose(rate, expected, rtol=0.003, atol=0)
