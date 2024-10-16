"""Tests for the diffusion coefficient property."""

import numpy as np
import pytest
from particula.particles.properties.diffusion_coefficient import (
    particle_diffusion_coefficient,
)


def test_particle_diffusion_coefficient_float():
    """Test the diffusion coefficient for a single float value."""
    temperature = 298.15
    particle_aerodynamic_mobility = 1e-9
    expected_diffusion_coefficient = 4.116404993e-30
    result = particle_diffusion_coefficient(
        temperature, particle_aerodynamic_mobility
    )
    assert result == pytest.approx(  # type: ignore
        expected_diffusion_coefficient
    )


def test_particle_diffusion_coefficient_array():
    """Test the diffusion coefficient for a numpy array."""
    temperature = np.array([298.15, 300.0, 310.0])
    particle_aerodynamic_mobility = np.array([1e-9, 2e-9, 3e-9])
    expected_diffusion_coefficient = np.array(
        [1.380649e-32, 2.761298e-32, 4.141947e-32]
    )
    result = particle_diffusion_coefficient(
        temperature, particle_aerodynamic_mobility
    )
    assert np.allclose(result, expected_diffusion_coefficient)


def test_particle_diffusion_coefficient_invalid():
    """Test the diffusion coefficient for invalid input."""
    temperature = "invalid"
    particle_aerodynamic_mobility = 1e-9
    with pytest.raises(TypeError):
        particle_diffusion_coefficient(
            temperature, particle_aerodynamic_mobility  # type: ignore
        )
