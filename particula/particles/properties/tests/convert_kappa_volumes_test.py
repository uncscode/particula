"""Test the kappa convert module."""

import numpy as np

from particula.particles.properties.convert_kappa_volumes import (
    get_kappa_from_volumes,
    get_solute_volume_from_kappa,
    get_water_volume_from_kappa,
)


def test_kappa_volume_solute():
    """Test the kappa_volume_solute function."""
    # Test with water_activity = 0.95 and kappa = 0.4
    assert np.allclose(
        get_solute_volume_from_kappa(100, 0.4, 0.95), 11.6279, rtol=1e-4
    )

    # Test with water_activity = 1 and kappa = 0.0 (zero kappa correction)
    assert np.allclose(
        get_solute_volume_from_kappa(200, 0.0, 1), 0.0, rtol=1e-4
    )


def test_kappa_volume_water():
    """Test the kappa_volume_water function."""
    # Test with water_activity = 0.95 and kappa = 0.4
    assert np.allclose(
        get_water_volume_from_kappa(5, 0.5, 0.99), 247.49, rtol=1e-4
    )


def test_kappa_volume_solute_zero_correction():
    """Test the kappa_volume_solute_zero_correction function."""
    # Test with water_activity = 1 and kappa = 0.0 (zero correction)
    assert np.allclose(
        get_solute_volume_from_kappa(5, 0.5, 1), 1.1258e16, rtol=1e14
    )


def test_kappa_from_volume():
    """Test the kappa_from_volume function."""
    # Test with water_activity = 0.95 and kappa = 0.4
    assert np.allclose(get_kappa_from_volumes(100, 200, 0.5), 2.0, rtol=1e-2)

    # Test with water_activity = 1 (zero kappa correction)
    assert np.allclose(
        get_kappa_from_volumes(100, 200, 1), 4.44089e-16, rtol=1e-17
    )
