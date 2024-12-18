"""Test of the ratio function in the coulomb_enhancement module."""

import numpy as np
from particula.particles.properties import coulomb_enhancement


def test_ratio():
    """Test the ratio function in the coulomb_enhancement module."""
    # Test case 1: Single radius and charge
    radius = 1.0
    charge = 2
    temperature = 298.15
    expected_result = -1.1203242349904997e-07
    value = coulomb_enhancement.ratio(radius, charge, temperature)
    assert np.isclose(value, expected_result)

    # Test case 2: Array of radii and charges
    radius = np.array([1.0, 2.0, 3.0])
    charge = np.array([-2, 3, 4])
    expected_result = np.array(
        [
            [-1.12032423e-07, 1.12032423e-07, 1.12032423e-07],
            [1.12032423e-07, -1.26036476e-07, -1.34438908e-07],
            [1.12032423e-07, -1.34438908e-07, -1.49376565e-07],
        ]
    )
    value = coulomb_enhancement.ratio(radius, charge, temperature)
    assert np.allclose(value, expected_result)


def test_kinetic():
    """Test the kinetic function in the coulomb_enhancement module."""
    # Test case 1: Positive coulomb potential
    coulomb_potential = 0.5
    expected_result = 1.5
    value = coulomb_enhancement.kinetic(coulomb_potential)
    assert np.isclose(value, expected_result)

    # Test case 2: Negative coulomb potential
    coulomb_potential = -0.5
    expected_result = 0.60653066
    value = coulomb_enhancement.kinetic(coulomb_potential)
    assert np.isclose(value, expected_result)


def test_continuum():
    """Test the continuum function in the coulomb_enhancement module."""
    # Test case 1: Non-zero coulomb potential
    coulomb_potential = 0.5
    expected_result = 1.27074704
    value = coulomb_enhancement.continuum(coulomb_potential)
    assert np.isclose(value, expected_result)

    # Test case 2: Zero coulomb potential
    coulomb_potential = 0.0
    expected_result = 1.0
    value = coulomb_enhancement.continuum(coulomb_potential)
    assert np.isclose(value, expected_result)
