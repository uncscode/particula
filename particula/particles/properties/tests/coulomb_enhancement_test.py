"""Test of the ratio function in the coulomb_enhancement module."""

import numpy as np

from particula.particles.properties import coulomb_enhancement


def test_ratio_lower_limit():
    """Test the ratio function for ratio_lower_limit functionality."""
    # Test case where the calculated ratio is below the lower limit
    radius = 1e-9  # Very small radius to increase the Coulomb potential
    charge = 100  # Large charge to increase the Coulomb potential
    temperature = 298.15
    expected_result = -200  # Default ratio_lower_limit

    value = coulomb_enhancement.get_coulomb_enhancement_ratio(
        radius, charge, temperature
    )
    assert np.isclose(value, expected_result)

    # Test with a custom ratio_lower_limit
    custom_limit = -100
    expected_custom_result = custom_limit
    value_custom = coulomb_enhancement.get_coulomb_enhancement_ratio(
        radius, charge, temperature, ratio_lower_limit=custom_limit
    )
    assert np.isclose(value_custom, expected_custom_result)


def test_ratio():
    """Test the ratio function in the coulomb_enhancement module."""
    # Test case 1: Single radius and charge
    radius = 1.0
    charge = 2
    temperature = 298.15
    expected_result = -1.1203242349904997e-07
    value = coulomb_enhancement.get_coulomb_enhancement_ratio(
        radius, charge, temperature
    )
    assert np.isclose(value, expected_result)

    # Ensure existing test cases do not trigger the ratio_lower_limit
    # so they test the normal functionality of the ratio function.

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
    value = coulomb_enhancement.get_coulomb_enhancement_ratio(
        radius, charge, temperature
    )
    assert np.allclose(value, expected_result)


def test_kinetic():
    """Test the kinetic function in the coulomb_enhancement module."""
    # Test case 1: Positive coulomb potential
    coulomb_potential = 0.5
    expected_result = 1.5
    value = coulomb_enhancement.get_coulomb_kinetic_limit(coulomb_potential)
    assert np.isclose(value, expected_result)

    # Test case 2: Negative coulomb potential
    coulomb_potential = -0.5
    expected_result = 0.60653066
    value = coulomb_enhancement.get_coulomb_kinetic_limit(coulomb_potential)
    assert np.isclose(value, expected_result)


def test_continuum():
    """Test the continuum function in the coulomb_enhancement module."""
    # Test case 1: Non-zero coulomb potential
    coulomb_potential = 0.5
    expected_result = 1.27074704
    value = coulomb_enhancement.get_coulomb_continuum_limit(coulomb_potential)
    assert np.isclose(value, expected_result)

    # Test case 2: Zero coulomb potential
    coulomb_potential = 0.0
    expected_result = 1.0
    value = coulomb_enhancement.get_coulomb_continuum_limit(coulomb_potential)
    assert np.isclose(value, expected_result)
