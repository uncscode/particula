"""Test for activity_coefficients function."""

import numpy as np

from particula.activity.activity_coefficients import bat_activity_coefficients


def test_activity_coefficients():
    """Test for activity_coefficents function."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_with_functional_group():
    """Test for activity_coefficents function with functional group."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.linspace(0.1, 1, 10)
    oxygen2carbon = 0.3
    density = 2500
    functional_group = "alcohol"

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=functional_group,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_edge_cases():
    """Test for activity_coefficents function with edge cases."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.array([1e-12, 1.0])
    oxygen2carbon = 0.3
    density = 2500

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_with_arrays():
    """Test for activity_coefficents function with np.array inputs."""
    molar_mass_ratio = np.array([18.016 / 250, 18.016 / 200])
    organic_mole_fraction = np.array([0.2, 0.8])
    oxygen2carbon = np.array([0.3, 0.4])
    density = np.array([2500, 2400])

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)


def test_activity_coefficients_with_mixed_inputs():
    """Test for activity_coefficents function with mixed inputs."""
    molar_mass_ratio = 18.016 / 250
    organic_mole_fraction = np.array([0.1, 0.9])
    oxygen2carbon = 0.3
    density = np.array([2500, 2400])

    activity_coefficients = bat_activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        organic_mole_fraction=organic_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,
    )
    assert np.all(activity_coefficients[0] >= 0)
    assert np.all(activity_coefficients[1] >= 0)
    assert np.all(activity_coefficients[2] >= 0)
    assert np.all(activity_coefficients[3] >= 0)
