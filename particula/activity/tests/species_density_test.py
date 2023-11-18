"""Test the species_density module."""

from particula.activity.species_density import organic_density_estimate


def test_organic_density_estimate():
    """Test the organic_density_estimate function."""
    # Test case 1: Known values
    molar_mass = 180.16
    oxygen2carbon = 0.5
    hydrogen2carbon = 1.0
    nitrogen2carbon = 0.1
    expected_density = 1.23456789

    assert organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon,
        nitrogen2carbon) == expected_density

    # Test case 2: Unknown hydrogen2carbon
    molar_mass = 200.0
    oxygen2carbon = 0.4
    hydrogen2carbon = -1.0
    nitrogen2carbon = 0.2
    expected_density = 1.3456789

    assert organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon,
        nitrogen2carbon) == expected_density

    # Test case 3: No nitrogen2carbon
    molar_mass = 150.0
    oxygen2carbon = 0.6
    hydrogen2carbon = 0.8
    expected_density = 1.123456789

    assert organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon) == expected_density

    # Test case 4: Mass ratio conversion
    molar_mass = 250.0
    oxygen2carbon = 0.3
    hydrogen2carbon = 1.2
    nitrogen2carbon = 0.05
    expected_density = 1.456789

    assert organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon,
        nitrogen2carbon,
        mass_ratio_convert=True) == expected_density
