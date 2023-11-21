"""Test the species_density module if it runs without errors."""

import numpy as np
from particula.activity import species_density


def test_organic_density_estimate():
    """Test the organic_density_estimate function."""
    # Test case 1: Known values
    molar_mass = 180.16
    oxygen2carbon = 0.5
    hydrogen2carbon = 1.0
    nitrogen2carbon = 0.1

    assert species_density.organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon,
        nitrogen2carbon) > 0.0

    # Test case 2: Unknown hydrogen2carbon
    molar_mass = 200.0
    oxygen2carbon = 0.4
    nitrogen2carbon = 0.2

    assert species_density.organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        nitrogen2carbon) > 0.0

    # Test case 3: No nitrogen2carbon
    molar_mass = 150.0
    oxygen2carbon = 0.6
    hydrogen2carbon = 0.8

    assert species_density.organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon) > 0.0

    # Test case 4: Mass ratio conversion
    molar_mass = 18/250.0
    oxygen2carbon = 0.3
    hydrogen2carbon = 1.2
    nitrogen2carbon = 0.05

    assert species_density.organic_density_estimate(
        molar_mass,
        oxygen2carbon,
        hydrogen2carbon,
        nitrogen2carbon,
        mass_ratio_convert=True) > 0.0


def test_organic_array():
    """test the organic_array function."""
    molar_mass_array = [180.16, 200.0, 150.0]
    oxygen2carbon_array = [0.5, 0.4, 0.6]
    hydrogen2carbon_array = [1.0, 1.0, 0.8]
    nitrogen2carbon_array = None

    assert np.all(species_density.organic_array(
        molar_mass=molar_mass_array,
        oxygen2carbon=oxygen2carbon_array,
        hydrogen2carbon=hydrogen2carbon_array,
        nitrogen2carbon=nitrogen2carbon_array) > 0.0)
