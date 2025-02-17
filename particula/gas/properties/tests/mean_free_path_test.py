""" testing the mean free path calculation
"""

import pytest
import numpy as np
from particula.gas.properties import molecule_mean_free_path


def test_molecule_mean_free_path():
    """Testing the mean free path of a molecule compare with mfp"""

    a_mfp = 6.52805868e-08  # at stp
    b_molecule_mfp = molecule_mean_free_path(
        temperature=298, pressure=101325, molar_mass=0.03
    )
    assert pytest.approx(a_mfp, rel=1e-6) == b_molecule_mfp


def test_dynamic_viscosity_provided():
    """Test when dynamic viscosity is explicitly provided"""
    dynamic_viscosity = 5 * 1.78e-5  # 5x Value for air at stp
    result = molecule_mean_free_path(dynamic_viscosity=dynamic_viscosity)
    assert result > 0


def test_array_input():
    """Test when array inputs are provided for temperature, pressure,
    and molar mass"""
    molar_masses = np.array([0.028, 0.044])
    results = molecule_mean_free_path(molar_mass=molar_masses)
    # All calculated mean free paths should be positive
    assert np.all(results > 0)


@pytest.mark.parametrize("temperature", [None, -1, "a"])
def test_invalid_temperature(temperature):
    """Test when invalid temperature values are provided to the function"""
    with pytest.raises((TypeError, ValueError)):
        molecule_mean_free_path(temperature=temperature)


@pytest.mark.parametrize("pressure", [None, -1, "a"])
def test_invalid_pressure(pressure):
    """Test when invalid pressure values are provided to the function"""
    with pytest.raises((TypeError, ValueError)):
        molecule_mean_free_path(pressure=pressure)


@pytest.mark.parametrize(
    "molar_mass", [None, -1, "a", np.array([0.028, -0.044])]
)
def test_invalid_molar_mass(molar_mass):
    """Test when invalid molar mass values are provided to the function"""
    with pytest.raises((TypeError, ValueError)):
        molecule_mean_free_path(molar_mass=molar_mass)
