""" testing the mean free path calculation
"""

import pytest
import numpy as np
from particula import u
from particula.util.mean_free_path import mfp, molecule_mean_free_path


def test_mfp():
    """ Testing the mean free path:

        1. test unitless and unit inputs
        2. test correct units
        3. test the calculated value (ref: ~66e-9 m mfp at sdt)
        4. test errors for invalid inputs

    """

    a_mfp = mfp(temperature=298*u.K, pressure=101325 * u.Pa)
    b_mfp = mfp(temperature=298, pressure=101325)
    c_mfp = mfp(temperature=298, pressure=101325, molecular_weight=0.03)

    assert a_mfp == b_mfp
    assert a_mfp.units == u.m
    assert a_mfp.magnitude == pytest.approx(66.4e-9, rel=1e-1)
    assert c_mfp <= a_mfp

    assert mfp(temperature=[200, 300]).m.shape == (2,)
    assert mfp(pressure=[1e5, 1.1e5]).m.shape == (2,)
    assert mfp(temperature=[200, 300], pressure=[1e5, 1.1e5]).m.shape == (2,)

    with pytest.raises(ValueError):
        mfp(temperature=5*u.m, pressure=101325*u.Pa)

    with pytest.raises(ValueError):
        mfp(temperature=298*u.K, pressure=5*u.m)

    with pytest.raises(ValueError):
        mfp(temperature=300*u.K,
            pressure=101325*u.Pa,
            molecular_weight=0.03*u.m/u.mol,
            )


def test_molecule_mean_free_path():
    """ Testing the mean free path of a molecule compare with mfp"""

    a_mfp = 6.52805868e-08  # at stp
    b_molecule_mfp = molecule_mean_free_path(
        temperature=298, pressure=101325, molar_mass=0.03
    )
    assert pytest.approx(a_mfp, rel=1e-6) == b_molecule_mfp


def test_dynamic_viscosity_provided():
    """ Test when dynamic viscosity is explicitly provided"""
    dynamic_viscosity = 5*1.78e-5  # 5x Value for air at stp
    result = molecule_mean_free_path(dynamic_viscosity=dynamic_viscosity)
    assert result > 0


def test_array_input():
    """Test when array inputs are provided for temperature, pressure,
    and molar mass"""
    molar_masses = np.array([0.028, 0.044])
    results = molecule_mean_free_path(molar_mass=molar_masses)
    # All calculated mean free paths should be positive
    assert all(results > 0)


@pytest.mark.parametrize("temperature", [None, -1, 'a'])
def test_invalid_temperature(temperature):
    """Test when invalid temperature values are provided to the function"""
    with pytest.raises((TypeError, ValueError)):
        molecule_mean_free_path(temperature=temperature)


@pytest.mark.parametrize("pressure", [None, -1, 'a'])
def test_invalid_pressure(pressure):
    """Test when invalid pressure values are provided to the function"""
    with pytest.raises((TypeError, ValueError)):
        molecule_mean_free_path(pressure=pressure)


@pytest.mark.parametrize("molar_mass",
                         [None, -1, 'a', np.array([0.028, -0.044])])
def test_invalid_molar_mass(molar_mass):
    """Test when invalid molar mass values are provided to the function"""
    with pytest.raises((TypeError, ValueError)):
        molecule_mean_free_path(molar_mass=molar_mass)
