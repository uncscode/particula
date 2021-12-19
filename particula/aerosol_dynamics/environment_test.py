"""
Tests for the Environment class.
"""

import pytest

from particula.aerosol_dynamics import environment

from particula.aerosol_dynamics import u

standard_environment = environment.Environment(
    temperature=298,  # * u.K,
    pressure=101325,  # * u.Pa,
)


def test_getters():
    """
    Tests the getters for the Environment class.
    """
    assert standard_environment.temperature() == 298 * u.K
    assert standard_environment.pressure() == 101325 * u.Pa


def test_dynamic_viscosity_air():
    """
    Tests that the calculation for the dynamic viscosity of air works.
    """
    assert (
        standard_environment.dynamic_viscosity_air().magnitude ==
        pytest.approx(1.716e-5, rel=1e-1)
    )


def test_mean_free_path_air():
    """
    Test that the calculation for the mean free path of air works.
    """
    assert (
        standard_environment.mean_free_path_air().magnitude ==
        pytest.approx(6.644e-8, rel=1e-1)
    )
