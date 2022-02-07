""" tests for the Environment class.

    1. Test getters for the Environment class.
    2. Test calculation for the dynamic viscosity of air.
    3. Test calculation for the mean free path of air.
"""

import pytest
from particula import u
from particula.aerosol_dynamics import environment

standard_environment = environment.Environment(
    temperature=298,  # * u.K,
    pressure=101325,  # * u.Pa,
)

standard_environment_default = environment.Environment()


def test_getters():
    """ tests the getters for the Environment class.
    """

    assert standard_environment.temperature() == 298 * u.K
    assert standard_environment.pressure() == 101325 * u.Pa


def test_getters_default():
    """ tests the getters for the (default) Environment class.
    """

    assert standard_environment_default.temperature() == 298 * u.K
    assert standard_environment_default.pressure() == 101325 * u.Pa


def test_dynamic_viscosity_air():
    """ tests calculation for the dynamic viscosity of air.
    """

    assert (
        standard_environment.dynamic_viscosity_air().magnitude ==
        pytest.approx(1.716e-5, rel=1e-1)
    )


def test_mean_free_path_air():
    """ test calculation for the mean free path of air.
    """

    assert (
        standard_environment.mean_free_path_air().magnitude ==
        pytest.approx(6.644e-8, rel=1e-1)
    )
