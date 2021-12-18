"""
testing new env class
"""

import pytest

from aerosol_dynamics import environment

from . import u


standard_environment = environment.Environment(
    temperature=298 * u.K,
    pressure=101325 * u.Pa,
)

def test_getters():
    """testing getters
    """
    assert standard_environment.temperature() == 298 * u.K
    assert standard_environment.pressure() == 101325 * u.Pa

def test_dynamic_viscosity_air():
    """testing dynamic visc
    """
    assert (standard_environment.dynamic_viscosity_air().magnitude ==
        pytest.approx(1.836e-5, rel=1e-3)
        )

def test_mean_free_path_air():
    """testing mfp
    """
    assert (standard_environment.mean_free_path_air().magnitude ==
        pytest.approx(6.644e-8, rel=1e-3)
        )
