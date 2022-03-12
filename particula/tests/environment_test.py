""" testing the environment class.
"""

import pytest
from particula import environment, u

inputs = {
    "temperature": 298.15,
    "pressure": 101325
}
inputs33 = {
    "temperature": [298.15, 299, 300],
    "pressure": [101325, 101330, 101335]
}
standard_environment = environment.Environment(
    **inputs
)


def test_getters():
    """ Tests the getters for the Environment class.
    """
    assert standard_environment.temperature == inputs["temperature"] * u.K
    assert standard_environment.pressure == inputs["pressure"] * u.Pa


def test_dynamic_viscosity_air():
    """ Tests the dynamic viscosity.
    """
    assert (
        standard_environment.dynamic_viscosity().magnitude ==
        pytest.approx(1.716e-5, rel=1e-1)
    )


def test_mean_free_path_air():
    """ Test the mean free path.
    """
    assert (
        standard_environment.mean_free_path().magnitude ==
        pytest.approx(6.644e-8, rel=1e-1)
    )


def test_shapes():
    """ testing the shapes, allowing list inputs
    """

    assert (
        environment.Environment(**inputs33).pressure.m.shape
        ==
        (len(inputs33["pressure"]),)
    )
    assert (
        environment.Environment(**inputs33).dynamic_viscosity().m.shape
        ==
        (len(inputs33["pressure"]),)
    )
