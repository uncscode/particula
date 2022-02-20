""" Testing the dimensionless coagulation module.
"""

import pytest
from particula import u
from particula.util.dimensionless_coagulation import less_coag, full_coag


def test_less():
    """ testing hardsphere dimensionless coagulation
    """

    assert (
        less_coag(radius=1e-9, other_radius=1e-8).units ==
        u.dimensionless
    )
    assert (
        less_coag(radius=1e-9, other_radius=1e-8).magnitude ==
        pytest.approx(18, rel=1e0)
    )

    assert (
        less_coag(radius=1e-9, other_radius=1e-8).magnitude <=
        less_coag(radius=1e-9, other_radius=1e-9).magnitude
    )

    assert (
        less_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1
        ).magnitude >=
        less_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=0
        ).magnitude
    )

def test_full():
    """ testing hardsphere dimensioned coagulation
    """

    assert (
        full_coag(radius=1e-9, other_radius=1e-8).units ==
        u.m**3 / u.s
    )

    assert (
        full_coag(radius=1e-9, other_radius=1e-8).m_as(u.cm**3/u.s) ==
        pytest.approx(1.8e-8, rel=1e-1)
    )

    assert (
        full_coag(radius=1e-9, other_radius=1e-8).magnitude >=
        full_coag(radius=1e-9, other_radius=1e-9).magnitude
    )

    assert (
        full_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1
        ).magnitude >=
        full_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=0
        ).magnitude
    )
