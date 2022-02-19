""" Testing the dimensionless coagulation module.
"""

import pytest
from particula import u
from particula.util.dimensionless_coagulation import hardsphere_coag 


def test_hsc():
    """ testing hardsphere coagulation
    """

    assert (
        hardsphere_coag(radius=1e-9, other_radius=1e-8).units ==
        u.dimensionless
    )
    assert (
        hardsphere_coag(radius=1e-9, other_radius=1e-8).magnitude ==
        pytest.approx(18, rel=1e0)
    )

    assert (
        hardsphere_coag(radius=1e-9, other_radius=1e-8).magnitude <=
        hardsphere_coag(radius=1e-9, other_radius=1e-9).magnitude
    )

    assert (
        hardsphere_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1
        ).magnitude >=
        hardsphere_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=0
        ).magnitude
    )
