""" test the diffusive knudsen number calculation
"""

import pytest
from particula import u
from particula.utils.particle_ import (
    diffusive_knudsen
)


def test_diffusive_knudsen():
    """ test the diffusive knudsen number calculation

        What is the diffusive knudsen number?
        What does it mean?
        Why is calculated like this?
        How can we test it properly?
        Any rules of thumb here?
    """

    assert (
        diffusive_knudsen(1e-9, 1e-8).units ==
        u.dimensionless
    )
    assert (
        diffusive_knudsen(1e-9, 1e-8).magnitude ==
        pytest.approx(4, rel=1e0)
    )

    assert (
        diffusive_knudsen(1e-9, 1e-8).magnitude <=
        diffusive_knudsen(1e-9, 1e-9).magnitude
    )

    assert (
        diffusive_knudsen(
            1e-9, 1e-8, charge=-1, other_charge=1
        ).magnitude >=
        diffusive_knudsen(
            1e-9, 1e-8, charge=-1, other_charge=0
        ).magnitude
    )
