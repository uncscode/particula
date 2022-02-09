""" testing the dimensionless coag calc
"""

import pytest
from particula import u

from particula.utils.coagulation_ import (
    DimensionlessCoagulation as CC,
)


def test_coag():

    """ testing coag
    """

    assert (
        CC(1e-9, 1e-8).hard_sphere().units == u.dimensionless
    )
    # check defaults:
    # how does this make sense?
    assert (
        CC(1e-9, 1e-8).hard_sphere().magnitude
        <=
        CC(1e-9, 1e-9).hard_sphere().magnitude
    )

    assert (
        CC(
            1e-9, 1e-8, charge=0, other_charge=0,
        ).hard_sphere().magnitude
        ==
        CC(
            1e-9, 1e-8, charge=0, other_charge=1,
        ).hard_sphere().magnitude
    )
    assert (
        CC(
            1e-8, 1e-9, charge=-1, other_charge=1,
        ).hard_sphere().magnitude
        ==
        CC(
            1e-9, 1e-8, charge=1, other_charge=-1,
        ).hard_sphere().magnitude
    )
