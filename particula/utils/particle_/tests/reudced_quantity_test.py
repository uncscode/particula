""" testing the utility calculating reduced quantities
"""

import pytest
from particula import u
from particula.utils.particle_.calc_reduced_quantity import (
    reduced_quantity
)


def test_reduced_quantity():
    """ Test that the reduced quantity is calculated correctly.
    """

    assert (
        reduced_quantity(1, 2) ==
        pytest.approx(2/3)
    )
    assert (
        reduced_quantity(1 * u.kg, 2 * u.kg).magnitude ==
        pytest.approx(2/3)
    )

    with pytest.raises(TypeError):
        reduced_quantity(1, 2 * u.kg)
    with pytest.raises(TypeError):
        reduced_quantity(1 * u.kg, 2 * u.m)
