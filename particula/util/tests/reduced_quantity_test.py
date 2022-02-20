""" testing the utility calculating reduced quantities
"""

import numpy as np
import pytest
from particula import u
from particula.util.reduced_quantity import reduced_quantity


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

    assert reduced_quantity([1, 2], [2, 3]).shape == (2, )

    assert reduced_quantity(
        np.array([1, 2]), np.transpose([np.array([2, 3])])
    ).shape == (2, 2)

    with pytest.raises(TypeError):
        reduced_quantity(1, 2 * u.kg)
    with pytest.raises(TypeError):
        reduced_quantity(1 * u.kg, 2 * u.m)
