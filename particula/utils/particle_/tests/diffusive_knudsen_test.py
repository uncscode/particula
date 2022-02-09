""" test the diffusive knudsen number calculation
"""

import pytest
from particula import u
from particula.utils.particle_ import (
    diffusive_knudsen
)


def test_diffusive_knudsen():
    """ test the diffusive knudsen number calculation
    """

    assert (
        diffusive_knudsen(1e-9, 1e-8).units == u.dimensionless
    )
