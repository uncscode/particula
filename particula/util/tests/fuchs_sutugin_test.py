""" test the fuchs_sutugin calc
"""

import pytest
from particula.util.fuchs_sutugin import fsc
from particula import u


def test_fuchs_sutugin():
    """ test
    """

    assert fsc(radius=1e-9).u == u.dimensionless
    assert fsc(radius=1).m == pytest.approx(0, abs=1e-7)
    assert fsc(radius=1e-10).m == pytest.approx(1, abs=1e-3)
    assert fsc(radius=[1, 2], alpha=1).m.shape == (2,)
    assert fsc(radius=[1, 2], alpha=[1, 2]).m.shape == (2, 2)
    assert fsc(radius=1, alpha=[1, 0.9]).m.shape == (1, 2)
    assert fsc(radius=1e-6, alpha=0).m == 0
