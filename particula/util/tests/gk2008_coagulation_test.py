""" test the hardsphere approx utility
"""

import pytest

from particula.util.gk2008_coagulation import gk2008_coag_less
from particula import u


def test_hardsphere_coag_less():
    """ testing
    """

    assert gk2008_coag_less(
        diff_knu=1e-9, cpr=1).u == u.dimensionless
    assert gk2008_coag_less(
        diff_knu=1e-32, cpr=1).m == pytest.approx(0)
