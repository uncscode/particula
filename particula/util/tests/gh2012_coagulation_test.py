""" test the hardsphere approx utility
"""

import pytest

from particula.util.gh2012_coagulation import gh2012_coag_less
from particula import u


def test_hardsphere_coag_less():
    """ testing
    """

    assert gh2012_coag_less(
        diff_knu=1e-9, cpr=1).u == u.dimensionless
    assert gh2012_coag_less(
        diff_knu=0, cpr=1).m == pytest.approx(0)
