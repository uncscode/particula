""" test the hardsphere approx utility
"""

import pytest

from particula.util.hardsphere_coagulation import hardsphere_coag_less
from particula import u

def test_hardsphere_coag_less():
    """ testing
    """

    assert hardsphere_coag_less(diff_knu=1e-9).u == u.dimensionless
    assert hardsphere_coag_less(diff_knu=0).m == pytest.approx(0)
