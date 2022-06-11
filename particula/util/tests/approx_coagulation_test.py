
""" test the coag approx utility
"""

import pytest

from particula.util.approx_coagulation import approx_coag_less
from particula import u


def test_approx_coag_less():
    """ testing
    """

    assert approx_coag_less(
        diff_knu=1e-9, cpr=1, approx="dy2007").u == u.dimensionless
    assert approx_coag_less(
        diff_knu=1e-32, cpr=1, approx="dy2007").m == pytest.approx(0)

    assert approx_coag_less(
        diff_knu=1e-9, approx="hardsphere").u == u.dimensionless
    assert approx_coag_less(
        diff_knu=0, approx="hardsphere").m == pytest.approx(0)
    assert approx_coag_less(
        diff_knu=1e-32, approx="hardsphere").m == pytest.approx(0)

    assert approx_coag_less(
        diff_knu=1e-9, cpr=1, approx="gk2008").u == u.dimensionless
    assert approx_coag_less(
        diff_knu=1e-32, cpr=1, approx="gk2008").m == pytest.approx(0)

    assert approx_coag_less(
        diff_knu=1e-9, cpr=1, approx="gh2012").u == u.dimensionless
    assert approx_coag_less(
        diff_knu=0, cpr=1, approx="gh2012").m == pytest.approx(0)
    assert approx_coag_less(
        diff_knu=1e-32, cpr=1, approx="gh2012").m == pytest.approx(0)

    assert approx_coag_less(
        diff_knu=1e-9, cpr=1, approx="cg2019").u == u.dimensionless
    assert approx_coag_less(
        diff_knu=1e-32, cpr=1, approx="cg2019").m == pytest.approx(0)
