""" A quick test of strip_units.unitless utility
"""

from particula import u
from particula.utils._common.strip_units import unitless


def test_strip_units():

    """ testing getting rid of units of an input quantity

        Tests:
            * see if a quantity with units is stripped
            * see if a quantity without units is returned
            * see if manipulation of quantities is also ok
    """

    assert unitless(1) == 1
    assert unitless(1 * u.kg) == 1
    assert unitless(5 * u.m) == 5
    assert unitless((5 * u.kg) * (1 * u.m)) == 5
