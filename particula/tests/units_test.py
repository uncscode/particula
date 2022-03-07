""" a quick test of units.strip utility
"""

from particula import u
from particula.units import strip as us


def test_units():
    """ Testing getting rid of units of an input quantity:
            * see if a quantity with units is stripped
            * see if a quantity without units is returned
            * see if manipulation of quantities is also ok
    """

    assert us(1) == 1

    assert us(1 * u.kg) == 1

    assert us(5 * u.m) == 5

    assert us((5 * u.kg) * (1 * u.m)) == 5
