""" quick test of strip_units functionality
"""

from particula import u
from particula.utils import unitless

# define some quantities and strip their units
a_quantity = 1 * u.m
b_quantity = 1 * u.kg
c_quantity = a_quantity * b_quantity


def test_strip_units():
    """ test that the strip_units function works
    """
    assert unitless(a_quantity) == 1
    assert unitless(b_quantity) == 1
    assert unitless(c_quantity) == 1
