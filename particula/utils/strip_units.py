""" utility to strip units from unit-registry quantities
"""


def make_unitless(quantity):
    """ take the magnitude
    """
    return quantity.magnitude
