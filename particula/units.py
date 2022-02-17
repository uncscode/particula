""" a simple utility to get rid of units if desired.
"""

from particula import u


def strip(quantity):

    """ This simple utility will return:
            * the magnitude of a quantity if it has units
            * the quantity itself if it does not have units
    """

    return (
        quantity.magnitude if isinstance(quantity, u.Quantity)
        else quantity
    )
