""" calculate the knudsen number
"""

from particula import u

def knudsen_number(radius, mean_free_path_air) -> float:

    """ Returns particle's Knudsen number.

    Parameters:
        radii_array         (float)     [m]
        mean_free_path_air  (float)     [m]

    Returns:
        knudsen_number      (float)     [unitless]

    The Knudsen number reflects the relative length scales of
    the particle and the suspending fluid (air, water, etc.).
    This is calculated by the mean free path of the medium
    divided by the particle radius.
    """

    if isinstance(radius, u.Quantity):
        radius = radius.to_base_units()
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(mean_free_path_air, u.Quantity):
        mean_free_path_air = mean_free_path_air.to_base_units()
    else:
        mean_free_path_air = u.Quantity(mean_free_path_air, u.kg/u.m**3)

    return mean_free_path_air / radius
