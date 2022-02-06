""" calculate the knudsen number
"""

def knudsen_number(radii_array, mean_free_path_air) -> float:

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

    return mean_free_path_air / radii_array
