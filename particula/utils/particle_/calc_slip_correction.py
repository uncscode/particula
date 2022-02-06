""" calculate slip correction
"""

import numpy as np
from particula.utils import knudsen_number


def slip_correction_factor(radius, mean_free_path_air) -> float:

    """ Returns particle's Cunningham slip correction factor.

    Parameters:
        radii_array                 (float) [m]
        mean_free_path_air          (float) [m]

    Returns:
        slip_correction_factor      (float) [unitless]

    Dimensionless quantity accounting for non-continuum effects
    on small particles. It is a deviation from Stokes' Law.
    Stokes assumes a no-slip condition that is not correct at
    high Knudsen numbers. The slip correction factor is used to
    calculate the friction factor.
    """

    return 1 + knudsen_number(radius, mean_free_path_air) * (
        1.257 + 0.4*np.exp(
            -1.1/knudsen_number(radius, mean_free_path_air)
        )
    )
