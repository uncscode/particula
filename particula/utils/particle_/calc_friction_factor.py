""" Calculate friction factor
"""

import numpy as np
from particula.utils import slip_correction


def friction_factor(
    radius,
    mean_free_path_air,
    dynamic_viscosity_air,
) -> float:

    """ Returns a particle's friction factor.

    Parameters:
        radii_array                 (np array)  [kg]
        mean_free_path_air          (float)     [m]
        dynamic_viscosity_air       (float)     [N*s/m]

    Returns:
        friction_factor             (array)     [N*s/m]

    Property of the particle's size and surrounding medium.
    Multiplying the friction factor by the fluid velocity
    yields the drag force on the particle.
    """

    return (
        6 * np.pi * dynamic_viscosity_air * radius /
        slip_correction(radius, mean_free_path_air)
    )
