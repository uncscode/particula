""" calculate the mass of a particle
"""

import numpy as np
from particula import u


def particle_mass(radius, density=1000):
    """ Returns particle's mass.

        Parameters:
            radius                  (float) [m]
            density (default: 1000) (float) [kg/m^3]

        Returns:
            mass                    (float) [kg]
    """

    if isinstance(radius, u.Quantity):
        radius = radius.to_base_units()
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(density, u.Quantity):
        density = density.to_base_units()
    else:
        density = u.Quantity(density, u.kg/u.m**3)

    return density * (4*np.pi/3) * (radius**3)
