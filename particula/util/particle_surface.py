""" calculating the particle area
"""

import numpy as np
from particula.util.input_handling import in_radius, in_scalar


def area(**kwargs):
    """ Returns particle's surface area: 4 pi r^2 .

        Parameters:
            radius       (float) [m]
            density      (float) [kg/m^3] (default: 1000)
            area_factor  (float) [ ]      (default: 1)

        Returns:
                         (float) [m^2]
    """

    radius = kwargs.get("radius", None)
    area_factor = kwargs.get("area_factor", 1)

    radius = in_radius(radius)
    area_factor = in_scalar(area_factor)

    return (
        (4*np.pi) * (radius**2)
        * area_factor
    )
