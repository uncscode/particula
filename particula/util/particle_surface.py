""" calculating the particle area
"""

import numpy as np
from particula.util.input_handling import in_radius, in_scalar


def area(
    radius=None,
    area_factor=1,
):
    """ Returns particle's surface area: 4 pi r^2 .

        Parameters:
            radius       (float) [m]
            area_factor  (float) [ ]      (default: 1)

        Returns:
                         (float) [m^2]
    """

    radius = in_radius(radius)
    area_factor = in_scalar(area_factor)

    return 4*np.pi*(radius**2)*area_factor
