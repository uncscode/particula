""" Calculate Couolomb potential ratio
"""

import numpy as np
from particula.utils import (
    BOLTZMANN_CONSTANT,
    ELEMENTARY_CHARGE_VALUE,
    ELECTRIC_PERMITTIVITY,
)
from particula.utils import unitless


def coulomb_potential_ratio(
    charge, other_charge,
    radius, other_radius,
    temperature,
) -> float:

    """ Calculates the Coulomb potential ratio.

    Parameters:
        charges_array                       (np array)  [unitless]
        charge_other                        (float)     [unitless]
        radii_array                         (np array)  [m]
        radius_other                        (float)     [m]
        temperature                         (float)     [K]

    Returns:
        coulomb_potential_ratio             (array)     [unitless]
    """

    numerator = -1 * charge * other_charge * (
        unitless(ELEMENTARY_CHARGE_VALUE) ** 2
    )
    denominator = 4 * np.pi * unitless(ELECTRIC_PERMITTIVITY) * (
        radius + other_radius
    )

    return numerator / (
        denominator * unitless(BOLTZMANN_CONSTANT) * temperature
    )
