""" Calculate coulomb-base enhancements
"""

import numpy as np
from particula.utils import (
    coulomb_ratio
)


def coulomb_enhancement_kinetic_limit(
    charge, other_charge,
    radius, other_radius,
    temperature,
) -> float:

    """ Coulombic coagulation enhancement kinetic limit.

    Parameters:
        charge          (np array)  [unitless]
        other_charge    (float)     [unitless]
        radius          (np array)  [m]
        other_radiu     (float)     [m]
        temperature     (float)     [K]

    Returns:
                        (array)     [unitless]
    """

    ret = coulomb_ratio(
        charge, other_charge,
        radius, other_radius,
        temperature,
    )

    return (
        1 + ret if ret >= 0 else np.exp(ret)
    )


def coulomb_enhancement_continuum_limit(
    charge, other_charge,
    radius, other_radius,
    temperature
) -> float:

    """ Coulombic coagulation enhancement continuum limit.

    Parameters:
        charge          (np array)  [unitless]
        other_charge    (float)     [unitless]
        radius          (np array)  [m]
        other_radius    (float)     [m]
        temperature     (float)     [K]

    Returns:
                        (array)     [unitless]
    """

    ret = coulomb_ratio(
        charge, other_charge,
        radius, other_radius,
        temperature,
    )

    return ret/(1-np.exp(-1*ret)) if ret != 0 else 1
