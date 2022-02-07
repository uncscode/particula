""" calculate the mean free path of air
"""

import numpy as np
from particula import u
from particula.utils.environment_.calc_dynamic_viscosity import (
    dynamic_viscosity_air as dyn_vis_air,
)
from particula.utils import (
    GAS_CONSTANT as GAS_CON,
    MOLECULAR_WEIGHT_AIR as MOL_WT_AIR,
)


def mean_free_path_air(temperature, pressure) -> float:

    """ Returns the mean free path: [m]

        The mean free path is the average distance
        traveled by a molecule between collisions
        with other molecules present in a medium (air).
    """

    if isinstance(temperature, u.Quantity):
        temperature = temperature.to_base_units()
    else:
        temperature = u.Quantity(temperature, u.K)

    if isinstance(pressure, u.Quantity):
        pressure = pressure.to_base_units()
    else:
        pressure = u.Quantity(pressure, u.Pa)

    return (
        (2*dyn_vis_air(temperature)/pressure) /
        (8*MOL_WT_AIR/(np.pi*GAS_CON*temperature))**0.5
    ).to_base_units()
