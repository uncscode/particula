"""Module for aerodynamic mobility of a particle in a fluid.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def particle_aerodynamic_mobility(
    radius: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the aerodynamic mobility of a particle, defined as the ratio
    of the slip correction factor to the product of the dynamic viscosity of
    the fluid, the particle radius, and a slip correction constant derived.

    This mobility quantifies the ease with which a particle can move through
    a fluid.

    Arguments:
        radius : The radius of the particle (m).
        slip_correction_factor : The slip correction factor for the particle
            in the fluid (dimensionless).
        dynamic_viscosity : The dynamic viscosity of the fluid (Pa.s).

    Returns:
        The particle aerodynamic mobility (m^2/s).
    """
    return slip_correction_factor / (6 * np.pi * dynamic_viscosity * radius)
