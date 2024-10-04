""" Module contains the function for calculating the mean thermal speed
of particles in a fluid."""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.constants import BOLTZMANN_CONSTANT


def mean_thermal_speed(
    mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Returns the particles mean thermal speed. Due to the the impact
    of air molecules on the particles, the particles will have a mean
    thermal speed.

    Args
    ----
    mass : The per particle mass of the particles [kg].
    temperature : The temperature of the air [K].

    Returns
    -------
    The mean thermal speed of the particles [m/s].

    References
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Section 9.5.3 Mean Free Path of an Aerosol Particle Equation 9.87.
    """
    return np.sqrt((8 * BOLTZMANN_CONSTANT.m * temperature) / (np.pi * mass))
