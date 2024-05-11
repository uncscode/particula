""" Module for calculating Knudsen number
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def calculate_knudsen_number(
    mean_free_path: Union[float, NDArray[np.float_]],
    particle_radius: Union[float, NDArray[np.float_]]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the Knudsen number using the mean free path of the gas and the
    radius of the particle. The Knudsen number is a dimensionless number that
    indicates the regime of gas flow relative to the size of particles.

    Args:
    -----
    - mean_free_path (Union[float, NDArray[np.float_]]): The mean free path of
    the gas molecules [meters (m)].
    - particle_radius (Union[float, NDArray[np.float_]]): The radius of the
    particle [meters (m)].

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The Knudsen number, which is the
    ratio of the mean free path to the particle radius.

    References:
    -----------
    - For more information at https://en.wikipedia.org/wiki/Knudsen_number
    """
    return mean_free_path / particle_radius
