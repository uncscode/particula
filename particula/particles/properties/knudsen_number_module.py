""" Module for calculating Knudsen number
"""

from typing import Union
import logging
from numpy.typing import NDArray
import numpy as np

logger = logging.getLogger("particula")


def calculate_knudsen_number(
    mean_free_path: Union[float, NDArray[np.float64]],
    particle_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Knudsen number using the mean free path of the gas and the
    radius of the particle. The Knudsen number is a dimensionless number that
    indicates the regime of gas flow relative to the size of particles.

    Args:
    -----
    - mean_free_path (Union[float, NDArray[np.float64]]): The mean free path of
    the gas molecules [meters (m)].
    - particle_radius (Union[float, NDArray[np.float64]]): The radius of the
    particle [meters (m)].

    Returns:
    --------
    - Union[float, NDArray[np.float64]]: The Knudsen number, which is the
    ratio of the mean free path to the particle radius.

    References:
    -----------
    - For more information at https://en.wikipedia.org/wiki/Knudsen_number
    """
    if not isinstance(mean_free_path, (float, np.ndarray)) or not isinstance(
        particle_radius, (float, np.ndarray)
    ):
        message = "The input must be a float or a numpy array"
        logger.error(message)
        raise TypeError(message)
    if isinstance(mean_free_path, float) and isinstance(
        particle_radius, (float, np.ndarray)
    ):
        return mean_free_path / particle_radius

    if isinstance(mean_free_path, np.ndarray) and isinstance(
        particle_radius, np.ndarray
    ):
        if (
            (mean_free_path.size == particle_radius.size)
            or (mean_free_path.size == 1)
            or (particle_radius.size == 1)
        ):
            return mean_free_path / particle_radius

        # Reshape to (n, 1) and vector_b to (1, m) to broadcast (n, m)
        particle_radius = particle_radius[
            :, np.newaxis
        ]  # Adds a new axis, creating a column vector
        mean_free_path = mean_free_path[
            np.newaxis, :
        ]  # Adds a new axis, creating a row vector
        return mean_free_path / particle_radius

    message = (
        "The input arrays must have the same size"
        " or one of them must have size 1"
    )
    logger.error(message)
    raise ValueError(message)
