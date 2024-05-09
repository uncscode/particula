"""
The coagulation gain, loss, net rate calculations.

These are separate from the strategies to isolate behavior, calculation
definitions from the usages. Which allows for easier testing and
charry-picking of code snips."""

from typing import Union
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import RectBivariateSpline  # type: ignore


def loss_rate(
    radius: Union[float, NDArray[np.float_]],
    concentration: Union[float, NDArray[np.float_]],
    kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the coagulation loss rate, via the integration method.

    Args:
    -----
    - radius: The radius of the particles.
    - concentraiton: The distribution of particles.
    - kernel: The coagulation kernel.

    Returns:
    --------
    - The coagulation loss rate.

    References:
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    return np.sum(concentration * kernel * np.transpose(concentration), axis=0)


def gain_rate(
    radius: Union[float, NDArray[np.float_]],
    concentration: Union[float, NDArray[np.float_]],
    kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the coagulation gain rate, via the integration method.

    Args:
    -----
    - radius: The radius of the particles.
    - concentration: The distribution of particles.
    - kernel: The coagulation kernel.

    Returns:
    --------
    - The coagulation gain rate.

    References:
    ----------
    - This equation necessitates the use of a for-loop due to the
    convoluted use of different radii at different stages. This is the
    most expensive step of all coagulation calculations. Using
    `RectBivariateSpline` accelerates this significantly.
    - Note, to estimate the kernel and distribution at
    (other_radius**3 - some_radius**3)*(1/3)
    we use interporlation techniques.
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    # gain
    # 0.5* C_i * C_j * K_ij
    gain_matrix = 0.5 * kernel * concentration * np.transpose(concentration)

    # select the diagonal to sum over, skip the first one.
    # rotate matrix
    flipped_matrix = np.fliplr(gain_matrix)

    # Generate offsets
    offsets = len(flipped_matrix) - 1 - np.arange(len(flipped_matrix))

    # Calculate traces of each diagonal
    gain = np.array([np.trace(flipped_matrix, offset=off)
                     for off in offsets[:-1]])
    # prepend the first element
    return np.insert(gain, 0, 0)


def net_rate(
    radius: Union[float, NDArray[np.float_]],
    concentration: Union[float, NDArray[np.float_]],
    kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the coagulation net rate, via the integration method.

    Args:
    -----
    - radius: The radius of the particles.
    - dist: The distribution of particles.
    - kernel: The coagulation kernel.

    Returns:
    --------
    - The coagulation net rate.

    References:
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    return gain_rate(radius, concentration, kernel) - loss_rate(
        radius, concentration, kernel)
