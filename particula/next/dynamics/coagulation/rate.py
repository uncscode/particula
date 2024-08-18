"""
The coagulation gain, loss, net rate calculations.

These are separate from the strategies to isolate behavior, calculation
definitions from the usages. Which allows for easier testing and
charry-picking of code snips.

The are discrete and continuous versions of the gain and loss rates.
The discrete versions are calculated via summation, while the continuous
versions are calculated via integration.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np
from scipy.interpolate import RectBivariateSpline  # type: ignore


def discrete_loss(
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the coagulation loss rate, via the summation method.

    Args:
        concentraiton : The distribution of particles.
        kernel : The coagulation kernel.

    Returns:
        The coagulation loss rate.

    References:
        Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
        physics, Chapter 13 Equations 13.61
    """
    return np.sum(kernel * np.outer(concentration, concentration), axis=0)


def discrete_gain(
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the coagulation gain rate, via the summation method.

    Args:
    -----
    - concentration: The distribution of particles.
    - kernel: The coagulation kernel.

    Returns:
    --------
    - The coagulation gain rate.

    References:
    ----------
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    # gain
    # 0.5* C_i * C_j * K_ij
    # outer replaces, concentration * np.transpose([concentration])
    gain_matrix = 0.5 * kernel * np.outer(concentration, concentration)

    # select the diagonal to sum over, skip the first one, size as no particles
    # will coagulate into it.
    # rotate matrix
    flipped_matrix = np.fliplr(gain_matrix)

    # Generate offsets
    offsets = len(flipped_matrix) - 1 - np.arange(len(flipped_matrix))

    # Calculate traces of each diagonal
    gain = np.array(
        [np.trace(flipped_matrix, offset=off) for off in offsets[:-1]]
    )
    # prepend the first element, as zero
    return np.insert(gain, 0, 0)


def continuous_loss(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the coagulation loss rate, via the integration method.

    Args:
    -----
    - radius: The radius of the particles.
    - concentration: The distribution of particles.
    - kernel: The coagulation kernel.

    Returns:
    --------
    - The coagulation loss rate.

    References:
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    # concentration (n,) and kernel (n,n)
    return concentration * np.trapz(y=kernel * concentration, x=radius)


def continuous_gain(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
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
    - Seinfeld, J. H., & Pandis, S. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """

    # outer replaces, concentration * np.transpose([concentration])
    interp = RectBivariateSpline(
        x=radius, y=radius, z=kernel * np.outer(concentration, concentration)
    )

    dpd = np.linspace(0, radius / 2 ** (1 / 3), radius.size)  # type: ignore
    dpi = (np.transpose(radius) ** 3 - dpd**3) ** (1 / 3)

    return radius**2 * np.trapz(
        interp.ev(dpd, dpi) / dpi**2, dpd, axis=0  # type: ignore
    )
