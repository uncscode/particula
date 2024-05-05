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
    distribution: Union[float, NDArray[np.float_]],
    kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the coagulation loss rate, via the integration method.

    Args:
    -----
    - radius: The radius of the particles.
    - dist: The distribution of particles.
    - kernel: The coagulation kernel.

    Returns:
    --------
    - The coagulation loss rate.

    References:
    ----------
    Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    return distribution * np.trapz(kernel * distribution, radius)


def gain_rate(
    radius: Union[float, NDArray[np.float_]],
    distribution: Union[float, NDArray[np.float_]],
    kernel: NDArray[np.float_]
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the coagulation gain rate, via the integration method.

    Args:
    -----
    - radius: The radius of the particles.
    - dist: The distribution of particles.
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
    interp = RectBivariateSpline(
        x=radius,
        y=radius,
        z=kernel * distribution * np.transpose(distribution)
    )

    dpd = np.linspace(0, radius / 2**(1 / 3), radius.size)
    dpi = ((np.transpose(radius))**3 - dpd**3)**(1 / 3)

    return radius**2 * np.trapz(interp.ev(dpd.m, dpi.m) / dpi**2, dpd, axis=0)


def net_rate(
    radius: Union[float, NDArray[np.float_]],
    distribution: Union[float, NDArray[np.float_]],
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
    return gain_rate(radius, distribution, kernel) - loss_rate(
        radius, distribution, kernel)
