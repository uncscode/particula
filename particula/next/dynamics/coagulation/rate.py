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

    Arguments:
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
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the coagulation gain rate, via the integration method, by
    converting to a continuous distribution.

    Arguments:
        radius : The radius of the particles.
        concentration : The distribution of particles.
        kernel : The coagulation kernel.

    Returns:
        The coagulation gain rate.

    References:
    ----------
    - This equation necessitates the use of a for-loop due to the
    convoluted use of different radii at different stages. This is the
    most expensive step of all coagulation calculations. Using
    `RectBivariateSpline` accelerates this significantly.
    - Note, to estimate the kernel and distribution at
    (other_radius**3 - some_radius**3)*(1/3) we use interporlation techniques.
    - Seinfeld, J. H., & Pandis, S. (2016). Atmospheric chemistry and
    physics, Chapter 13 Equations 13.61
    """
    # Calculate bin widths (delta_x_array)
    delta_x_array = np.diff(
        radius, append=2 * radius[-1] - radius[-2])  # type: ignore

    # Convert concentration to a probability density function (PDF)
    concentration_pdf = concentration / delta_x_array

    # Prepare interpolation for continuous distribution
    interp = RectBivariateSpline(
        x=radius,
        y=radius,
        z=kernel * np.outer(concentration_pdf, concentration_pdf),
    )

    # Define dpd and dpi for integration
    # integration variable
    dpd = np.linspace(0, radius / 2 ** (1 / 3), radius.size)  # type: ignore
    # adjusted for broadcasting
    dpi = (np.transpose(radius) ** 3 - dpd**3) ** (1 / 3)

    # Compute gain using numerical integration
    gain = radius**2 * np.trapz(
        interp.ev(dpd, dpi) / dpi**2, dpd, axis=0)  # type: ignore

    # Convert back to original scale (from PDF to PMF)
    return gain * delta_x_array


def continuous_loss(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the coagulation loss rate, via the integration method.

    Arguments:
        radius : The radius of the particles.
        concentration : The distribution of particles.
        kernel : The coagulation kernel.

    Returns:
        The coagulation loss rate.

    References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
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

    Arguments:
        radius : The radius of the particles.
        concentration : The distribution of particles.
        kernel : The coagulation kernel.

    Returns:
        The coagulation gain rate.

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
    # continuous distribution, kernel (n,n)
    # outer replaces, concentration * np.transpose([concentration])
    interp = RectBivariateSpline(
        x=radius, y=radius, z=kernel * np.outer(concentration, concentration)
    )

    dpd = np.linspace(0, radius / 2 ** (1 / 3), radius.size)  # type: ignore
    dpi = (np.transpose(radius) ** 3 - dpd**3) ** (1 / 3)

    return radius**2 * np.trapz(
        interp.ev(dpd, dpi) / dpi**2,  # type: ignore
        dpd,
        axis=0  # type: ignore
    )
