"""Module to calculate the Kelvin effect on vapor pressure."""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.constants import GAS_CONSTANT


def kelvin_radius(
    effective_surface_tension: Union[float, NDArray[np.float64]],
    effective_density: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Kelvin radius which determines the curvature effect on
    vapor pressure.

    Args:
    -----
    - surface_tension (float or NDArray[float]): Surface tension of the
    mixture [N/m].
    - molar_mass (float or NDArray[float]): Molar mass of the species
    [kg/mol].
    - mass_concentration (float or NDArray[float]): Concentration of the
    species [kg/m^3].
    - temperature (float): Temperature of the system [K].

    Returns:
    --------
    - float or NDArray[float]: Kelvin radius [m].

    References:
    -----------
    - Based on Neil Donahue's approach to the Kelvin equation:
    r = 2 * surface_tension * molar_mass / (R * T * density)
    See more: https://en.wikipedia.org/wiki/Kelvin_equation
    """
    return (2 * effective_surface_tension * molar_mass) / (
        GAS_CONSTANT.m * temperature * effective_density
    )


def kelvin_term(
    radius: Union[float, NDArray[np.float64]],
    kelvin_radius_value: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the Kelvin term, which quantifies the effect of particle
    curvature on vapor pressure.

    Args:
    -----
    - radius (float or NDArray[float]): Radius of the particle [m].
    - kelvin_radius (float or NDArray[float]): Kelvin radius [m].

    Returns:
    --------
    - float or NDArray[float]: The exponential factor adjusting vapor
    pressure due to curvature.

    References:
        Based on Neil Donahue's collection of terms in the Kelvin equation:
        exp(kelvin_radius / particle_radius)
        See more: https://en.wikipedia.org/wiki/Kelvin_equation
    """
    kelvin_expand = False
    # Broadcast the arrays if necessary np.isscalar(kelvin_radius_value)
    if isinstance(kelvin_radius_value, np.ndarray) and (
        kelvin_radius_value.size > 1
    ):
        kelvin_expand = True
        kelvin_radius_value = kelvin_radius_value[np.newaxis, :]
    if isinstance(radius, np.ndarray) and not kelvin_expand:
        return np.exp(kelvin_radius_value / radius)
    if isinstance(radius, np.ndarray) and (radius.size > 1) and kelvin_expand:
        radius = radius[:, np.newaxis]
    return np.exp(kelvin_radius_value / radius)
