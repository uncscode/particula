"""Thermal Conductivity of air."""

import logging
from typing import Union
from numpy.typing import NDArray
import numpy as np

logger = logging.getLogger("particula")  # get instance of logger


def get_thermal_conductivity(
    temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the thermal conductivity of air as a function of temperature.
    Based on a simplified linear relation from atmospheric science literature.
    Only valid for temperatures within the range typically found on
    Earth's surface.

    Args:
    -----
    - temperature (Union[float, NDArray[np.float64]]): The temperature at which
    the thermal conductivity of air is to be calculated, in Kelvin (K).

    Returns:
    --------
    - Union[float, NDArray[np.float64]]: The thermal conductivity of air at the
    specified temperature in Watts per meter-Kelvin (W/m·K) or J/(m s K).

    Raises:
    ------
    - ValueError: If the temperature is below absolute zero (0 K).

    References:
    ----------
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.
    """
    if np.any(temperature < 0):
        logger.error("Temperature must be greater than or equal to 0 Kelvin.")
        raise ValueError(
            "Temperature must be greater than or equal to 0 Kelvin."
        )
    return 1e-3 * (4.39 + 0.071 * temperature)
