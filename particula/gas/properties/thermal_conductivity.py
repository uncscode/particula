"""Thermal Conductivity of air."""

import logging
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")  # get instance of logger


@validate_inputs({"temperature": "nonnegative"})
def get_thermal_conductivity(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the thermal conductivity of air as a function of temperature.
    Based on a simplified linear relation from atmospheric science literature.
    Only valid for temperatures within the range typically found on
    Earth's surface.

    Long Description:
        This function uses a simplified linear relation from
        atmospheric science literature. Valid for Earth-like surface
        temperatures, typically 200–330 K.

    Equation:
        - k(T) = 1e-3 × (4.39 + 0.071 × T)

    Where:
        - k(T) : Thermal conductivity [W/(m·K)].
        - T : Temperature [K].

    Arguments:
        - temperature : The temperature in Kelvin (K).

    Returns:
        - The thermal conductivity [W/(m·K)] or [J/(m·s·K)].

    Examples:
        ```py
        k_300K = get_thermal_conductivity(300)
        # ~0.449 W/(m·K)
        ```

    References:
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.
    """
    return 1e-3 * (4.39 + 0.071 * temperature)
