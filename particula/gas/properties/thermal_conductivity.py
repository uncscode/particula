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

    Args:
    -----
    - temperature (Union[float, NDArray[np.float64]]): The temperature at which
    the thermal conductivity of air is to be calculated, in Kelvin (K).

    Returns:
    --------
    - Union[float, NDArray[np.float64]]: The thermal conductivity of air at the
    specified temperature in Watts per meter-Kelvin (W/m·K) or J/(m s K).

    References:
    ----------
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.
    """
    return 1e-3 * (4.39 + 0.071 * temperature)
