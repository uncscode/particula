"""Thermal Conductivity of air."""

import logging
from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")  # get instance of logger


@validate_inputs({"temperature": "nonnegative"})
def get_thermal_conductivity(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Thermal conductivity of air as a function of temperature.

    Calculate the thermal conductivity of air as a function of temperature.
    Based on a simplified linear relation from atmospheric science literature.
    Only valid for temperatures within the range typically found on
    Earth's surface.

    - k(T) = 1e-3 × (4.39 + 0.071 × T)
        - k(T) is Thermal conductivity [W/(m·K)].
        - T is Temperature [K].

    Arguments:
        - temperature : The temperature in Kelvin (K).

    Returns:
        - The thermal conductivity [W/(m·K)] or [J/(m·s·K)].

    Examples:
        ``` py title="Example Usage"
        import particula as par
        par.gas.get_thermal_conductivity(300)
        # ~0.449 W/(m·K)
        ```

    References:
    - Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.
    """
    return 1e-3 * (4.39 + 0.071 * temperature)
