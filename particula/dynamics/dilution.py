"""General dilution rate for closed systems."""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def get_volume_dilution_coefficient(
    volume: Union[float, NDArray[np.float64]],
    input_flow_rate: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the volume dilution coefficient.

    This coefficient represents how quickly a substance is diluted within
    a system of a given volume when a known input flow rate is supplied.
    The equation is:

    - α = Q / V
        - α is the volume dilution coefficient [s⁻¹],
        - Q is the input flow rate [m³/s],
        - V is the system volume [m³].

    Arguments:
        - volume : The volume of the system in cubic meters (m³).
        - input_flow_rate : The flow rate entering the system in
            cubic meters per second (m³/s).

    Returns:
        - The volume dilution coefficient in inverse seconds (s⁻¹).

    Examples:
        ``` py title="Example (float input)"
        get_volume_dilution_coefficient(volume=10, input_flow_rate=0.1)
        # Returns 0.01
        ```

        ``` py title="Example (array input)"
        get_volume_dilution_coefficient(
            volume=np.array([10, 20, 30]),
            input_flow_rate=np.array([0.1, 0.2, 0.3]),
        )
        # Returns array([0.01, 0.01, 0.01])
        ```

    References:
        - O. Levenspiel, "Chemical Reaction Engineering," 3rd ed., Wiley, 1999.
        [check]
    """
    return input_flow_rate / volume


def get_dilution_rate(
    coefficient: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the dilution rate of a substance in a system.

    The dilution rate describes how quickly the concentration of a
    substance decreases due to the volume dilution coefficient and
    the current concentration. The calculation is:

    - R = -(α × c)
        - R is the dilution rate [s⁻¹],
        - α is the volume dilution coefficient [s⁻¹],
        - c is the current concentration [#/m³].

    Arguments:
        - coefficient : The volume dilution coefficient in inverse
            seconds (s⁻¹).
        - concentration : The concentration of the substance in #/m³
            (or relevant units).

    Returns:
        - The dilution rate in s⁻¹, returned as a negative value
          to indicate a decrease in concentration.

    Examples:
        ``` py title="Example (float input)"
        get_dilution_rate(coefficient=0.01, concentration=100)
        # Returns -1.0
        ```

        ``` py title="Example (array input)"
        get_dilution_rate(
            coefficient=0.01,
            concentration=np.array([100, 200, 300]),
        )
        # Returns array([-1., -2., -3.])
        ```

    References:
        - H. Fogler, "Elements of Chemical Reaction Engineering,"
          5th ed., Prentice Hall, 2016. [check]
    """
    return -coefficient * concentration
