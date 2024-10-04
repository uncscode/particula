"""
General dilution rate for closed systems.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def volume_dilution_coefficient(
    volume: Union[float, NDArray[np.float64]],
    input_flow_rate: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the volume dilution coefficient.

    The volume dilution coefficient is a measure of how quickly a substance
    is diluted within a given volume due to an incoming flow. It is defined as
    the ratio of the flow rate to the volume.

    Arguments:
        volume: The volume of the system in cubic meters (m³).
        input_flow_rate: The flow rate of the substance entering the system
            in cubic meters per second (m³/s).

    Returns:
        The volume dilution coefficient in inverse seconds (s⁻¹).

    Examples:
        ``` py title="float input"
        volume_dilution_coefficient(
            volume=10,
            input_flow_rate=0.1,
        )
        # Returns 0.01
        ```

        ``` py title="array input"
        volume_dilution_coefficient(
            volume=np.array([10, 20, 30]),
            input_flow_rate=np.array([0.1, 0.2, 0.3]),
        )
        # Returns array([0.01, 0.01, 0.01])
        ```
    """
    return input_flow_rate / volume


def dilution_rate(
    coefficient: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the dilution rate of a substance.

    The dilution rate quantifies the rate at which the concentration of a
    substance decreases due to dilution, based on the volume dilution
    coefficient and the current concentration of the substance.

    Arguments:
        coefficient: The volume dilution coefficient in inverse seconds (s⁻¹).
        concentration: The concentration of the substance in the system
            in particles per cubic meter (#/m³) or any other relevant units.

    Returns:
        The dilution rate, which is the rate of decrease in concentration
        in inverse seconds (s⁻¹). The value is returned as negative, indicating
        a reduction in concentration over time.

    Examples:
        ``` py title="float input"
        dilution_rate(
            coefficient=0.01,
            concentration=100,
        )
        # Returns -1.0
        ```

        ``` py title="array input"
        dilution_rate(
            coefficient=0.01,
            concentration=np.array([100, 200, 300]),
        )
        # Returns array([-1., -2., -3.])
        ```
    """
    return -coefficient * concentration
