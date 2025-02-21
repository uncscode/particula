"""Functions for calculating the partial pressure of a gas."""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.constants import GAS_CONSTANT


def calculate_partial_pressure(
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the partial pressure of a gas from its concentration, molar mass,
    and temperature.

    Long Description:
        This function uses the ideal gas relation:
        p = (c × R × T) / M

    Equation:
        - p = (c × R × T) / M

    Where:
        - p : Partial pressure [Pa].
        - c : Gas concentration [kg/m³].
        - R : Universal gas constant [J/(mol·K)].
        - T : Temperature [K].
        - M : Molar mass [kg/mol].

    Arguments:
        - concentration : Concentration of the gas [kg/m³].
        - molar_mass : Molar mass of the gas [kg/mol].
        - temperature : Temperature [K].

    Returns:
        - Partial pressure of the gas [Pa].

    Examples:
        ```py title="Example usage"
        partial = calculate_partial_pressure(1.2, 0.02897, 298)
        # Output: ~986.4 Pa
        ```

    References:
        - Wikipedia contributors, "Ideal gas law," Wikipedia,
          https://en.wikipedia.org/wiki/Ideal_gas_law
    """
    # Input validation
    if np.any(concentration < 0):
        raise ValueError("Concentration must be positive")
    if np.any(molar_mass <= 0):
        raise ValueError("Molar mass must be positive")
    if np.any(temperature <= 0):
        raise ValueError("Temperature must be positive")

    # Calculate the partial pressure
    return (concentration * float(GAS_CONSTANT) * temperature) / molar_mass


def calculate_saturation_ratio(
    partial_pressure: Union[float, NDArray[np.float64]],
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the saturation ratio of the gas at a given partial pressure and
    pure vapor pressure.

    Long Description:
        The saturation ratio is defined as the ratio of partial pressure to the
        pure vapor pressure.

    Equation:
        - S = p / p_vap

    Where:
        - S : Saturation ratio (dimensionless).
        - p : Partial pressure [Pa].
        - p_vap : Pure vapor pressure [Pa].

    Arguments:
        - partial_pressure : Partial pressure [Pa].
        - pure_vapor_pressure : Pure vapor pressure [Pa].

    Returns:
        - Saturation ratio of the gas (dimensionless).

    Examples:
        ```py title="Example usage"
        ratio = calculate_saturation_ratio(800.0, 1000.0)
        # Output: 0.8
        ```

    References:
        - Wikipedia contributors, "Relative humidity," Wikipedia,
          https://en.wikipedia.org/wiki/Relative_humidity
    """
    return partial_pressure / pure_vapor_pressure
