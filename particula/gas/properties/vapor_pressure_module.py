"""
Vapor pressure modules for calculating the vapor pressure of a gas.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
from particula.util.constants import GAS_CONSTANT


# Antoine equation function
def get_antoine_vapor_pressure(
    a: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
    c: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using the Antoine equation.

    Long Description:
        The Antoine equation relates the logarithm of vapor pressure to
        temperature for a pure substance.

    Equation:
        - P = 10^(a - b / (T - c)) × 133.322

    Where:
        - P : Vapor pressure [Pa].
        - a, b, c : Antoine equation parameters (dimensionless).
        - T : Temperature [K].

    Arguments:
        - a : Antoine parameter a (dimensionless).
        - b : Antoine parameter b (dimensionless).
        - c : Antoine parameter c (dimensionless).
        - temperature : Temperature in Kelvin [K].

    Returns:
        - Vapor pressure in Pascals [Pa].

    Examples:
        ```py title="Example usage"
        p_antoine = get_antoine_vapor_pressure(8.07131, 1730.63, 233.426, 373.15)
        # Output: ~101325 Pa (roughly 1 atm)
        ```

    References:
        - Equation: log10(P) = a - b / (T - c)
        - https://en.wikipedia.org/wiki/Antoine_equation
        - Kelvin conversion details:
          https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1
    """
    vapor_pressure_log = a - (b / (temperature - c))
    vapor_pressure = 10**vapor_pressure_log
    return vapor_pressure * 133.32238741499998  # Convert mmHg to Pa


# Clausius-Clapeyron equation function
def get_clausius_clapeyron_vapor_pressure(
    latent_heat: Union[float, NDArray[np.float64]],
    temperature_initial: Union[float, NDArray[np.float64]],
    pressure_initial: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    gas_constant: float = GAS_CONSTANT,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using Clausius-Clapeyron equation.

    Long Description:
        This function calculates the final vapor pressure based on an initial
        temperature/pressure pair and the latent heat of vaporization,
        assuming ideal gas behavior.

    Equation:
        - P_final = P_initial × exp( (L / R) × (1 / T_initial - 1 / T_final) )

    Where:
        - P_final : Final vapor pressure [Pa].
        - P_initial : Initial vapor pressure [Pa].
        - L : Latent heat of vaporization [J/mol].
        - R : Universal gas constant [J/(mol·K)].
        - T_initial : Initial temperature [K].
        - T_final : Final temperature [K].

    Arguments:
        - latent_heat : Latent heat of vaporization [J/mol].
        - temperature_initial : Initial temperature [K].
        - pressure_initial : Initial vapor pressure [Pa].
        - temperature : Final temperature [K].
        - gas_constant : Gas constant (default 8.314 J/(mol·K)).

    Returns:
        - Pure vapor pressure [Pa].

    Examples:
        ```py title="Example usage"
        p_final = get_clausius_clapeyron_vapor_pressure(
            40660, 373.15, 101325, 300
        )
        # Output: ~35307 Pa
        ```

    References:
        - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
    return pressure_initial * np.exp(
        (latent_heat / gas_constant)
        * (1 / temperature_initial - 1 / temperature)
    )


# Buck equation function
def get_buck_vapor_pressure(
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using the Buck equation for water vapor.

    Long Description:
        Uses separate empirical formulas below 0 °C and above 0 °C to compute
        water vapor pressure.

    Equation:
        For T < 0 °C:
            - p = 6.1115 × exp( (23.036 - T/333.7) × T / (279.82 + T ) ) × 100
        For T ≥ 0 °C:
            - p = 6.1121 × exp( (18.678 - T/234.5) × T / (257.14 + T ) ) × 100

    Where:
        - p : Vapor pressure [Pa].
        - T : Temperature in Celsius [°C] (converted internally from Kelvin).

    Arguments:
        - temperature : Temperature in Kelvin [K].

    Returns:
        - Vapor pressure in Pascals [Pa].

    Examples:
        ```py title="Example usage"
        p_buck = get_buck_vapor_pressure(273.15)
        # Output: ~611 Pa (around ice point)
        ```

    References:
        - Buck, A. L., (1981) ...
        - https://en.wikipedia.org/wiki/Arden_Buck_equation
    """
    temp = np.array(temperature) - 273.15  # Convert to Celsius
    temp_below_freezing = temp < 0.0
    temp_above_freezing = temp >= 0.0

    vapor_pressure_below_freezing = (
        6.1115 * np.exp((23.036 - temp / 333.7) * temp / (279.82 + temp)) * 100
    )  # hPa to Pa
    vapor_pressure_above_freezing = (
        6.1121 * np.exp((18.678 - temp / 234.5) * temp / (257.14 + temp)) * 100
    )  # hPa to Pa

    return (
        vapor_pressure_below_freezing * temp_below_freezing
        + vapor_pressure_above_freezing * temp_above_freezing
    )
