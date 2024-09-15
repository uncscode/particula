"""
Vapor pressure modules for calculating the vapor pressure of a gas.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
from particula.constants import GAS_CONSTANT

from particula.util.input_handling import convert_units  # type: ignore


# Antoine equation function
def antoine_vapor_pressure(
    a: Union[float, NDArray[np.float64]],
    b: Union[float, NDArray[np.float64]],
    c: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using the Antoine equation.

    Args:
        a, b, c: Antoine equation parameters.
        temperature: Temperature in Kelvin.

    Returns:
        Vapor pressure in Pascals.

    References:
        - Equation: log10(P) = a - b / (T - c)
        - https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
        - Kelvin form:
            https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1
    """
    vapor_pressure_log = a - (b / (temperature - c))
    vapor_pressure = 10**vapor_pressure_log
    return vapor_pressure * convert_units("mmHg", "Pa")  # Convert mmHg to Pa


# Clausius-Clapeyron equation function
def clausius_clapeyron_vapor_pressure(
    latent_heat: Union[float, NDArray[np.float64]],
    temperature_initial: Union[float, NDArray[np.float64]],
    pressure_initial: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
    gas_constant: float = GAS_CONSTANT.m,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using Clausius-Clapeyron equation.

    Args:
        latent_heat: Latent heat of vaporization in J/mol.
        temperature_initial: Initial temperature in Kelvin.
        pressure_initial: Initial vapor pressure in Pascals.
        temperature: Final temperature in Kelvin.
        gas_constant: gas constant (default is 8.314 J/(mol·K)).

    Returns:
        Pure vapor pressure in Pascals.

    References:
        - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation
    """
    return pressure_initial * np.exp(
        (latent_heat / gas_constant)
        * (1 / temperature_initial - 1 / temperature)
    )


# Buck equation function
def buck_vapor_pressure(
    temperature: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate vapor pressure using the Buck equation for water vapor.

    Args:
        temperature: Temperature in Kelvin.

    Returns:
        Vapor pressure in Pascals.

    References:
        - Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
            Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
            https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
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
