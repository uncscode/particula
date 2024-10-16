"""Functions for calculating the partial pressure of a gas."""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.constants import GAS_CONSTANT


def calculate_partial_pressure(
    concentration: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the partial pressure of a gas from its concentration, molar mass,
    and temperature.

    Parameters:
    - concentration (float): Concentration of the gas in kg/m^3.
    - molar_mass (float): Molar mass of the gas in kg/mol.
    - temperature (float): Temperature in Kelvin.

    Returns:
    - float: Partial pressure of the gas in Pascals (Pa).
    """
    # Calculate the partial pressure
    return (concentration * float(GAS_CONSTANT.m) * temperature) / molar_mass


def calculate_saturation_ratio(
    partial_pressure: Union[float, NDArray[np.float64]],
    pure_vapor_pressure: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the saturation ratio of the gas at a given pressure and
    temperature.

    Args:
        partial_pressure: Pressure in Pascals.
        pure_vapor_pressure: Pure vapor pressure of the gas in Pascals.

    Returns:
        saturation_ratio: The saturation ratio of the gas.
    """
    return partial_pressure / pure_vapor_pressure
