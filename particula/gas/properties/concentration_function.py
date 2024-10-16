"""Function for calculating the gas concentraions.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.constants import GAS_CONSTANT  # pyright: ignore


def calculate_concentration(
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the concentration of a gas from its partial pressure, molar mass,
    and temperature using the ideal gas law.

    Parameters:
    - pressure (float or NDArray[np.float64]): Partial pressure of the gas
    in Pascals (Pa).
    - molar_mass (float or NDArray[np.float64]): Molar mass of the gas in kg/mol
    - temperature (float or NDArray[np.float64]): Temperature in Kelvin.

    Returns:
    - concentration (float or NDArray[np.float64]): Concentration of the gas
    in kg/m^3.
    """
    return (partial_pressure * molar_mass) / (
        float(GAS_CONSTANT.m) * temperature
    )
