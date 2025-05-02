"""Function for calculating the gas concentrations."""

from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.constants import GAS_CONSTANT  # pyright: ignore


def get_concentration_from_pressure(
    partial_pressure: Union[float, NDArray[np.float64]],
    molar_mass: Union[float, NDArray[np.float64]],
    temperature: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the concentration of a gas using the ideal gas law.

    The concentration is determined from the partial pressure, molar mass,
    and temperature using the ideal gas equation:

    - C = (P × M) / (R × T)
        - C is the concentration in kg/m³,
        - P is the partial pressure in Pascals (Pa),
        - M is the molar mass in kg/mol,
        - R is the universal gas constant (J/(mol·K)),
        - T is the temperature in Kelvin.

    Arguments:
        partial_pressure : Partial pressure of the gas in Pascals (Pa).
        molar_mass : Molar mass of the gas in kg/mol.
        temperature : Temperature in Kelvin.

    Examples:
        ```py title="Floating-point Example Usage"
        import particula as par
        par.gas.get_concentration_from_pressure(101325, 0.02897, 298.15)
        # Output: 1.184587604735883
        ```

    Returns:
        Concentration of the gas in kg/m³.
    """
    return (partial_pressure * molar_mass) / (
        float(GAS_CONSTANT) * temperature
    )
