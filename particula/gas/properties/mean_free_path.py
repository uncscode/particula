"""
Calculating the mean free path of air

Long Description:
    The mean free path is the average distance traveled by a molecule
    between collisions with other molecules in a medium (air). Typical
    mean free path values in air are about 65 nm at 298 K and 101325 Pa.

References:
    - "Mean Free Path," Wikipedia, The Free Encyclopedia,
      https://en.wikipedia.org/wiki/Mean_free_path

"""

import logging
from typing import Union, Optional
from numpy.typing import NDArray
import numpy as np
from particula.util.constants import (
    GAS_CONSTANT,
    MOLECULAR_WEIGHT_AIR,
)  # type: ignore
from particula.gas.properties.dynamic_viscosity import (
    get_dynamic_viscosity,
)

logger = logging.getLogger("particula")  # get instance of logger


def get_molecule_mean_free_path(
    molar_mass: Union[
        float, NDArray[np.float64]
    ] = MOLECULAR_WEIGHT_AIR,  # type: ignore
    temperature: float = 298.15,
    pressure: float = 101325,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate the mean free path of a gas molecule in air.

    This function calculates λ based on the input conditions. If
    dynamic_viscosity is not provided, it is computed via
    get_dynamic_viscosity(temperature).

    - λ = (2 × μ / P) / √(8 × M / (π × R × T))
        - λ is Mean free path [m].
        - μ is Dynamic viscosity [Pa·s].
        - P is Gas pressure [Pa].
        - M is Molar mass [kg/mol].
        - R is Universal gas constant [J/(mol·K)].
        - T is Gas temperature [K].

    Arguments:
        - molar_mass : The molar mass of the gas molecule [kg/mol].
        - temperature : The temperature of the gas [K].
        - pressure : The pressure of the gas [Pa].
        - dynamic_viscosity : The dynamic viscosity of the gas [Pa·s].
            If None, it will be calculated based on the temperature.

    Returns:
        - Mean free path of the gas molecule in meters (m).

    Examples:
        ```py title="Example usage"
        import particula as par
        par.gas.get_molecule_mean_free_path()
        # Returns mean free path at ~298K and 101325Pa, ~6.5e-8 m
        ```

    References:
        - "Mean Free Path," Wikipedia, The Free Encyclopedia.
          https://en.wikipedia.org/wiki/Mean_free_path
    """
    # check inputs are positive
    if temperature <= 0:
        logger.error("Temperature must be positive [Kelvin]")
        raise ValueError("Temperature must be positive [Kelvin]")
    if pressure <= 0:
        logger.error("Pressure must be positive [Pascal]")
        raise ValueError("Pressure must be positive [Pascal]")
    if np.any(molar_mass <= 0):
        logger.error("Molar mass must be positive [kg/mol]")
        raise ValueError("Molar mass must be positive [kg/mol]")
    if dynamic_viscosity is None:
        dynamic_viscosity = get_dynamic_viscosity(temperature)
    gas_constant = float(GAS_CONSTANT)

    return (2 * dynamic_viscosity / pressure) / (
        8 * molar_mass / (np.pi * gas_constant * temperature)
    ) ** 0.5
