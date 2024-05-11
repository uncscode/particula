"""Module for calculating the partial pressure of a species in a
gas over particle phase."""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def partial_pressure_delta(
    partial_pressure_gas: Union[float, NDArray[np.float_]],
    partial_pressure_particle: Union[float, NDArray[np.float_]],
    kelvin_term: Union[float, NDArray[np.float_]],
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the difference in partial pressure of a species between the gas
    phase and the particle phase, which is used in the calculation of the rate
    of change of mass of an aerosol particle.

    Args:
    -----
    - partial_pressure_gas (Union[float, NDArray[np.float_]]): The partial
    pressure of the species in the gas phase.
    - partial_pressure_particle (Union[float, NDArray[np.float_]]): The partial
    pressure of the species in the particle phase.
    - kelvin_term (Union[float, NDArray[np.float_]]): Kelvin effect to account
    for the curvature of the particle.

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The difference in partial pressure
    between the gas phase and the particle phase.
    """
    return partial_pressure_gas - partial_pressure_particle * kelvin_term
