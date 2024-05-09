""" calculating the mean free path of air

    The mean free path is the average distance
    traveled by a molecule between collisions
    with other molecules present in a medium (air).

    The expeected mean free path of air is approx.
    65 nm at 298 K and 101325 Pa.

"""

from typing import Union, Optional
from numpy.typing import NDArray
import numpy as np
from particula.constants import (
    GAS_CONSTANT, MOLECULAR_WEIGHT_AIR)
from particula.util.dynamic_viscosity import dyn_vis


def molecule_mean_free_path(
    molar_mass: Union[
        float, NDArray[np.float_]] = MOLECULAR_WEIGHT_AIR.m,  # type: ignore
    temperature: float = 298.15,
    pressure: float = 101325,
    dynamic_viscosity: Optional[float] = None,
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the mean free path of a gas molecule in air based on the
    temperature, pressure, and molar mass of the gas. The mean free path
    is the average distance traveled by a molecule between collisions with
    other molecules present in a medium (air).

    Args:
    -----
    - molar_mass (Union[float, NDArray[np.float_]]): The molar mass
    of the gas molecule [kg/mol]. Default is the molecular weight of air.
    - temperature (float): The temperature of the gas [K]. Default is 298.15 K.
    - pressure (float): The pressure of the gas [Pa]. Default is 101325 Pa.
    - dynamic_viscosity (Optional[float]): The dynamic viscosity of the gas
    [Pa*s]. If not provided, it will be calculated based on the temperature.

    Returns:
    --------
    - Union[float, NDArray[np.float_]]: The mean free path of the gas molecule
    in meters (m).

    References:
    ----------
    - https://en.wikipedia.org/wiki/Mean_free_path
    """
    # check inputs are positive
    if temperature <= 0:
        raise ValueError("Temperature must be positive [Kelvin]")
    if pressure <= 0:
        raise ValueError("Pressure must be positive [Pascal]")
    if np.any(molar_mass <= 0):
        raise ValueError("Molar mass must be positive [kg/mol]")
    if dynamic_viscosity is None:
        dynamic_viscosity = dyn_vis(temperature)  # type: ignore
        dynamic_viscosity = float(dynamic_viscosity.m)  # type: ignore

    return np.array(
        (2 * dynamic_viscosity / pressure)
        / (8 * molar_mass / (np.pi * GAS_CONSTANT.m * temperature))**0.5,
        dtype=np.float_)
