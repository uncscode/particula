""" Module for calculating the dynamic viscosity

    The dynamic viscosity is calculated using the Sutherland formula,
    assuming ideal gas behavior, as a function of temperature.

    "The dynamic viscosity equals the product of the sum of
    Sutherland's constant and the reference temperature divided by
    the sum of Sutherland's constant and the temperature,
    the reference viscosity and the ratio to the 3/2 power
    of the temperature to reference temperature."

    https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
"""

import logging
from particula.constants import (REF_TEMPERATURE_STP, REF_VISCOSITY_AIR_STP,
                                 SUTHERLAND_CONSTANT)

logger = logging.getLogger("particula")  # get instance of logger


def get_dynamic_viscosity(
    temperature: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP.m,
    reference_temperature: float = REF_TEMPERATURE_STP.m
) -> float:
    """
    Calculates the dynamic viscosity of air via Sutherland's formula, which is
    a common method in fluid dynamics for gases that involves temperature
    adjustments.

    Args:
    -----
    - temperature: Desired air temperature [K]. Must be greater than 0.
    - reference_viscosity: Gas viscosity [Pa*s] at the reference temperature
    (default is STP).
    - reference_temperature: Gas temperature [K] for the reference viscosity
    (default is STP).

    Returns:
    --------
    - float: The dynamic viscosity of air at the given temperature [Pa*s].

    Raises:
    ------
    - ValueError: If the temperature is less than or equal to 0.

    References:
    ----------
    https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
    """
    if temperature <= 0:
        logger.error("Temperature must be greater than 0 Kelvin.")
        raise ValueError("Temperature must be greater than 0 Kelvin.")
    return (
        reference_viscosity * (temperature / reference_temperature)**1.5 *
        (reference_temperature + SUTHERLAND_CONSTANT.m) /
        (temperature + SUTHERLAND_CONSTANT.m)
    )
