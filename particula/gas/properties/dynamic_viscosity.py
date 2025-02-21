"""Module for calculating the dynamic viscosity

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

from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)

logger = logging.getLogger("particula")  # get instance of logger


def get_dynamic_viscosity(
    temperature: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
) -> float:
    """
    Calculate the dynamic viscosity of air using Sutherland's formula.

    Equation:
        μ(T) = μ₀ × (T / T₀)^(3/2) × (T₀ + S) / (T + S)

    where:
        - μ(T) is the dynamic viscosity at temperature T,
        - μ₀ is the reference viscosity at temperature T₀,
        - S is the Sutherland constant.

    Arguments:
        temperature : Desired air temperature in Kelvin. Must be > 0.
        reference_viscosity : Gas viscosity at the reference temperature (default is STP).
        reference_temperature : Gas temperature in Kelvin for the reference viscosity (default is STP).

    Returns:
        Dynamic viscosity of air at the given temperature in Pa·s.

    References:
        - Wolfram Formula Repository, "Sutherland's Formula," https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
    """
    if temperature <= 0:
        logger.error("Temperature must be greater than 0 Kelvin.")
        raise ValueError("Temperature must be greater than 0 Kelvin.")
    return (
        reference_viscosity
        * (temperature / reference_temperature) ** 1.5
        * (reference_temperature + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )
