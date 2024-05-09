""" calculating the dynamic viscosity

    The dynamic viscosity is calculated using the Sutherland formula,
    assuming ideal gas behavior, as a function of temperature.

    "The dynamic viscosity equals the product of the sum of
    Sutherland's constant and the reference temperature divided by
    the sum of Sutherland's constant and the temperature,
    the reference viscosity and the ratio to the 3/2 power
    of the temperature to reference temperature."

    https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
"""

from particula.constants import (REF_TEMPERATURE_STP, REF_VISCOSITY_AIR_STP,
                                 SUTHERLAND_CONSTANT)


def dynamic_viscosity(
    temperature: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP.m,
    reference_temperature: float = REF_TEMPERATURE_STP.m
) -> float:
    """
    The dynamic viscosity of air via Sutherland formula.
    This formula depends on temperature (temp) and the reference
    temperature (t_ref) as well as the reference viscosity (mu_ref).

    Args:
    -----
    - temperature: Desired air temperature [K]
    - reference_viscosity: Gas viscosity [Pa*s] (default: air)
    - reference_temperature: Gas temperature [K] (default: 298.15)

    Returns:
    --------
    - float: The dynamic viscosity of air [Pa*s]

    References:
    ----------
    https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
    """
    return (
        reference_viscosity * (temperature / reference_temperature)**(3/2)
        * (reference_temperature + SUTHERLAND_CONSTANT.m) /
        (temperature + SUTHERLAND_CONSTANT.m)
    )
