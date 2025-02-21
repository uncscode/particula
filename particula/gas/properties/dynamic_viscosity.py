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
from particula.util.validate_inputs import validate_inputs

logger = logging.getLogger("particula")  # get instance of logger


@validate_inputs({"temperature": "positive"})
def get_dynamic_viscosity(
    temperature: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
) -> float:
    """
    Calculate the dynamic viscosity of air using Sutherland's formula.

    - μ(T) = μ₀ × (T / T₀)^(3/2) × (T₀ + S) / (T + S)
        - μ(T) is the dynamic viscosity at temperature T (Pa·s).
        - μ₀ is the reference viscosity (Pa·s).
        - T is the temperature in Kelvin.
        - T₀ is the reference temperature in Kelvin.
        - S is the Sutherland constant in Kelvin.

    Arguments:
        - temperature : Desired air temperature in Kelvin. Must be > 0.
        - reference_viscosity : Gas viscosity at the reference temperature
            (default is STP).
        - reference_temperature : Gas temperature in Kelvin for the reference
            viscosity (default is STP).

    Returns:
        - Dynamic viscosity of air at the given temperature in Pa·s.

    Examples:
        ``` py title="Example Float Usage"
        import particula as par
        par.gas.get_dynamic_viscosity(300.0)
        # Output (approx.): 1.846e-05
        ```

    References:
        - Wolfram Formula Repository, "Sutherland's Formula,"
          https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
    """
    return (
        reference_viscosity
        * (temperature / reference_temperature) ** 1.5
        * (reference_temperature + SUTHERLAND_CONSTANT)
        / (temperature + SUTHERLAND_CONSTANT)
    )
