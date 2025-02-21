"""Kinematic viscosity for fluids.

Long Description:
    The kinematic viscosity (ν) is the ratio of the dynamic viscosity (μ)
    to the density (ρ).

Equation:
    - ν = μ / ρ

Where:
    - ν : Kinematic viscosity [m²/s].
    - μ : Dynamic viscosity [Pa·s].
    - ρ : Fluid density [kg/m³].

References:
    - "Viscosity Conversion Formula," Wolfram Formula Repository, 
      https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula
    - Wikipedia contributors, "Viscosity," Wikipedia,
      https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity

"""

from particula.util.validate_inputs import validate_inputs
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.util.constants import (
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
)


def get_kinematic_viscosity(
    dynamic_viscosity: float,
    fluid_density: float,
) -> float:
    """
    Calculate the kinematic viscosity of a fluid.

    Long Description:
        The function calculates ν by dividing the dynamic viscosity (μ)
        by the fluid density (ρ).

    Equation:
        ν = μ / ρ

    Arguments:
        - dynamic_viscosity : Dynamic viscosity of the fluid [Pa·s].
        - fluid_density : Density of the fluid [kg/m³].

    Returns:
        - The kinematic viscosity [m²/s].

    Examples:
        ```py title="Example usage"
        nu = get_kinematic_viscosity(1.8e-5, 1.2)
        # Output: ~1.5e-5
        ```

    References:
        - "Viscosity Conversion Formula," Wolfram Formula Repository.
          https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula
    """
    return dynamic_viscosity / fluid_density


@validate_inputs({"temperature": "positive"})
def get_kinematic_viscosity_via_system_state(
    temperature: float,
    fluid_density: float,
    reference_viscosity: float = REF_VISCOSITY_AIR_STP,
    reference_temperature: float = REF_TEMPERATURE_STP,
) -> float:
    """
    Calculate the kinematic viscosity of air by calculating dynamic
    viscosity of air.

    Equation:
        ν = μ / ρ

    Arguments:
        temperature : Desired air temperature [K]. Must be greater than 0.
        fluid_density : Density of the fluid [kg/m³].
        reference_viscosity : Gas viscosity [Pa*s] at the reference temperature
            (default is STP).
        reference_temperature : Gas temperature [K] for the reference viscosity
            (default is STP).

    Returns:
        The kinematic viscosity of air at the given temperature [m^2/s].

    Raises:
        - ValueError : If the temperature is less than or equal to 0.

    References:
        https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula
    """
    dynamic_viscosity = get_dynamic_viscosity(
        temperature=temperature,
        reference_viscosity=reference_viscosity,
        reference_temperature=reference_temperature,
    )
    return get_kinematic_viscosity(
        dynamic_viscosity=dynamic_viscosity, fluid_density=fluid_density
    )
