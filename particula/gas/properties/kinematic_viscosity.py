"""
Kinematic viscosity for fluids.

The kinematic viscosity is the ratio of the dynamic viscosity to the
density of a fluid. It is a measure of the fluid's resistance to flow
under the influence of gravity.

v = mu / rho

Where:
    - v : Kinematic viscosity [m^2/s].
    - mu : Dynamic viscosity [Pa*s].
    - rho : Density of the fluid [kg/m^3].

References:
https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula
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

    The kinematic viscosity is the ratio of the dynamic viscosity to the
    density of a fluid. It is a measure of the fluid's resistance to flow
    under the influence of gravity.

    Args:
    -----
        - dynamic_viscosity : Dynamic viscosity of the fluid [Pa*s].
        - fluid_density : Density of the fluid [kg/m³].

    Returns:
    --------
        - Kinematic viscosity of the fluid [m^2/s].

    References:
    ----------
    - Equation : v = mu / rho
            - v : Kinematic viscosity [m^2/s].
            - mu : Dynamic viscosity [Pa*s].
            - rho : Density of the fluid [kg/m^3].
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
    Calculate the kinematic viscosity of air by calculating dynamic viscosity
    of air.

    Args:
    -----
    - temperature: Desired air temperature [K]. Must be greater than 0.
    - fluid_density: Density of the fluid [kg/m³].
    - reference_viscosity: Gas viscosity [Pa*s] at the reference temperature
    (default is STP).
    - reference_temperature: Gas temperature [K] for the reference viscosity
    (default is STP).

    Returns:
    --------
    - float: The kinematic viscosity of air at the given temperature [m^2/s].

    Raises:
    ------
    - ValueError: If the temperature is less than or equal to 0.

    References:
    ----------
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
