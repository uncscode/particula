""" calculate the dynamic viscosity of air.
"""

from particula import u
from particula.utils import (
    REF_VISCOSITY_AIR,
    REF_TEMPERATURE,
    SUTHERLAND_CONSTANT,
)


def dynamic_viscosity_air(temperature) -> float:

    """ Returns the dynamic viscosity of air: [kg/m/s]

        The dynamic viscosity is calculated using
        the 3-parameter Sutherland Viscosity Law.
    """

    if isinstance(temperature, u.Quantity):
        temperature = temperature.to_base_units()
    else:
        temperature = u.Quantity(temperature, u.K)

    return (
        REF_VISCOSITY_AIR *
        (temperature/REF_TEMPERATURE)**1.5 *
        (REF_TEMPERATURE + SUTHERLAND_CONSTANT) /
        (temperature + SUTHERLAND_CONSTANT)
    ).to_base_units()
