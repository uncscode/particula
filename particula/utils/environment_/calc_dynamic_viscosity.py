""" calculate the dynamic viscosity of air.
"""

from particula import u
from particula.utils import (
    REF_VISCOSITY_AIR,
    REF_TEMPERATURE,
    SUTHERLAND_CONSTANT,
)


def dynamic_viscosity_air(temperature=298) -> float:

    """ Returns the dynamic viscosity of air

        Parameters:
            temperature (float) [K] (default: 298)

        Returns:
                        (float) [kg/m/s]

        The dynamic viscosity is calculated using
        the 3-parameter Sutherland Viscosity Law.

        The expected dynamic viscosity of air is approx.
        1.8e-05 kg/m/s at standard conditions (298 K).
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
