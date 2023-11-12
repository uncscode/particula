""" calculate kelvin correction """

import numpy as np

from particula import u
from particula.util.input_handling import in_temperature, in_density, in_radius
from particula.util.input_handling import in_handling, in_molecular_weight
from particula.constants import GAS_CONSTANT


def kelvin_radius(
    surface_tension=0.072 * u.N/u.m,
    molecular_weight=0.01815 * u.kg/u.mol,
    density=1000 * u.kg/u.m**3,
    temperature=298.15 * u.K
):
    """ Kelvin radius (Neil's definition)
        https://en.wikipedia.org/wiki/Kelvin_equation
    """

    temperature = in_temperature(temperature).to_base_units()
    molecular_weight = in_molecular_weight(molecular_weight).to_base_units()
    density = in_density(density).to_base_units()
    surface_tension = in_handling(surface_tension, u.N/u.m)  # type: ignore

    return 2 * surface_tension * molecular_weight / (
        GAS_CONSTANT * temperature * density
    )


def kelvin_term(radius=None, **kwargs):
    """ Kelvin term (Neil's definition)
        https://en.wikipedia.org/wiki/Kelvin_equation
    """

    rad = in_radius(radius)

    kelvin_rad = kelvin_radius(**kwargs)

    return np.exp(kelvin_rad / rad)
