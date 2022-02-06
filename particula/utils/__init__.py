""" expose useful utilites here
"""

# flake8: noqa: F401
from particula.utils.get_constants import(
    BOLTZMANN_CONSTANT,
    AVOGADRO_NUMBER,
    GAS_CONSTANT,
    ELEMENTARY_CHARGE_VALUE,
    RELATIVE_PERMITTIVITY_AIR,
    VACUUM_PERMITTIVITY,
    ELECTRIC_PERMITTIVITY,
)
from particula.utils.strip_units import make_unitless as unitless
from particula.utils.particle_ import (
    knudsen_number,
    slip_correction,
    friction_factor,
    reduced_mass,
    reduced_friction,
    coulomb_ratio,
    coulomb_kinetic,
    coulomb_continuum,
)
