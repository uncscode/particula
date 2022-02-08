""" module to perform calculations related to aerosol dynamics.

    This module contains functions to perform calculations related
    to aerosol dynamics. It inherits the unit registry, u, from
    particula. The pacakge contents below are callablle. For example,
    the particle module is callable as:

    >>> from particula.aerosol_dynamics import particle

    ...

    More details to follow.
"""

# expose physical parameters here
# flake8: noqa: F401
from particula.aerosol_dynamics.physical_parameters import (
    BOLTZMANN_CONSTANT,
    AVOGADRO_NUMBER,
    GAS_CONSTANT,
    ELEMENTARY_CHARGE_VALUE,
    RELATIVE_PERMITTIVITY_AIR,
    VACUUM_PERMITTIVITY,
    ELECTRIC_PERMITTIVITY,
)
from particula.aerosol_dynamics.particle import Particle
from particula.aerosol_dynamics.environment import Environment
