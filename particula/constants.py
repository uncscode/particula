""" a centralized location for important, unchanged physics parameters.

    This file contains constants that are used in multiple modules. Each
    constant has its own units and exported with them. The constants are
    mostly related to atmospheric aerosol particles in usual conditions.

"""

from particula import u

BOLTZMANN_CONSTANT = 1.380649e-23 * u.m**2 * u.kg / u.s**2 / u.K

AVOGADRO_NUMBER = 6.022140857e23 / u.mol

# Gas constant in J mol^-1 K^-1
GAS_CONSTANT = BOLTZMANN_CONSTANT * AVOGADRO_NUMBER

ELEMENTARY_CHARGE_VALUE = 1.60217662e-19 * u.C

# Relative permittivity of air at room temperature
# Previously known as the "dielectric constant"
# Often denoted as epsilon
RELATIVE_PERMITTIVITY_AIR = 1.0005  # unitless

# Permittivity of free space in F/m
# Also known as the electric constant, permittivity of free space
# Often denoted by epsilon_0
VACUUM_PERMITTIVITY = 8.85418782e-12 * u.F / u.m

ELECTRIC_PERMITTIVITY = RELATIVE_PERMITTIVITY_AIR * VACUUM_PERMITTIVITY

# These values are used to calculate the dynamic viscosity of air
REF_VISCOSITY_AIR = 1.716e-5 * u.Pa * u.s
REF_TEMPERATURE = 273.15 * u.K
SUTHERLAND_CONSTANT = 110.4 * u.K
MOLECULAR_WEIGHT_AIR = (28.9644 * u.g / u.mol).to_base_units()
