""" a centralized location for important, unchanged physics parameters.

    This file contains constants that are used in multiple modules. Each
    constant has its own units and exported with them. The constants are
    mostly related to atmospheric aerosol particles in usual conditions.

"""

from particula import u

BOLTZMANN_CONSTANT = 1.380649e-23 * u.m**2 * u.kg / u.s**2 / u.K

AVOGADRO_NUMBER = 6.022140857e23 / u.mol

# Gas constant in J mol^-1 K^-1 = m^2 kg mol^-1 s^-2 K^-1
# J = kg m^2 s^-2
GAS_CONSTANT = BOLTZMANN_CONSTANT * AVOGADRO_NUMBER

ELEMENTARY_CHARGE_VALUE = 1.60217662e-19 * u.C

# Relative permittivity of air at approx.
# 296.15 K and 101325 Pa and 40% RH
# See https://www.osti.gov/servlets/purl/1504063
# Previously known as the "dielectric constant"
# Often denoted as epsilon
RELATIVE_PERMITTIVITY_AIR_ROOM = 1.000530569  # unitless
# At STP (273.15 K, 1 atm):
# see: https://en.wikipedia.org/wiki/Relative_permittivity
RELATIVE_PERMITTIVITY_AIR_STP = 1.00058986  # unitless

# select one of the two:
RELATIVE_PERMITTIVITY_AIR = RELATIVE_PERMITTIVITY_AIR_ROOM

# Permittivity of free space in F/m
# Also known as the electric constant, permittivity of free space
# Often denoted by epsilon_0
VACUUM_PERMITTIVITY = 8.85418782e-12 * u.F / u.m

ELECTRIC_PERMITTIVITY = RELATIVE_PERMITTIVITY_AIR * VACUUM_PERMITTIVITY

# These values are used to calculate the dynamic viscosity of air
# Here, REF temperature and viscosity are at STP:
# Standard temperature and pressure (273.15 K and 101325 Pa)
REF_VISCOSITY_AIR_STP = 1.716e-5 * u.Pa * u.s
REF_TEMPERATURE_STP = 273.15 * u.K
SUTHERLAND_CONSTANT = 110.4 * u.K
MOLECULAR_WEIGHT_AIR = (28.9644 * u.g / u.mol).to_base_units()
