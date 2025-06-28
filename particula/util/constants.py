"""A centralized location for important, unchanged physics parameters.

All constants are given in their base units. We use scipy constants.
"""

import scipy

BOLTZMANN_CONSTANT = scipy.constants.k

AVOGADRO_NUMBER = scipy.constants.Avogadro

# Gas constant in J mol^-1 K^-1 = m^2 kg mol^-1 s^-2 K^-1
# J = kg m^2 s^-2
GAS_CONSTANT = BOLTZMANN_CONSTANT * AVOGADRO_NUMBER

ELEMENTARY_CHARGE_VALUE = scipy.constants.elementary_charge

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
VACUUM_PERMITTIVITY = scipy.constants.epsilon_0

ELECTRIC_PERMITTIVITY = RELATIVE_PERMITTIVITY_AIR * VACUUM_PERMITTIVITY

# These values are used to calculate the dynamic viscosity of air
# Here, REF temperature and viscosity are at STP:
# Standard temperature and pressure (273.15 K and 101325 Pa)
REF_VISCOSITY_AIR_STP = 1.716e-5  # Pa*s
REF_TEMPERATURE_STP = 273.15  # K
SUTHERLAND_CONSTANT = 110.4  # K
MOLECULAR_WEIGHT_AIR = 0.0289644  # kg/mol

STANDARD_GRAVITY = scipy.constants.g
