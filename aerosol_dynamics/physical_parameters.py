"""
Centralized location for physical parameters.
"""

from . import u

# air viscosity in kg/m-s
MEDIUM_VISCOSITY = 18.27e-6 * u.kg / u.m / u.s

# mean free path of air in m
MEAN_FREE_PATH_AIR = 65e-9 * u.m

# Boltzmann's constant in m^2 kg s^-2 K^-1
BOLTZMANN_CONSTANT = 1.380649e-23 * u.m ** 2 * u.kg / (u.s**2) / u.K

# temperature in K
TEMPERATURE = 300 * u.K

# elementary charge in C
ELEMENTARY_CHARGE_VALUE = 1.60217662e-19 * u.C

# Relative permittivity of air at room temperature
# Previously known as the "dielectric constant"
# Often denoted as epsilon
RELATIVE_PERMITTIVITY_AIR = 1.0005  # unitless

# permittivity of free space in F/m
# Also known as the electric constant, permittivity of free space
# Often denoted by epsilon_0
VACUUM_PERMITTIVITY = 8.85418782e-12 * u.F / u.m

# permittivity of air
ELECTRIC_PERMITTIVITY = RELATIVE_PERMITTIVITY_AIR * VACUUM_PERMITTIVITY
