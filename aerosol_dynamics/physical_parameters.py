"""
Centralized location for physical parameters.
"""

from . import ureg

# air viscosity in kg/m-s
MEDIUM_VISCOSITY = 18.27e-6 * ureg.kg / ureg.m / ureg.s

# mean free path of air in m
MEAN_FREE_PATH_AIR = 65e-9 * ureg.m

# Boltzmann's constant in m^2 kg s^-2 K^-1
BOLTZMANN_CONSTANT = 1.380649e-21 * ureg.m ** 2 * ureg.kg / ureg.s / ureg.K

# temperature in K
TEMPERATURE = 300 * ureg.K

# elementary charge in C
ELEMENTARY_CHARGE_VALUE = 1.60217662e-19 * ureg.C

# dielectric constant of air
EPSI = 1.0005 # unitless

# permittivity of free space in F/m
EPS0 = 8.85418782e-12 * ureg.F / ureg.m

# permittivity of air
ELECTRIC_PERMITTIVITY = EPSI*EPS0
