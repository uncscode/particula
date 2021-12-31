""" A simple, fast, and powerful particle simulator.

    particula is a simple, fast, and powerful particle simulator,
    or at least two of the three, we hope. It is a simple particle
    system that is designed to be easy to use and easy to extend.
    The goal is to have a robust aerosol (gas phase + particle phase)
    simulation system that can be used to answer scientific questions
    that arise for experiments and research discussions.

    The main features of particula are:
    ...

    More details to follow.
"""

# Import pint here to avoid using a different registry in each module.
from pint import UnitRegistry

# u is the unit registry name.
u = UnitRegistry()

# Temperature has the quirky issue that mathematical operations on temperature
# are problematic. e.g. 298.15K + 273.15K != 25C + 0C.
# The following setting converts all offset units to base units prior to
# any operations.
# https://pint.readthedocs.io/en/latest/nonmult.html
u.autoconvert_offset_to_baseunit = True

__version__ = '0.0.4'
