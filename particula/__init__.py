""" a simple, fast, and powerful particle simulator.

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
# from pint import UnitRegistry


# import main items to expose
from particula.environment import Environment  # noqa: F401
from particula.vapor import Vapor  # noqa: F401
from particula.particle import Particle  # noqa: F401
from particula.rates import Rates  # noqa: F401
from particula.dynamics import Solver  # noqa: F401

from particula.units import u  # noqa: F401
# u is the unit registry name.
# u = UnitRegistry(force_ndarray=True)

__version__ = "0.0.12"
