""" testing for the vapor class
"""

from particula import u
from particula.vapor import Vapor

inputs = {
    "vapor_radius": 1.5e-9,
    "vapor_density": 1500,
}

inputs2 = {
    "vapor_radius": [1.5e-9, 1.6e-9],
    "vapor_density": [1500, 1600],
    "vapor_molecular_weight": [200, 300],
    "vapor_concentration": [1, 2],
}

vaps = Vapor(**inputs)
vaps2 = Vapor(**inputs2)


def test_basics():
    """ testing basic stuff
    """
    assert vaps.vapor_radius == inputs["vapor_radius"] * u.m
    assert vaps.vapor_density == inputs["vapor_density"] * u.kg/u.m**3
    assert vaps.vapor_molec_wt.u == u.kg/u.mol


def test_shapes():
    """ testing sizes and shapes
    """
    assert vaps2.vapor_density.m.shape == (2,)
    assert vaps2.vapor_molec_wt.m.shape == (2,)
    assert vaps2.driving_force().m.shape == (1, 2)


def test_units():
    """ testing units
    """
    assert vaps2.vapor_density.u == u.kg/u.m**3
    assert vaps2.vapor_molec_wt.u == u.kg/u.mol
    assert vaps2.driving_force().u == u.kg/u.m**3
