""" testing f
"""

from particula import u
from particula.util.condensation_growth import CondensationGrowth

simple_dic_kwargs = {
    "mode": 200e-9,  # 200 nm median
    "nbins": 1000,  # 1000 bins
    "nparticles": 1e6,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.2,  # relatively narrow
    "cutoff": .99999,  # let's take it all lol
    "vapor_radius": 1.6e-9,  # 1.6 nm
    "vapor_density": 1400,  # 1400 kg/m^3
    "vapor_concentration": 1,  # 1 ug/m^3
    "vapor_attachment": 1,  # 1
    "vapor_molec_wt": 200,  # 200 g/mol
    "something_something": None
}

cond = CondensationGrowth(**simple_dic_kwargs)


def test_cond_base():
    """ testing units and shapes
    """

    assert cond.radius().m.shape == (1000,)
    assert cond.driving_force().u == u.kg/u.m**3
    # assert cond.driving_force().m.shape == (1,)
    assert cond.molecular_enhancement().u == u.dimensionless
    assert cond.molecular_enhancement().m.shape == (1000,)
    assert cond.red_mass().u == u.kg/u.mol
    assert cond.red_mass().m.shape == (1000,)
    assert cond.vapor_speed().u == u.m/u.s
    assert cond.vapor_speed().m.shape == (1000,)
    assert cond.vapor_flux().u == u.kg/u.s
    assert cond.vapor_flux().m.shape == (1000,)
    assert cond.particle_growth().u == u.m/u.s
    assert cond.particle_growth().m.shape == (1000,)
