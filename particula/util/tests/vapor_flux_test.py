""" testing vapor flux
"""

import numpy as np

from particula import u
from particula.util.vapor_flux import phi


def test_vapor_flux():
    """ testing vapor flux
    """

    assert phi(
        particle_area=1e-9*u.m**2,
        molecular_enhancement=1*u.dimensionless,
        vapor_attachment=1*u.dimensionless,
        vapor_speed=1*u.m/u.s,
        driving_force=1*u.kg/u.m**3,
        fsc=1*u.dimensionless
    ).m.size == 1

    assert phi(
        particle_area=[1e-9, 1e-9]*u.m**2,
        molecular_enhancement=np.array([[.1, .2, .3]]),
        vapor_attachment=1*u.dimensionless,
        vapor_speed=1*u.m/u.s,
        driving_force=1*u.kg/u.m**3,
        fsc=[1, 2]*u.dimensionless
    ).m.shape == (2, 3)
