""" Calculate friction factor
"""

import numpy as np
from particula import u
from particula.utils.particle_ import slip_correction


def friction_factor(
    radius,
    dyn_vis_air=1.8e-05,
    mfp_air=66.4e-9,
) -> float:

    """ Returns a particle's friction factor.

        Parameters:
            radius      (float)     [m]
            dyn_vis_air (float)     [N*s/m]
            mfp_air     (float)     [m]

        Returns:
                        (float)     [N*s]

        Property of the particle's size and surrounding medium.
        Multiplying the friction factor by the fluid velocity
        yields the drag force on the particle.

        It is best thought of as an inverse of mobility or
        the ratio between thermal energy and diffusion coefficient.
        The modified Stoke's diffusion coefficient is defined as
        kT / (6 * np.pi * dyn_vis_air * radius / slip_corr)
        and thus the friction factor can be defined as
        (6 * np.pi * dyn_vis_air * radius / slip_corr).

        In the continuum limit (Kn -> 0; Cc -> 1):
            6 * np.pi * dyn_vis_air * radius

        In the kinetic limit (Kn -> inf):
            8.39 * (dyn_vis_air/mfp_air) * const * radius**2

        See more: DOI: 10.1080/02786826.2012.690543 (const=1.36)
    """

    if isinstance(radius, u.Quantity):
        radius = radius.to_base_units()
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(dyn_vis_air, u.Quantity):
        dyn_vis_air = dyn_vis_air.to_base_units()
    else:
        dyn_vis_air = u.Quantity(dyn_vis_air, u.N * u.s / u.m)

    if isinstance(mfp_air, u.Quantity):
        mfp_air = mfp_air.to_base_units()
    else:
        mfp_air = u.Quantity(mfp_air, u.m)

    return (
        6 * np.pi * dyn_vis_air * radius /
        slip_correction(radius, mfp_air)
    )
