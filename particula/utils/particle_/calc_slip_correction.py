""" calculate slip correction
"""

import numpy as np
from particula import u
from particula.utils.particle_ import knudsen_number


def slip_correction_factor(radius, mfp_air=66.4e-9) -> float:

    """ Returns particle's Cunningham slip correction factor.

        Parameters:
            radius  (float) [m]
            mfp_air (float) [m] (default: 66.4e-9)

        Returns:
                    (float) [dimensionless]

        Dimensionless quantity accounting for non-continuum effects
        on small particles. It is a deviation from Stokes' Law.
        Stokes assumes a no-slip condition that is not correct at
        high Knudsen numbers. The slip correction factor is used to
        calculate the friction factor.

        Thus, the slip correction factor is about unity (1) for larger
        particles (Kn -> 0). Its behavior on the other end of the
        spectrum (smaller particles; Kn -> inf) is more nuanced, though
        it tends to scale linearly on a log-log scale, log Cc vs log Kn.
    """

    if isinstance(radius, u.Quantity):
        radius = radius.to_base_units()
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(mfp_air, u.Quantity):
        mfp_air = mfp_air.to_base_units()
    else:
        mfp_air = u.Quantity(mfp_air, u.m)

    return 1 + knudsen_number(radius, mfp_air) * (
        1.257 + 0.4*np.exp(-1.1/knudsen_number(radius, mfp_air))
    )
