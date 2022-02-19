""" calculate slip correction
"""

import numpy as np
from particula import u
from particula.util.knudsen_number import knudsen_number
from particula.util.mean_free_path import mean_free_path as mfp_def


def slip_correction_factor(radius, mfp=mfp_def()) -> float:

    """ Returns particle's Cunningham slip correction factor.

        Parameters:
            radius  (float) [m]
            mfp     (float) [m] (default: mfp_def())

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
        if radius.to_base_units().u == "meter":
            radius = radius.to_base_units()
        else:
            raise ValueError(f"{radius} must be in meters!")
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(mfp, u.Quantity):
        if mfp.to_base_units().u == "meter":
            mfp = mfp.to_base_units()
        else:
            raise ValueError(f"{mfp} must be in meters!")
    else:
        mfp = u.Quantity(mfp, u.m)

    return 1 + knudsen_number(radius, mfp) * (
        1.257 + 0.4*np.exp(-1.1/knudsen_number(radius, mfp))
    )
