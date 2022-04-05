""" determing the radius cutoff for particle distribution
"""

import numpy as np
from scipy.stats import lognorm

from particula.util.input_handling import in_scalar, in_radius


def cut_rad(
    cutoff=in_scalar(0.9999).m,
    gsigma=in_scalar(1.25).m,
    mode=in_radius(100e-9).m,
    **kwargs
):
    """ This routine determins the radius cutoff for the particle distribution

        Inputs:
            cutoff  (float) coverage cutoff (default: .9999)
            gsigma  (float) geometric standard deviation (default: 1.25)
            mode    (float) mean radius of the particles (default: 1e-7)

        Returns:
            (starting radius, ending radius) float tuple
    """

    _ = kwargs.get("something", None)

    if np.array([mode]).size == 1:
        (rad_start, rad_end) = lognorm.interval(
            alpha=cutoff,
            s=np.log(gsigma),
            scale=mode,
        )
    else:
        (rad_start_pre, rad_end_pre) = lognorm.interval(
            alpha=cutoff,
            s=np.log(gsigma),
            scale=mode,
        )
        rad_start = rad_start_pre.min()
        rad_end = rad_end_pre.max()

    return (rad_start, rad_end)
