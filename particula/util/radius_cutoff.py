""" determing the radius cutoff for particle distribution
"""

import numpy as np
from scipy.stats import lognorm

from particula import u
from particula.util.input_handling import in_scalar, in_radius


def cut_rad(
    cutoff=in_scalar(0.9999).m,
    gsigma=in_scalar(1.25).m,
    mode=in_radius(100e-9),
    force_radius_start=None,
    force_radius_end=None,
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

    _ = kwargs.get("something")
    if not isinstance(mode, u.Quantity):
        mode = in_radius(mode)

    if np.array([mode.m]).size == 1:
        (rad_start, rad_end) = lognorm.interval(
            cutoff,
            s=np.log(gsigma),
            scale=mode.m,
        )
    else:
        (rad_start_pre, rad_end_pre) = lognorm.interval(
            cutoff,
            s=np.log(gsigma),
            scale=mode.m,
        )
        rad_start = rad_start_pre.min() or force_radius_start.m
        rad_end = rad_end_pre.max() or force_radius_end.m

    if force_radius_start is not None:
        force_radius_start = in_radius(force_radius_start)
        rad_start = force_radius_start.m
    if force_radius_end is not None:
        force_radius_end = in_radius(force_radius_end)
        rad_end = force_radius_end.m

    return (rad_start, rad_end)
