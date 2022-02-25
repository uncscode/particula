""" discretization of the distribution of the particles
"""
import numpy as np
from scipy.stats import lognorm
from particula import u
from particula.util.input_handling import in_scalar, in_radius


def discretize(**kwargs):
    """ discretize the distribution of the particles

        Parameters:
            interval    (float) the size interval of the distribution
            distype     (str)   the type of distribution, "lognormal" for now
            scale       (float) pdf scale (corresponds to mode in lognormal)
    """

    interval = kwargs.get("interval", None)
    disttype = kwargs.get("disttype", "lognormal")
    gsigma = in_scalar(kwargs.get("gsigma", 1.25)).m
    mode = in_radius(kwargs.get("mode", 100e-9)).m

    if interval is None:
        raise ValueError("the 'interval' must be specified!")

    if not isinstance(interval, u.Quantity):
        interval = u.Quantity(interval, " ")

    if disttype == "lognormal":
        dist = lognorm.pdf(
            x=interval.m,
            s=np.log(gsigma),
            scale=mode,
        )/interval.u
    else:
        raise ValueError("the 'disttype' must be 'lognormal' for now!")

    return dist
