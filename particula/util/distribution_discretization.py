""" discretization of the distribution of the particles
"""
import numpy as np
from scipy.stats import lognorm
from particula import u
from particula.util.input_handling import in_scalar, in_radius


def discretize(
    interval=None,
    disttype="lognormal",
    gsigma=in_scalar(1.25).m,
    mode=in_radius(100e-9).m,
    nparticles=in_scalar(1e5).m,
    **kwargs
):
    """ discretize the distribution of the particles

        Parameters:
            interval    (float) the size interval of the distribution
            distype     (str)   the type of distribution, "lognormal" for now
            gsigma      (float) geometric standard deviation of distribution
            mode        (float) pdf scale (corresponds to mode in lognormal)
    """

    _ = kwargs.get("something", None)

    if interval is None:
        raise ValueError("the 'interval' must be specified!")

    if not isinstance(interval, u.Quantity):
        interval = u.Quantity(interval, " ")

    if disttype != "lognormal":
        raise ValueError("the 'disttype' must be 'lognormal' for now!")

    return ((
        lognorm.pdf(
            x=interval.m,
            s=np.reshape(np.log(gsigma), (np.array([gsigma]).size, 1)),
            scale=np.reshape([mode], (np.array([mode]).size, 1)),
        ) / interval.u
        * np.reshape([nparticles], (np.array([nparticles]).size, 1))
        ).sum(axis=0) /
        np.array([nparticles]).sum() /
        np.max([np.array([mode]).size, np.array([gsigma]).size])
    )
