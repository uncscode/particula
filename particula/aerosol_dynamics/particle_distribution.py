""" Constructing a particle distribution

    This module contains the functions to construct a particle distribution
    from a given set of parameters, like the number of particles, the
    distribution of the particle sizes, etc
"""

from scipy.stats import lognorm
import numpy as np
from particula.util.input_handling import in_scalar, in_volume
from particula.util.radius_cutoff import cut_rad


class ParticleDistribution:
    """ make a class
    """

    def __init__(self, **kwargs):
        """ constructing
        """

        self.mode = kwargs.get("mode", None)
        self.nbins = kwargs.get("nbins", 1000)
        self.nparticles = in_scalar(kwargs.get("nparticles", 1e5))
        self.volume = in_volume(kwargs.get("volume", 1e-6))
        self.gsigma = kwargs.get("gsigma", 1.25)

        self.kwargs = kwargs

    def rad(self):
        """ construct the radius

            this gets 99.99% of the distribution, then makes a radius from it
        """

        (rad_start, rad_end) = cut_rad(.9999, np.log(self.gsigma), self.mode)

        return np.linspace(rad_start, rad_end, self.nbins)

    def lnd(self):
        """ discritize the distribution
        """

        return lognorm.pdf(
            x=self.rad(),
            s=np.log(self.gsigma),
            scale=self.mode,
        )
