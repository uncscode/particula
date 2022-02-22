""" Constructing a particle distribution

    This module contains the functions to construct a particle distribution
    from a given set of parameters. These parameters often include:
        - mode          mean radius of the particles
        - nbins         number of bins in the radius distribution
        - nparticles    number of particles
        - volume        volume in which particles exist
        - gsigma        geometric standard deviation of distribution

    The distribution is constructed following the probability density
    function of a lognormal distribution. This is customary in the
    aerosol dynamics literature.

"""

from scipy.stats import lognorm
import numpy as np
from particula.util.input_handling import in_scalar, in_volume, in_radius
from particula.util.radius_cutoff import cut_rad


class ParticleDistribution:
    """ Constructing the ParticleDistribution class
    """

    def __init__(self, **kwargs):
        """ creating the particle distribution via **kwargs

            the kwargs are:
                - mode          mean radius of the particles
                - nbins         number of bins in the radius distribution
                - nparticles    number of particles
                - volume        volume in which particles exist
                - gsigma        geometric standard deviation of distribution

        """

        self.mode = in_radius(kwargs.get("mode", None)).m
        self.nbins = in_scalar(kwargs.get("nbins", 1000)).m
        self.nparticles = in_scalar(kwargs.get("nparticles", 1e5))
        self.volume = in_volume(kwargs.get("volume", 1e-6))
        self.gsigma = in_scalar(kwargs.get("gsigma", 1.25)).m

        self.kwargs = kwargs

    def rad(self):
        """ Returns the radius space of the particles

            Utilizing the utility cut_rad to get 99.99% of the distribution.
            From this interval, radius is made on a linspace with nbins points.
            Note: linspace is used here to practical purposes --- often, the
            logspace treatment will return errors in the discretization due
            to the asymmetry across the interval (finer resolution for smaller
            particles, but much coarser resolution for larger particles).
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
