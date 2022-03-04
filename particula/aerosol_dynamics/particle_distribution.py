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

import numpy as np
from particula import u
from particula.util.distribution_discretization import discretize
from particula.util.input_handling import in_scalar, in_volume
from particula.util.radius_cutoff import cut_rad


class ParticleDistribution:
    """ Constructing the ParticleDistribution class
    """

    def __init__(self, **kwargs):
        """ creating the particle distribution via **kwargs

            Parameters:
                spacing (str): "logspace" or "linspace" gradation

            Utilizes:
                - particula.util.radius_cutoff.cut_rad(**kwargs)
                  (rad_start, rad_end) = cut_rad(**kwargs)
                  (kwargs: cutoff, gsigma, mode)
                - particula.util.distribution_discretization.discretize
                  distribution = discretize(interval, **kwargs)
                  (kwargs: gsigma, mode)

        """

        self.spacing = kwargs.get("spacing", "linspace")
        self.nbins = in_scalar(kwargs.get("nbins", 1000)).m
        self.nparticles = in_scalar(kwargs.get("nparticles", 1e5))
        self.volume = in_volume(kwargs.get("volume", 1e-6))

        self.kwargs = kwargs

    def radius(self):
        """ Returns the radius space of the particles

            Utilizing the utility cut_rad to get 99.99% of the distribution.
            From this interval, radius is made on a linspace with nbins points.
            Note: linspace is used here to practical purposes --- often, the
            logspace treatment will return errors in the discretization due
            to the asymmetry across the interval (finer resolution for smaller
            particles, but much coarser resolution for larger particles).
        """

        (rad_start, rad_end) = cut_rad(**self.kwargs)

        if self.spacing == "logspace":
            radius = np.logspace(
                np.log10(rad_start),
                np.log10(rad_end),
                self.nbins
            )
        elif self.spacing == "linspace":
            radius = np.linspace(
                rad_start,
                rad_end,
                self.nbins
            )
        else:
            raise ValueError("Spacing must be 'logspace' or 'linspace'!")

        return radius*u.m

    def discretize(self):
        """ Returns a distribution pdf of the particles

            Utilizing the utility discretize to get make a lognorm distribution
            via scipy.stats.lognorm.pdf:
                interval: the size interval of the distribution
                gsigma  : geometric standard deviation of distribution
                mode    : geometric mean radius of the particles
        """

        return discretize(interval=self.radius(), **self.kwargs)

    def distribution(self):
        """ Returns a distribution pdf of the particles

            Utilizing the utility discretize to get make a lognorm distribution
            via scipy.stats.lognorm.pdf:
                interval: the size interval of the distribution
                gsigma  : geometric standard deviation of distribution
                mode    : geometric mean radius of the particles
        """

        return self.nparticles*self.discretize()/self.volume
