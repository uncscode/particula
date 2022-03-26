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
from particula.particle import BasePreParticle


class ParticleDistribution(BasePreParticle):
    """ Constructing the ParticleDistribution class
    """
    def radius(self):
        """ Returns the radius space of the particles

            Utilizing the utility cut_rad to get 99.99% of the distribution.
            From this interval, radius is made on a linspace with nbins points.
            Note: linspace is used here to practical purposes --- often, the
            logspace treatment will return errors in the discretization due
            to the asymmetry across the interval (finer resolution for smaller
            particles, but much coarser resolution for larger particles).
        """
        return self.pre_radius()

    def discretize(self):
        """ Returns a distribution pdf of the particles

            Utilizing the utility discretize to get make a lognorm distribution
            via scipy.stats.lognorm.pdf:
                interval: the size interval of the distribution
                gsigma  : geometric standard deviation of distribution
                mode    : geometric mean radius of the particles
        """
        return self.pre_discretize()

    def distribution(self):
        """ Returns a distribution pdf of the particles

            Utilizing the utility discretize to get make a lognorm distribution
            via scipy.stats.lognorm.pdf:
                interval: the size interval of the distribution
                gsigma  : geometric standard deviation of distribution
                mode    : geometric mean radius of the particles
        """
        return self.pre_distribution()
