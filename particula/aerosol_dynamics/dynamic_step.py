""" step dynamically
"""

import numpy as np
from scipy.stats import lognorm

from particula.aerosol_dynamics.particle_distribution import ParticleDistribution
from particula.util.radius_cutoff import cut_rad


class DynamicSteps(ParticleDistribution):
    """ step fwd
    """

    def __init__(self, **kwargs):
        """ constructing
        """

        super().__init__(**kwargs)

        self.mode = kwargs.get("mode", None)
        self.nbins = kwargs.get("nbins", 1000)
        self.nparticles = kwargs.get("nparticles", 1e5)
        self.gsigma = kwargs.get("gsigma", 1.25)

        self.kwargs = kwargs

    