""" calculate the coag rate
"""

import numpy as np

from particula.aerosol_dynamics.particle_distribution import ParticleDistribution
from particula.util.dimensionless_coagulation import full_coag


class CoagulationRate(ParticleDistribution):
    """ calculate the coag rate
    """

    def __init__(self, **kwargs):
        """ constructing
        """

        super().__init__(**kwargs)
        self.kwargs = kwargs

    def coag_kern(self):
        """ get coag kernel
        """

        return full_coag(radius=self.rad(), **self.kwargs)

    def coag_prep(self):
        """ return vals for integration
        """

        nums = self.lnd()*self.nparticles/self.volume
        rads = self.rad()
        kern = self.coag_kern()

        return nums, rads, kern

    def coag_loss(self):
        """ get coag loss
        """

        nums, rads, kern = self.coag_prep()

        return nums*np.trapz(kern*nums, rads)

    def coag_gain(self):
        """ get coag gain
        """

        nums, rads, kern = self.coag_prep()

        return nums*np.trapz(kern*nums, rads)
