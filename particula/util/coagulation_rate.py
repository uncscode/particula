""" calculate the coag rate
"""

import numpy as np
from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution
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

        dps = rads

    # dpd = np.linspace(0, dps/2**(1/3), fine)
    # dpi = (dps**3 - dpd**3)**(1/3)
    # num_oth = np.triu(np.interp(dpi, dps, nums), k=1) + np.tril(nums)
    # ker_oth = interp(dpi, dps, kern)

        gain = np.zeros_like(dps)

        for i, dpa in enumerate(dps):
            dpd = np.linspace(0, dpa/2**(1/3), 1000)
            dpi = (dpa**3 - dpd**3)**(1/3)

            num_rep = np.interp(dpd, dps, nums)
            num_oth = np.interp(dpi, dps, nums)

            ker_oth = np.interp(dpi, dps, kern[:, i])

            dss = (dpa**3 - dpd**3)**(2/3)

            test = dpa*np.trapz(ker_oth*num_oth*num_rep/dss, dpd)

            gain[i] = test.m

        return gain*test.u
