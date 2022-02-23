""" calculate the coag rate
"""

import numpy as np


class CoagulationRate:
    """ calculate the coag rate
    """

    def __init__(self, distribution, radius, kernel):
        """ constructing
        """

        self.distribution = distribution
        self.radius = radius
        self.kernel = kernel

    def coag_prep(self):
        """ return vals for integration
        """

        nums = self.distribution
        rads = self.radius
        kern = self.kernel

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

        gain = np.zeros_like(dps)

        for i, dpa in enumerate(dps):
            dpd = np.linspace(0, dpa/2**(1/3), 1000)
            dpi = (dpa**3 - dpd**3)**(1/3)

            num_rep = np.interp(dpd, dps, nums)
            num_oth = np.interp(dpi, dps, nums)

            ker_oth = np.interp(dpi, dps, kern[:, i])

            dss = (dpa**3 - dpd**3)**(2/3)

            test = (dpa**2)*np.trapz(ker_oth*num_oth*num_rep/dss, dpd)

            gain[i] = test.m

        return gain*test.u
