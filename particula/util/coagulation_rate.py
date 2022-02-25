""" Calculate the coagulation rate.
"""

import numpy as np


class CoagulationRate:
    """ A class to calculate the coagulation rate.

        Parameters:
            distribution (float): The distribution of particles.
            radius       (float): The radius of the particles.
            kernel       (float): The kernel of the particles.
    """

    def __init__(self, distribution, radius, kernel):
        """ Constructing the class, needing already built
            distribution, radius and kernel.

            * the distribution has units of m**-4
            * the radius has units of m
            * the kernel has units of m**3/s
        """

        self.distribution = distribution
        self.radius = radius
        self.kernel = kernel

    def coag_prep(self):
        """ Repackage the parameters
        """

        nums = self.distribution
        rads = self.radius
        kern = self.kernel

        return nums, rads, kern

    def coag_loss(self):
        """ Returns the coagulation loss rate

            Equation:

            loss_rate(other_radius) = (
                dist(other_radius) *
                integral( # over all space
                    kernel(radius, other_radius) *
                    dist(radius),
                    radius
                )

            Units:

            m**-4/s
        """

        nums, rads, kern = self.coag_prep()

        return nums*np.trapz(kern*nums, rads)

    def coag_gain(self):
        """ Returns the coagulation gain rate

            Equation:

            gain_rate(other_radius) = (
                other_radius**2 *
                integral( # from some_radius=0 to other_radius/2**(1/3)
                    kernel(some_radius, (other_radius**3-some_radius**3)*(1/3)*
                    dist(some_radius) *
                    dist((other_radius**3 - some_radius**3)*(1/3)) /
                    (other_radius**3 - some_radius**3)*(2/3),
                    some_radius
                )
            )

            Units:
                m**-4/s

            This equation necessitates the use of a for-loop due to the
            convoluted use of different radii at different stages.
            This is the costliest step of all coagulation calculations.
            Note, to estimate the kernel and distribution at
            (other_radius**3 - some_radius**3)*(1/3)
            we use interporlation techniques.

            This could be made better in the future.
        """

        # get the parameters
        nums, rads, kern = self.coag_prep()

        # set a radius value
        dps = rads

        # make a sekeloton array for the gain
        gain = np.zeros_like(dps)

        # loop over the radius of interest (dps above)
        for i, dpa in enumerate(dps):

            # make the dummy radius for integration
            dpd = np.linspace(0, dpa/2**(1/3), 1000)

            # get the dummy radius for interpolation
            dpi = (dpa**3 - dpd**3)**(1/3)

            # interpolate the distribution to the dummy radii
            num_rep = np.interp(dpd, dps, nums)
            num_oth = np.interp(dpi, dps, nums)

            # interpolate the kernel to the dummy radius
            # this is WRONG, it needs to be fixed...
            # ker_oth = np.interp(dpd, dps, kern[:, i])
            # PLACEHOLDER FOR NOW... (error is managable)
            ker_oth = np.interp(dpi, dps, kern[i, :])

            # calculate last term
            dss = (dpa**3 - dpd**3)**(2/3)

            # calculate the gain
            test = (dpa**2)*np.trapz(ker_oth*num_oth*num_rep/dss, dpd)

            # store the gain
            gain[i] = test.m

        # return it
        return gain*test.u
