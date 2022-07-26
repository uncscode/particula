""" Calculate the coagulation rate.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


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

            Note: we strip the units first because we want
                  to allow solvers to use this without issue.
                  These solvers often need pure ndarrays.
                  It would work fine with the units, but
                  it will be throwing warning about stripping
                  them, which is annoying and maybe not good
                  for performance.
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

            Using `RectBivariateSpline` accelerates this significantly.
        """

        nums, rads, kern = self.coag_prep()

        interp = RectBivariateSpline(
            rads.m, rads.m, kern.m * nums.m * np.transpose([nums.m])
        )

        dpd = np.linspace(0, rads.m/2**(1/3), rads.m.size)*rads.u
        dpi = ((np.transpose(rads.m)*rads.u)**3 - dpd**3)**(1/3)

        return rads**2 * np.trapz(
            interp.ev(dpd.m, dpi.m) * kern.u * nums.u * nums.u / dpi**2,
            dpd,
            axis=0
        )
