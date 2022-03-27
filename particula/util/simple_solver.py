""" a simple solver.

    For now, this takes as input:
    - the initial distribution (m**-4)
    - the radius of said distribution (m)
    - the coagulation kernel associated with it (m**3/s)
    - a desired time span in the form of ndarray (unitless, assumed in seconds)

    It returns an array of the with the dimensions
    (time_span, distribution)
    with the units of distribution (m**-4).
"""

import numpy as np
from scipy.integrate import odeint
from particula.util.coagulation_rate import CoagulationRate
from particula import u


def ode_func(_nums, _, _rads, _coag):
    """ function to integrate
    """
    coag = CoagulationRate(
        distribution=_nums * u.m**-4,
        radius=_rads * u.m,
        kernel=_coag * u.m**-4,
    )

    return coag.coag_gain().m - coag.coag_loss().m


class SimpleSolver:
    """ a class to solve the ODE:

        Need:
        1. initial distribution
        2. associated radius
        3. associated coagulation kernel

        Also:
        1. desired time span in seconds (given unitless)
    """

    def __init__(self, **kwargs):
        """ constructor

            kwargs:
            - distribution: initial distribution (m**-4)
            - radius: associated radius (m)
            - kernel: associated coagulation kernel (m**3/s)
            - tspan: desired time span (s)
        """
        self.nums_init = kwargs.get("distribution", None)
        self.rads_init = kwargs.get("radius", None)
        self.coag_kern = kwargs.get("kernel", None)
        self.time_span = kwargs.get("tspan", np.linspace(0, 10, 1000))
        self.kwargs = kwargs

    def prep_inputs(self):
        """ strip units, etc.
        """

        return (
            self.nums_init.m,
            self.rads_init.m,
            self.coag_kern.m,
            self.time_span,
        )

    def solution(self):
        """ utilize scipy.integrate.odeint
        """

        nums, rads, kern, time = self.prep_inputs()

        return odeint(
            ode_func,
            nums,
            time,
            args=(rads, kern)
        )*self.nums_init.u
