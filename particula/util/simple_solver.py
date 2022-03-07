""" a simple solver
"""

import numpy as np
from scipy.integrate import odeint
from particula.util.coagulation_rate import CoagulationRate
from particula.units import strip


def ode_func(_nums, _, _rads, _coag):
    """ function to integrate
    """

    coag = CoagulationRate(
        distribution=_nums,
        radius=_rads,
        kernel=_coag,
    )

    return coag.coag_gain().m - coag.coag_loss().m


class SimpleSolver:
    """ a class to solve the ODE:

        Need:
        1. initial distribution
        2. initial radius
        3. time span

        Also:
        1. coagulation kernel
    """

    def __init__(self, **kwargs):
        """ constructor
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
            strip(self.nums_init),
            strip(self.rads_init),
            strip(self.coag_kern),
            strip(self.time_span)
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
        )
