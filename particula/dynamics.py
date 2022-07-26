""" statics -> dynamics
"""
from scipy.integrate import odeint

from particula.particle import Particle
from particula.util.coagulation_rate import CoagulationRate
from particula.rates import Rates


class Solver(Rates):
    """ dynamic solver
    """

    def __init__(
        self,
        time_span=None,
        **kwargs,
    ):
        """ constructor
        """
        super().__init__(**kwargs)

        if time_span is None:
            raise ValueError("You must provide a time span!")

        self.time_span = time_span

    def _ode_func(self, _nums, _, _rads, _coag):
        """ ode_func
        """
        return self.coagulation_gain().m - self.coagulation_loss().m

    def solution(self):
        """ solve the equation
        """

        return odeint(
            func=self._ode_func,
            y0=self.particle.particle_distribution(),
            t=self.time_span,
            args=(
                self.particle.particle_radius,
                self.particle.coagulation(),
            ),
        )*self.particle.particle_distribution().u
