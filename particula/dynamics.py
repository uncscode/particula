""" statics -> dynamics
"""
from scipy.integrate import odeint

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

    def _ode_func(self, _nums, _,):
        """ ode_func
        """
        setattr(
            self,
            'particle_distribution',
            _nums*self.particle_distribution.u
        )

        return self.sum_rates().m

    def solution(self):
        """ solve the equation
        """

        return odeint(
            func=self._ode_func,
            y0=self.particle.particle_distribution().m,
            t=self.time_span,
        )*self.particle.particle_distribution().u
