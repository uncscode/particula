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

    def _ode_func(self, _nums, _, _rads, _coag):
        """ ode_func
        """
        setattr(
            self,
            'particle_distribution',
            _nums*self.particle_distribution.u
        )
        setattr(
            self,
            'particle_coagulation',
            _coag*self.particle_coagulation.u
        )
        setattr(
            self,
            'particle_coagulation',
            _rads*self.particle_radius.u
        )

        return self.coagulation_gain().m - self.coagulation_loss().m

    def solution(self):
        """ solve the equation
        """

        return odeint(
            func=self._ode_func,
            y0=self.particle.particle_distribution().m,
            t=self.time_span,
            args=(
                self.particle_radius,
                self.particle_coagulation,
            ),
        )*self.particle.particle_distribution().u
