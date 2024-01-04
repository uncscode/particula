""" statics -> dynamics
"""
from scipy.integrate import odeint, solve_ivp

from particula.rates import Rates


class Solver(Rates):
    """ dynamic solver
    """

    def __init__(
        self,
        time_span=None,
        do_coagulation=True,
        do_condensation=True,
        do_nucleation=True,
        do_dilution=False,
        do_wall_loss=False,
        **kwargs
    ):  # pylint: disable=too-many-arguments
        """ constructor
        """
        super().__init__(**kwargs)

        if time_span is None:
            raise ValueError("You must provide a time span!")

        self.time_span = time_span
        self.do_coagulation = do_coagulation
        self.do_condensation = do_condensation
        self.do_nucleation = do_nucleation
        self.do_dilution = do_dilution
        self.do_wall_loss = do_wall_loss

    def _ode_func(self, _nums, _,):
        """ ode_func
        """
        setattr(
            self,
            'particle_distribution',
            _nums*self.particle_distribution.u
        )

        return self.sum_rates(
            coagulation=self.do_coagulation,
            condensation=self.do_condensation,
            nucleation=self.do_nucleation,
            dilution=self.do_dilution,
            wall_loss=self.do_wall_loss,
        ).m

    def solution(
            self,
            method='odeint',
            **kwargs_ode
    ):
        """ solve the equation
        """
        if method == 'odeint':
            return odeint(
                func=self._ode_func,
                y0=self.particle.particle_distribution().m,
                t=self.time_span,
                **kwargs_ode
            )*self.particle.particle_distribution().u

        if method == 'solve_ivp':
            if 'method' not in kwargs_ode:
                kwargs_ode['method'] = 'BDF'  # Choose an appropriate method
            return solve_ivp(
                fun=self._ode_func,
                t_span=(self.time_span[0], self.time_span[-1]),
                y0=self.particle.particle_distribution().m,
                t_eval=self.time_span,
                **kwargs_ode
            ).y.T*self.particle.particle_distribution().u

        raise ValueError("Invalid method!")
