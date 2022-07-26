""" statics -> dynamics
"""
from scipy.integrate import odeint

from particula.particle import Particle
from particula.util.coagulation_rate import CoagulationRate


class Rates:
    """ The class to calculate the rates
    """

    def __init__(
        self,
        particle=None,
    ):
        """ setting up the class
        """

        if particle is None or not isinstance(particle, Particle):
            raise ValueError("You must provide a baseline Particle object!")

        self.particle = particle

    def coagulation_loss(self):
        """ get the coagulation loss rate from:

            particula.util.coagulation_rate:
            class CoagulationRate
            function coag_loss
        """

        return CoagulationRate(
            distribution=self.particle.particle_distribution(),
            radius=self.particle.particle_radius,
            kernel=self.particle.coagulation(),
        ).coag_loss()

    def coagulation_gain(self):
        """ get coagulation gain rate from:

            particula.util.coagulation_rate:
            class CoagulationRate
            function coag_gain
        """

        return CoagulationRate(
            distribution=self.particle.particle_distribution(),
            radius=self.particle.particle_radius,
            kernel=self.particle.coagulation(),
        ).coag_gain()

    def coagulation_rate(self):
        """ get the coagulation rate by summing the loss and gain rates
        """

        return self.coagulation_gain() - self.coagulation_loss()

    def condensation_growth_rate(self):
        """ condensation
        """

        return self.particle.particle_growth()


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
