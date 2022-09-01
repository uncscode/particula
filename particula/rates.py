""" statics -> dynamics
"""

import numpy as np

from hypersolver.derivative import ord1_acc4

from particula.particle import Particle
from particula.util.coagulation_rate import CoagulationRate


class Rates:
    """ The class to calculate the rates
    """

    def __init__(
        self,
        particle=None,
        lazy=True,
    ):
        """ setting up the class
        """

        if particle is None or not isinstance(particle, Particle):
            raise ValueError("You must provide a baseline Particle object!")

        self.particle = particle
        self.lazy = lazy
        self.particle_distribution = self.particle.particle_distribution()
        self.particle_radius = self.particle.particle_radius
        self.particle_coagulation = self.particle.coagulation()
        self.particle_formation_rate = self.particle.particle_formation_rate

        if not self.lazy:
            self.eager_coags = CoagulationRate(
                distribution=self.particle_distribution,
                radius=self.particle_radius,
                kernel=self.particle_coagulation,
                lazy=self.lazy
            ).eager_coags

    def _coag_loss_gain(self):
        """ get both loss and gain
        """
        return CoagulationRate(
            distribution=self.particle_distribution,
            radius=self.particle_radius,
            kernel=self.particle_coagulation,
        )

    def coagulation_loss(self):
        """ get the coagulation loss rate
        """
        return self._coag_loss_gain().coag_loss() if self.lazy \
            else self.eager_coags[0]

    def coagulation_gain(self):
        """ get coagulation gain rate
        """
        return self._coag_loss_gain().coag_gain() if self.lazy \
            else self.eager_coags[1]

    def coagulation_rate(self):
        """ get the coagulation rate by summing the loss and gain rates
        """

        return self.coagulation_gain() - self.coagulation_loss()

    def condensation_growth_speed(self):
        """ condensation speed
        """

        return self.particle.particle_growth()

    def condensation_growth_rate(self):
        """ condensation rate
        """
        return ord1_acc4(
            - self.condensation_growth_speed().m * self.particle_distribution.m,
            self.particle_radius.m
        ) * (
            self.condensation_growth_speed().u * self.particle_distribution.u /
            self.particle_radius.u
        )

    def nucleation_rate(self):
        """ nucleation rate
        """
        result = np.zeros(
            self.condensation_growth_rate().m.shape
        )*self.particle_formation_rate.u
        result[0] = self.particle_formation_rate
        return result

    def dilution_rate(self):
        """ dilution rate
        """
        return (
            - self.particle.dilution_rate_coefficient() *
            self.particle_distribution
        )

    def wall_loss_rate(self):
        """ wall loss rate
        """
        return (
            - self.particle.wall_loss_coefficient() *
            self.particle_distribution
        )

    def sum_rates(
        self,
        coagulation=1,
        condensation=1,
        nucleation=1,
    ):
        """ sum rates
        """

        return (
            self.coagulation_rate() * coagulation +
            self.condensation_growth_rate() * condensation +
            self.nucleation_rate() * nucleation
        )
