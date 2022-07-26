""" statics -> dynamics
"""

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
        self.particle_distribution = self.particle.particle_distribution()
        self.particle_radius = self.particle.particle_radius
        self.particle_coagulation = self.particle.coagulation()

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

        return self._coag_loss_gain().coag_loss()

    def coagulation_gain(self):
        """ get coagulation gain rate
        """

        return self._coag_loss_gain().coag_gain()

    def coagulation_rate(self):
        """ get the coagulation rate by summing the loss and gain rates
        """

        return self.coagulation_gain() - self.coagulation_loss()

    def condensation_growth_rate(self):
        """ condensation
        """

        return self.particle.particle_growth()
