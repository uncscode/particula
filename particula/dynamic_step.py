""" stepping dynamically... one step at a time
"""


from particula.particle_distribution import ParticleDistribution
from particula.util.coagulation_rate import CoagulationRate
from particula.util.dimensionless_coagulation import full_coag


class DynamicStep(ParticleDistribution):
    """ The class to calculate the rate at one step
    """

    def __init__(self, **kwargs):
        """ Constructiong the class and inheriting from the
            ParticleDistribution class.

            See the ParticleDistribution class for the parameters:
            under particula.aerosol_dynamics.particle_distribution
        """

        super().__init__(**kwargs)

        self.kwargs = kwargs

    def coag_kern(self):
        """ get coagulation kernel

            see:
            particula.util.dimensionless_coagulation:
            function full_coag
        """

        return full_coag(radius=self.radius(), **self.kwargs)

    def coag_loss(self):
        """ get the coagulation loss rate from:

            particula.util.coagulation_rate:
            class CoagulationRate
            function coag_loss
        """

        return CoagulationRate(
            distribution=self.distribution(),
            radius=self.radius(),
            kernel=self.coag_kern(),
        ).coag_loss()

    def coag_gain(self):
        """ get coagulation gain rate from:

            particula.util.coagulation_rate:
            class CoagulationRate
            function coag_gain
        """

        return CoagulationRate(
            distribution=self.distribution(),
            radius=self.radius(),
            kernel=self.coag_kern(),
        ).coag_gain()

    def coag_rate(self):
        """ get the coagulation rate by summing the loss and gain rates
        """

        return self.coag_gain() - self.coag_loss()
