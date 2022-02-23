""" step dynamically
"""


from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution
from particula.util.coagulation_rate import CoagulationRate
from particula.util.dimensionless_coagulation import full_coag


class DynamicStep(ParticleDistribution):
    """ step fwd
    """

    def __init__(self, **kwargs):
        """ constructing
        """

        super().__init__(**kwargs)

        self.kwargs = kwargs

    def coag_kern(self):
        """ get coag kernel
        """

        return full_coag(radius=self.radius(), **self.kwargs)

    def coag_loss(self):
        """ get coag loss
        """

        return CoagulationRate(
            distribution=self.distribution(),
            radius=self.radius(),
            kernel=self.coag_kern(),
        ).coag_loss()

    def coag_gain(self):
        """ get coag gain
        """

        return CoagulationRate(
            distribution=self.distribution(),
            radius=self.radius(),
            kernel=self.coag_kern(),
        ).coag_gain()

    def coag_rate(self):
        """ get coag rate
        """

        return self.coag_gain() - self.coag_loss()
