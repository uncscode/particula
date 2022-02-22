""" step dynamically
"""

from particula.aerosol_dynamics.particle_distribution import \
    ParticleDistribution
from particula.util.dimensionless_coagulation import full_coag


class DynamicStep(ParticleDistribution):
    """ step fwd
    """

    def __init__(self, **kwargs):
        """ constructing
        """

        super().__init__(**kwargs)

        self.radius = self.rad()
        self.kwargs = kwargs

    def coag_kern(self):
        """ get coag kernel
        """

        return full_coag(radius=self.rad(), **self.kwargs)

    def coag_rate(self):
        """ get coag rate
        """

        return (
            self.coag_kern() *
            self.nparticles *
            self.lnd()
        )
