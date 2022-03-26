""" calculating the condensation
"""
import numpy as np
from particula import u
from particula.constants import AVOGADRO_NUMBER as avg_num
from particula.particle_distribution import ParticleDistribution
from particula.util.fuchs_sutugin import fsc
from particula.util.input_handling import (in_concentration, in_density,
                                           in_length, in_molecular_weight,
                                           in_scalar)
from particula.util.molecular_enhancement import mol_enh
from particula.util.particle_mass import mass
from particula.util.particle_surface import area
from particula.util.reduced_quantity import reduced_quantity as redq
from particula.util.rms_speed import cbar


class CondensationGrowth(ParticleDistribution):
    """ a class for the condensing vapor

        TODO:
            - Add documentation.
            - Add extra dim for the condensing vapor (redq is problematic).
    """

    def __init__( # pylint: disable=too-many-arguments
        self,
        vapor_radius=1.6e-9*u.m,
        vapor_density=1400*u.kg/u.m**3,
        vapor_concentration=1*u.kg/u.m**3,
        vapor_attachment=1,
        vapor_molecular_weight=200*u.g/u.mol,
        **kwargs
    ):
        """ initializing the class
        """

        super().__init__(**kwargs)

        self.vapor_radius = in_length(vapor_radius)
        self.vapor_density = in_density(vapor_density)
        self.vapor_concentration = in_concentration(vapor_concentration)
        self.vapor_attachment = in_scalar(vapor_attachment)
        self.vapor_molec_wt = in_molecular_weight(vapor_molecular_weight)

        self.kwargs = kwargs

    def driving_force(self):
        """ condensation driving force
        """

        return self.vapor_concentration

    def molecular_enhancement(self):
        """ molecular enhancement
        """

        return mol_enh(
            vapor_size=self.vapor_radius,
            particle_size=self.radius()
        )

    def red_mass(self):
        """ red mass
        """

        return redq(
            np.transpose([self.vapor_molec_wt.m])*self.vapor_molec_wt.u,
            mass(radius=self.radius(), **self.kwargs)*avg_num
        )

    def vapor_speed(self):
        """ vapor speed
        """

        return cbar(molecular_weight=self.red_mass())/4

    def vapor_flux(self):
        """ vapor flux
        """

        return (
            area(radius=self.radius()) *
            self.molecular_enhancement() *
            self.vapor_attachment *
            self.vapor_speed() *
            self.driving_force() *
            fsc(radius=self.radius(), **self.kwargs)
        )

    def particle_growth(self):
        """ particle growth in m/s
        """

        shape_factor = 1
        return (
            self.vapor_flux() * 2 / (
                self.vapor_density *
                self.radius()**2 *
                np.pi *
                shape_factor
            )
        )
