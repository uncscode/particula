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

    def __init__(self, **kwargs):
        """ initializing the class
        """

        super().__init__(**kwargs)

        self.vapor_radius = in_length(
            kwargs.get('vapor_radius', 1.6e-9)
        )
        self.vapor_density = in_density(
            kwargs.get('vapor_density', 1400)
        )
        self.vapor_concentration = in_concentration(
            kwargs.get('vapor_concentration', 1)
        )
        self.vapor_attachment = in_scalar(
            kwargs.get('vapor_attachment', 1)
        )
        self.vapor_molec_wt = in_molecular_weight(
            kwargs.get('vapor_molecular_weight', 200*u.g/u.mol)
        )

        self.kwargs = kwargs

    def driving_force(self):
        """ condensation driving force
        """

        return np.array(
            [self.vapor_concentration.m]
        )*self.vapor_concentration.u

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
