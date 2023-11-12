""" a class to initiate vapors condensing onto particles
"""
import numpy as np

from particula import u
from particula.environment import Environment
from particula.util.input_handling import (in_concentration, in_density,
                                           in_length, in_molecular_weight,
                                           in_scalar)
from particula.util.species_properties import vapor_concentration


class Vapor(Environment):
    """ based on the Environment class
    """

    def __init__(self, **kwargs):
        """ initiating the vapor class
        to add ability for mulitple vapors, with different properties
        """
        super().__init__(**kwargs)

        self.vapor_radius = in_length(
            kwargs.get('vapor_radius', 1.6e-9)
        )
        self.vapor_density = in_density(
            kwargs.get('vapor_density', 1400)
        )
        self.vapor_concentration = in_concentration(
            kwargs.get('vapor_concentration', 0.025e-9)
        )
        self.vapor_attachment = in_scalar(
            kwargs.get('vapor_attachment', 1)
        )
        self.vapor_molec_wt = in_molecular_weight(
            kwargs.get('vapor_molecular_weight', 200*u.g/u.mol)
        )
        self.species = kwargs.get('species_list', ['generic'])

        self.kwargs = kwargs

    def driving_force(self, species=None, surface_saturation_ratio=1):
        """ condensation driving force
        """
        if species == "water":
            # condensation driving force of water vapor
            # gas phase concentration above the particle surface
            particle_surface_concentration = vapor_concentration(
                    saturation_ratio=surface_saturation_ratio,
                    temperature=self.temperature,
                    species="water"
                )

            # the difference between the gas phase concentration and the
            # vapor concentration at the surface is the driving force for
            # condensation or evaporation
            driving_force = (self.water_vapor_concentration()
                             - particle_surface_concentration)

            return np.array(
                [driving_force.m]
            )*driving_force.u
        return np.array(
            [self.vapor_concentration.m]
        )*self.vapor_concentration.u
