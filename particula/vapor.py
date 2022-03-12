""" a class to initiate vapors condensing onto particles
"""
import numpy as np

from particula import u
from particula.environment import Environment
from particula.util.input_handling import (in_concentration, in_density,
                                           in_length, in_molecular_weight,
                                           in_scalar)


class Vapor(Environment):
    """ based on the Environment class
    """

    def __init__(self, **kwargs):
        """ initiating the vapor class
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
