""" a class to initiate vapors condensing onto particles
"""
import numpy as np

from particula import u
from particula.environment import Environment
from particula.util.input_handling import (in_concentration, in_density,
                                           in_length, in_molecular_weight,
                                           in_scalar)
from particula.util.kelvin_correction import kelvin_term


class Vapor(Environment):
    """ based on the Environment class
    """

    def __init__(self, **kwargs):
        """ initiating the vapor class
        TODO: add ability for mulitple vapors, with different properties
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
        self.kappa = in_scalar(
            kwargs.get('kappa', 0.2)
        )

        self.kwargs = kwargs

    def driving_force(self):
        """ condensation driving force
        """
        return np.array(
            [self.vapor_concentration.m]
        )*self.vapor_concentration.u

    # def driving_force_water(self, radius, dry_radius):
    #     """ condensation driving force of water
    #     """
    #     # Calculate the Kelvin effect
    #     kelvin_effect = kelvin_term(radius, **self.kwargs)

    #     # gas phase concentration of a species
    #     C_g[species]=y[gas]

    #     # gas phase concentration above the droplet surface
    #     if species == 'H2O':
    #         # for water:
        
    #         # Saturation ratio of water
    #         S_w = (radius**3 - dry_radius**3) / np.maximum(
    #                 radius**3-dry_radius**3*(1.0-self.kappa),
    #                 1.e-30
    #             ) * kelvin_effect
    #         # Equilibrium concentration on the surface
    #         C_surf = S_w * C_star[species](T)

    #     else:
    #         # for other species 
    #         C_surf = K * X[species]*C_star[species](T)
            
    #     # Calculate the diffusion coefficient for each individual species
    #     D_eff = diffusion_coefficient(N_m, R_wet, T, S_w, species)

    #     # condensation equation for individual particle species
    #     dydt[bins] = N_m * 4.0 * np.pi * R_wet * D_eff * (C_g[species]-C_surf)

    #     # condensation equation for gases
    #     dydt[gas] = - sum(dydt[bins])
