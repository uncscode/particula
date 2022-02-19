""" calculate coulombic enhancements
"""

import numpy as np
from particula import u
from particula.constants import (BOLTZMANN_CONSTANT, ELECTRIC_PERMITTIVITY,
                                 ELEMENTARY_CHARGE_VALUE)
from particula.util.input_handling import in_radius, in_scalar, in_temperature


class CoulombEnhancement:

    """ Calculate coulomb-related enhancements

        Attributes:
            radius          (float) [m]
            other_radius    (float) [m]             (default: radius)
            charge          (float) [dimensionless] (default: 0)
            other_charge    (float) [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)
    """

    def __init__(self, **kwargs):

        radius = kwargs.get("radius", "None")
        other_radius = kwargs.get("other_radius", radius)
        charge = kwargs.get("charge", 0)
        other_charge = kwargs.get("other_charge", 0)
        temperature = kwargs.get("temperature", 298)

        self.radius = in_radius(radius)
        self.other_radius = in_radius(other_radius)
        self.charge = in_scalar(charge)
        self.other_charge = in_scalar(other_charge)
        self.temperature = in_temperature(temperature)

    def coulomb_potential_ratio(self) -> float:
        """ Calculates the Coulomb potential ratio.

            Parameters:
                radius          (float) [m]
                other_radius    (float) [m]             (default: radius)
                charge          (int)   [dimensionless] (default: 0)
                other_charge    (int)   [dimensionless] (default: 0)
                temperature     (float) [K]             (default: 298)

            Returns:
                                (float) [dimensionless]
        """

        elem_char_val = ELEMENTARY_CHARGE_VALUE
        elec_perm = ELECTRIC_PERMITTIVITY
        boltz_const = BOLTZMANN_CONSTANT

        numerator = -1 * self.charge * self.other_charge * (
            elem_char_val ** 2
        )
        denominator = 4 * np.pi * elec_perm * (
            self.radius + self.other_radius
        )

        return numerator / (
            denominator * boltz_const * self.temperature
        )

    def coulomb_enhancement_kinetic_limit(self) -> float:
        """ Coulombic coagulation enhancement kinetic limit.

            Parameters:
                radius          (float) [m]
                other_radius    (float) [m]             (default: radius)
                charge          (float) [dimensionless] (default: 0)
                other_charge    (float) [dimensionless] (default: 0)
                temperature     (float) [K]             (default: 298)

            Returns:
                                (float) [dimensionless]
        """

        ratio = self.coulomb_potential_ratio()

        return (
            1 + ratio if ratio >= 0
            else np.exp(ratio)
        )

    def coulomb_enhancement_continuum_limit(self) -> float:
        """ Coulombic coagulation enhancement continuum limit.

            Parameters:
                radius          (float) [m]
                other_radius    (float) [m]             (default: radius)
                charge          (float) [dimensionless] (default: 0)
                other_charge    (float) [dimensionless] (default: 0)
                temperature     (float) [K]             (default: 298)

            Returns:
                                (float) [dimensionless]
        """

        ratio = self.coulomb_potential_ratio()

        return (
            ratio/(1-np.exp(-1*ratio)) if ratio != 0
            else u.Quantity(1, u.dimensionless)
        )


def cekl(**kwargs):
    """ Calculate coulombic enhancement kinetic limit

        Parameters:
            radius          (float) [m]
            other_radius    (float) [m]             (default: radius)
            charge          (float) [dimensionless] (default: 0)
            other_charge    (float) [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)

        Returns:
                                (float) [dimensionless]
    """

    return CoulombEnhancement(**kwargs).coulomb_enhancement_kinetic_limit()


def cecm(**kwargs):
    """ Calculate coulombic enhancement continuum limit

        Parameters:
            radius          (float) [m]
            other_radius    (float) [m]             (default: radius)
            charge          (float) [dimensionless] (default: 0)
            other_charge    (float) [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)

        Returns:
                                (float) [dimensionless]
    """

    return CoulombEnhancement(**kwargs).coulomb_enhancement_continuum_limit()
