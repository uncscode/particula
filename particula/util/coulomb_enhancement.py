""" calculate coulombic enhancements

    TODO:
        * add clarification about charge values (scalar or array)
        * for now, assume scalar charge values only (i.e. one charge at a time)
        * explain ratio.all() or ratio.any() control flow
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

        # self.elem_char_val = ELEMENTARY_CHARGE_VALUE
        # self.elec_perm = ELECTRIC_PERMITTIVITY
        # self.boltz_const = BOLTZMANN_CONSTANT

        if np.array(self.temperature.m).size > 1:
            raise ValueError(
                f"\t\n"
                f"\tTemperature {self.temperature} must be scalar for this!\n"
                f"\tThis is to allow calculation of the coagulation kernel.\n"
                f"\tYou can repeat this routine for different temperatures."
            )

        if np.array(
            self.charge.m
        ).size + np.array(self.other_charge.m).size > 2:
            raise ValueError(
                f"\t\n"
                f"\t Charges {self.charge} and {self.other_charge} must\n"
                f"\tbe scalars for this to work!\n"
                f"\tYou can repeat this routine for different charges."
            )

    def coulomb_potential_ratio(self):
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
            np.transpose([self.radius.m])*self.radius.u +
            self.other_radius
        )

        return numerator / (
            denominator * boltz_const * self.temperature
        )

    def coulomb_enhancement_kinetic_limit(self):
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
            (1 + ratio) if ratio.all() >= 0
            else np.exp(ratio)
        ).to_base_units()

    def coulomb_enhancement_continuum_limit(self):
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
            ratio/(1-np.exp(-1*ratio)) if ratio.all() != 0
            else u.Quantity(1, u.dimensionless)
        ).to_base_units()


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


def cecl(**kwargs):
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
