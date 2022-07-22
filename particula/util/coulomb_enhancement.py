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
from particula.util.input_handling import (in_handling, in_radius, in_scalar,
                                           in_temperature)


class CoulombEnhancement:  # pylint: disable=too-many-instance-attributes

    """ Calculate coulomb-related enhancements

        Attributes:
            radius          (float) [m]
            other_radius    (float) [m]             (default: radius)
            charge          (float) [dimensionless] (default: 0)
            other_charge    (float) [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        radius=None,
        other_radius=None,
        charge=0,
        other_charge=0,
        temperature=298,
        elementary_charge_value=ELEMENTARY_CHARGE_VALUE,
        electric_permittivity=ELECTRIC_PERMITTIVITY,
        boltzmann_constant=BOLTZMANN_CONSTANT,
        **kwargs
    ):
        """ Initialize CoulombEnhancement object
        """

        other_radius = radius if other_radius is None else other_radius

        self.radius = in_radius(radius)
        self.other_radius = in_radius(other_radius)
        self.charge = in_scalar(charge)
        self.other_charge = in_scalar(other_charge)
        self.temperature = in_temperature(temperature)
        self.elem_char_val = in_handling(elementary_charge_value, u.C)
        self.elec_perm = in_handling(electric_permittivity, u.F/u.m)
        self.boltz_const = in_handling(
            boltzmann_constant, u.m**2*u.kg/u.s**2/u.K
        )

        self.kwargs = kwargs

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

        numerator = -1 * self.charge * self.other_charge * (
            self.elem_char_val ** 2
        )
        denominator = 4 * np.pi * self.elec_perm * (
            np.transpose([self.radius.m])*self.radius.u +
            self.other_radius
        )

        return numerator / (
            denominator * self.boltz_const * self.temperature
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

        # return 1 + ratio if ratio >=0, otherwise np.exp(ratio)
        return (
            (1 + ratio) * (ratio >= 0) + np.exp(ratio) * (ratio < 0)
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

        # return ratio/(1-np.exp(-1*ratio)) if ratio != 0, otherwise 1
        return (
            (ratio*(ratio != 0)+1*(ratio == 0))
            / (1-np.exp(-1*ratio)*(ratio != 0))
        ).to_base_units()


def cpr(**kwargs):
    """ Calculate coulomb potential ratio
    """
    return CoulombEnhancement(**kwargs).coulomb_potential_ratio()


def cekl(**kwargs):
    """ Calculate coulombic enhancement kinetic limit
    """
    return CoulombEnhancement(**kwargs).coulomb_enhancement_kinetic_limit()


def cecl(**kwargs):
    """ Calculate coulombic enhancement continuum limit
    """
    return CoulombEnhancement(**kwargs).coulomb_enhancement_continuum_limit()


def coulomb_enhancement_all(**kwargs):
    """ Return all the values above in one call
    """
    cea = CoulombEnhancement(**kwargs)
    return (
        cea.coulomb_potential_ratio(),
        cea.coulomb_enhancement_kinetic_limit(),
        cea.coulomb_enhancement_continuum_limit()
    )
