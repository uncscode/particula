""" Calculate coulombic enhancements
"""

import numpy as np

from particula import u
from particula.utils import (
    BOLTZMANN_CONSTANT,
    ELEMENTARY_CHARGE_VALUE,
    ELECTRIC_PERMITTIVITY,
)


class CoulombEnhancement:

    """ Calculate coulomb-base enhancements

        Attributes:
            radius          (float) [m]
            other_radius    (float) [m]
            charge          (float) [dimensionless] (default: 0)
            other_charge    (float) [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)
    """

    def __init__( # pylint: disable=too-many-arguments
        self,
        radius, other_radius,
        charge=0, other_charge=0,
        temperature=298,
    ):

        if isinstance(radius, u.Quantity):
            self.radius = radius.to_base_units()
        else:
            self.radius = u.Quantity(radius, u.m)

        if isinstance(other_radius, u.Quantity):
            self.other_radius = other_radius.to_base_units()
        else:
            self.other_radius = u.Quantity(other_radius, u.m)

        if isinstance(temperature, u.Quantity):
            self.temperature = temperature.to_base_units()
        else:
            self.temperature = u.Quantity(temperature, u.K)

        for i in [charge]:
            self.charge = i.m if isinstance(i, u.Quantity) else i
        for i in [other_charge]:
            self.other_charge = i.m if isinstance(i, u.Quantity) else i

    def coulomb_potential_ratio(self) -> float:

        """ Calculates the Coulomb potential ratio.

            Parameters:
                radius          (float) [m]
                other_radius    (float) [m]
                charge          (int)   [dimensionless] (default: 0)
                other_charge    (int)   [dimensionless] (default: 0)
                temperature     (float) [K]             (default: 298)

            Returns:
                                (float) [dimensionless]
        """

        numerator = -1 * self.charge * self.other_charge * (
            ELEMENTARY_CHARGE_VALUE ** 2
        )
        denominator = 4 * np.pi * ELECTRIC_PERMITTIVITY * (
            self.radius + self.other_radius
        )

        return numerator / (
            denominator * BOLTZMANN_CONSTANT * self.temperature
        )

    def coulomb_enhancement_kinetic_limit(self) -> float:

        """ Coulombic coagulation enhancement kinetic limit.

            Parameters:
                radius          (float) [m]
                other_radius    (float) [m]
                charge          (float) [dimensionless] (default: 0)
                other_charge    (float) [dimensionless] (default: 0)
                temperature     (float) [K]             (default: 298)

            Returns:
                                (float) [dimensionless]
        """

        ret = self.coulomb_potential_ratio()

        return (
            1 + ret if ret >= 0
            else np.exp(ret)
        )

    def coulomb_enhancement_continuum_limit(self) -> float:

        """ Coulombic coagulation enhancement continuum limit.

            Parameters:
                radius          (float) [m]
                other_radius    (float) [m]
                charge          (float) [dimensionless] (default: 0)
                other_charge    (float) [dimensionless] (default: 0)
                temperature     (float) [K]             (default: 298)

            Returns:
                                (float) [dimensionless]
        """

        ret = self.coulomb_potential_ratio()

        return (
            ret/(1-np.exp(-1*ret)) if ret != 0
            else u.Quantity(1, u.dimensionless)
        )
