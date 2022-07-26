""" Calculate the diffusive knudsen number
"""

import numpy as np
from particula.constants import BOLTZMANN_CONSTANT
from particula.util.coulomb_enhancement import (CoulombEnhancement,
                                                coulomb_enhancement_all)
from particula.util.friction_factor import frifac
from particula.util.input_handling import in_density
from particula.util.particle_mass import mass
from particula.util.reduced_quantity import reduced_quantity


class DiffusiveKnudsen(CoulombEnhancement):
    """ A class for Diff..Knu
    """

    def __init__(
        self,
        density=1000,
        other_density=None,
        **kwargs
    ):
        """ define properties """

        super().__init__(**kwargs)

        self.density = in_density(density)
        self.other_density = self.density if other_density is None \
            else in_density(other_density)

        self.kwargs = kwargs

    def get_red_mass(self):
        """ get the reduced mass
        """

        mass_dummy = mass(radius=self.radius, density=self.density)

        return reduced_quantity(
            np.transpose([mass_dummy.m])*mass_dummy.u,
            mass(radius=self.other_radius, density=self.other_density)
        )

    def get_rxr(self):
        """ add two radii
        """
        return np.transpose([self.radius.m])*self.radius.u + self.other_radius

    def get_red_frifac(self):
        """ get the reduced friction factor
        """

        frifac_kwargs = self.kwargs.copy()
        other_frifac_kwargs = self.kwargs.copy()

        frifac_kwargs.pop("radius", None)
        other_frifac_kwargs.pop("radius", None)

        dummy_frifac = frifac(radius=self.radius, **frifac_kwargs)
        return reduced_quantity(
            np.transpose([dummy_frifac.m])*dummy_frifac.u,
            frifac(radius=self.other_radius, **other_frifac_kwargs)
        )

    def get_ces(self):
        """ get coulomb enhancement parameters
        """
        lkwargs = self.kwargs.copy()
        return coulomb_enhancement_all(**lkwargs)

    def get_celimits(self):
        """ get coag enh limits
        """

        lkwargs = self.kwargs.copy()
        return coulomb_enhancement_all(**lkwargs)[1:]

    def get_diff_knu(self):
        """ calculate it
        """
        boltz_const = BOLTZMANN_CONSTANT
        rmass = self.get_red_mass()
        rfrifac = self.get_red_frifac()
        (cekl_val, cecl_val) = self.get_celimits()
        temp = self.temperature
        return (
            ((temp * boltz_const * rmass)**0.5 / rfrifac) /
            (self.get_rxr() * cecl_val / cekl_val)
        )


def diff_knu(**kwargs):
    """ Diffusive Knudsen number.

        The *diffusive* Knudsen number is different from Knudsen number.
        Ratio of:
            - numerator: mean persistence of one particle
            - denominator: effective length scale of
                particle--particle Coulombic interaction

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.diffusive_knudsen import diff_knu
        >>> # with only one radius
        >>> diff_knu(radius=1e-9)
        <Quantity(29.6799, 'dimensionless')>
        >>> # with two radii
        >>> diff_knu(radius=1e-9, other_radius=1e-8)
        <Quantity(3.85387845, 'dimensionless')>
        >>> # with radii and charges
        >>> diff_knu(radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1)
        <Quantity(4.58204028, 'dimensionless')>
        ```
        Parameters:
            radius          (float) [m]
            other_radius    (float) [m]             (default: radius)
            density         (float) [kg/m^3]        (default: 1000)
            other_density   (float) [kg/m^3]        (default: density)
            charge          (int)   [dimensionless] (default: 0)
            other_charge    (int)   [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)

        Returns:
                            (float) [dimensionless]

        Notes:
            this function uses the friction factor and
            the coulomb enhancement calculations; for more information,
            please see the documentation of the respective functions:
                - particula.util.friction_factor.frifac(**kwargs)
                - particula.util.coulomb_enhancement.cekl(**kwargs)
                - particula.util.coulomb_enhancement.cecl(**kwargs)

    """

    return DiffusiveKnudsen(**kwargs).get_diff_knu()


def red_mass(**kwargs):
    """ get the reduced mass
    """

    return DiffusiveKnudsen(**kwargs).get_red_mass()


def red_frifac(**kwargs):
    """ get the reduced friction factor
    """

    return DiffusiveKnudsen(**kwargs).get_red_frifac()


def ces(**kwargs):
    """ get the coulomb enhancement limits
    """

    return DiffusiveKnudsen(**kwargs).get_ces()


def celimits(**kwargs):
    """ get coag enh limits
    """

    return DiffusiveKnudsen(**kwargs).get_celimits()


def rxr(**kwargs):
    """ add two radii
    """

    return DiffusiveKnudsen(**kwargs).get_rxr()
