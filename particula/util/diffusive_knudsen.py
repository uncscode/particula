""" Calculate the diffusive knudsen number
"""

from particula.constants import BOLTZMANN_CONSTANT
from particula.util.coulomb_enhancement import cecl, cekl
from particula.util.friction_factor import frifac
from particula.util.input_handling import (in_density, in_radius,
                                           in_temperature)
from particula.util.particle_mass import mass
from particula.util.reduced_quantity import reduced_quantity


class DiffusiveKnudsen:
    """ A class for Diff..Knu
    """

    def __init__(self, **kwargs):
        """ define properties """

        radius = kwargs.get("radius", "None")
        other_radius = kwargs.get("other_radius", radius)
        density = kwargs.get("density", 1000)
        other_density = kwargs.get("other_density", density)
        # charge = kwargs.get("charge", 0)
        # other_charge = kwargs.get("other_charge", charge)
        temperature = kwargs.get("temperature", 298)

        self.radius = in_radius(radius)
        self.other_radius = in_radius(other_radius)
        self.density = in_density(density)
        self.other_density = in_density(other_density)
        # self.charge = in_scalar(charge)
        # self.other_charge = in_scalar(other_charge)
        self.temperature = in_temperature(temperature)
        self.kwargs = kwargs

    def get_red_mass(self):
        """ get the reduced mass
        """
        return reduced_quantity(
            mass(radius=self.radius, density=self.density),
            mass(radius=self.other_radius, density=self.other_density)
        )

    def get_rxr(self):
        """ add two radii
        """
        return self.radius + self.other_radius

    def get_red_frifac(self):
        """ get the reduced friction factor
        """

        frifac_kwargs = self.kwargs.copy()
        other_frifac_kwargs = self.kwargs.copy()

        frifac_kwargs.pop("radius", None)
        other_frifac_kwargs.pop("radius", None)

        return reduced_quantity(
            frifac(radius=self.radius, **frifac_kwargs),
            frifac(radius=self.other_radius, **other_frifac_kwargs)
        )

    def get_celimits(self):
        """ get coag enh limits
        """

        lkwargs = self.kwargs.copy()
        return (cekl(**lkwargs), cecl(**lkwargs))

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
            ((self.radius + self.other_radius) * cecl_val / cekl_val)
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


def celimits(**kwargs):
    """ get coag enh limits
    """

    return DiffusiveKnudsen(**kwargs).get_celimits()


def rxr(**kwargs):
    """ add two radii
    """
    print(kwargs)
    return DiffusiveKnudsen(**kwargs).get_rxr()
