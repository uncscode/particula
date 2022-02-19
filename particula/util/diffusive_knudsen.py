""" Calculate the diffusive knudsen number
"""

from particula.constants import BOLTZMANN_CONSTANT
from particula.util.coulomb_enhancement import cecl, cekl
from particula.util.friction_factor import frifac
from particula.util.input_handling import (in_density, in_radius, in_scalar,
                                           in_temperature)
from particula.util.particle_mass import mass
from particula.util.reduced_quantity import reduced_quantity


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

    radius = kwargs.get("radius", "None")
    other_radius = kwargs.get("other_radius", radius)
    density = kwargs.get("density", 1000)
    other_density = kwargs.get("other_density", density)
    charge = kwargs.get("charge", 0)
    other_charge = kwargs.get("other_charge", charge)
    temperature = kwargs.get("temperature", 298)

    radius = in_radius(radius)
    other_radius = in_radius(other_radius)
    density = in_density(density)
    other_density = in_density(other_density)
    charge = in_scalar(charge)
    other_charge = in_scalar(other_charge)
    temperature = in_temperature(temperature)

    red_mass = reduced_quantity(
        mass(radius=radius, density=density),
        mass(radius=other_radius, density=other_density)
    )

    frifac_kwargs = kwargs.copy()
    other_frifac_kwargs = kwargs.copy()

    frifac_kwargs.pop("radius", None)
    other_frifac_kwargs.pop("radius", None)

    red_frifac = reduced_quantity(
        frifac(radius=radius, **frifac_kwargs),
        frifac(radius=other_radius, **other_frifac_kwargs)
    )

    cekl_val = cekl(**kwargs)
    cecl_val = cecl(**kwargs)

    print(cekl_val.u)
    print(cecl_val.u)

    boltz_const = BOLTZMANN_CONSTANT

    return (
        ((temperature * boltz_const * red_mass)**0.5 / red_frifac) /
        ((radius + other_radius) * cecl_val / cekl_val)
    )
