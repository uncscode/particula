""" Calculate the diffusive knudsen number
"""

from particula import u
from particula.utils import (
    BOLTZMANN_CONSTANT as BOL,
)
from particula.utils.particle_ import (
    particle_mass as pm,
    reduced_quantity as rq,
    friction_factor as ff,
    CoulombEnhancement as CE,
)


# pylint: disable=too-many-arguments, too-many-branches, too-many-locals
def diffusive_knudsen_number(
    radius, other_radius,
    density=1000, other_density=1000,
    charge=0, other_charge=0,
    temperature=298,
    mfp_air=66.4e-9,
    dyn_vis_air=1.8e-05,
) -> float:

    """ Diffusive Knudsen number.

        Parameters:
            radius          (float) [m]
            other_radius    (float) [m]
            density         (float) [kg/m^3]        (default: 1000)
            other_density   (float) [kg/m^3]        (default: 1000)
            charge          (int)   [dimensionless] (default: 0)
            other_charge    (int)   [dimensionless] (default: 0)
            temperature     (float) [K]             (default: 298)
            mfp_air         (float) [m]             (default: 66.4e-9)
            dyn_vis_air     (float) [kg/m/s]        (default: 1.8e-05)

        Returns:
                            (float) [dimensionless]

        The *diffusive* Knudsen number is different from Knudsen number.
        Ratio of:
            - numerator: mean persistence of one particle
            - denominator: effective length scale of
                particle--particle Coulombic interaction
    """

    if isinstance(radius, u.Quantity):
        radius = radius.to_base_units()
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(other_radius, u.Quantity):
        other_radius = other_radius.to_base_units()
    else:
        other_radius = u.Quantity(other_radius, u.m)

    if isinstance(density, u.Quantity):
        density = density.to_base_units()
    else:
        density = u.Quantity(density, u.kg*u.m**-3)

    if isinstance(other_density, u.Quantity):
        other_density = other_density.to_base_units()
    else:
        other_density = u.Quantity(other_density, u.kg*u.m**-3)

    if isinstance(temperature, u.Quantity):
        temperature = temperature.to_base_units()
    else:
        temperature = u.Quantity(temperature, u.K)

    for i in [charge, other_charge]:
        i = i.m if isinstance(i, u.Quantity) else i

    if isinstance(mfp_air, u.Quantity):
        mfp_air = mfp_air.to_base_units()
    else:
        mfp_air = u.Quantity(mfp_air, u.m)

    if isinstance(dyn_vis_air, u.Quantity):
        dyn_vis_air = dyn_vis_air.to_base_units()
    else:
        dyn_vis_air = u.Quantity(dyn_vis_air, u.kg/u.m/u.s)

    r_mass = rq(
        pm(radius, density),
        pm(other_radius, other_density)
    )
    r_frictionf = rq(
        ff(radius, dyn_vis_air, mfp_air),
        ff(other_radius, dyn_vis_air, mfp_air)
    )
    cecl = CE(
        radius, other_radius, charge, other_charge, temperature
    ).coulomb_enhancement_continuum_limit()
    cekl = CE(
        radius, other_radius, charge, other_charge, temperature
    ).coulomb_enhancement_kinetic_limit()

    return (
        ((temperature * BOL * r_mass)**0.5 / r_frictionf) /
        ((radius + other_radius) * cecl / cekl)
    )
