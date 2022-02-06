"""Particle derived properties and interactions.
"""
import numpy as np
from particula.utils import unitless

from particula.aerosol_dynamics import (
    BOLTZMANN_CONSTANT,
#    AVOGADRO_NUMBER,
#    GAS_CONSTANT,
    ELEMENTARY_CHARGE_VALUE,
#    RELATIVE_PERMITTIVITY_AIR,
#    VACUUM_PERMITTIVITY,
    ELECTRIC_PERMITTIVITY,
)

def knudsen_number(radius, mean_free_path_air) -> float:
    """Returns particle's Knudsen number.

    Parameters:
        radius              (float) [m]
        mean_free_path_air  (float) [m]
    returns:
        knudsen_number (float) [unitless]

    The Knudsen number reflects the relative length scales of
    the particle and the suspending fluid (air, water, etc.).
    This is calculated by the mean free path of the medium
    divided by the particle radius.
    """

    return mean_free_path_air / radius


def slip_correction_factor(radius, mean_free_path_air) -> float:
    """Returns particle's Cunningham slip correction factor.

    Parameters:
        radius              (float) [m]
        mean_free_path_air  (float) [m]
    returns:
        slip_correction_factor (float) [unitless]

    Dimensionless quantity accounting for non-continuum effects
    on small particles. It is a deviation from Stokes' Law.
    Stokes assumes a no-slip condition that is not correct at
    high Knudsen numbers. The slip correction factor is used to
    calculate the friction factor.

    See Eq 9.34 in Atmos. Chem. & Phys. (2016) for more information."""

    return 1 + knudsen_number(radius, mean_free_path_air) * (
        1.257 + 0.4*np.exp(-1.1/knudsen_number(radius, mean_free_path_air))
    )


def friction_factor(
    radius,
    mean_free_path_air, dynamic_viscosity_air
) -> float:
    """Returns a particle's friction factor.

    Checks units: [N*s/m]

    Property of the particle's size and surrounding medium.
    Multiplying the friction factor by the fluid velocity
    yields the drag force on the particle.
    """

    return (
        6 * np.pi * dynamic_viscosity_air * radius /
        slip_correction_factor(radius, mean_free_path_air)
    )


def reduced_mass(mass_array, mass_other) -> float:
    """Returns the reduced mass of two particles.

    Parameters:
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
    returns:
        reduced_mass (array) [unitless]

    The reduced mass is an "effective inertial" mass.
    Allows a two-body problem to be solved as a one-body problem.
    """

    return mass_array * mass_other / (mass_array + mass_other)


def reduced_friction_factor(
    radii_array, radius_other, mean_free_path_air, dynamic_viscosity_air
) -> float:
    """Returns the reduced friction factor between two particles.

    Parameters:
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
    returns:
        reduced_friction_factor (array) [unitless]

    Similar to the reduced mass.
    The reduced friction factor allows a two-body problem
    to be solved as a one-body problem.
    """

    return (
        friction_factor(radii_array, mean_free_path_air, dynamic_viscosity_air)
        * friction_factor(radius_other, mean_free_path_air, dynamic_viscosity_air)
        / (
            friction_factor(radii_array, mean_free_path_air, dynamic_viscosity_air)
            + friction_factor(radius_other, mean_free_path_air, dynamic_viscosity_air)
        )
    )


def coulomb_potential_ratio(
    charges_array, charge_other, radii_array, radius_other, temperature
) -> float:
    """Calculates the Coulomb potential ratio.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        temperature  (float) [K]
    returns:
        coulomb_potential_ratio (array) [unitless]
    """

    numerator = -1 * charges_array * charge_other * (
        unitless(ELEMENTARY_CHARGE_VALUE) ** 2
    )
    denominator = 4 * np.pi * unitless(ELECTRIC_PERMITTIVITY) * (
        radii_array + radius_other
    )
    return (
        numerator /
        (denominator * unitless(BOLTZMANN_CONSTANT) * temperature)
    )


def coulomb_enhancement_kinetic_limit(
    charges_array, charge_other, radii_array, radius_other, temperature
) -> float:
    """Kinetic limit of Coulomb enhancement for particle--particle coagulation.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        temperature  (float) [K]
    returns:
        coulomb_enhancement_kinetic_limit (array) [unitless]
    """

    coulomb_potential_ratio_initial = coulomb_potential_ratio(
        charges_array, charge_other, radii_array, radius_other, temperature
    )

    return (
    np.array([1+x if x >= 0 else np.exp(x) for x in coulomb_potential_ratio_initial])
    )


def coulomb_enhancement_continuum_limit(
    charges_array, charge_other, radii_array, radius_other, temperature
) -> float:
    """Continuum limit of Coulomb enhancement for particle--particle coagulation.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        temperature  (float) [K]
    returns:
        coulomb_enhancement_continuum_limit (array) [unitless]    """

    coulomb_potential_ratio_initial = coulomb_potential_ratio(
        charges_array, charge_other, radii_array,
        radius_other, temperature,
    )
    return(
        np.array([(x / 1-np.exp(-1*x)) if x != 0 else 1 for x in coulomb_potential_ratio_initial])
    )

def diffusive_knudsen_number(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
) -> float:
    """Diffusive Knudsen number.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
        temperature  (float) [K]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
    returns:
        diffusive_knudsen_number (array) [unitless]

    The *diffusive* Knudsen number is different from Knudsen number.
    Ratio of:
        - numerator: mean persistence of one particle
        - denominator: effective length scale of
            particle--particle Coulombic interaction
    """

    numerator = (
        (
            temperature * BOLTZMANN_CONSTANT
            * reduced_mass(mass_array, mass_other)
        )**0.5
        / reduced_friction_factor(
            radii_array, radius_other, mean_free_path_air, dynamic_viscosity_air
        )
    )
    denominator = (
        (radii_array + radius_other)
        * coulomb_enhancement_kinetic_limit(
            charges_array, charge_other, radii_array, radius_other, temperature
        )
        / coulomb_enhancement_continuum_limit(
            charges_array, charge_other, radii_array, radius_other, temperature
        )
    )
    return numerator / denominator


def dimensionless_coagulation_kernel_hard_sphere(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
) -> float:
    """Dimensionless particle--particle coagulation kernel.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
        temperature  (float) [K]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
    returns:
        dimensionless_coagulation_kernel_hard_sphere (array) [unitless]
    """

    # Constants for the chargeless hard-sphere limit
    # see doi:
    hsc1 = 25.836
    hsc2 = 11.211
    hsc3 = 3.502
    hsc4 = 7.211
    diffusive_knudsen_number_initial = diffusive_knudsen_number(
        charges_array, charge_other,
        radii_array, radius_other,
        mass_array, mass_other,
        temperature, mean_free_path_air, dynamic_viscosity_air,
    )

    numerator = (
        (4 * np.pi * diffusive_knudsen_number_initial**2)
        + (hsc1 * diffusive_knudsen_number_initial**3)
        + ((8 * np.pi)**(1/2) * hsc2 * diffusive_knudsen_number_initial**4)
    )
    denominator = (
        1
        + hsc3 * diffusive_knudsen_number_initial
        + hsc4 * diffusive_knudsen_number_initial**2
        + hsc2 * diffusive_knudsen_number_initial**3
    )
    return numerator / denominator


def collision_kernel_continuum_limit(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
) -> float:
    """Continuum limit of collision kernel.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
        temperature  (float) [K]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
    returns:
        collision_kernel_continuum_limit (array) [unitless]
    """

    diffusive_knudsen_number_initial = diffusive_knudsen_number(
        charges_array, charge_other,
        radii_array, radius_other,
        mass_array, mass_other,
        temperature, mean_free_path_air, dynamic_viscosity_air,
    )
    return 4 * np.pi * (diffusive_knudsen_number_initial**2)


def collision_kernel_kinetic_limit(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
) -> float:
    """Kinetic limit of collision kernel.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
        temperature  (float) [K]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
    returns:
        collision_kernel_kinetic_limit (array) [unitless]
    """

    diffusive_knudsen_number_initial = diffusive_knudsen_number(
        charges_array, charge_other,
        radii_array, radius_other,
        mass_array, mass_other,
        temperature, mean_free_path_air, dynamic_viscosity_air,
    )
    return np.sqrt(8 * np.pi) * diffusive_knudsen_number_initial


def dimensionless_coagulation_kernel_parameterized(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
    authors: str = "cg2019",
) -> float:
    """Dimensionless particle--particle coagulation kernel.

    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
        temperature  (float) [K]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
        authors:        authors of the parameterization
            - gh2012    doi.org:10.1103/PhysRevE.78.046402
            - cg2019    doi:10.1080/02786826.2019.1614522
            - hard_sphere
            (default: cg2019)
    returns:
        dimensionless_coagulation_kernel_parameterized (array) [dimensionless]
    """
    coulomb_potential_ratio_initial = coulomb_potential_ratio(
        charges_array, charge_other, radii_array, radius_other, temperature
    )

    diffusive_knudsen_number_initial = diffusive_knudsen_number(
        charges_array, charge_other,
        radii_array, radius_other,
        mass_array, mass_other,
        temperature, mean_free_path_air, dynamic_viscosity_air,
    )
    dimensionless_coagulation_kernel_hard_sphere_initial = \
    dimensionless_coagulation_kernel_hard_sphere(
        charges_array, charge_other,
        radii_array, radius_other,
        mass_array, mass_other,
        temperature, mean_free_path_air, dynamic_viscosity_air,
    )

    if authors == "cg2019":
        # some parameters

        corra = 2.5
        corrb = (
            4.528*np.exp(-1.088*coulomb_potential_ratio_initial)
        ) + (
            0.7091*np.log(1 + 1.527*coulomb_potential_ratio_initial)
        )

        corrc = (11.36)*(coulomb_potential_ratio_initial**0.272) - 10.33
        corrk = - 0.003533*coulomb_potential_ratio_initial + 0.05971

        # mu for the parameterization
        corr_mu = (corrc/corra)*(
            (1 + corrk*((np.log(
                diffusive_knudsen_number_initial
            ) - corrb)/corra))**((-1/corrk) - 1)
        ) * (
            np.exp(-(1 + corrk*(np.log(
                diffusive_knudsen_number_initial
            ) - corrb)/corra)**(- 1/corrk))
        )

        answer = (
            # self.dimensionless_coagulation_kernel_hard_sphere(
            #     other, environment
            # ) if self.coulomb_potential_ratio(
            #     other, environment
            # ) <= 0 else
            dimensionless_coagulation_kernel_hard_sphere_initial
            * np.exp(corr_mu)
        )

    elif authors == "gh2012":
        numerator = coulomb_enhancement_continuum_limit(
                charges_array, charge_other, radii_array, radius_other, temperature
        )

        denominator = 1 + 1.598*(np.minimum(
            diffusive_knudsen_number_initial,
            3*diffusive_knudsen_number_initial/2
            / coulomb_potential_ratio_initial
        ))**1.1709

        answer = (
            dimensionless_coagulation_kernel_hard_sphere_initial
            if coulomb_potential_ratio_initial <= 0.5 else
            numerator / denominator
        )

    elif authors == "hard_sphere":
        answer = dimensionless_coagulation_kernel_hard_sphere_initial

    if authors not in ["gh2012", "hard_sphere", "cg2019"]:
        raise ValueError("We don't have this parameterization.")

    return answer


def dimensioned_coagulation_kernel(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
    authors: str = "cg2019",
) -> float:
    """Dimensioned particle--particle coagulation kernel.
    Parameters:
        charges_array  (np array) [dimensionless]
        charge_other  (float) [dimensionless]
        radii_array  (np array) [m]
        radius_other  (float) [m]
        mass_array  (np array) [kg]
        mass_other  (float) [kg]
        temperature  (float) [K]
        mean_free_path_air  (float) [m]
        dynamic_viscosity_air  (float) [N*s/m]
        authors:        authors of the parameterization
            - gh2012    doi.org:10.1103/PhysRevE.78.046402
            - cg2019    doi:10.1080/02786826.2019.1614522
            - hard_sphere
            (default: cg2019)
    returns:
        dimensioned_coagulation_kernel (array) [m**3/s]

    """

    return (
        dimensionless_coagulation_kernel_parameterized(
            charges_array, charge_other,
            radii_array, radius_other,
            mass_array, mass_other,
            temperature, mean_free_path_air, dynamic_viscosity_air,
            authors,
        )
        * reduced_friction_factor(
            radii_array, radius_other,
            mean_free_path_air, dynamic_viscosity_air
        )
        * (
            radii_array + radius_other
        )**3
        * coulomb_enhancement_kinetic_limit(
            charges_array, charge_other, radii_array, radius_other, temperature
        )**2
        / reduced_mass(mass_array, mass_other)
        / coulomb_enhancement_continuum_limit(
            charges_array, charge_other, radii_array, radius_other, temperature
        )
    )
