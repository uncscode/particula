""" Calculate the diffusive knudsen number
"""

def diffusive_knudsen_number_numerator(
    mass, other_mass,
    radius, other_radius,
    temperature, mean_free_path_air, dynamic_viscosity_air,
) -> float:
    """ something
    """

    numerator = (
        (
            temperature * unitless(BOLTZMANN_CONSTANT)
            * reduced_mass(mass, other_mass)
        )**0.5
        / reduced_friction_factor(
            radius, other_radius,
            mean_free_path_air,
            dynamic_viscosity_air
        )
    )

def diffusive_knudsen_number(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    temperature, mean_free_path_air, dynamic_viscosity_air,
) -> float:

    """Diffusive Knudsen number.

    Parameters:
        charges_array                   (np array)  [unitless]
        charge_other                    (float)     [unitless]
        radii_array                     (np array)  [m]
        radius_other                    (float)     [m]
        mass_array                      (np array)  [kg]
        mass_other                      (float)     [kg]
        temperature                     (float)     [K]
        mean_free_path_air              (float)     [m]
        dynamic_viscosity_air           (float)     [N*s/m]

    Returns:
                                        (array)     [unitless]

    The *diffusive* Knudsen number is different from Knudsen number.
    Ratio of:
        - numerator: mean persistence of one particle
        - denominator: effective length scale of
            particle--particle Coulombic interaction
    """

    numerator = (
        (
            temperature * unitless(BOLTZMANN_CONSTANT)
            * reduced_mass(mass_array, mass_other)
        )**0.5
        / reduced_friction_factor(
            radii_array, radius_other,
            mean_free_path_air,
            dynamic_viscosity_air
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