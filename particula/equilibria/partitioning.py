"""Equilibrium calculations for the particula thermodynamic model."""

from scipy.optimize import minimize, Bounds
import numpy as np

from particula.activity import binary_activity


def liquid_vapor_obj_function(
    e_j_partition_guess,
    c_star_j_dry,
    concentration_organic_matter,
    gamma_organic_ab,
    mass_fraction_water_ab,
    q_ab,
    molar_mass,
    error_only=True,
):  # pylint: disable=too-many-positional-arguments, too-many-arguments
    # pylint: disable=too-many-locals
    """Objective function for liquid-vapor partitioning."""
    # objective function

    # guess the liquid phase concentration, from partition coefficient
    c_j_liquid = e_j_partition_guess * concentration_organic_matter
    # using the guess, calculate the total liquid phase concentration
    c_liquid_guess = np.sum(c_j_liquid)

    # calculate the aerosol liquid phase (no water) concentration
    # of the alpha phase
    c_j_alpha = c_j_liquid * q_ab[:, 0]
    # calculate the total water concentration of the alpha phase
    # zero division error catch
    denominator_alpha = [
        1 / (1 - mass_factions) if mass_factions < 1 else 0
        for mass_factions in mass_fraction_water_ab[:, 0]
    ]
    c_j_aq_alpha = np.sum(
        c_j_alpha * mass_fraction_water_ab[:, 0] * denominator_alpha
    )  # type: ignore
    # total liquid phase concentration of the alpha phase, including water
    c_j_alpha_total = np.sum(c_j_alpha) + c_j_aq_alpha

    # calculate the aerosol liquid phase (no water) concentration
    # of the beta phase
    c_j_beta = c_j_liquid * q_ab[:, 1]
    # calculate the total water concentration of the beta phase
    # zero division error catch
    denominator_beta = [
        1 / (1 - mass_factions) if mass_factions < 1 else 0
        for mass_factions in mass_fraction_water_ab[:, 1]
    ]
    c_j_aq_beta = np.sum(
        c_j_beta * mass_fraction_water_ab[:, 1] * denominator_beta
    )  # type: ignore
    # total liquid phase concentration of the beta phase, including water
    c_j_beta_total = np.sum(c_j_beta) + c_j_aq_beta

    c_liquid_total = c_j_alpha_total + c_j_beta_total

    # find the mass weighted molar mass of each phase
    mass_weighted_molar_mass_alpha = (
        np.sum(c_j_alpha / molar_mass) + c_j_aq_alpha / 18.015
    )
    mass_weighted_molar_mass_beta = (
        np.sum(c_j_beta / molar_mass) + c_j_aq_beta / 18.015
    )

    # calculate the c_star_j (C*j) of each phase, if the mass weighted molar
    # mass is zero, then the c_star_j is zero
    if mass_weighted_molar_mass_alpha > 0:
        c_star_j_via_alpha = (
            c_star_j_dry
            * gamma_organic_ab[:, 0]
            * q_ab[:, 0]
            * c_liquid_total
            / (molar_mass * mass_weighted_molar_mass_alpha)
        )
    else:
        c_star_j_via_alpha = np.zeros_like(c_star_j_dry)

    if mass_weighted_molar_mass_beta > 0:
        c_star_j_via_beta = (
            c_star_j_dry
            * gamma_organic_ab[:, 1]
            * q_ab[:, 1]
            * c_liquid_total
            / (molar_mass * mass_weighted_molar_mass_beta)
        )
    else:
        c_star_j_via_beta = np.zeros_like(c_star_j_dry)

    # calculate the new ci_star_j (Ci*j) of each phase, weighted by the
    # by the q_ab value for each phase
    c_star_j_new = (
        c_star_j_via_alpha * q_ab[:, 0] + c_star_j_via_beta * q_ab[:, 1]
    )

    # with this new value we can calculate the new e_j_partition (Ej)
    e_j_partition_new = (1 + c_star_j_new / (c_liquid_total + 1e-16)) ** -1

    # with the new e_j_partition (Ej) we can calculate the new c_j_liquid (Cj)
    c_j_liquid_new = e_j_partition_new * concentration_organic_matter
    # The total liquid phase concentration (excluding water) is the sum of
    # the new c_j_liquid (Cj)
    c_liquid_total_new = np.sum(c_j_liquid_new)

    # the error in the guess can be calculated by the difference between the
    # partition coefficient guess/new and the liquid phase concentration
    # guess/new
    error_out = np.sum(
        (
            (e_j_partition_guess - e_j_partition_new) ** 2
            + (c_liquid_guess - c_liquid_total_new) ** 2
        )
    )
    if error_only:
        return error_out

    # now we can calculate the new phase concentrations
    # alpha
    c_j_liquid_new_alpha = c_j_liquid_new * q_ab[:, 0]
    c_j_aq_new_alpha = (
        c_j_liquid_new_alpha * mass_fraction_water_ab[:, 0] * denominator_alpha
    )
    c_liquid_new_alpha = np.sum(c_j_liquid_new_alpha)
    c_aq_new_alpha = np.sum(c_j_aq_new_alpha)
    # beta
    c_j_liquid_new_beta = c_j_liquid_new * q_ab[:, 1]
    c_j_aq_new_beta = (
        c_j_liquid_new_beta * mass_fraction_water_ab[:, 1] * denominator_beta
    )
    c_liquid_new_beta = np.sum(c_j_liquid_new_beta)  # aerosol/organics
    c_aq_new_beta = np.sum(c_j_aq_new_beta)  # water in aerosol
    # total water
    c_liquid_total_water_new = c_aq_new_alpha + c_aq_new_beta

    alpha_phase_output = (
        c_j_liquid_new_alpha,
        c_j_aq_new_alpha,
        c_liquid_new_alpha,
        c_aq_new_alpha,
    )
    beta_phase_output = (
        c_j_liquid_new_beta,
        c_j_aq_new_beta,
        c_liquid_new_beta,
        c_aq_new_beta,
    )
    system_output = (
        c_liquid_total_new,
        c_liquid_total_water_new,
        e_j_partition_new,
        error_out,
    )
    return (alpha_phase_output, beta_phase_output, system_output)


def liquid_vapor_partitioning(
    c_star_j_dry,
    concentration_organic_matter,
    molar_mass,
    gamma_organic_ab,
    mass_fraction_water_ab,
    q_ab,
    partition_coefficient_guess=None,
):  # pylint: disable=too-many-positional-arguments, too-many-arguments
    # pylint: disable=too-many-locals
    """Thermodynamic equilibrium between liquid and vapor phase.
    with activity coefficients,"""

    # clean up nan values
    gamma_organic_ab = np.nan_to_num(gamma_organic_ab)
    mass_fraction_water_ab = np.nan_to_num(mass_fraction_water_ab)
    q_ab = np.nan_to_num(q_ab)

    if partition_coefficient_guess is None:
        partition_coefficient_guess = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    bounds = Bounds(lb=0, ub=1)

    problem = {
        "fun": lambda x: liquid_vapor_obj_function(
            e_j_partition_guess=x,
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            molar_mass=molar_mass,
            error_only=True,
        ),
        "x0": partition_coefficient_guess,
        "bounds": bounds,
    }

    fit_result = minimize(**problem)

    # run the objective function with the optimized parameters
    alpha, beta, system = liquid_vapor_obj_function(
        e_j_partition_guess=fit_result.x,
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic_matter,
        gamma_organic_ab=gamma_organic_ab,
        mass_fraction_water_ab=mass_fraction_water_ab,
        q_ab=q_ab,
        molar_mass=molar_mass,
        error_only=False,
    )
    return alpha, beta, system, fit_result


def get_properties_for_liquid_vapor_partitioning(
    water_activity_desired,
    molar_mass,
    oxygen2carbon,
    density,
):
    """Get properties for liquid-vapor partitioning."""
    oxygen2carbon = np.asarray(oxygen2carbon, dtype=float)
    molar_mass = np.asarray(molar_mass, dtype=float)
    density = np.asarray(density, dtype=float)
    water_activity_desired = np.asarray(water_activity_desired, dtype=float)

    gamma_organic_ab = np.empty([len(oxygen2carbon), 2], dtype=float)
    mass_fraction_water_ab = np.empty([len(oxygen2carbon), 2], dtype=float)
    q_ab = np.empty([len(oxygen2carbon), 2], dtype=float)

    molar_mass_ratio = 18.015 / np.array(molar_mass)

    for i, oxy in enumerate(oxygen2carbon):

        alpha, beta, q_alpha = binary_activity.fixed_water_activity(
            water_activity=water_activity_desired,
            molar_mass_ratio=molar_mass_ratio[i],
            oxygen2carbon=oxy,
            density=density[i],
        )

        gamma_organic_ab[i, 0] = alpha[-1][0]
        mass_fraction_water_ab[i, 0] = alpha[2]
        if beta is None:
            gamma_organic_ab[i, 1] = 0
            mass_fraction_water_ab[i, 1] = 0
        else:
            gamma_organic_ab[i, 1] = beta[-1]
            mass_fraction_water_ab[i, 1] = beta[2]
        q_ab[i, 0] = q_alpha
        q_ab[i, 1] = 1 - q_alpha
    return gamma_organic_ab, mass_fraction_water_ab, q_ab
