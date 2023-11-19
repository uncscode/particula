# linting disabled until reformatting of this file
# pylint: disable=all
# flake8: noqa
# pytype: skip-file

# get Csat, mass fractions a/b, activity coefficients a/b
# optimize Ej once
# repeat for next RH value

from scipy.optimize import minimize, Bounds
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize, Bounds
from scipy.optimize import basinhopping, differential_evolution
import numpy as np


def VBS_equilibration_withLLEpartition_objFun_KGv3(
        guess_Ej, C_OM_ugPm3, Cstar_dry, activity_coefficient_AB, q_alpha_molefrac_phase_split_org, mass_fraction_water_AB, molecular_weight):

    mass_fraction_water_alpha = mass_fraction_water_AB[:, 0]
    mass_fraction_water_beta = mass_fraction_water_AB[:, 1]
    activity_coefficient_alpha = activity_coefficient_AB[:, 0]
    activity_coefficient_beta = activity_coefficient_AB[:, 1]

    q_alpha = q_alpha_molefrac_phase_split_org
    q_beta = 1 - q_alpha_molefrac_phase_split_org

    mass_fraction_water_alpha_denominator = (1 - mass_fraction_water_alpha)
    mass_fraction_water_beta_denominator = (1 - mass_fraction_water_beta)

    alpha_good_denominators = mass_fraction_water_alpha_denominator > 0
    beta_good_denominators = mass_fraction_water_beta_denominator > 0

    mass_fraction_water_alpha_denominator = abs(
        mass_fraction_water_alpha_denominator) * alpha_good_denominators + np.logical_not(alpha_good_denominators)
    mass_fraction_water_beta_denominator = abs(
        mass_fraction_water_beta_denominator) * beta_good_denominators + np.logical_not(beta_good_denominators)

    Coa_j = guess_Ej * C_OM_ugPm3
    Coa_guess_viaEj = np.sum(Coa_j)

    Coa_j_alpha = Coa_j * q_alpha
    Caq_alpha = np.sum(
        Coa_j_alpha *
        mass_fraction_water_alpha /
        mass_fraction_water_alpha_denominator)
    Coaaq_alpha = np.sum(Coa_j_alpha) + Caq_alpha

    Coa_j_beta = Coa_j * q_beta
    Caq_beta = np.sum(
        Coa_j_beta *
        mass_fraction_water_beta /
        mass_fraction_water_beta_denominator)
    Coaaq_beta = np.sum(Coa_j_beta) + Caq_beta

    Coaaq_via_Ej_guess = Coaaq_beta + Coaaq_alpha

    massweighted_molar_weight_alpha = np.sum(
        Coa_j_alpha / molecular_weight) + Caq_alpha / 18.015

    if massweighted_molar_weight_alpha > 0:
        Cstar_j_via_alpha = Cstar_dry * activity_coefficient_alpha * q_alpha * \
            Coaaq_via_Ej_guess / (molecular_weight *
                                  massweighted_molar_weight_alpha)
    else:
        Cstar_j_via_alpha = q_alpha * 0

    massweighted_molar_weight_beta = np.sum(
        Coa_j_beta / molecular_weight) + Caq_beta / 18.015

    if massweighted_molar_weight_beta > 0:
        Cstar_j_via_beta = Cstar_dry * activity_coefficient_beta * q_beta * \
            Coaaq_via_Ej_guess / (molecular_weight *
                                  massweighted_molar_weight_beta)
    else:
        Cstar_j_via_beta = q_beta * 0

    Cstar_j_used = Cstar_j_via_alpha * q_alpha + Cstar_j_via_beta * q_beta

    Ej_new = (1 + Cstar_j_used / Coaaq_via_Ej_guess)**-1

    Coa_j_new = Ej_new * C_OM_ugPm3
    Coa_new_viaEj = np.sum(Coa_j_new)

    Coa_j_alpha_new = Coa_j_new * q_alpha
    Caq_j_alpha_new = Coa_j_alpha_new * mass_fraction_water_alpha / \
        mass_fraction_water_alpha_denominator
    Caq_alpha_new = np.sum(Caq_j_alpha_new)
    Coa_alpha_new = np.sum(Coa_j_alpha_new)

    Coa_j_beta_new = Coa_j_new * q_beta
    Caq_j_beta_new = Coa_j_beta_new * mass_fraction_water_beta / \
        mass_fraction_water_beta_denominator
    Caq_beta_new = np.sum(Caq_j_beta_new)
    Coa_beta_new = np.sum(Coa_j_beta_new)

    q_alpha_water_new = Caq_alpha_new / (Caq_alpha_new + Caq_beta_new)

    error_out = np.sum((guess_Ej - Ej_new)**2 +
                       (Coa_guess_viaEj - Coa_new_viaEj)**2)

    partition_coefficients = [Ej_new]
    q_alpha_water = q_alpha_water_new
    Coa_j_AB = [Coa_j_alpha_new, Coa_j_beta_new]
    Caq_j_AB = [Caq_j_alpha_new, Caq_j_beta_new]
    Coa_AB = [Coa_alpha_new, Coa_beta_new]
    Caq_AB = [Caq_alpha_new, Caq_beta_new]
    Coa = np.sum(Coa_AB)
    Cstar_j = Cstar_j_used

    return partition_coefficients, Coa_j_AB, Caq_j_AB, Cstar_j, Coa_AB, Caq_AB, Coa, q_alpha_water, error_out


def VBS_equilibration_withLLEpartition_KGv2(guess_C_OAalpha_ugPm3, guess_C_OAbeta_ugPm3, guess_partition_coefficients, C_OM_ugPm3, Cstar_dry, activity_coefficient_AB,
                                            q_alpha_molefrac_phase_split_org, mass_fraction_water_AB, molecular_weight, a_water, O2C_values, BAT_functional_group, VBSBAT_options):

    Molar_mass_ratios = 18.015 / molecular_weight
    if guess_C_OAbeta_ugPm3 > 0:
        guess_C_oaaqbeta_ugPm3 = guess_C_OAbeta_ugPm3 / \
            (1 - np.mean(mass_fraction_water_AB[:, 1]))
    elif guess_C_OAalpha_ugPm3 > 0:
        guess_C_oaaqbeta_ugPm3 = guess_C_OAalpha_ugPm3 / \
            (1 - np.mean(mass_fraction_water_AB[:, 1]))
    else:
        guess_C_oaaqbeta_ugPm3 = 1 / \
            (1 - np.mean(mass_fraction_water_AB[:, 1]))

    if np.sum(
            guess_partition_coefficients) == 0 or VBSBAT_options['optimization']['independent_aw'].lower() == 'yes':

        if VBSBAT_options['VBSBAT_NN_options']['use_NN_for_VBS_initial_guess'].lower(
        ) == 'no':  # guess for Ej using approximation
            Cstar_j_guess = Cstar_dry * activity_coefficient_AB[:, 1] / (
                1 + mass_fraction_water_AB[:, 1] * (1 / Molar_mass_ratios - 1))  # not right but fine for guess
            Ej_guess = (1 + Cstar_j_guess / guess_C_oaaqbeta_ugPm3) ** -1

    # use previous Ej_guess results
    elif VBSBAT_options['optimization']['independent_aw'].lower() == 'no':
        Ej_guess = guess_partition_coefficients
    else:
        raise ValueError(
            ' Select VBSBAT_options.optimization.independent_aw option either yes or no')

    # error in guess, runs the cost funciton to see if there is an error in
    # guess.
    guess_error = VBS_equilibration_withLLEpartition_objFun_KGv3(
        Ej_guess,
        C_OM_ugPm3,
        Cstar_dry,
        activity_coefficient_AB,
        q_alpha_molefrac_phase_split_org,
        mass_fraction_water_AB,
        molecular_weight)[
        -1]

    # check if refinement is needed
    if VBSBAT_options['optimization']['guess_refinement_threshold'] <= guess_error:
        S_om = C_OM_ugPm3.shape
        bounds = Bounds(
            np.ones(
                S_om[0]) * 10**-10,
            np.ones(
                S_om[0]))  # Coa qAlpha and Ej bounds

        # options for fmincon optimization
        options = {'disp': False, 'adaptive': True}

        problem = {
            'fun': lambda x: VBS_equilibration_withLLEpartition_objFun_KGv3(x, C_OM_ugPm3, Cstar_dry, activity_coefficient_AB, q_alpha_molefrac_phase_split_org, mass_fraction_water_AB, molecular_weight)[-1],
            'x0': Ej_guess,
            'bounds': bounds,
            'options': options,
        }

        if VBSBAT_options['optimization']['opt_method'].lower(
        ) == 'fmincon':  # a single initialization point which is default
            fit_result = minimize(**problem)
            fit_CoaEj, fit_exit_flag = fit_result.x, fit_result.success
        # multiple start points taken at random
        elif VBSBAT_options['optimization']['opt_method'].lower() == 'global':
            fit_result = basinhopping(**problem)
            fit_CoaEj, fit_exit_flag = fit_result.x, fit_result.message
        elif VBSBAT_options['optimization']['opt_method'].lower() == 'none':
            fit_CoaEj = Ej_guess
            fit_exit_flag = np.NaN
        else:
            raise ValueError(
                'Invalid VBSBAT_options.optimization.opt_method... possible options fmincon, global, none')
    else:  # guess has a lower error than guess_refinement_threshold
        fit_CoaEj = Ej_guess
        fit_exit_flag = np.NaN

    # runs a final calcuation for output
    partition_coefficients_AB, Coa_j_AB, Caq_j_AB, Cstar_j, Coa_AB, Caq_AB, Coa, q_alpha_water, error_out = VBS_equilibration_withLLEpartition_objFun_KGv3(
        fit_CoaEj, C_OM_ugPm3, Cstar_dry, activity_coefficient_AB, q_alpha_molefrac_phase_split_org, mass_fraction_water_AB, molecular_weight)

    return partition_coefficients_AB, Coa_j_AB, Caq_j_AB, Cstar_j, Coa_AB, Caq_AB, Coa, q_alpha_water, fit_exit_flag, error_out


def VBS_equilibration_extractCsat_withLLEpartition_KGv2(
        Cp_j_VBSold, Cstar_j_VBSold, aw_measurment, molecular_weight, O2C_values, H2C_values, BAT_functional_group, BAT_refinement_mode, N2C_values_denistyOnly=None):

    if N2C_values_denistyOnly is None:
        N2C_values_denistyOnly = np.zeros_like(molecular_weight)

    S = Cp_j_VBSold.shape

    Molar_mass_ratios = 18.016 / molecular_weight

    aw_vec = np.full((S[0], 1), aw_measurment)

    McGlashan_fit_tolerance = 10**-8

    _, mole_frac_org_beta = inverted_NNMcGlashan_v8(
        O2C_values, H2C_values, Molar_mass_ratios, aw_vec, BAT_functional_group)

    mass_fraction_water_beta = np.zeros_like(molecular_weight)
    activity_coefficient_beta = np.zeros_like(mass_fraction_water_beta)

    for i in range(S[0]):
        mole_frac_bounds_beta = [0, 1]
        if BAT_refinement_mode.lower() == 'interpolate':
            _, _, _, activity_coefficient_beta[i, 0], _, _, mass_fraction_water_beta[i, 0], _, _, _, _ = \
                BAT_activity_calc_with_refinement_v1(mole_frac_org_beta[i, 0], O2C_values[i, 0], H2C_values[i, 0],
                                                     Molar_mass_ratios[i, 0], BAT_functional_group, None, [BAT_refinement_mode, 'beta'], aw_vec[i, 0], N2C_values_denistyOnly)
        else:
            _, _, _, activity_coefficient_beta[i, 0], _, _, mass_fraction_water_beta[i, 0], _, _, _, _ = \
                BAT_activity_calc_with_refinement_v1(mole_frac_org_beta[i, 0], O2C_values[i, 0], H2C_values[i, 0],
                                                     Molar_mass_ratios[i, 0], BAT_functional_group, None, BAT_refinement_mode, aw_vec[i, 0], N2C_values_denistyOnly)

    q_beta = 1

    # New Coa j
    Coa_j = Cp_j_VBSold

    # beta
    Coa_j_beta = Coa_j * q_beta
    Caq_beta = np.sum(Coa_j_beta * mass_fraction_water_beta /
                      (1 - mass_fraction_water_beta))
    Coaaq_beta = np.sum(Coa_j_beta) + Caq_beta

    Coaaq_via_Ej_guess = Coaaq_beta

    # C* via beta phase
    massweighted_molar_weight_beta = np.sum(
        Coa_j_beta / (molecular_weight)) + Caq_beta / 18.015

    Csat_approx = Cstar_j_VBSold / (activity_coefficient_beta * q_beta *
                                    Coaaq_via_Ej_guess / (molecular_weight * massweighted_molar_weight_beta))

    return Csat_approx
