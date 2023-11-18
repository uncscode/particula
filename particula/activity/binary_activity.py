"""Binary activity coefficient model for organic-water mixtures.

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Optional
import numpy as np

from particula.activity.machine_limit import safe_exp
from particula.activity.phase_separation import organic_water_single_phase


# the fit values for the activity model
FIT_LOW = {'a1': [7.089476E+00, -7.711860E+00, -3.885941E+01, -1.000000E+02],
           'a2': [-6.226781E-01, -1.000000E+02, 3.081244E-09, 6.188812E+01],
           's': [-5.988895E+00, 6.940689E+00]}
FIT_MID = {'a1': [5.872214E+00, -4.535007E+00, -5.129327E+00, -2.809232E+01],
           'a2': [-9.740486E-01, -1.000000E+02, 2.109751E+00, -2.367683E+01],
           's': [-1.219164E+00, 4.742729E+00]}
FIT_HIGH = {'a1': [5.921550E+00, -2.528295E+00, -3.883017E+00, -7.898128E+00],
            'a2': [-1.000000E+02, -1.000000E+02, 1.353916E+00, -1.160145E+01],
            's': [-7.868187E-02, 3.650860E+00]}


def activity_coefficients(
        molar_mass_ratio,
        org_mole_fraction,
        oxygen2carbon,
        density,
        functional_group=None,
):
    """
    Activity coefficients for water and organic matter, see Gorkowski (2019)

    Args:
        molar mass ratio (float): water MW / organic MW
        org mole fraction (float): fraction of organic matter
        oxygen2carbon (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficient (dict): dictionary of fit values for low oxygen2carbon
        region

    Returns:
        activity_water (float): activity coefficient of water
        activity_organic (float): activity coefficient of organic matter
        mass_water (float): mass fraction of water
        mass_organic (float): mass fraction of organic matter
    """
    oxygen2carbon, molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon,
        molar_mass_ratio=molar_mass_ratio,
        functional_group=functional_group
    )
    gibbs_mix, derivative_gibbs = gibbs_mix_weight(
        molar_mass_ratio,
        org_mole_fraction,
        oxygen2carbon,
        density,
    )
    # equations S8 S10
    # the func value for component 1 = LOG(activity coeff. water)
    ln_gamma_water = gibbs_mix - org_mole_fraction * derivative_gibbs
    # the func value of the component 2 = LOG(activity coeff. of the organic)
    ln_gamma_org = gibbs_mix + (1.0 - org_mole_fraction) * derivative_gibbs

    gamma_water = safe_exp(ln_gamma_water)
    gamma_org = safe_exp(ln_gamma_org)

    activity_water = gamma_water * (1.0 - org_mole_fraction)
    activity_organic = gamma_org * org_mole_fraction

    mass_water = (1.0 - org_mole_fraction) * molar_mass_ratio / (
        (1.0 - org_mole_fraction) * (molar_mass_ratio - 1) + 1
    )
    mass_organic = 1 - mass_water

    return activity_water, activity_organic, mass_water, mass_organic


def gibbs_of_mixing(
        molar_mass_ratio,
        org_mole_fraction,
        oxygen2carbon,
        density,
        fit_dict
):
    """
    Gibbs free energy of mixing, see Gorkowski (2019). equation S4.

    Args:
        molar mass ratio (float): water MW / organic MW
        org mole fraction (float): fraction of organic matter
        oxygen2carbon (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficient (dict): dictionary of fit values for low oxygen2carbon
        region
    """
    c1 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict['a1'])
    c2 = coefficients_c(molar_mass_ratio, oxygen2carbon, fit_dict['a2'])

    rhor = 0.997 / density  # assumes water is the other fluid

    # equation S3
    # the scaled molar mass ratio of this mixture's components.
    scaled_molar_mass_ratio = molar_mass_ratio * fit_dict['s'][1] \
        * (1.0 + oxygen2carbon) ** fit_dict['s'][0]

    # phi2 is a scaled volume fraction
    phi2 = org_mole_fraction / (
        org_mole_fraction + (1.0 - org_mole_fraction)
        * scaled_molar_mass_ratio / rhor
    )

    # equation S4
    sum1 = c1 + c2 * (1 - 2 * phi2)
    gibbs_mix = phi2 * (1.0 - phi2) * sum1

    # equation s6 the derivative of phi2 with respect to organic x2
    dphi2dx2 = (scaled_molar_mass_ratio / rhor) * (phi2 / org_mole_fraction)**2

    # equation S7
    derivative_gibbs_mix = (
        (1.0 - 2.0 * phi2) * sum1 - 2 * c2 * phi2 * (1.0 - phi2)
    ) * dphi2dx2

    return gibbs_mix, derivative_gibbs_mix


def gibbs_mix_weight(
        molar_mass_ratio,
        org_mole_fraction,
        oxygen2carbon,
        density,
        functional_group: Optional[str] = None,
):
    """
    Gibbs free energy of mixing, see Gorkowski (2019), with weighted
    oxygen2carbon regions

    Args:
        molar mass ratio (float): water MW / organic MW
        org mole fraction (float): fraction of organic matter
        oxygen2carbon (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficient (dict): dictionary of fit values for low oxygen2carbon
        region

    Returns:
        gibbs_mix (float): Gibbs energy of mixing (including 1/RT)
        derivative_gibbs (float): derivative of Gibbs energy with respect to
        mole fraction of organics (includes 1/RT)
    """
    oxygen2carbon, molar_mass_ratio = convert_to_oh_equivalent(
        oxygen2carbon,
        molar_mass_ratio,
        functional_group=functional_group
    )

    weights = bat_blending_weights(molar_mass_ratio, oxygen2carbon)

    if weights[1] > 0:  # if mid region is used
        gibbs_mix_mid, derivative_gibbs_mid = gibbs_of_mixing(
            molar_mass_ratio,
            org_mole_fraction,
            oxygen2carbon,
            density,
            FIT_MID
        )

        if weights[0] > 0:  # if paired with low oxygen2carbon region
            gibbs_mix_low, derivative_gibbs_low = gibbs_of_mixing(
                molar_mass_ratio,
                org_mole_fraction,
                oxygen2carbon,
                density,
                FIT_LOW
            )
            gibbs_mix = weights[0] * gibbs_mix_low + weights[1] * gibbs_mix_mid
            derivative_gibbs = weights[0] * derivative_gibbs_low \
                + weights[1] * derivative_gibbs_mid
        else:  # else paired with high oxygen2carbon region
            gibbs_mix_high, derivative_gibbs_high = gibbs_of_mixing(
                molar_mass_ratio,
                org_mole_fraction,
                oxygen2carbon,
                density,
                FIT_HIGH
            )
            gibbs_mix = weights[2] * gibbs_mix_high + \
                weights[1] * gibbs_mix_mid
            derivative_gibbs = weights[2] * derivative_gibbs_high \
                + weights[1] * derivative_gibbs_mid
    else:  # when only high 2OC region is used
        gibbs_mix, derivative_gibbs = gibbs_of_mixing(
            molar_mass_ratio,
            org_mole_fraction,
            oxygen2carbon,
            density,
            FIT_HIGH
        )
    return gibbs_mix, derivative_gibbs


def convert_to_oh_equivalent(
        oxygen2carbon, molar_mass_ratio, functional_group=None):
    """
    add the OH equivalent conversion to the docstring
    """
    # sourcery skip
    if functional_group is None:
        return oxygen2carbon, molar_mass_ratio
    if functional_group == 'alcohol':
        return oxygen2carbon + 1, molar_mass_ratio + 16  # fix this from SI
    raise ValueError('BAT functional group not recognized')


def bat_blending_weights(molar_mass_ratio, oxygen2carbon):
    """
    Function to estimate the blending weights for the BAT model.

    Args:
    molar_mass_ratio (float): Molar mass ratio of the organic compound.

    Returns:
    blending_weights (array): List of blending weights for the BAT model
        in the low, mid, and high oxygen2carbon regions.
    """

    oxygen2carbon_ml = organic_water_single_phase(molar_mass_ratio)

    blending_weights = np.zeros(3)  # [low, mid, high] oxygen2carbon regions

    # lower to middle oxygen2carbon region
    if oxygen2carbon <= oxygen2carbon_ml * 0.75:
        b_ml = 0.189974476118418
        b_1 = 79.2606902175984
        b_2 = 0.0604293454322489

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml * b_ml
        weight_b = 1 / (1 + np.exp(
            - b_1 * (oxygen2carbon_b - b_2)
        ))  # logistic transfer function

        oxygen2carbon_b_norm = oxygen2carbon - (0.75 * oxygen2carbon_ml * b_ml)

        weight_norm = 1 / (1 + np.exp(
            - b_1 * (oxygen2carbon_b_norm - b_2)
        ))

        blending_weights[1] = weight_b / weight_norm
        blending_weights[0] = 1 - blending_weights[1]

    # middle to high oxygen2carbon region
    elif oxygen2carbon <= oxygen2carbon_ml * 2:
        b_1 = 75.0159268221068
        b_2 = 0.000947111285750515

        oxygen2carbon_b = oxygen2carbon - oxygen2carbon_ml
        blending_weights[2] = 1 / (1 + np.exp(
            - b_1 * (oxygen2carbon_b - b_2)
        ))  # logistic transfer function

        blending_weights[1] = 1 - blending_weights[2]

    else:  # high only region
        blending_weights[2] = 1

    return blending_weights


def coefficients_c(
        molar_mass_ratio,
        oxygen2carbon,
        fit_values
):
    """
    Coefficients for activity model, see Gorkowski (2019). equation S1 S2.

    Args:
        molar mass ratio (float): water MW / organic MW
        fit_values (list): a_n1, a_n2, a_n3, a_n4
    """
    return fit_values[0] * np.exp(fit_values[1] * oxygen2carbon) + fit_values[
        2
    ] * np.exp(fit_values[3] * molar_mass_ratio)
