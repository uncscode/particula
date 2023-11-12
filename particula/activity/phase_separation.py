
# %%
# consider changing o2c to oxygen2carbon
# define convert_to_OH_equivalent
# add test

# Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
# Relative-humidity-dependent organic aerosol thermodynamics
# via an efficient reduced-complexity model.
# Atmospheric Chemistry and Physics
# https://doi.org/10.5194/acp-19-13383-2019

# linting disabled until reformatting of this file
# pylint: disable=all
# flake8: noqa
# pytype: skip-file

import numpy as np


def to_molarmass_ratio(molar_mass, other_molar_mass=18.01528):
    """
    Convert the given molar mass to a molar mass ratio with respect to water.
    (MW water / MW organic)

    Parameters:
    molar_mass (np.array): The molar mass of the organic compound.
    other_molar_mass (float, optional): The molar mass of the other compound.
        Defaults to 18.01528.

    Returns:
    np.array: The molar mass ratio with respect to water.
    """
    if isinstance(molar_mass, list):
        return [other_molar_mass / mm for mm in molar_mass]
    else:
        return other_molar_mass / molar_mass


def from_molarmass_ratio(molar_mass_ratio, other_molar_mass=18.01528):
    """
    Convert the given molar mass ratio (MW water / MW organic) to a
    molar mass with respect to the other compound.

    Parameters:
    molar_mass_ratio (np.array): The molar mass ratio with respect to water.
    other_molar_mass (float, optional): The molar mass of the other compound.
        Defaults to 18.01528.

    Returns:
    np.array: The molar mass of the organic compound.
    """
    if isinstance(molar_mass_ratio, list):
        return [other_molar_mass * mm for mm in molar_mass_ratio]
    else:
        return other_molar_mass * molar_mass_ratio


def organic_water_single_phase(molar_mass_ratio):
    """
    Convert the given molar mass ratio (MW water / MW organic) to a
    and O2C value were above is a single phase with water and below
    phase separation is possible.

    Parameters:
    molar_mass_ratio np.: The molar mass ratio with respect to water.

    Returns:
    float: The single phase cross point.
    """

    return (
        0.205 / (1 + np.exp(26.6 * (molar_mass_ratio - 0.12))) ** 0.843 + 0.225
    )


def convert_to_OH_equivalent(O2C, molarmass_ratio, BAT_functional_group=None):
    # Define this function
    return O2C, molarmass_ratio


def organic_density_estimate(
        M,
        O2C,
        H2C=None,
        N2C=None,
        mass_ratio_convert=False
):
    """
    Function to estimate the density of organic compounds based on the simple
    model by Girolami (1994). The input parameters include molar mass, O:C
    and H:C ratios. If the H:C ratio is unknown at input, enter a negative
    value. The actual H:C will then be estimated based on an initial assumption
    of H:C = 2. The model also estimates the number of carbon atoms per
    molecular structure based on molar mass, O:C, and H:C.
    The density is then approximated by the formula of Girolami.

    Reference:
    Girolami, G. S.: A Simple 'Back of the Envelope' Method for Estimating
    the Densities and Molecular Volumes of Liquids and Solids,
    J. Chem. Educ., 71(11), 962, doi:10.1021/ed071p962, 1994.

    Parameters:
        M (float): Molar mass.
        O2C (float): O:C ratio.
        H2C (float): H:C ratio. If unknown, provide a negative value.
        N2C (float, optional): N:C ratio. Defaults to None.

    Returns:
        densityEst (float): Estimated density in g/cm^3.
    """
    if N2C is None:
        N2C = M * 0
    if H2C is None:
        H2C = M * 0
    if mass_ratio_convert:
        M = from_molarmass_ratio(M)

    mass_C = 12.01  # the molar masses in [g/mol]
    mass_O = 16.0
    mass_H = 1.008
    mass_N = 14.0067

    # 1) Estimate the H2C value if not provided from input
    # Assuming an aliphatic compound with H2C = 2.0 in the absence of
    # functional groups, then correct for oxygen content assuming a linear
    # -1 slope (Van Krevelen diagram for typical SOA)
    H2Cest = 2.0 - O2C if H2C < 0.1 else H2C
    # 2) Compute the approximate number of carbon atoms per organic molecule
    NC = M / (mass_C + H2Cest * mass_H + O2C * mass_O + N2C * mass_N)

    # 3) Compute density estimate based on method by Girolami (1994)
    # Here no correction is applied for rings and aromatic compounds
    # (due to limited info at input)
    rho1 = M / (5.0 * NC * (2.0 + H2Cest + O2C * 2.0 + N2C * 2.0))
    # the returned denisty is in [g/cm^3]; and scaled assuming that most
    # that most of the oxygen atoms are able to make H-bonds 
    # (donor or acceptor)
    return rho1 * (1.0 + min(NC * O2C * 0.1 + NC * N2C * 0.1, 0.3))


def bat_blending_weights(molarmass_ratio, O2C):
    """
    Function to estimate the blending weights for the BAT model.

    Parameters:
    -----------
    molarmass_ratio (float): Molar mass ratio of the organic compound.

    Returns:
    --------
    blending_weights (array): List of blending weights for the BAT model
        in the low, mid, and high O2C regions.
    """

    O2C_ml = organic_water_single_phase(molarmass_ratio)

    blending_weights = np.zeros(3)  # [low, mid, high] O2C regions

    if O2C <= O2C_ml * 0.75:  # lower to mid O2C region
        b_ml = 0.189974476118418
        b_1 = 79.2606902175984
        b_2 = 0.0604293454322489

        O2C_b = O2C - O2C_ml * b_ml
        weight_b = 1 / (1 + np.exp(
            - b_1 * (O2C_b - b_2)
        ))  # logistic transfer function

        O2C_b_norm = O2C - (0.75 * O2C_ml * b_ml)

        weight_norm = 1 / (1 + np.exp(
            - b_1 * (O2C_b_norm - b_2)
        ))

        blending_weights[1] = weight_b / weight_norm
        blending_weights[0] = 1 - blending_weights[1]

    elif O2C <= O2C_ml * 2:  # mid to high O2C region
        b_1 = 75.0159268221068
        b_2 = 0.000947111285750515

        O2C_b = O2C - O2C_ml
        blending_weights[2] = 1 / (1 + np.exp(
            - b_1 * (O2C_b - b_2)
        ))  # logistic transfer function

        blending_weights[1] = 1 - blending_weights[2]

    else:  # high only region
        blending_weights[2] = 1

    return blending_weights

# %%


FIT_LOW = {'a1': [7.089476E+00, -7.711860E+00, -3.885941E+01, -1.000000E+02],
           'a2': [-6.226781E-01, -1.000000E+02, 3.081244E-09, 6.188812E+01],
           's': [-5.988895E+00, 6.940689E+00]}
FIT_MID = {'a1': [5.872214E+00, -4.535007E+00, -5.129327E+00, -2.809232E+01],
           'a2': [-9.740486E-01, -1.000000E+02, 2.109751E+00, -2.367683E+01],
           's': [-1.219164E+00, 4.742729E+00]}
FIT_HIGH = {'a1': [5.921550E+00, -2.528295E+00, -3.883017E+00, -7.898128E+00],
            'a2': [-1.000000E+02, -1.000000E+02, 1.353916E+00, -1.160145E+01],
            's': [-7.868187E-02, 3.650860E+00]}


def coefficients_c(
        molarmass_ratio,
        O2C,
        fit_values
        ):
    """
    Coefficients for activity model, see Gorkowski (2019). equation S1 S2.

    Parameters:
    ---------
        molar mass ratio (float): water MW / organic MW
        fit_values (list): a_n1, a_n2, a_n3, a_n4
    """
    return fit_values[0] * np.exp(fit_values[1] * O2C) + fit_values[
        2
    ] * np.exp(fit_values[3] * molarmass_ratio)


def exp_limited(value):
    """
    np.exp with limits for machine precision max input value of 690.

    Parameters:
    value (array): Input array.

    Returns:
    array: Exponential of the input array with a limit for machine precision.
    """
    return np.exp(np.where(value > 690, 690, value))


def log_limited(value):
    """
    np.log with limits for machine precision min input value of 1e-300.

    Parameters:
    value (array): Input array.

    Returns:
    array: Log of the input array with a limit for machine precision.
    """
    return np.log(np.where(value < 1e-300, 1e-300, value))


def gibbs_of_mixing(
        molarmass_ratio,
        org_mole_fraction,
        O2C,
        density,
        fit_dict
):
    """
    Gibbs free energy of mixing, see Gorkowski (2019). equation S4.

    Parameters:
    ---------
        molar mass ratio (float): water MW / organic MW
        org mole fraction (float): fraction of organic matter
        O2C (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficient (dict): dictionary of fit values for low O2C region
    """
    c1 = coefficients_c(molarmass_ratio, O2C, fit_dict['a1'])
    c2 = coefficients_c(molarmass_ratio, O2C, fit_dict['a2'])

    rhor = 0.997 / density  # assumes water is the other fluid

    # equation S3
    scaledMr = molarmass_ratio * fit_dict['s'][1] \
        * (1.0 + O2C) ** fit_dict['s'][0]
    # the scaled molar mass ratio of this mixture's components.
    phi2 = org_mole_fraction / (
        org_mole_fraction + (1.0 - org_mole_fraction) * scaledMr / rhor
        )  # phi2 is a scaled volume fraction

    # equation S4
    sum1 = c1 + c2*(1-2*phi2)
    gibbs_mix = phi2 * (1.0 - phi2) * sum1

    # equation s6 the derivative of phi2 with respect to organic x2
    dphi2dx2 = (scaledMr / rhor) * (phi2 / org_mole_fraction) ** 2

    # equation S7
    derivative_gibbs_mix = (
        (1.0 - 2.0 * phi2) * sum1 - 2*c2*phi2 * (1.0 - phi2)
        ) * dphi2dx2

    return gibbs_mix, derivative_gibbs_mix


def gibbs_mix_weight(
        molarmass_ratio,
        org_mole_fraction,
        O2C,
        density,
        BAT_functional_group=None,
):
    """
    Gibbs free energy of mixing, see Gorkowski (2019), with weighted
    O2C regions

    Parameters:
    ---------
        molar mass ratio (float): water MW / organic MW
        org mole fraction (float): fraction of organic matter
        O2C (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficient (dict): dictionary of fit values for low O2C region

    Returns:
    -------
        gibbs_mix (float): Gibbs energy of mixing (including 1/RT)
        derivative_gibbs (float): derivative of Gibbs energy with respect to
        mole fraction of organics (includes 1/RT)
    """
    O2C, molarmass_ratio = convert_to_OH_equivalent(
        O2C,
        molarmass_ratio,
        BAT_functional_group=None
    )

    weights = bat_blending_weights(molarmass_ratio, O2C)

    if weights[1] > 0:  # if mid region is used
        gibbs_mix_mid, derivative_gibbs_mid = gibbs_of_mixing(
            molarmass_ratio,
            org_mole_fraction,
            O2C,
            density,
            FIT_MID
        )

        if weights[0] > 0:  # if paired with low O2C region
            gibbs_mix_low, derivative_gibbs_low = gibbs_of_mixing(
                molarmass_ratio,
                org_mole_fraction,
                O2C,
                density,
                FIT_LOW
            )
            gibbs_mix = weights[0]*gibbs_mix_low + weights[1]*gibbs_mix_mid
            derivative_gibbs = weights[0]*derivative_gibbs_low \
                + weights[1]*derivative_gibbs_mid
        else:  # else paired with high O2C region
            gibbs_mix_high, derivative_gibbs_high = gibbs_of_mixing(
                molarmass_ratio,
                org_mole_fraction,
                O2C,
                density,
                FIT_HIGH
            )
            gibbs_mix = weights[2]*gibbs_mix_high + weights[1]*gibbs_mix_mid
            derivative_gibbs = weights[2]*derivative_gibbs_high \
                + weights[1]*derivative_gibbs_mid
    else:  # when only high 2OC region is used
        gibbs_mix, derivative_gibbs = gibbs_of_mixing(
            molarmass_ratio,
            org_mole_fraction,
            O2C,
            density,
            FIT_HIGH
        )
    return gibbs_mix, derivative_gibbs


def activity_coefficients(
        molarmass_ratio,
        org_mole_fraction,
        O2C,
        density,
        BAT_functional_group=None,
):
    """
    Activity coefficients for water and organic matter, see Gorkowski (2019)

    Parameters:
    ---------
        molar mass ratio (float): water MW / organic MW
        org mole fraction (float): fraction of organic matter
        O2C (float): oxygen to carbon ratio
        density (float): density of mixture
        fit_coefficient (dict): dictionary of fit values for low O2C region

    Returns:
    -------
        activity_water (float): activity coefficient of water
        activity_organic (float): activity coefficient of organic matter
        mass_water (float): mass fraction of water
        mass_organic (float): mass fraction of organic matter
    """
    O2C, molarmass_ratio = convert_to_OH_equivalent(
        O2C,
        molarmass_ratio,
        BAT_functional_group=None
    )
    gibbs_mix, derivative_gibbs = gibbs_mix_weight(
            molarmass_ratio,
            org_mole_fraction,
            O2C,
            density,
        )
    # equations S8 S10
    # the func value for component 1 = LOG(activity coeff. water)
    ln_gamma_water = gibbs_mix - org_mole_fraction * derivative_gibbs
    # the func value of the component 2 = LOG(activity coeff. of the organic)
    ln_gamma_org = gibbs_mix + (1.0 - org_mole_fraction) * derivative_gibbs

    gamma_water = exp_limited(ln_gamma_water)
    gamma_org = exp_limited(ln_gamma_org)

    activity_water = gamma_water * (1.0 - org_mole_fraction)
    activity_organic = gamma_org * org_mole_fraction

    mass_water = (1.0 - org_mole_fraction) * molarmass_ratio / (
            (1.0 - org_mole_fraction) * (molarmass_ratio - 1) + 1
        )
    mass_organic = 1 - mass_water

    return activity_water, activity_organic, mass_water, mass_organic


def gibbs_free_engery(
    org_mole_fraction,
    gibbs_mix,
):
    """
    Calculate the gibbs free energy of the mixture. Ideal and non-ideal.

    Parameters:
    org_mole_fraction (np.array): A numpy array of organic mole fractions.
    gibbs_mix (np.array): A numpy array of gibbs free energy of mixing.

    Returns:
    gibbs_ideal (np.array): The ideal gibbs free energy of mixing.
    gibbs_real (np.array): The real gibbs free energy of mixing.
    """

    gibbs_ideal = (1-org_mole_fraction) * log_limited(1-org_mole_fraction) \
        + org_mole_fraction * log_limited(org_mole_fraction)
    gibbs_real = gibbs_ideal + gibbs_mix
    return gibbs_ideal, gibbs_real


def find_phase_sep_index(activity_data):
    """
    This function finds phase separation using activity>1 and
    inflections in the activity curve data.
    In physical systems activity can not be above one and
    curve should be monotonic. Or else there will be phase separation.

    Parameters:
    activity_data (np.array): A numpy array of activity data.

    Returns:
    tuple: The phase separation via activity,
    phase separation via activity curvature,
    index phase separation starts, index phase separation end.
    """

    # Compute difference between consecutive elements in the array
    activity_diff = np.diff(activity_data)
    data_length = len(activity_data)

    # Check if the data length is more than 3
    if data_length > 3:
        min_value = np.min(activity_diff)
        max_value = np.max(activity_diff)

        # Check if the min and max differences have the same sign
        if np.sign(min_value) == np.sign(max_value):
            # If so, no phase separation via activity curvature
            phase_sep_curve = 0
            index_phase_sep_starts = np.nan
            index_phase_sep_end = np.nan
        else:
            # If signs differ, phase separation via activity curvature occurs
            phase_sep_curve = 1

            # Find where the sign changes in the activity difference
            activity_diff_sign_change = np.sign(
                np.concatenate(([activity_diff[0]], activity_diff))
                ) != np.sign(activity_diff[0])

            # Find the first change in sign
            index_start = np.where(activity_diff_sign_change)[0][0]
            # Find the last change in sign
            back_index = index_start - 1 + np.where(
                ~activity_diff_sign_change[index_start:])[0][0]

            # Find closest match to restart the process
            if back_index < data_length:
                activity_data_gap = np.argmin(
                    np.abs(
                        activity_data[back_index:] - activity_data[index_start]
                    ))
                restart_match_index = activity_data_gap + back_index - 1
            else:
                restart_match_index = data_length

            # Check if any activity data is greater than 1
            if sum(activity_data > 1):
                # Find minimum activity corresponding index
                min_index_idilute = np.argmin(
                    activity_data[index_start:]) + index_start - 1

                # Find where activity data matches the minimum value
                activity_data_gap_start = np.argmin(
                    np.abs(
                        activity_data[:index_start]
                        - activity_data[min_index_idilute]
                    ))

                # Assign appropriate indices for phase separation
                index_phase_sep_starts = min(
                    activity_data_gap_start, index_start)
                index_phase_sep_end = min(
                    min_index_idilute, restart_match_index)
            else:
                index_phase_sep_starts = index_start
                index_phase_sep_end = restart_match_index
    else:
        phase_sep_activity = activity_data
        phase_sep_curve = 0
        index_phase_sep_starts = np.nan
        index_phase_sep_end = np.nan

    # Assign phase separation via activity based on data being greater than 1
    phase_sep_activity = 1 if sum(activity_data > 1) else 0
    return {'phase_sep_activity': phase_sep_activity,
            'phase_sep_curve': phase_sep_curve,
            'index_phase_sep_starts': index_phase_sep_starts,
            'index_phase_sep_end': index_phase_sep_end}


def find_phase_separation(activity_water, activity_org):
    """
    This function checks for phase separation in each activity curve.

    Parameters:
    activity_water (np.array): A numpy array of water activity values.
    activity_org (np.array): A numpy array of organic activity values.

    Returns:
    dic: 'phase_sep_check': phase_sep_check,
            'lower_a_w_sep_index': lower_a_w_sep_index,
            'upper_a_w_sep_index': upper_a_w_sep_index,
            'matching_upper_a_w_sep_index': matching_upper_a_w_sep_index
    """

    # check for phase separation in each activity curve
    water_sep = find_phase_sep_index(activity_water)
    organic_sep = find_phase_sep_index(activity_org)

    # gather all the indexes into a list for easier access
    indexes = [water_sep['index_phase_sep_starts'],
               water_sep['index_phase_sep_end'],
               organic_sep['index_phase_sep_starts'],
               organic_sep['index_phase_sep_end']]

    # If there is a phase separation curve in the water activity data
    if water_sep['phase_sep_curve'] == 1:
        phase_sep_check = 1

        # Check for the direction of the curve (increasing or decreasing)
        if activity_water[0] < activity_water[-1]:  # increasing a_w with index
            # find the min and max indexes
            lower_a_w_sep_index = min(indexes)
            upper_a_w_sep_index = max(indexes)

            # calculate the mid index
            mid_sep_index = (lower_a_w_sep_index + upper_a_w_sep_index) // 2
            # slice the data upto mid index
            activity_water_beta = activity_water[:mid_sep_index]
        else:  # decreasing a_w with index
            # find the min and max indexes
            lower_a_w_sep_index = max(indexes)
            upper_a_w_sep_index = min(indexes)

            # calculate the mid index
            mid_sep_index = (lower_a_w_sep_index + upper_a_w_sep_index) // 2
            # slice the data upto mid index
            activity_water_beta = activity_water[mid_sep_index:]
        match_a_w = activity_water[upper_a_w_sep_index]
        # find the index where the difference is greater than 0
        match_index_prime = np.where((activity_water_beta - match_a_w) > 0)

        # if no such index found, assign the index where the max
        # difference is located
        if len(match_index_prime[0]) == 0:
            match_index_prime = np.argmax(activity_water_beta - match_a_w)

    else:
        upper_a_w_sep_index = 2
        lower_a_w_sep_index = 2
        match_index_prime = 2
        phase_sep_check = 0  # no phase sep

    return {'phase_sep_check': phase_sep_check,
            'lower_a_w_sep_index': lower_a_w_sep_index,
            'upper_a_w_sep_index': upper_a_w_sep_index,
            'matching_upper_a_w_sep_index': match_index_prime,
            'lower_a_w_sep': activity_water[lower_a_w_sep_index],
            'upper_a_w_sep': activity_water[upper_a_w_sep_index],
            'matching_upper_a_w_sep': activity_water[match_index_prime]}


def phase_separation_q_alpha(
        a_w_sep,
        aw_series,
        VBSBAT_options=None
):
    """
    This function makes a squeezed logistic function to transfer for
    q_alpha ~0 to q_alpha ~1,

    Parameters:
    a_w_sep (np.array): A numpy array of values.
    aw_series (np.array): A numpy array of values.
    VBSBAT_options (dict): A dictionary of options.

    Returns:
    np.array: The q_alpha value.
    """
    if a_w_sep == 0:
        return np.ones_like(aw_series)
    # spread in transfer from 50/50 point
    delta_a_w_sep = 1 - a_w_sep

    MIN_SPREAD_IN_AW = 10**-6
    # check min value allowed
    above_min_delta_a_w_sep_value = delta_a_w_sep > MIN_SPREAD_IN_AW
    delta_a_w_sep = delta_a_w_sep * above_min_delta_a_w_sep_value + \
        ~above_min_delta_a_w_sep_value * MIN_SPREAD_IN_AW

    # calculate curve parameter of sigmoid
    Q_ALPHA_AT_1PHASE_AW = 0.99  # can be changed
    sigmoid_curve_parameter = log_limited(
        1 / (1 - Q_ALPHA_AT_1PHASE_AW) - 1) / delta_a_w_sep

    # calculate q_alpha return value
    return 1 - 1 / (
        1
        + exp_limited(
            sigmoid_curve_parameter * (aw_series - a_w_sep + delta_a_w_sep)
        )
    )


def biphasic_to_single_phase_RH_point(
    O2C,
    H2C,
    Mratio,
    BAT_functional_group=None
):
    """
    This function computes the biphasic to single phase RH.

    Parameters:
    O2C (np.array): An array representing O2C values.
    H2C (np.array): An array representing H2C values.
    Mratio (np.array): An array representing molar mass ratio values.
    BAT_functional_group (str/list): The BAT functional group(s).

    Returns:
    np.array: The RH cross point array.


    """

    RH_cross_point = np.zeros_like(O2C)

    interpolate_step_numb = 200  # interpolation points
    # mole_frac = np.linspace(1e-12, 1, interpolate_step_numb + 1)
    mole_frac = np.logspace(-6, 0, interpolate_step_numb + 1)

    for i in range(len(O2C)):
        density = organic_density_estimate(
            Mratio[i],
            O2C[i],
            H2C[i],
            mass_ratio_convert=True)
        activities = activity_coefficients(
            molarmass_ratio=Mratio[i],
            org_mole_fraction=mole_frac,
            O2C=O2C[i],
            density=density,
            BAT_functional_group=BAT_functional_group
        )

        if np.isnan(activities[0]).any():
            raise ValueError('water activity is NaN, check inputs')

        phase_check = find_phase_separation(activities[0], activities[1])

        if phase_check['phase_sep_check'] == 1:
            RH_cross_point[i] = phase_check['upper_a_w_sep']
        else:
            RH_cross_point[i] = 0  # no phase separation

    # Checks outputs with in physical limits
    # round to zero
    RH_cross_point[RH_cross_point < 0] = 0
    # round max to 1
    RH_cross_point[RH_cross_point > 1] = 1

    return RH_cross_point
