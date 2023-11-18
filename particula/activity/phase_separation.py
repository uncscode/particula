
"""
This module contains functions to calculate the phase separation of organic
compounds in water. The functions are based on the BAT model by Gorkowski
et al. (2019).

Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""
# consider changing o2c to oxygen2carbon
# define convert_to_OH_equivalent
# add test

import numpy as np

from particula.activity.machine_limit import safe_exp, safe_log
from particula.activity.species_density import organic_density_estimate
from particula.activity.binary_activity import activity_coefficients

MIN_SPREAD_IN_AW = 10**-6
Q_ALPHA_AT_1PHASE_AW = 0.99


def organic_water_single_phase(molar_mass_ratio):
    """
    Convert the given molar mass ratio (MW water / MW organic) to a
    and oxygen2carbon value were above is a single phase with water and below
    phase separation is possible.

    Args:
    molar_mass_ratio np.: The molar mass ratio with respect to water.

    Returns:
    float: The single phase cross point.
    """

    return (
        0.205 / (1 + np.exp(26.6 * (molar_mass_ratio - 0.12))) ** 0.843 + 0.225
    )


def find_phase_sep_index(activity_data):  # pylint: disable=too-many-locals
    """
    This function finds phase separation using activity>1 and
    inflections in the activity curve data.
    In physical systems activity can not be above one and
    curve should be monotonic. Or else there will be phase separation.

    Args:
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

    Args:
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


def q_alpha(
        a_w_sep,
        aw_series,
):
    """
    This function makes a squeezed logistic function to transfer for
    q_alpha ~0 to q_alpha ~1,

    Args:
    a_w_sep (np.array): A numpy array of values.
    aw_series (np.array): A numpy array of values.

    Returns:
    np.array: The q_alpha value.
    """
    if a_w_sep == 0:
        return np.ones_like(aw_series)
    # spread in transfer from 50/50 point
    delta_a_w_sep = 1 - a_w_sep

    # check min value allowed
    above_min_delta_a_w_sep_value = delta_a_w_sep > MIN_SPREAD_IN_AW
    delta_a_w_sep = delta_a_w_sep * above_min_delta_a_w_sep_value + \
        ~above_min_delta_a_w_sep_value * MIN_SPREAD_IN_AW

    # calculate curve parameter of sigmoid
    sigmoid_curve_parameter = safe_log(
        1 / (1 - Q_ALPHA_AT_1PHASE_AW) - 1) / delta_a_w_sep

    # calculate q_alpha return value
    return 1 - 1 / (
        1
        + safe_exp(
            sigmoid_curve_parameter * (aw_series - a_w_sep + delta_a_w_sep)
        )
    )


def biphasic_water_activity_point(
    oxygen2carbon,
    hydrogen2carbon,
    molar_mass_ratio,
    functional_group=None
):
    """
    This function computes the biphasic to single phase
    water activity (RH*100).

    Args:
    oxygen2carbon (np.array): An array representing oxygen2carbon values.
    hydrogen2carbon (np.array): An array representing hydrogen2carbon values.
    molar_mass_ratio (np.array): An array representing molar mass ratio values.
    functional_group (str/list): The BAT functional group(s).

    Returns:
    np.array: The RH cross point array.
    """

    water_activity_cross_point = np.zeros_like(oxygen2carbon)

    interpolate_step_numb = 200  # interpolation points
    # mole_frac = np.linspace(1e-12, 1, interpolate_step_numb + 1)
    mole_frac = np.logspace(-6, 0, interpolate_step_numb + 1)

    for i, _ in enumerate(oxygen2carbon):
        density = organic_density_estimate(
            molar_mass_ratio[i],
            oxygen2carbon[i],
            hydrogen2carbon[i],
            mass_ratio_convert=True)
        activities = activity_coefficients(
            molar_mass_ratio=molar_mass_ratio[i],
            org_mole_fraction=mole_frac,
            oxygen2carbon=oxygen2carbon[i],
            density=density,
            functional_group=functional_group
        )

        if np.isnan(activities[0]).any():
            raise ValueError('water activity is NaN, check inputs')

        phase_check = find_phase_separation(activities[0], activities[1])

        if phase_check['phase_sep_check'] == 1:
            water_activity_cross_point[i] = phase_check['upper_a_w_sep']
        else:
            water_activity_cross_point[i] = 0  # no phase separation

    # Checks outputs with in physical limits
    # round to zero
    water_activity_cross_point[water_activity_cross_point < 0] = 0
    # round max to 1
    water_activity_cross_point[water_activity_cross_point > 1] = 1

    return water_activity_cross_point
