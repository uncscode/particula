"""Calculate the phase separation of organic compounds in water.

The functions are based on the BAT model.

Gorkowski, K., Preston, T. C., Zuend, A. (2019).
Relative-humidity-dependent organic aerosol thermodynamics
Via an efficient reduced-complexity model.
Atmospheric Chemistry and Physics
https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from particula.util.machine_limit import get_safe_exp, get_safe_log

MIN_SPREAD_IN_AW = 10**-6
Q_ALPHA_AT_1PHASE_AW = 0.99


def organic_water_single_phase(
    molar_mass_ratio: Union[int, float, list, np.ndarray],
) -> np.ndarray:
    """Solubility limit of organic compounds in water.

    Convert the given molar mass ratio (MW water / MW organic) to a
    and oxygen2carbon value were above is a single phase with water and below
    phase separation is possible.

    Args:
        - molar_mass_ratio: The molar mass ratio with respect to water.

    Returns:
        - The single phase cross point.

    References:
        - Gorkowski, K., Preston, T. C., &#38; Zuend, A. (2019).
          Relative-humidity-dependent organic aerosol thermodynamics
          Via an efficient reduced-complexity model.
          Atmospheric Chemistry and Physics
          https://doi.org/10.5194/acp-19-13383-2019
    """
    # check inputs
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)

    return (
        0.205 / (1 + get_safe_exp(26.6 * (molar_mass_ratio - 0.12))) ** 0.843
        + 0.225
    )


# pylint: disable=too-many-locals
def find_phase_sep_index(activity_data: NDArray[np.float64]) -> dict:
    """Find phase separation in activity data.

    This function finds phase separation using activity>1 and
    inflections in the activity curve data.
    In physical systems activity can not be above one and
    curve should be monotonic. Or else there will be phase separation.

    Args:
    - activity_data: A array of activity data.

    Returns:
    dict: A dictionary containing the following keys:
        - 'phase_sep_activity': Phase separation via activity
            (1 if there is phase separation, 0 otherwise)
        - 'phase_sep_curve': Phase separation via activity curvature
            (1 if there is phase separation, 0 otherwise)
        - 'index_phase_sep_starts': Index where phase separation starts
        - 'index_phase_sep_end': Index where phase separation ends
    """
    # check inputs
    activity_data = np.asarray(activity_data, dtype=np.float64)

    # Compute difference between consecutive elements in the array
    activity_diff = np.diff(activity_data)
    data_length = len(activity_data)

    # Declare variables with Union types for type checking
    index_phase_sep_starts: Union[int, float]
    index_phase_sep_end: Union[int, float]
    phase_sep_activity: int

    # Check if the data length is more than 3
    if data_length > 3:
        min_value = np.min(activity_diff)
        max_value = np.max(activity_diff)

        # Check if the min and max differences have the same sign
        if np.sign(min_value) == np.sign(max_value):
            # If so, no phase separation via activity curvature
            phase_sep_curve = 0
            # find the index where the activity is closest to 1
            index_phase_sep_starts = int(np.argmin(np.abs(activity_data - 1)))
            index_phase_sep_end = index_phase_sep_starts
        else:
            # If signs differ, phase separation via activity curvature occurs
            phase_sep_curve = 1
            # Find indices where the sign of the second derivative changes
            sign_changes = np.diff(np.sign(activity_diff))

            # The first index where a sign change occurs
            inflection_index = np.where(sign_changes)[0]  # all indices
            index_start = int(
                inflection_index[0]
                if len(inflection_index) > 0
                else data_length
            )
            # The last index where a sign change occurs
            back_index = int(
                inflection_index[-1]
                if len(inflection_index) > 0
                else data_length
            )

            # Check if first section of activity data is greater than 1
            if np.any(activity_data[:index_start] > 1):
                index_phase_sep_starts = int(
                    np.argmin(np.abs(activity_data[:index_start] - 1))
                )
            else:
                index_phase_sep_starts = index_start

            # Check if second section of activity data is greater than 1
            if np.any(activity_data[back_index:] > 1):
                index_phase_sep_end = int(
                    np.argmin(np.abs(activity_data[back_index:] - 1))
                    + back_index
                )
            else:
                index_phase_sep_end = back_index
    else:
        phase_sep_curve = 0
        index_phase_sep_starts = float(data_length)
        index_phase_sep_end = float(data_length)

    # Assign phase separation via activity based on data being greater than 1
    phase_sep_activity = 1 if sum(activity_data > 1) else 0
    return {
        "phase_sep_activity": phase_sep_activity,
        "phase_sep_curve": phase_sep_curve,
        "index_phase_sep_starts": index_phase_sep_starts,
        "index_phase_sep_end": index_phase_sep_end,
    }


def find_phase_separation(
    activity_water: NDArray[np.float64], activity_org: NDArray[np.float64]
) -> dict:
    """This function checks for phase separation in each activity curve.

    Args:
    - activity_water (np.array): A numpy array of water activity values.
    - activity_org (np.array): A numpy array of organic activity values.

    Returns:
    dict: A dictionary containing the following keys:
        - 'phase_sep_check': An integer indicating whether phase separation
                is present (1) or not (0).
        - 'lower_seperation_index': The index of the lower separation point
                in the activity curve.
        - 'upper_seperation_index': The index of the upper separation point in
                the activity curve.
        - 'matching_upper_seperation_index': The index where the difference
                between activity_water_beta and match_a_w is greater than 0.
        - 'lower_seperation': The value of water activity at the lower
                separation point.
        - 'upper_seperation': The value of water activity at the upper
                separation point.
        - 'matching_upper_seperation': The value of water activity at the
                matching upper separation point.
    """
    # check for phase separation in each activity curve
    water_sep = find_phase_sep_index(activity_water)
    organic_sep = find_phase_sep_index(activity_org)

    # gather all the indexes into a list for easier access
    indexes = [
        water_sep["index_phase_sep_starts"],
        water_sep["index_phase_sep_end"],
        organic_sep["index_phase_sep_starts"],
        organic_sep["index_phase_sep_end"],
    ]

    # If there is a phase separation curve in the water activity data
    if water_sep["phase_sep_curve"] == 1:
        phase_sep_check = 1

        # Check for the direction of the curve (increasing or decreasing)
        if activity_water[0] < activity_water[-1]:  # increasing a_w with index
            # find the min and max indexes
            lower_seperation_index: int = int(min(indexes))
            upper_seperation_index: int = int(max(indexes))
            match_a_w = activity_water[upper_seperation_index]

            # start from the lower_seperation_index and find the index where
            # the difference between activity_water and match_a_w changes sign
            match_slice = np.sign(
                match_a_w - activity_water[lower_seperation_index:]
            )
            match_index_prime_arr = np.where(match_slice == -1)
            if len(match_index_prime_arr[0]) == 0:
                match_index_prime: int = lower_seperation_index
            else:
                match_index_prime = int(
                    match_index_prime_arr[0][0] + lower_seperation_index
                )
        else:  # decreasing a_w with index
            # find the min and max indexes
            lower_seperation_index = int(max(indexes))
            upper_seperation_index = int(min(indexes))
            match_a_w = activity_water[upper_seperation_index]

            # start from the lower_seperation_index and find the index where
            # the difference between activity_water and match_a_w changes sign
            match_slice = np.sign(
                activity_water[:lower_seperation_index] - match_a_w
            )
            match_index_prime_arr = np.where(match_slice == -1)
            if len(match_index_prime_arr[0]) == 0:
                match_index_prime = lower_seperation_index
            else:
                match_index_prime = int(match_index_prime_arr[0][0])

    else:
        upper_seperation_index = 2
        lower_seperation_index = 2
        match_index_prime = 2
        phase_sep_check = 0  # no phase seperation

    return {
        "phase_sep_check": phase_sep_check,
        "lower_seperation_index": lower_seperation_index,
        "upper_seperation_index": upper_seperation_index,
        "matching_upper_seperation_index": match_index_prime,
        "lower_seperation": activity_water[int(lower_seperation_index)],
        "upper_seperation": activity_water[int(upper_seperation_index)],
        "matching_upper_seperation": activity_water[int(match_index_prime)],
    }


def q_alpha(
    seperation_activity: NDArray[np.float64],
    activities: NDArray[np.float64],
) -> np.ndarray:
    """Calculates the q_alpha value using a squeezed logistic function.

    Args:
        - seperation_activity (np.array): A numpy array of values representing
            the separation activity.
        - activities (np.array): A numpy array of activity values.

    Returns:
        np.array: The q_alpha value.

    Notes:
        - The q_alpha value represents the transfer from
            q_alpha ~0 to q_alpha ~1.
        - The function uses a sigmoid curve parameter to calculate the
            q_alpha value.
    """
    # check inputs
    if seperation_activity == 0:
        return np.ones_like(activities)

    # spread in transfer from 50/50 point
    delta_seperation = 1 - seperation_activity

    # check min value allowed
    above_min_delta_seperation_value = delta_seperation > MIN_SPREAD_IN_AW
    delta_seperation = (
        delta_seperation * above_min_delta_seperation_value
        + (not above_min_delta_seperation_value) * MIN_SPREAD_IN_AW
    )

    # calculate curve parameter of sigmoid
    sigmoid_curve_parameter = (
        get_safe_log(1 / (1 - Q_ALPHA_AT_1PHASE_AW) - 1) / delta_seperation
    )

    # calculate q_alpha return value
    return 1 - 1 / (
        1
        + get_safe_exp(
            sigmoid_curve_parameter
            * (activities - seperation_activity + delta_seperation)
        )
    )
