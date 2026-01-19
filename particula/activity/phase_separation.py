r"""Phase separation utilities for organic–water systems.

Implements BAT (Binary Activity Thermodynamics) phase-separation helpers used
by the activity module to identify liquid–liquid phase separation (LLPS) and
compute transfer functions.

Constants:
    MIN_SPREAD_IN_AW: Minimum allowed spread in water activity when computing
        the transition width for :func:`q_alpha`. Prevents numerical issues
        in the logistic transfer when activity ranges are extremely small.
        Value: 1e-6 (dimensionless).

    Q_ALPHA_AT_1PHASE_AW: Target :math:`q_\alpha` value at the single-phase
        activity boundary. Used to calibrate the sigmoid steepness in
        :func:`q_alpha`. Value: 0.99 (dimensionless).

References:
    Gorkowski, K., Preston, T. C., & Zuend, A. (2019).
    Relative-humidity-dependent organic aerosol thermodynamics via an
    efficient reduced-complexity model. Atmospheric Chemistry and Physics.
    https://doi.org/10.5194/acp-19-13383-2019
"""

from typing import Dict, Union

import numpy as np
from numpy.typing import NDArray

from particula.util.machine_limit import get_safe_exp, get_safe_log

MIN_SPREAD_IN_AW: float = 10**-6
Q_ALPHA_AT_1PHASE_AW: float = 0.99


def organic_water_single_phase(
    molar_mass_ratio: Union[int, float, list, np.ndarray],
) -> np.ndarray:
    """Compute single-phase O:C limit for organic–water mixtures.

    Converts the molar mass ratio (:math:`MW_{water} / MW_{organic}`) to the
    oxygen-to-carbon threshold above which mixtures remain single phase.
    Below this threshold, liquid–liquid phase separation (LLPS) is possible.

    Args:
        molar_mass_ratio: Molar mass ratio with respect to water.

    Returns:
        Single-phase O:C crossover point as a NumPy array.

    Examples:
        >>> organic_water_single_phase(molar_mass_ratio=0.1)
        array(...)

    References:
        Gorkowski et al. (2019).
        https://doi.org/10.5194/acp-19-13383-2019
    """
    molar_mass_ratio = np.asarray(molar_mass_ratio, dtype=np.float64)

    return (
        0.205 / (1 + get_safe_exp(26.6 * (molar_mass_ratio - 0.12))) ** 0.843
        + 0.225
    )


# pylint: disable=too-many-locals
def find_phase_sep_index(
    activity_data: Union[float, NDArray[np.float64]],
) -> Dict[str, Union[int, float]]:
    """Detect phase separation using activity monotonicity and limits.

    Identifies potential phase separation by checking for activities above one
    and curvature inflections in the activity curve. Physical systems should
    have activity ≤ 1 and monotonic behavior; violations indicate LLPS.

    Args:
        activity_data: Activity values, scalars or arrays promoted to NumPy
            arrays respecting the same dtype and shape.

    Returns:
        Dictionary with detection flags and indices:
            phase_sep_activity: 1 if any activity exceeds 1, else 0.
            phase_sep_curve: 1 if curvature changes sign, else 0.
            index_phase_sep_starts: Index where separation starts.
            index_phase_sep_end: Index where separation ends.
    """
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
    activity_water: Union[float, NDArray[np.float64]],
    activity_org: Union[float, NDArray[np.float64]],
) -> Dict[str, Union[int, float]]:
    """Check for phase separation across water and organic activity curves.

    Applies :func:`find_phase_sep_index` to both water and organic activities,
    then combines the detected indices into a consolidated separation report.

    Args:
        activity_water: Water activity values. Scalars are promoted to NumPy
            arrays.
        activity_org: Organic activity values. Scalars are promoted to NumPy
            arrays.

    Returns:
        Dictionary containing:
            phase_sep_check: 1 if any phase separation is detected, else 0.
            lower_seperation_index: Lower separation index across both series.
            upper_seperation_index: Upper separation index across both series.
            matching_upper_seperation_index: Index where water activity crosses
                the matched separation value.
            lower_seperation: Water activity at lower separation index.
            upper_seperation: Water activity at upper separation index.
            matching_upper_seperation: Water activity at the matched index.
    """
    activity_water_arr = np.asarray(activity_water, dtype=np.float64)
    activity_org_arr = np.asarray(activity_org, dtype=np.float64)

    water_sep = find_phase_sep_index(activity_water_arr)
    organic_sep = find_phase_sep_index(activity_org_arr)

    # gather all the indexes into a list for easier access
    indexes = [
        water_sep["index_phase_sep_starts"],
        water_sep["index_phase_sep_end"],
        organic_sep["index_phase_sep_starts"],
        organic_sep["index_phase_sep_end"],
    ]

    lower_seperation_index = 2
    upper_seperation_index = 2
    match_index_prime = 2

    # If there is a phase separation curve in the water activity data
    if water_sep["phase_sep_curve"] == 1:
        phase_sep_check = 1

        if activity_water_arr[0] < activity_water_arr[-1]:  # increasing order
            lower_seperation_index = int(min(indexes))
            upper_seperation_index = int(max(indexes))
            match_a_w = activity_water_arr[upper_seperation_index]

            match_slice = np.sign(
                match_a_w - activity_water_arr[lower_seperation_index:]
            )
            match_index_prime_arr = np.where(match_slice == -1)
            if len(match_index_prime_arr[0]) == 0:
                match_index_prime = lower_seperation_index
            else:
                match_index_prime = int(
                    match_index_prime_arr[0][0] + lower_seperation_index
                )
        else:  # decreasing order
            lower_seperation_index = int(max(indexes))
            upper_seperation_index = int(min(indexes))
            match_a_w = activity_water_arr[upper_seperation_index]

            match_slice = np.sign(
                activity_water_arr[:lower_seperation_index] - match_a_w
            )
            match_index_prime_arr = np.where(match_slice == -1)
            if len(match_index_prime_arr[0]) == 0:
                match_index_prime = lower_seperation_index
            else:
                match_index_prime = int(match_index_prime_arr[0][0])
    else:
        phase_sep_check = 0  # no phase seperation

    lower_seperation_activity = activity_water_arr[int(lower_seperation_index)]
    upper_seperation_activity = activity_water_arr[int(upper_seperation_index)]
    matching_upper_activity = activity_water_arr[int(match_index_prime)]

    return {
        "phase_sep_check": phase_sep_check,
        "lower_seperation_index": lower_seperation_index,
        "upper_seperation_index": upper_seperation_index,
        "matching_upper_seperation_index": match_index_prime,
        "lower_seperation": lower_seperation_activity,
        "upper_seperation": upper_seperation_activity,
        "matching_upper_seperation": matching_upper_activity,
    }


def q_alpha(
    seperation_activity: Union[float, NDArray[np.float64]],
    activities: Union[float, NDArray[np.float64]],
) -> np.ndarray:
    r"""Compute :math:`q_\alpha` transition using a squeezed logistic curve.

    Maps activity values to a smooth transition between phase-separated and
    single-phase regimes. The sigmoid is calibrated so that
    :data:`Q_ALPHA_AT_1PHASE_AW` is reached at the single-phase boundary and
    the transition width is bounded below by :data:`MIN_SPREAD_IN_AW`.

    Args:
        seperation_activity: Activity at which the mixture transitions
            between phases. Scalars or arrays are flattened internally.
        activities: Activity values to evaluate. Scalars are promoted to NumPy
            arrays before computing the logistic curve.

    Returns:
        NumPy array of :math:`q_\alpha` values with the same shape as
        ``activities``.

    Examples:
        >>> q_alpha(0.8, np.array([0.7, 0.8, 0.9]))
        array(...)
    """
    separation_activity_array = np.asarray(
        seperation_activity, dtype=np.float64
    )
    activities_array = np.asarray(activities, dtype=np.float64)

    if np.all(separation_activity_array == 0):
        return np.ones_like(activities_array)

    delta_seperation = 1 - separation_activity_array

    above_min_delta_seperation_value = delta_seperation > MIN_SPREAD_IN_AW
    delta_seperation = (
        delta_seperation * above_min_delta_seperation_value
        + (not above_min_delta_seperation_value) * MIN_SPREAD_IN_AW
    )

    sigmoid_curve_parameter = (
        get_safe_log(1 / (1 - Q_ALPHA_AT_1PHASE_AW) - 1) / delta_seperation
    )

    return 1 - 1 / (
        1
        + get_safe_exp(
            sigmoid_curve_parameter
            * (activities_array - separation_activity_array + delta_seperation)
        )
    )
