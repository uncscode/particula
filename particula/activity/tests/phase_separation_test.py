"""test for phase_separation.py."""

import numpy as np

from particula.activity.phase_separation import (
    find_phase_sep_index,
    find_phase_separation,
    organic_water_single_phase,
    q_alpha,
)


def test_organic_water_single_phase():
    """Test for organic_water_single_phase function."""
    # Test with positive values
    assert np.all(organic_water_single_phase([1, 2, 3]) >= 0)

    # Test with large values
    assert np.all(organic_water_single_phase([1000, 2000, 3000]) >= 0)


def test_find_phase_sep_index():
    """Test for find_phase_sep_index function."""
    # Test with positive values
    activity = [0.2, 0.3, 0.4, 0.35, 0.5, 0.6, 0.9]
    sep_dic = find_phase_sep_index(activity_data=activity)

    assert sep_dic == {
        "phase_sep_activity": 0,
        "phase_sep_curve": 1,
        "index_phase_sep_starts": 1,
        "index_phase_sep_end": 2,
    }

    # Test with values above 1
    activity = [0.2, 0.3, 0.4, 1, 1.2, 1.3, 1.2, 1.1, 1.0]
    sep_dic = find_phase_sep_index(activity_data=activity)
    assert sep_dic == {
        "phase_sep_activity": 1,
        "phase_sep_curve": 1,
        "index_phase_sep_starts": 4,
        "index_phase_sep_end": 8,
    }


def test_find_phase_separation():
    """Test for find_phase_separation function."""
    # Test with positive values

    activity_water = np.array([0.2, 0.3, 0.4, 0.35, 0.5, 0.6, 0.9])
    activity_org = np.array([0.8, 0.7, 0.6, 0.65, 0.5, 0.4, 0.1])
    sep_dic = find_phase_separation(
        activity_water=activity_water, activity_org=activity_org
    )

    assert sep_dic == {
        "phase_sep_check": 1,
        "lower_seperation_index": 1,
        "upper_seperation_index": 2,
        "matching_upper_seperation_index": 4,
        "lower_seperation": 0.3,
        "upper_seperation": 0.4,
        "matching_upper_seperation": 0.5,
    }


def test_q_alpha():
    """Test for q_alpha function."""
    # Test with positive values
    assert np.all(
        q_alpha(
            seperation_activity=0.5,
            activities=np.array([0.2, 0.3, 0.4, 0.35, 0.5, 0.6, 0.9]),
        )
        >= 0
    )
