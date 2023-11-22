"""Test the time_manage module."""

import numpy as np
from particula.util import time_manage
from particula.util.time_manage import time_str_to_epoch


def test_time_str_to_epoch():
    """Test the time_str_to_epoch function."""
    # Test with a time in New York
    time = "2021-06-18 22:34:21"
    time_format = "%Y-%m-%d %H:%M:%S"
    timezone_identifier = "America/New_York"
    expected = 1624070061.0
    assert time_str_to_epoch(
        time,
        time_format,
        timezone_identifier) == expected

    # Test with a time in Los Angeles
    timezone_identifier = "America/Los_Angeles"
    expected = 1624080861.0
    assert time_str_to_epoch(
        time,
        time_format,
        timezone_identifier) == expected

    # Test with a time in London
    timezone_identifier = "Europe/London"
    expected = 1624052061.0
    assert time_str_to_epoch(
        time,
        time_format,
        timezone_identifier) == expected

    # Test with a time in Sydney
    timezone_identifier = "Australia/Sydney"
    expected = 1624019661.0
    assert time_str_to_epoch(
        time,
        time_format,
        timezone_identifier) == expected


def test_datetime64_from_epoch_array():
    """Test the datetime64_from_epoch_array function."""
    # Test with a simple example array
    epoch_array = np.array([0, 1, 2, 3, 4])
    expected_result = np.array(['1970-01-01T00:00:00', '1970-01-01T00:00:01',
                                '1970-01-01T00:00:02', '1970-01-01T00:00:03',
                                '1970-01-01T00:00:04'], dtype='datetime64[s]')
    assert np.array_equal(
            time_manage.datetime64_from_epoch_array(epoch_array),
            expected_result
        )

    # Test with a non-zero delta
    delta = 3600  # 1 hour in seconds
    expected_result = np.array(['1970-01-01T01:00:00', '1970-01-01T01:00:01',
                                '1970-01-01T01:00:02', '1970-01-01T01:00:03',
                                '1970-01-01T01:00:04'], dtype='datetime64[s]')
    assert np.array_equal(
            time_manage.datetime64_from_epoch_array(epoch_array, delta),
            expected_result
        )

    # Test with an empty array
    empty_array = np.array([])
    try:
        time_manage.datetime64_from_epoch_array(empty_array)
        assert False, \
            "Function should raise an AssertionError for empty array."
    except AssertionError:
        pass

    # Test with a non-integer array
    float_array = np.array([0.0, 1.0, 2.0])
    try:
        time_manage.datetime64_from_epoch_array(float_array)
        assert False, \
            "Function should raise an AssertionError for non-integer array."
    except AssertionError:
        pass
