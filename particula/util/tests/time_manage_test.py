"""Test the time_manage module."""

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
