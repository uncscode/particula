"""Test the loader module."""


from datetime import datetime
import pytz
import numpy as np
from particula.data import loader


def test_filter_list():
    """Test the filter_list function."""
    data = ['apple,banana,orange', 'pear,kiwi,plum', 'grapefruit,lemon']
    char_counts = {',': 2}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == ['apple,banana,orange', 'pear,kiwi,plum']

    data = ['apple,banana,orange', 'pear,kiwi,plum', 'grapefruit,lemon']
    char_counts = {',': 3}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == []

    data = ['apple,banana,orange', 'pear,kiwi,plum', 'grapefruit,lemon']
    char_counts = {',': 1}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == ['grapefruit,lemon']

    data = ['apple,banana,orange', 'pear,kiwi,plum', 'grapefruit,lemon']
    char_counts = {'z': 1}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == []


def test_parse_time_column():
    """Test the parse_time_column function."""
    line = np.array('2022-01-01 12:00:00,0.5,0.6'.split(','))
    time_format = '%Y-%m-%d %H:%M:%S'

    expected_timestamp = datetime(
        2022, 1, 1, 12, 0, 0,
        tzinfo=pytz.timezone('UTC')).timestamp()

    # Test case with a single time column as integer index
    time_column_int = 0
    assert loader.parse_time_column(
        time_column=time_column_int,
        time_format=time_format,
        line=line,
    ) == expected_timestamp

    # Test case with two time columns as list of indices
    line = np.array('2022-01-01,12:00:00,0.5,0.6'.split(','))
    time_column_int = [0, 1]
    assert loader.parse_time_column(
        time_column=time_column_int,
        time_format=time_format,
        line=line,
    ) == expected_timestamp

    # Test case with invalid time column
    time_column_invalid = 3
    try:
        loader.parse_time_column(time_column_invalid, time_format, line)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

    # Test case with invalid time format
    time_format_invalid = '%Y'
    try:
        loader.parse_time_column(time_column_int, time_format_invalid, line)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
