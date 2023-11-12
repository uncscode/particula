"""Test the loader module."""
# pytype: skip-file

import numpy as np
from particula.data import merger


def create_sample_data():
    """Create sample data for testing."""
    data = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
    time = np.array([0, 1, 2, 3, 4])
    header_list = ['header1', 'header2']
    return data, time, header_list


def test_combine_data_with_2d_data():
    """Test with 2d data."""
    # Setup
    data, time, header_list = create_sample_data()
    data_new = np.array([[7, 7], [8, 8]])
    time_new = np.array([1, 4])
    header_new = ['header3', 'header4']

    # Execution
    merged_data, merged_header_list, _ = merger.combine_data(
        data, time, header_list, data_new, time_new, header_new)

    # Verification
    expected_data = np.array([
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [8, 8, 8, 8, 8]
    ])
    assert np.array_equal(merged_data, expected_data)
    expected_header_list = ['header1', 'header2', 'header3', 'header4']
    assert np.all(merged_header_list == expected_header_list)


def test_combine_data_with_1d_data():
    """Test with 1d data."""
    # Setup
    data, time, header_list = create_sample_data()
    data_new = np.array([7, 7])
    time_new = np.array([1, 4])
    header_new = ['header3']

    # Execution
    merged_data, merged_header_list, _ = merger.combine_data(
        data, time, header_list, data_new, time_new, header_new)

    # Verification
    expected_data = np.array([
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7],
    ])
    assert np.array_equal(merged_data, expected_data)
    expected_header_list = ['header1', 'header2', 'header3']
    assert np.all(merged_header_list == expected_header_list)


def test_combine_data_with_nan_values():
    """Test with nan values."""
    # Setup
    data, time, header_list = create_sample_data()
    data_new = np.array([
        [np.nan, 6, np.nan], [7, 7, np.nan], [np.nan, np.nan, np.nan]])
    time_new = np.array([1, 2, 3])
    header_new = ['header3', 'header4', 'header5']

    # Execution
    merged_data, merged_header_list, _ = merger.combine_data(
        data, time, header_list, data_new, time_new, header_new)

    # Verification
    expected_data = np.array([
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7], [np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    assert np.array_equal(
        np.nan_to_num(merged_data), np.nan_to_num(expected_data)
        )
    expected_header_list = ['header1', 'header2', 'header3', 'header4',
                            'header5']
    assert np.all(merged_header_list == expected_header_list)


def test_combine_data_with_transposed_input():
    """Test with transposed input."""
    # Setup
    data, time, header_list = create_sample_data()
    data_new = np.array([
        [np.nan, 6, np.nan], [7, 7, np.nan], [np.nan, np.nan, np.nan]])
    time_new = np.array([1, 2, 3])
    header_new = ['header3', 'header4', 'header5']

    # Execution
    merged_data, merged_header_list, _ = merger.combine_data(
        data, time, header_list, data_new, time_new, header_new)

    # Verification
    expected_data = np.array([
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7], [np.nan, np.nan, np.nan, np.nan, np.nan]
    ])
    assert np.array_equal(np.nan_to_num(merged_data),
                          np.nan_to_num(expected_data))
    expected_header_list = ['header1', 'header2', 'header3',
                            'header4', 'header5']
    assert np.all(merged_header_list == expected_header_list)


def test_combine_data_same_time_2d_data():
    """Test with 2d data and same time."""
    # Setup
    data, time, header_list = create_sample_data()
    data_new = np.array([[7, 8], [7, 8], [7, 8], [7, 8], [7, 8]])
    time_new = np.array([0, 1, 2, 3, 4])
    header_new = ['header3', 'header4']

    # Execution
    merged_data, merged_header_list, _ = merger.combine_data(
        data, time, header_list, data_new, time_new, header_new)

    # Verification
    expected_data = np.array([
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [8, 8, 8, 8, 8]
    ])
    assert np.array_equal(merged_data, expected_data)
    expected_header_list = ['header1', 'header2', 'header3', 'header4']
    assert np.all(merged_header_list == expected_header_list)


def test_combine_data_same_time_1d_data():
    """Test with 1d data and same time."""
    # Setup
    data, time, header_list = create_sample_data()
    data_new = np.array([7, 7, 7, 7, 7])
    time_new = np.array([0, 1, 2, 3, 4])
    header_new = ['header3']

    # Execution
    merged_data, merged_header_list, _ = merger.combine_data(
        data, time, header_list, data_new, time_new, header_new)

    # Verification
    expected_data = np.array([
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7],
    ])
    assert np.array_equal(merged_data, expected_data)
    expected_header_list = ['header1', 'header2', 'header3']
    assert np.all(merged_header_list == expected_header_list)
