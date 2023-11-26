"""Test the convert module."""

import numpy as np
from particula.util import stats


def test_merge_format_str_headers():
    """Test the merge_different_headers function."""
    # Create example input data
    data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    header = ['a', 'b', 'c']
    data_add = np.array([[5, 8], [5, 8], [5, 8]])
    header_add = ['c', 'd']

    # Call function to merge the data and headers
    data, header, merged_data, merged_header = stats.merge_formatting(
        data_current=data,
        header_current=header,
        data_new=data_add,
        header_new=header_add
    )

    # Test merged data shape
    assert merged_data.shape == (3, 4)

    # Test merged header
    assert merged_header == ['a', 'b', 'c', 'd']


def test_merge_format_num_headers():
    """Test the merge_different_headers function."""
    # Create example input data
    data = np.array([[1, 2, 3], [1, 2, 3]])
    header = ['1', '2', '3']
    data_add = np.array([[5, 8], [5, 8]])
    header_add = ['3', '4']

    # Call function to merge the data and headers
    data, header, merged_data, merged_header = stats.merge_formatting(
        data_current=data,
        header_current=header,
        data_new=data_add,
        header_new=header_add
    )

    # Test merged data shape
    assert merged_data.shape == (2, 4)

    # Test merged header
    assert merged_header == ['1', '2', '3', '4']


def test_average_to_interval():
    """Test the average_to_interval function."""
    # Set up test data
    time_stream = np.arange(0, 1000, 10)
    average_base_sec = 10
    average_base_time = np.arange(0, 1000, 60)
    data_stream = np.linspace(0, 1000, len(time_stream)).reshape(-1, 1)
    data_stream = np.concatenate([data_stream, data_stream, data_stream], axis=1)
    average_base_data = np.zeros((len(average_base_time), 3))
    average_base_data_std = np.zeros((len(average_base_time), 3))

    # Call the function
    average_base_data, average_base_data_std = stats.average_to_interval(
        time_raw=time_stream,
        data_raw=data_stream,
        average_interval=average_base_sec,
        average_interval_array=average_base_time,
        average_data=average_base_data,
        average_data_std=average_base_data_std
    )
    expected_data = np.array(
            [
                [0.0, 25.252, 85.858, 146.464,
                    207.070, 267.676, 328.282, 388.888,
                    449.494, 510.101, 570.707, 631.313,
                    691.919, 752.525, 813.131, 873.737,
                    934.3434],
                [0.0, 25.252, 85.858, 146.464,
                    207.070, 267.676, 328.282, 388.888,
                    449.494, 510.101, 570.707, 631.313,
                    691.919, 752.525, 813.131, 873.737,
                    934.3434],
                [0.0, 25.252, 85.858, 146.464,
                    207.070, 267.676, 328.282, 388.888,
                    449.494, 510.101, 570.707, 631.313,
                    691.919, 752.525, 813.131, 873.737,
                    934.34343434]
            ]
        ).T

    expected_std = np.array(
            [
                [0.0, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250],
                [0.0, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250],
                [0.0, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250, 17.250, 17.250, 17.250,
                    17.250, 17.250]
            ]
        ).T

    assert np.allclose(average_base_data, expected_data, rtol=1e-3)
    assert np.allclose(average_base_data_std, expected_std, rtol=1e-3)


def test_mask_outliers_bottom():
    """Test the mask_outliers function."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bottom = 3

    expected_mask = np.array(
        [False, False, True, True, True, True, True, True, True, True])

    assert np.allclose(stats.mask_outliers(data, bottom=bottom), expected_mask)


def test_mask_outliers_top():
    """Test the mask_outliers function."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    top = 7

    expected_mask = np.array(
        [True, True, True, True, True, True, True, False, False, False])

    assert np.allclose(stats.mask_outliers(data, top=top), expected_mask)


def test_mask_outliers_value():
    """Test the mask_outliers function."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    value = 5

    expected_mask = np.array(
        [True, True, True, True, False, True, True, True, True, True])

    assert np.allclose(stats.mask_outliers(data, value=value), expected_mask)


def test_mask_outliers_invert():
    """Test the mask_outliers function."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bottom = 3
    top = 7
    invert = True

    expected_mask = np.array(
        [True, True, False, False, False, False, False, True, True, True])

    assert np.allclose(
        stats.mask_outliers(data, bottom=bottom, top=top, invert=invert),
        expected_mask)


def test_threshold_outliers_all():
    """Test the threshold_outliers function."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bottom = 3
    top = 7
    value = 5
    invert = True

    expected_mask = np.array(
        [True, True, False, False, True, False, False, True, True, True])

    assert np.allclose(
        stats.mask_outliers(
            data, bottom=bottom, top=top, value=value, invert=invert),
        expected_mask)
