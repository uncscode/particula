"""Test the loader module."""

import os
import tempfile
from datetime import datetime
import pytest
import pytz
import numpy as np
from particula.data import loader
from particula.data.stream import Stream

# sample stream
stream = Stream(
    header=["header1", "header3"],
    data=np.array([1, 2, 4]),
    time=np.array([1.0, 2.0, 4.0]),
    files=["file1", "file3"],
)


def test_filter_list():
    """Test the filter_list function."""
    data = ["apple,banana,orange", "pear,kiwi,plum", "grapefruit,lemon"]
    char_counts = {",": 2}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == ["apple,banana,orange", "pear,kiwi,plum"]

    data = ["apple,banana,orange", "pear,kiwi,plum", "grapefruit,lemon"]
    char_counts = {",": 3}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == []

    data = ["apple,banana,orange", "pear,kiwi,plum", "grapefruit,lemon"]
    char_counts = {",": 1}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == ["grapefruit,lemon"]

    data = ["apple,banana,orange", "pear,kiwi,plum", "grapefruit,lemon"]
    char_counts = {"z": 1}
    filtered_data = loader.filter_list(data, char_counts)
    assert filtered_data == []


def test_parse_time_column():
    """Test the parse_time_column function."""
    line = np.array("2022-01-01 12:00:00,0.5,0.6".split(","))
    time_format = "%Y-%m-%d %H:%M:%S"

    expected_timestamp = datetime(
        2022, 1, 1, 12, 0, 0, tzinfo=pytz.timezone("UTC")
    ).timestamp()

    # Test case with a single time column as integer index
    time_column_int = 0
    assert (
        loader.parse_time_column(
            time_column=time_column_int,
            time_format=time_format,
            line=line,
        )
        == expected_timestamp
    )

    # Test case with two time columns as list of indices
    line = np.array("2022-01-01,12:00:00,0.5,0.6".split(","))
    time_column_int = [0, 1]
    assert (
        loader.parse_time_column(
            time_column=time_column_int,
            time_format=time_format,
            line=line,
        )
        == expected_timestamp
    )

    # Test case with invalid time column
    time_column_invalid = 3
    try:
        loader.parse_time_column(time_column_invalid, time_format, line)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

    # Test case with invalid time format
    time_format_invalid = "%Y"
    try:
        loader.parse_time_column(time_column_int, time_format_invalid, line)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_data_raw_loader():
    """Test data loader"""
    # Define the expected data
    expected_data = ["line 1", "line 2", "line 3"]

    # Use tempfile.NamedTemporaryFile to create a temp file for the test
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as tmp_file:
        # Write the expected data to the temp file
        tmp_file.write("\n".join(expected_data))
        # Make sure data is written to disk
        tmp_file.flush()

        # Get the temporary file's name
        file_path = tmp_file.name

    try:
        # Load the data using the loader function
        loaded_data = loader.data_raw_loader(file_path)

        # Compare the loaded data with the expected data
        assert (
            loaded_data == expected_data
        ), "The loaded data does not match the expected data."
    finally:
        # Clean up: remove the temporary file after the test
        os.remove(file_path)


def test_data_format_checks():
    """Checks the rows of data, filtering out invalid rows based on
    predefined checks."""
    # Extended data with a mix of valid and invalid rows based on the criteria
    data = [
        "row, 1",  # Invalid: Only 1 comma
        "row, 2,",  # Valid: Exactly 2 commas, no slashes or colons, 0-10 char
        "row, 3, extra",  # Invalid: More than 10 characters
        "row/ 4",  # Invalid: Contains a slash
        "row: 5",  # Invalid: Contains a colon
        "row, 6,",  # Valid: Exactly 2 commas, no slashes or colons, 0-10 char
        "row, 7, ",  # Valid: Exactly 2 commas, no slashes or colons, 0-10 char
    ]
    data_checks = {
        "characters": [0, 10],  # Character count range (inclusive)
        # Required counts for specific characters
        "char_counts": {",": 2, "/": 0, ":": 0},
        "skip_rows": 0,  # Not skipping any starting rows
        "skip_end": 0,  # Not skipping any ending rows
    }
    expected_formatted_data = [
        "row, 2,",  # Meets all criteria
        "row, 6,",  # Meets all criteria
        "row, 7,",  # Meets all criteria
    ]

    # Perform data format checks
    clean_data = loader.data_format_checks(data, data_checks)

    # Assert the cleaned data matches the expected formatted data
    assert (
        clean_data == expected_formatted_data
    ), "Filtered data does not match the expected data."


def test_sample_data():
    """test sampling of data rows"""
    # Input data setup
    data = ["2022-01-01 12:00:00,1,2", "2022-01-01 12:01:00,3,4"]
    time_column = 0
    time_format = "%Y-%m-%d %H:%M:%S"
    data_columns = [1, 2]
    delimiter = ","

    # Expected results
    expected_epoch_time = np.array([1641033600.0, 1641033660.0])
    expected_data_array = np.array([[1, 2], [3, 4]])

    # Call the function under test
    epoch_time, data_array = loader.sample_data(
        data, time_column, time_format, data_columns, delimiter
    )

    # Assertions
    assert np.allclose(
        epoch_time, expected_epoch_time, atol=1e-6
    ), "Epoch times do not match expected values."
    assert np.allclose(
        data_array, expected_data_array, atol=1e-6
    ), "Data arrays do not match expected values."
    # Check the type of returned values to ensure they are numpy arrays
    assert isinstance(
        epoch_time, np.ndarray
    ), "Returned epoch time is not a numpy array."
    assert isinstance(
        data_array, np.ndarray
    ), "Returned data array is not a numpy array."


@pytest.mark.parametrize(
    "keyword, expected, raises_exception",
    [
        (1, 1, False),  # integer index, valid
        ("B", 1, False),  # string keyword, valid
        (3, None, True),  # integer index, invalid
        ("D", None, True),  # string keyword, invalid
    ],
)
def test_keyword_to_index_variations(keyword, expected, raises_exception):
    """Test the keyword_to_index function with various keyword types
    and values."""
    header = ["A", "B", "C"]

    if raises_exception:
        with pytest.raises(ValueError):
            loader.keyword_to_index(keyword, header)
    else:
        assert loader.keyword_to_index(keyword, header) == expected


def test_save_stream_to_csv(tmpdir):
    """Test save to csv"""
    # Create a temporary directory for testing
    output_dir = tmpdir.mkdir("output")
    # Define the expected file path
    expected_file_path = os.path.join(output_dir, "data.csv")
    # Call the function
    loader.save_stream_to_csv(stream, str(tmpdir))
    # Assert that the file was created
    assert os.path.isfile(expected_file_path)
    # options for suffix
    expected_file_path = os.path.join(output_dir, "data_suffix.csv")
    # Call the function with optional parameters
    loader.save_stream_to_csv(stream, str(tmpdir), suffix_name="_suffix")
    # Assert that the file was created with the expected name and location
    assert os.path.isfile(expected_file_path)

    # Call the function with an invalid path
    with pytest.raises(ValueError):
        loader.save_stream_to_csv(stream, "invalid/path")


def test_save_stream(tmpdir):
    """ "Test saving a stream to a pickle"""
    # Create a temporary directory for testing
    test_dir = tmpdir.mkdir("output")

    # Test saving stream without suffix
    loader.save_stream(str(tmpdir), stream)
    file_path = os.path.join(test_dir, "stream.pk")
    assert os.path.isfile(file_path)

    # Test saving stream with suffix
    loader.save_stream(str(tmpdir), stream, suffix_name="_test")
    file_path = os.path.join(test_dir, "stream_test.pk")
    assert os.path.isfile(file_path)

    # Test saving stream to non-existent directory
    invalid_dir = os.path.join(test_dir, "non_existent_dir")
    with pytest.raises(ValueError):
        loader.save_stream(invalid_dir, stream)

    # Test saving stream with custom folder name
    custom_folder = "custom_folder"
    loader.save_stream(str(tmpdir), stream, folder=custom_folder)
    file_path = os.path.join(str(tmpdir), custom_folder, "stream.pk")
    assert os.path.isfile(file_path)


def test_load_stream_valid_path(tmpdir):
    """Test loading a stream with a valid path."""
    # Test saving stream without suffix
    loader.save_stream(str(tmpdir), stream)
    # load test save
    result = loader.load_stream(str(tmpdir))

    assert np.allclose(result.data, stream.data)


def test_load_stream_valid_path_with_suffix(tmpdir):
    """Test loading a stream with a suffix."""
    # Arrange
    suffix_name = "_suffix"
    # Test saving stream without suffix
    loader.save_stream(str(tmpdir), stream, suffix_name)
    result = loader.load_stream(str(tmpdir), suffix_name)
    # Assert
    assert np.allclose(result.data, stream.data)
    assert result.header == stream.header


def test_load_stream_invalid_path():
    """Test loading a stream with an invalid path."""
    # Arrange
    path = "/invalid/path"
    # Act & Assert
    with pytest.raises(ValueError):
        loader.load_stream(path)
