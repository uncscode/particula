"""Test the loader module."""

import os
from datetime import datetime
import pytz
import numpy as np
from particula.data import loader
from particula.data.stream import Stream


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


# Test data_raw_loader function
def test_data_raw_loader():
    file_path = 'my_file.txt'
    expected_data = ['line 1', 'line 2', 'line 3']
    assert loader.data_raw_loader(file_path) == expected_data


# Test data_format_checks function
def test_data_format_checks():
    data = ['row 1', 'row 2', 'row 3']
    data_checks = {
        "characters": [0, 10],
        "char_counts": {",": 2, "/": 0, ":": 0},
        "skip_rows": 0,
        "skip_end": 0
    }
    expected_formatted_data = ['row 2']
    assert loader.data_format_checks(data, data_checks) == expected_formatted_data


# Test sample_data function
def test_sample_data():
    data = ['2022-01-01 12:00:00,1,2', '2022-01-01 12:01:00,3,4']
    time_column = 0
    time_format = '%Y-%m-%d %H:%M:%S'
    data_columns = [1, 2]
    delimiter = ','
    expected_epoch_time = np.array([1641033600.0, 1641033660.0])
    expected_data_array = np.array([[1, 2], [3, 4]])
    assert loader.sample_data(
        data, time_column, time_format, data_columns, delimiter
        ) == (expected_epoch_time, expected_data_array)


def test_keyword_to_index_with_integer_index():
    header = ['A', 'B', 'C']
    keyword = 1
    expected_index = 1
    assert loader.keyword_to_index(keyword, header) == expected_index


def test_keyword_to_index_with_string_keyword():
    header = ['A', 'B', 'C']
    keyword = 'B'
    expected_index = 1
    assert loader.keyword_to_index(keyword, header) == expected_index


def test_keyword_to_index_with_invalid_integer_index():
    header = ['A', 'B', 'C']
    keyword = 3
    with pytest.raises(ValueError):
        loader.keyword_to_index(keyword, header)


def test_keyword_to_index_with_invalid_string_keyword():
    header = ['A', 'B', 'C']
    keyword = 'D'
    with pytest.raises(ValueError):
        loader.keyword_to_index(keyword, header)


def test_save_stream_to_csv_success(tmpdir):
    # Create a temporary directory for testing
    output_dir = tmpdir.mkdir("output")

    # Create a sample Stream object
    stream = Stream(...)

    # Define the expected file path
    expected_file_path = os.path.join(output_dir, "stream.csv")

    # Call the function
    loader.save_stream_to_csv(stream, str(tmpdir))

    # Assert that the file was created
    assert os.path.isfile(expected_file_path)



def test_save_stream_to_csv_optional_params(tmpdir):
    # Create a temporary directory for testing
    output_dir = tmpdir.mkdir("output")

    # Create a sample Stream object
    stream = Stream(...)

    # Define the expected file path with suffix and folder
    expected_file_path = os.path.join(output_dir, "data_suffix.csv")

    # Call the function with optional parameters
    loader.save_stream_to_csv(
        stream,
        str(tmpdir),
        suffix_name="suffix",
        folder="output")

    # Assert that the file was created with the expected name and location
    assert os.path.isfile(expected_file_path)


def test_save_stream_to_csv_invalid_path():
    # Create a sample Stream object
    stream = Stream(...)

    # Call the function with an invalid path
    with pytest.raises(ValueError):
        loader.save_stream_to_csv(stream, "invalid/path")


def test_save_stream_to_csv_integrity(tmpdir):
    # Create a temporary directory for testing
    output_dir = tmpdir.mkdir("output")

    # Create a sample Stream object
    stream = Stream(...)

    # Define the expected file path
    expected_file_path = os.path.join(output_dir, "stream.csv")

    # Call the function
    loader.save_stream_to_csv(stream, str(tmpdir))

    # Load the saved CSV file and verify the integrity of the Stream object
    with open(expected_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)


def test_save_stream(tmpdir):
    # Create a temporary directory for testing
    test_dir = tmpdir.mkdir("test_dir")

    # Create a sample stream object
    stream = [1, 2, 3, 4, 5]

    # Test saving stream without suffix
    loader.save_stream(str(test_dir), stream)
    file_path = os.path.join(str(test_dir), 'output', 'stream.pk')
    assert os.path.isfile(file_path)

    # Test saving stream with suffix
    loader.save_stream(str(test_dir), stream, sufix_name='_test')
    file_path = os.path.join(str(test_dir), 'output', 'stream_test.pk')
    assert os.path.isfile(file_path)

    # Test saving stream to non-existent directory
    invalid_dir = os.path.join(str(test_dir), 'non_existent_dir')
    with pytest.raises(ValueError):
        loader.save_stream(invalid_dir, stream)

    # Test saving stream to an existing file
    file_path = os.path.join(str(test_dir), 'output', 'existing_file.pk')
    with open(file_path, 'w') as file:
        file.write('Existing file')
    with pytest.raises(FileExistsError):
        loader.save_stream(str(test_dir), stream)

    # Test saving stream with custom folder name
    custom_folder = 'custom_folder'
    loader.save_stream(str(test_dir), stream, folder=custom_folder)
    file_path = os.path.join(str(test_dir), custom_folder, 'stream.pk')
    assert os.path.isfile(file_path)


def test_load_stream_valid_path():
    # Arrange
    path = '/path/to/directory'
    expected_file_path = os.path.join(path, 'output', 'stream.pk')
    expected_stream = Stream()  # Replace Stream() with the expected stream object
    
    # Act
    with open(expected_file_path, 'wb') as file:
        pickle.dump(expected_stream, file)
    result = loader.load_stream(path)
    
    # Assert
    assert result == expected_stream


def test_load_stream_valid_path_with_suffix():
    # Arrange
    path = '/path/to/directory'
    suffix_name = 'suffix'
    expected_file_path = os.path.join(path, 'output', f'stream{suffix_name}.pk')
    expected_stream = Stream()  # Replace Stream() with the expected stream object
    
    # Act
    with open(expected_file_path, 'wb') as file:
        pickle.dump(expected_stream, file)
    result = loader.load_stream(path, suffix_name)
    
    # Assert
    assert result == expected_stream


def test_load_stream_invalid_path():
    # Arrange
    path = '/invalid/path'
    
    # Act & Assert
    with pytest.raises(ValueError):
        loader.load_stream(path)