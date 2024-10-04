"""File readers and loaders for datacula."""

from typing import Union, List
from typing import List, Union, Tuple, Dict, Any, Optional
import warnings
import codecs
import glob
import os
import pickle
import netCDF4 as nc
import numpy as np
import csv

from particula.util import convert
from particula.util.time_manage import time_str_to_epoch
from particula.data.lake import Lake
from particula.data.stream import Stream

FILTER_WARNING_FRACTION = 0.5


def data_raw_loader(file_path: str) -> list:
    """
    Load raw data from a file at the specified file path and return it as a
    list of strings. Attempts to handle UTF-8, UTF-16, and UTF-32 encodings.
    Defaults to UTF-8 if no byte order mark (BOM) is found.

    Args:
        file_path (str): The file path of the file to read.

    Returns:
        list: The raw data read from the file as a list of strings.

    Examples:
        ``` py title="Load my_file.txt"
        data = data_raw_loader('my_file.txt')
        Loading data from: my_file.txt
        print(data)
        ['line 1', 'line 2', 'line 3']
        ```
    """
    try:
<<<<<<< HEAD
        # Read a small part of the file to detect BOM (byte order mark)
        with open(file_path, "rb") as f:
            raw_bytes = f.read(4)

        # Determine encoding based on BOM
        if raw_bytes.startswith(codecs.BOM_UTF16_LE) or raw_bytes.startswith(
            codecs.BOM_UTF16_BE
        ):
            encoding = "utf-16"
        elif raw_bytes.startswith(codecs.BOM_UTF32_LE) or raw_bytes.startswith(
            codecs.BOM_UTF32_BE
        ):
            encoding = "utf-32"
        else:
            encoding = "utf8"  # Default to utf-8 if no BOM is found

        # Read file with the detected encoding
        with open(file_path, "r", encoding=encoding, errors="replace") as file:
=======
        with open(file_path, 'r', encoding='utf8', errors='replace') as file:
>>>>>>> upstream/main
            data = [line.rstrip() for line in file]

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        data = []

    return data


def filter_list(data: List[str], char_counts: dict) -> List[str]:
    """
    Filter rows from a list of strings based on character counts.

    Each row must contain a specified number of certain characters to pass
    the filter. The `char_counts` dictionary specifies the characters to count
    and the exact count required for each character in each row.

    Arguments:
        data: A list of strings to be filtered.
        char_counts: A dictionary specifying character counts for filtering.
            The keys are the characters to count, and the values are the
            required counts for each character in a row.

    Returns:
        A new list of strings containing only the rows that meet the
        character count requirements.

    Raises:
        UserWarning: If more than 90% of the rows are filtered out, indicating
            that the filter may be too strict based on the specified
            character(s).

    Examples:
        ``` py title="Filter rows based on comma counts"
        data = ['apple,banana,orange',
                 'pear,kiwi,plum', 'grapefruit,lemon']
        char_counts = {',': 2}
        filtered_data = filter_list(data, char_counts)
        print(filtered_data)
        ['apple,banana,orange', 'pear,kiwi,plum']
        ```
    """
    filtered_data = data
    for char, count in char_counts.items():
        if count > -1:
            filtered_data = [
                row for row in filtered_data if row.count(char) == count]
        if len(filtered_data) / len(data) < FILTER_WARNING_FRACTION:
            warnings.warn(
                f"More than {FILTER_WARNING_FRACTION} of the rows have " +
                f"been filtered out based on the character: {char}.")
    return filtered_data


def replace_list(data: List[str], replace_dict: Dict[str, str]) -> List[str]:
    """
    Replace characters in each string of a list based on a replacement
    dictionary.

    Each character specified in the `replace_dict` will be replaced with the
    corresponding value in every string in the input list.

    Arguments:
        data: A list of strings in which the characters will be replaced.
        replace_dict: A dictionary specifying character replacements.
            The keys are the characters to be replaced, and the values are the
            replacement characters or strings.

    Returns:
        A new list of strings with the replacements applied.

    Examples:
        ``` py title="Replace characters in a list of strings"
        data = ['apple[banana]orange', '[pear] kiwi plum']
        replace_dict = {'[': '', ']': ''}
        replaced_data = replace_list(data, replace_dict)
        print(replaced_data)
        ['applebananaorange', 'pear kiwi plum']
        ```
    """
    replaced_data = []
    for row in data:
        modified_row = row
        for old_char, new_char in replace_dict.items():
            modified_row = modified_row.replace(old_char, new_char)
        replaced_data.append(modified_row)
    return replaced_data


def data_format_checks(data: List[str], data_checks: dict) -> List[str]:
    """
    Validate and format raw data according to specified checks.

    Arguments:
        data: List of strings containing the raw data to be checked.
        data_checks: Dictionary specifying the format checks to apply,
            such as character limits, character counts, and rows to skip.

    Returns:
        A list of strings containing the validated and formatted data.

    Raises:
        TypeError: If `data` is not provided as a list.

    Examples:
        ``` py title="Validate line based on counts"
        data = ['row 1', 'row 2', 'row 3']
        data_checks = {
            "characters": [0, 10],
            "char_counts": {",": 2, "/": 0, ":": 0},
            "skip_rows": 0,
            "skip_end": 0
        }
        formatted_data = data_format_checks(data, data_checks)
        print(formatted_data)
        ['row 2']
        ```
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    length_initial = len(data)
    if data_checks.get('skip_rows', 0) > 0:
        data = data[data_checks['skip_rows']:]
    if data_checks.get('skip_end', 0) > 0:
        data = data[:-data_checks['skip_end']]
    if len(data_checks.get('characters', [])) == 1:
        # Filter out any rows with fewer than the specified number of
        # characters.
        data = [
            x for x in data
            if (
                len(x)
                > data_checks['characters'][0]
            )
        ]
    elif len(data_checks.get('characters', [])) == 2:
        # Filter out any rows with fewer than the minimum or more than the
        # maximum number of characters.
        data = [
            x for x in data
            if (
                data_checks['characters'][0]
                < len(x)
                < data_checks['characters'][1]
            )
        ]
    if len(data) / length_initial < FILTER_WARNING_FRACTION:
        warnings.warn(
            f"More than {FILTER_WARNING_FRACTION} rows are filtered based on "
            + f"{data_checks['characters']} or skip rows."
        )
    if 'char_counts' in data_checks:
        char_counts = data_checks.get('char_counts', {})
        data = filter_list(data, char_counts)
    if "replace_chars" in data_checks:
        replace_dict = data_checks.get("replace_chars", {})
        data = replace_list(data, replace_dict)
    if data := [x.strip() for x in data]:
        return data
    else:
        raise ValueError('No data left in file')


def parse_time_column(
    time_column: Union[int, List[int]],
    time_format: str,
    line: np.ndarray,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = 'UTC'
) -> float:
    """
    Parse the time column(s) from a data line and return the timestamp.

    Arguments:
        time_column: Index or list of indices identifying the column(s)
            containing the time information.
        time_format: String specifying the format of the time information,
            e.g., '%Y-%m-%d %H:%M:%S'.
        line: A numpy array representing the data line to parse.
        date_offset: Optional string representing a fixed offset to add
            to the timestamp. Default is None.
        seconds_shift: Number of seconds to add to the timestamp. Default is 0.
        timezone_identifier: Timezone identifier for the timestamp.
            Default is 'UTC'.

    Returns:
        A float representing the timestamp in seconds since the epoch.

    Raises:
        ValueError: If the specified time column or format is invalid.
    """
    if time_format == 'epoch':
        return float(line[time_column]) + seconds_shift
    if date_offset:
        # if the time is in one column, and the date is fixed
        if isinstance(time_column, int):
            time_str = f"{date_offset} {line[time_column]}"
        else:
            time_str = f"{date_offset} {line[time_column[0]]}"
        return (
            time_str_to_epoch(time_str, time_format, timezone_identifier)
            + seconds_shift
        )
    if isinstance(time_column, int):
        return (
            time_str_to_epoch(
                line[time_column], time_format, timezone_identifier
            )
            + seconds_shift
        )
    if isinstance(time_column, list) and len(time_column) == 1:
        return (
            time_str_to_epoch(
                line[time_column[0]], time_format, timezone_identifier
            )
            + seconds_shift
        )
    if isinstance(time_column, list) and len(time_column) == 2:
        # if the time and date are in two column
        time_str = f"{line[time_column[0]]} {line[time_column[1]]}"
        return (
            time_str_to_epoch(time_str, time_format, timezone_identifier)
            + seconds_shift
        )
    raise ValueError(
<<<<<<< HEAD
        f"Invalid time column or format: {time_column}, {time_format}"
        f"{line}"
    )
=======
        f"Invalid time column or format: {time_column}, {time_format}")
>>>>>>> upstream/main


def sample_data(
    data: List[str],
    time_column: Union[int, List[int]],
    time_format: str,
    data_columns: List[int],
    delimiter: str,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = 'UTC'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract time and data streams from input data.

    Arguments:
        data: List of strings containing the input data.
        time_column: Index or list of indices indicating the column(s)
            containing the time values.
        time_format: Format string specifying the time format, e.g.,
            '%Y-%m-%d %H:%M:%S'.
        data_columns: List of indices identifying the columns containing
            the data values.
        delimiter: Character used to separate columns in the input data.
        date_offset: Optional string representing an offset to apply to
            the date, in the format 'days:hours:minutes:seconds'.
            Default is None.
        seconds_shift: Number of seconds to shift the timestamps. Default is 0.
        timezone_identifier: Timezone of the data. Default is 'UTC'.

    Returns:
        Tuple (np.ndarray, np.ndarray):
            - `epoch_time`: A 1-D numpy array of epoch times.
            - `data_array`: A 2-D numpy array of data values.

    Raises:
        ValueError: If the data is not in the expected format or
            if no matching data value is found.
    """
    # flake8: noqa
    # pylint disable: too-many-positional-arguments
    epoch_time = np.zeros(len(data))
    epoch_time = np.zeros(len(data))
    data_array = np.zeros((len(data), len(data_columns)))

    for i, line in enumerate(data):
        # split the line into an array
        line_array = np.array(line.split(delimiter))
        # split the line into an array
        line_array = np.array(line.split(delimiter))

        epoch_time[i] = parse_time_column(
            time_column=time_column,
            time_format=time_format,
            line=line_array,
            date_offset=date_offset,
            seconds_shift=seconds_shift,
            timezone_identifier=timezone_identifier
        )

        for j, col in enumerate(data_columns):
            value = line_array[col].strip() if col < len(line_array) else ''
            if value in ['', '.']:  # no data
                data_array[i, j] = np.nan
            elif value.count('�') > 0:
                data_array[i, j] = np.nan
            elif value[0].isnumeric():  # if the first character is a number
                data_array[i, j] = float(value)
            elif value[-1].isnumeric():
                data_array[i, j] = float(value)
            elif value[0] == '-':
                data_array[i, j] = float(value)
            elif value[0] == '+':
                data_array[i, j] = float(value)
            elif value[0] == '.':
                try:
                    data_array[i, j] = float(value)
                except ValueError as exc:
                    print(line_array)
                    raise ValueError(
                        f'Data is not a float: row {i}, col {j}, value {value}'
                    ) from exc

            elif value.isalpha():
                true_match = [
<<<<<<< HEAD
                    "ON",
                    "on",
                    "On",
                    "oN",
                    "1",
                    "True",
                    "true",
                    "TRUE",
                    "tRUE",
                    "t",
                    "T",
                    "Yes",
                    "yes",
                    "YES",
                    "yES",
                    "y",
                    "Y",
                    "OK",
                    "ok",
                    "Ok",
                    "Okay",
=======
                    'ON', 'on', 'On', 'oN', '1', 'True', 'true',
                    'TRUE', 'tRUE', 't', 'T', 'Yes', 'yes', 'YES',
                    'yES', 'y', 'Y'
>>>>>>> upstream/main
                ]
                false_match = [
                    'OFF', 'off', 'Off', 'oFF', '0',
                    'False', 'false', 'FALSE', 'fALSE', 'f',
                    'F', 'No', 'no', 'NO', 'nO', 'n', 'N'
                ]
                nan_match = [
                    'NaN', 'nan', 'Nan', 'nAN', 'NAN', 'NaN',
                    'nAn', 'naN', 'NA', 'Na', 'nA', 'na',
                    'N', 'n', '', 'aN', 'null', 'NULL', 'Null',
                    '-99999', '-9999', '.'
                ]
                if value in true_match:
                    data_array[i, j] = 1
                elif value in false_match:
                    data_array[i, j] = 0
                elif value in nan_match:
                    data_array[i, j] = np.nan
                else:
                    raise ValueError(
                        f'No match for data value: row {i}, \
                             col {j}, value {value}'
                    )

    return epoch_time, data_array


def general_data_formatter(
    data: list,
    data_checks: dict,
    data_column: list,
    time_column: Union[int, List[int]],
    time_format: str,
    delimiter: str = ',',
    header_row: int = 0,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = 'UTC'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Format and sample data to extract time and data streams.

    Arguments:
        data: List of strings containing the raw data.
        data_checks: Dictionary specifying validation rules for the data.
        data_column: List of indices identifying the columns containing the
            data.
        time_column: Index or list of indices identifying the column(s)
            containing the time information.
        time_format: String specifying the format of the time information,
            e.g., '%Y-%m-%d %H:%M:%S'.
        delimiter: String used to separate columns in the data. Default is ','.
        header_row: Index of the row containing column names. Default is 0.
        date_offset: Optional string to add as a fixed offset to the timestamp.
            Default is None.
        seconds_shift: Number of seconds to add to the timestamp. Default is 0.
        timezone_identifier: Timezone identifier for the timestamps.
            Default is 'UTC'.

    Returns:
        Tuple (np.ndarray, np.ndarray):
            - The first array contains the epoch times.
            - The second array contains the corresponding data values.
    """

    # find str matching in header row and gets index
    if isinstance(data_column[0], str):
        data_header = data[header_row].split(delimiter)
        # Get data column indices
<<<<<<< HEAD
        try:
            data_column = [data_header.index(x) for x in data_column]
        except ValueError as exc:
            raise ValueError(
                f"Header column not found in header: {data_header}"
            ) from exc
=======
        data_column = [data_header.index(x)
                       for x in data_column]
>>>>>>> upstream/main

    # Check the data format
    data = data_format_checks(data, data_checks)

    # Sample the data to get the epoch times and the data
    epoch_time, data_array = sample_data(
        data,
        time_column,
        time_format,
        data_column,
        delimiter,
        date_offset,
        seconds_shift,
        timezone_identifier
    )

    return epoch_time, data_array


def keyword_to_index(keyword: Union[str, int], header: List[str]) -> int:
    """
    Convert a keyword representing a column position in the header to
    its index.

    This function processes a keyword that can either be an integer index
    or a string corresponding to a column name. If the keyword is an integer,
    it is treated as the direct index of the column. If the keyword is a
    string, the function searches the header list for the column name
    and returns its index.

    Arguments:
        keyword: The keyword representing the column's position in the header.
            It can be an integer index or a string specifying the column name.
        header: A list of column names (header) in the data.

    Returns:
        The index of the column in the header.

    Raises:
        ValueError: If the keyword is a string and is not found in the header,
            or if the keyword is an integer but is out of the header's
            index range.
    """

    if isinstance(keyword, int):
        if keyword < 0 or keyword >= len(header):
            raise ValueError(
                f"Index {keyword} is out of range for the header.")
        return keyword
    elif keyword in header:
        return header.index(keyword)
    else:
        raise ValueError(f"Cannot find '{keyword}' in header {header[:20]}...")


def sizer_data_formatter(
    data: List[str],
    data_checks: Dict[str, Any],
    data_sizer_reader: Dict[str, str],
    time_column: Union[int, List[int]],
    time_format: str,
    delimiter: str = ',',
    header_row: int = 0,
    date_offset: Optional[str] = None,
    seconds_shift: int = 0,
    timezone_identifier: str = 'UTC'
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Format data from a particle sizer into structured arrays.

    Arguments:
        data: List of raw data strings to be formatted.
        data_checks: Dictionary specifying validation rules for the data.
        data_sizer_reader: Dictionary containing mappings for interpreting
            the sizer data format.
        time_column: Index or list of indices indicating the position of
            the time column(s) in the data.
        time_format: Format string for parsing time information in the data.
        delimiter: Delimiter used to separate values in the data.
            Default is ','.
        header_row: Row index of the header containing column names.
            Default is 0.
        date_offset: Optional string representing an offset to add to
            timestamps. Default is None.
        seconds_shift: Number of seconds to shift the timestamps.
            Default is 0.
        timezone_identifier: Timezone identifier for the data timestamps.
            Default is 'UTC'.

    Returns:
        Tuple(np.ndarray, np.ndarray, list):
            - A numpy array of epoch times.
            - A numpy array of Dp header values.
            - A list of numpy arrays representing the data.
    """
    # Split header data using the provided delimiter
    data_header = data[header_row].split(delimiter)
    # Convert start and end keywords to indices
    dp_start_index = keyword_to_index(
        data_sizer_reader["Dp_start_keyword"], data_header)
    dp_end_index = keyword_to_index(
        data_sizer_reader["Dp_end_keyword"], data_header)

    # Ensure dp_start_index and dp_end_index are within valid range
    if dp_start_index > dp_end_index:
        raise ValueError(
            "Dp_start_keyword must come before Dp_end_keyword in the header")
    # Generate the range of column indices to include
    dp_columns = list(
        range(dp_start_index, dp_end_index + 1)
        )  # +1 to include the end index
    # Extract headers for the specified range
    header = [data_header[i] for i in dp_columns]

    # Format data
    data = data_format_checks(data, data_checks)

    # Get data arrays
    epoch_time, data_2d = sample_data(
        data,
        time_column,
        time_format,
        dp_columns,
        delimiter,
        date_offset,
        seconds_shift=seconds_shift,
        timezone_identifier=timezone_identifier
    )

    if "convert_scale_from" in data_sizer_reader:
        if data_sizer_reader["convert_scale_from"] == "dw":
            for i in range(len(epoch_time)):
                data_2d[i, :] = convert.convert_sizer_dn(
                    diameter=np.array(header).astype(float),
                    dn_dlogdp=data_2d[i, :],
                    inverse=True
                )

    return epoch_time, data_2d, header


def non_standard_date_location(
    data: list,
    date_location: dict
) -> str:
    """
    Extracts the date from a non-standard location in the data.

    Arguments:
        data: A list of strings representing the data.
        date_location: A dictionary specifying the method for extracting the
            date from the data.
                - 'file_header_block': The date is located in the file header
                    block, and its position is specified by the 'row',
                    'delimiter', and 'index' keys.

    Returns:
        str: The date extracted from the specified location in the data.

    Raises:
        ValueError: If an unsupported or invalid method is specified in
            date_location.
    """
    if date_location['method'] != 'file_header_block':
        raise ValueError('Invalid date location method specified')

    row_index = date_location['row']
    delimiter = date_location['delimiter']
    index = date_location['index']
    return data[row_index].split(delimiter)[index].strip()


def get_files_in_folder_with_size(
    path: str,
    subfolder: str,
    filename_regex: str,
    min_size: int = 10,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Returns a list of files in the specified folder and subfolder that
    match the given filename pattern and have a size greater than the
    specified minimum size.

    Args:
        path : str
            The path to the parent folder.
        subfolder : str
            The name of the subfolder containing the files.
        filename_regex : str
            A regular expression pattern for matching the filenames.
        min_size : int, optional
            The minimum file size in bytes (default is 10).

    Returns:
        Tuple(List[str], List[str], List[int]):
        - file_list: The filenames that match the pattern and size criteria.
        - full_path: The full paths to the files.
        - file_size: The file sizes in bytes.
    """
    search_path = os.path.join(path, subfolder)

    if not os.path.isdir(search_path):
        raise ValueError(f"{search_path} is not a directory")

    file_list = glob.glob(os.path.join(search_path, filename_regex))

    # filter the files by size
    full_path = [
        file for file in file_list
        if os.path.getsize(os.path.join(search_path, file)) > min_size
    ]

    # get the file names only
    file_list = [os.path.split(path)[-1]
                 for path in full_path]
    file_size_in_bytes = [os.path.getsize(path) for path in full_path]

    return file_list, full_path, file_size_in_bytes


def save_stream_to_csv(
    stream: Stream,
    path: str,
    suffix_name: Optional[str] = None,
<<<<<<< HEAD
    folder: str = "output",
=======
    folder: Optional[str] = 'output',
>>>>>>> upstream/main
    include_time: bool = True,
) -> None:
    """
    Save stream object as a CSV file, with an option to include formatted time.
    
    Args:
    stream : Stream
        Stream object to be saved.
    path : str
        Path where the CSV file will be saved.
    suffix_name : str, optional
        Suffix to add to CSV file name. The default is None.
    folder : str, optional
        Subfolder within path to save the CSV file. The default is 'output'.
    include_time : bool, optional
        Whether to include time data in the first column. The default is True.
    """
    # Validate path
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")
    # Create the output folder if it does not exist
    output_folder = os.path.join(path, folder)
    os.makedirs(output_folder, exist_ok=True)

    # Add suffix to file name if present
    file_name = f'data{suffix_name}.csv' \
        if suffix_name is not None else 'data.csv'
    file_path = os.path.join(output_folder, file_name)

    try:
        # Save stream data to CSV
        with open(file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Prepare header
            header = stream.header
            if include_time:
                header = ['Epoch_UTC'] + header
            csv_writer.writerow(header)
            
            # Write data rows
            for i in range(len(stream.data)):
                row = stream.data[i, :].tolist()
                if include_time and len(stream.time) == len(stream.data):
                    time_val = stream.time[i]
                    row = [time_val] + row
                csv_writer.writerow(row)
        print(f"Stream saved to CSV: {file_name}")
    except (FileNotFoundError, PermissionError, IOError, OSError) as e:
        print(f"Failed to save the stream to CSV: {e}")
    except ValueError as e:
        print(f"Data format error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def save_stream(
    path: str,
    stream: Stream,
<<<<<<< HEAD
    suffix_name: Optional[str] = None,
    folder: str = "output",
=======
    sufix_name: Optional[str] = None,
    folder: Optional[str] = 'output'
>>>>>>> upstream/main
) -> None:
    """
    Save stream object as a pickle file.
    
    Args
    ----------
    stream : Stream
        Stream object to be saved.
    path : str
        Path to save pickle file.
    suffix_name : str, optional
        Suffix to add to pickle file name. The default is None.
    """
    # Validate path
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")

    # create output folder if it does not exist
    output_folder = os.path.join(path, folder)
    os.makedirs(output_folder, exist_ok=True)

    # add suffix to file name if present
<<<<<<< HEAD
    file_name = (
        f"stream{suffix_name}.pk" if suffix_name is not None else "stream.pk"
    )
=======
    file_name = f'stream{sufix_name}.pk' \
        if sufix_name is not None else 'stream.pk'
>>>>>>> upstream/main
    # path to save pickle file
    file_path = os.path.join(output_folder, file_name)

    try:
        # Attempt to save the stream
        with open(file_path, 'wb') as file:
            pickle.dump(stream, file)
        print(f"Stream saved: {file_name}")
    except IOError as e:
        # Handles I/O errors (e.g., file not found, no permissions)
        print(f"Failed to save the stream due to an I/O error: {e}")
    except pickle.PickleError as e:
        # Handles errors specifically related to the pickling process
        print(f"Failed to save the stream due to a pickling error: {e}")
    except Exception as e:
        # Handles any other unexpected errors
        print(f"An unexpected error occurred: {e}")


def load_stream(
    path: str,
<<<<<<< HEAD
    suffix_name: Optional[str] = None,
    folder: str = "output",
=======
    sufix_name: Optional[str] = None,
    folder: Optional[str] = 'output'
>>>>>>> upstream/main
) -> Stream:
    """
    Load stream object from a pickle file.
    
    Args
    ----------
    path : str
        Path to load pickle file.
    suffix_name : str, optional
        Suffix to add to pickle file name. The default is None.
    folder : str, optional
        Folder to load pickle file from. The default is 'output'.
    
    Returns
    -------
    Stream
        Loaded Stream object.
    """
    # Validate path
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")
    # add suffix to file name if present
<<<<<<< HEAD
    file_name = (
        f"stream{suffix_name}.pk" if suffix_name is not None else "stream.pk"
    )
=======
    file_name = f'stream{sufix_name}.pk' \
        if sufix_name is not None else 'stream.pk'
>>>>>>> upstream/main
    # path to load pickle file
    file_path = os.path.join(path, folder, file_name)

    # load stream
    with open(file_path, 'rb') as file:
        stream = pickle.load(file)

    return stream


def save_lake(
    path: str,
    lake: Lake,
<<<<<<< HEAD
    suffix_name: Optional[str] = None,
    folder: str = "output",
=======
    sufix_name: Optional[str] = None,
    folder: Optional[str] = 'output'
>>>>>>> upstream/main
) -> None:
    """
    Save each stream in the lake as separate pickle files.

    Arguments:
        path: Path to save pickle files.
        lake: Lake object to be saved.
        suffix_name: Suffix to add to pickle file names. The default is None.
        folder: Folder to save pickle files. The default is 'output'.
    """
<<<<<<< HEAD
    print("Saving lake...")

=======
    print('Saving lake...')
>>>>>>> upstream/main
    # Validate path
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")

    # Create output folder if it does not exist
    output_folder = os.path.join(path, folder)
    os.makedirs(output_folder, exist_ok=True)

<<<<<<< HEAD
    # Save each stream as a separate file
    for i, (stream_name, stream) in enumerate(lake.items(), start=1):
        file_name = (
            f"lake_part{i:02d}_{suffix_name}.pk"
            if suffix_name
            else f"lake_part{i:02d}.pk"
        )
        file_path = os.path.join(output_folder, file_name)
        try:
            with open(file_path, "wb") as file:
                pickle.dump(stream, file)
            print(f"Saved stream '{stream_name}' as '{file_name}'")
        except IOError as e:
            print(
                f"Failed to save stream '{stream_name}' "
                "due to an I/O error: {e}"
            )
        except pickle.PickleError as e:
            print(
                f"Failed to save stream '{stream_name}' "
                "due to a pickling error: {e}"
            )
        except Exception as e:
            print(
                f"An unexpected error occurred while saving stream"
                f" '{stream_name}': {e}"
            )
=======
    # add suffix to file name if present
    file_name = f'lake{sufix_name}.pk' \
        if sufix_name is not None else 'lake.pk'
    # path to save pickle file
    file_path = os.path.join(output_folder, file_name)
>>>>>>> upstream/main

    # Save the lake metadata (just the stream names) in a separate file
    metadata_file = (
        f"lake_metadata_{suffix_name}.pk"
        if suffix_name
        else "lake_metadata.pk"
    )
    metadata_path = os.path.join(output_folder, metadata_file)
    try:
<<<<<<< HEAD
        with open(metadata_path, "wb") as file:
            pickle.dump(list(lake.streams.keys()), file)
        print(f"Lake metadata saved as '{metadata_file}'")
=======
        # Attempt to save the datalake
        with open(file_path, 'wb') as file:
            pickle.dump(lake, file)
        print(f"Lake saved: {file_name}")
>>>>>>> upstream/main
    except IOError as e:
        print(f"Failed to save lake metadata due to an I/O error: {e}")
    except pickle.PickleError as e:
        print(f"Failed to save lake metadata due to a pickling error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving lake metadata: {e}")


def load_lake(
    path: str,
<<<<<<< HEAD
    suffix_name: Optional[str] = None,
    folder: str = "output",
=======
    sufix_name: Optional[str] = None
>>>>>>> upstream/main
) -> Lake:
    """
    Load a lake object by loading individual streams from separate pickle files.

    Arguments:
        path: Path to load pickle files.
        suffix_name: Suffix to add to pickle file names. The default is None.
        folder: Folder to load pickle files from. The default is 'output'.

<<<<<<< HEAD
    Returns:
        Lake: Reconstructed Lake object.
    """
    print("Loading lake...")

    # Validate path
    if not os.path.isdir(path):
        raise ValueError(f"Provided path '{path}' is not a directory.")
=======
    Returns
    -------
    data_lake : DataLake
        Loaded DataLake object.
    """
    file_name = f'lake{sufix_name}.pk' \
        if sufix_name is not None else 'lake.pk'
    # path to load pickle file
    file_path = os.path.join(path, 'output', file_name)

    # load datalake
    with open(file_path, 'rb') as file:
        lake = pickle.load(file)
>>>>>>> upstream/main

    # Path to the folder where streams are stored
    load_folder = os.path.join(path, folder)

    # Load lake metadata (stream names)
    metadata_file = (
        f"lake_metadata_{suffix_name}.pk"
        if suffix_name
        else "lake_metadata.pk"
    )
    metadata_path = os.path.join(load_folder, metadata_file)

    try:
        with open(metadata_path, "rb") as file:
            stream_names = pickle.load(file)
        print(f"Loaded lake metadata from '{metadata_file}'")
    except IOError as e:
        raise ValueError(
            f"Failed to load lake metadata due to an I/O error: {e}"
        )
    except pickle.PickleError as e:
        raise ValueError(
            f"Failed to load lake metadata due to a pickling error: {e}"
        )
    except Exception as e:
        raise ValueError(
            f"An unexpected error occurred while loading lake metadata: {e}"
        )

    # Initialize an empty lake
    lake = Lake()

    # Load each stream and add to the lake
    for i, stream_name in enumerate(stream_names, start=1):
        file_name = (
            f"lake_part{i:02d}_{suffix_name}.pk"
            if suffix_name
            else f"lake_part{i:02d}.pk"
        )
        file_path = os.path.join(load_folder, file_name)

        try:
            with open(file_path, "rb") as file:
                stream = pickle.load(file)
            lake.add_stream(stream, stream_name)
            print(f"Loaded stream '{stream_name}' from '{file_name}'")
        except IOError as e:
            print(
                f"Failed to load stream '{stream_name}' "
                "due to an I/O error: {e}"
            )
        except pickle.PickleError as e:
            print(
                f"Failed to load stream '{stream_name}' "
                "due to a pickling error: {e}"
            )
        except Exception as e:
            print(
                "An unexpected error occurred while loading stream "
                f"'{stream_name}': {e}"
            )
    return lake


def netcdf_get_epoch_time(
        file_path: str,
        settings: dict
) -> np.ndarray:
    """
    Given a netCDF file path and settings, returns an array of epoch times in
    seconds as a float.

    Currently only uses ARM 1.2 netCDF files (base_time + time_offset)

    Args:
        file_path (str): The path to the netCDF file.
        settings (dict): A dictionary containing settings for the instrument.

    Returns:
        np.ndarray: An array of epoch times, in seconds as a float.
    """
    nc_file = nc.Dataset(file_path)  # type: ignore

    epoch_time = np.zeros(nc_file.dimensions['time'].size)

    for time_col in settings['time_column']:
        epoch_time += nc_file.variables.get(time_col)[:]
    epoch_time = np.array(epoch_time.astype(float))
    nc_file.close()

    return epoch_time


def netcdf_data_1d_load(
        file_path: str,
        settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Given a netCDF file path and settings, returns a tuple containing the
    epoch time, header, and data as a numpy array. We do apply the mask to the
    data, and fill the masked values with nan.

    Args:
        file_path (str): The path to the netCDF file.
        settings (dict): A dictionary containing settings for the instrument.

    Returns:
        Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
        header, and data as a numpy array.

    Errors:
        KeyError: If the settings dictionary does not contain 'data_1d'.
    """
    # check if data_1d is in the settings dic
    if 'data_1d' not in settings['netcdf_reader']:
        raise KeyError("data_1d not in settings['netcdf_reader']")

    # get header
    header_1d = settings['netcdf_reader']['header_1d']

    nc_file = nc.Dataset(file_path)  # type: ignore
    # get epoch time
    epoch_time = netcdf_get_epoch_time(file_path, settings)

    # empty array to store data
    data_1d = np.zeros(
        (len(settings['netcdf_reader']['data_1d']),
         nc_file.dimensions['time'].size)
    )
    # select and fill masked array with nan
    for i, data_col in enumerate(settings['netcdf_reader']['data_1d']):
        try:
            data = nc_file.variables.get(data_col)[:]
            data_1d[i, :] = np.ma.filled(data.astype(float), np.nan)
        except (TypeError, KeyError):
            data_1d[i, :] = np.nan
            warnings.warn(data_col + " not found in the netCDF file")
    nc_file.close()

    # check data shape, transpose if necessary so that time is last dimension
    data_1d = convert.data_shape_check(
        time=epoch_time,
        data=data_1d,
        header=header_1d)

    return epoch_time, header_1d, data_1d


# pylint: disable-all
def netcdf_data_2d_load(
        file_path: str,
        settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Given a netCDF file path and settings, returns a tuple containing the
    epoch time, header, and data as a numpy array. We do apply the mask to the
    data, and fill the masked values with nan.

    Args:
        file_path (str): The path to the netCDF file.
        settings (dict): A dictionary containing settings for the instrument.

    Returns:
        Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
        header, and data as a numpy array.

    Errors:
        KeyError: If the settings dictionary does not contain 'data_2d'.
    """
    # check if data_1d is in the settings dic
    if 'data_2d' not in settings['netcdf_reader']:
        raise KeyError("data_2d not in settings['netcdf_reader']")

    # get epoch time
    epoch_time = netcdf_get_epoch_time(file_path, settings)
    # load netcdf file
    nc_file = nc.Dataset(file_path)  # type: ignore

    # select data_2d
    data_2d = nc_file.variables.get(settings['netcdf_reader']['data_2d'])[:]
    # convert masked array to numpy array
    data_2d = np.ma.filled(data_2d.astype(float), np.nan)
    # get header
    header_2d = nc_file.variables.get(
        settings['netcdf_reader']['header_2d']
    )[:]
    nc_file.close()

    # convert header to list of strings
    header_2d = [str(item) for item in header_2d.tolist()]

    # check data shape, transpose if necessary so that time is last dimension
    data_2d = convert.data_shape_check(
        time=epoch_time,
        data=data_2d,
        header=header_2d)

    return epoch_time, header_2d, data_2d


def netcdf_info_print(file_path, file_return=False):
    """
    Prints information about a netCDF file. Useful for generating settings
    dictionaries.

    Args:
        file_path (str): The path to the netCDF file.
        file_return (bool): If True, returns the netCDF file object.
            Defaults to False.

    Returns:
        nc_file (netCDF4.Dataset): The netCDF file object.
    """

    nc_file = nc.Dataset(file_path)  # type: ignore
    print("Dimensions:")
    for dim in nc_file.dimensions:
        print(dim, len(nc_file.dimensions[dim]))
    print("\nVariables:")
    for var in nc_file.variables:
        print(var,
              nc_file.variables[var].shape,
              nc_file.variables[var].dtype)
    print("\nHeaders:")
    for attr in nc_file.ncattrs():
        print(attr, "=", getattr(nc_file, attr))

    if file_return:
        return nc_file
    nc_file.close()
    return None
