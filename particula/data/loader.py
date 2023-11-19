"""File readers and loaders for datacula."""

from typing import List, Union, Tuple, Dict, Any, Optional
import warnings
import glob
import os
import pickle
import netCDF4 as nc
import numpy as np
import pandas as pd

from particula.util import convert
from particula.util.time_manage import time_str_to_epoch
from particula.data.lake import Lake

FILTER_WARNING_FRACTION = 0.5


def data_raw_loader(file_path: str) -> list:
    """
    Load raw data from a file at the specified file path and return it as a
    list of strings.

    Args:
        file_path (str): The file path of the file to read.

    Returns:
        list: The raw data read from the file as a list of strings.

    Examples:
        >>> data = data_raw_loader('my_file.txt')
        Loading data from: my_file.txt
        >>> print(data)
        ['line 1', 'line 2', 'line 3']
    """
    try:
        with open(file_path, 'r', encoding='utf8', errors='replace') as file:
            data = [line.rstrip() for line in file]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        data = []
    return data


def filter_list(data: List[str], char_counts: dict) -> List[str]:
    """
    A pass filter of rows from a list of strings.
    Each row must contain a specified number of characters to pass the filter.
    The number of characters to count is specified in the char_counts
    dictionary. The keys are the characters to count, and the values are the
    exact count required for each character in each row.

    Args:
    ----------
        data (List[str]): A list of strings to filter.
            A list of strings to filter.
        char_counts (dict): A dictionary of character counts to select by.
            The keys are the characters to count, and the values are the
            count required for each character.

    Returns:
    ----------
        List[str]: A new list of strings containing only the rows that meet the
        character count requirements.

    Raises:
    ----------
        UserWarning: If more than 90% of the rows are filtered out, and it
            includes the character(s) used in the filter.

    Examples:
    ----------
        >>> data = ['apple,banana,orange', 'pear,kiwi,plum',
                    'grapefruit,lemon']
        >>> char_counts = {',': 2}
        >>> filtered_data = filter_rows_by_count(data, char_counts)
        >>> print(filtered_data)
        ['apple,banana,orange', 'pear,kiwi,plum']
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


def data_format_checks(data: List[str], data_checks: dict) -> List[str]:
    """
    Check if the data is in the correct format.

    Args:
        data (List[str]): A list of strings containing the raw data.
        data_checks (dict): Dictionary containing the format checks.

    Returns:
        List[str]: A list of strings containing the formatted data.

    Raises:
        TypeError: If data is not a list.

    Examples:
        >>> data = ['row 1', 'row 2', 'row 3']
        >>> data_checks = {
        ...     "characters": [0, 10],
        ...     "char_counts": {",": 2, "/": 0, ":": 0},
        ...     "skip_rows": 0,
        ...     "skip_end": 0
        ... }
        >>> formatted_data = data_format_checks(data, data_checks)
        >>> print(formatted_data)
        ['row 2']
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
    Parses the time column of a data line and returns it as a timestamp.

    Args:
    ----------
    time_column : Union[int, List[int]]
        The index or indices of the column(s) containing the time information.
    time_format : str
        The format of the time information, e.g. '%Y-%m-%d %H:%M:%S'.
    line : str
        The data line to parse.
    date_offset : Optional[str], default=None
        A fixed date offset to add to the timestamp in front.
    seconds_shift : int, default=0
        A number of seconds to add to the timestamp.

    Returns:
    -------
    float
        The timestamp corresponding to the time information in the data line,
        in seconds since the epoch.

    Raises:
    ------
    ValueError
        If an invalid time column or format is specified.
    """
    if time_format == 'epoch':
        return float(line[time_column]) + seconds_shift
    if date_offset:
        # if the time is in one column, and the date is fixed
        time_str = f"{date_offset} {line[time_column]}"
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
    if isinstance(time_column, list) and len(time_column) == 2:
        # if the time and date are in two column
        time_str = f"{line[time_column[0]]} {line[time_column[1]]}"
        return (
            time_str_to_epoch(time_str, time_format, timezone_identifier)
            + seconds_shift
        )
    raise ValueError(
        f"Invalid time column or format: {time_column}, {time_format}")


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
    Samples the data to get the time and data streams.

    Args:
    -----------
    data : List[str]
        The input data in the form of a list of strings.
    time_column : int
        The index of the column that contains the time values.
    time_format : str
        The format string that specifies the time format.
    data_columns : List[int]
        The indices of the columns that contain the data values.
    delimiter : str
        The delimiter character used to separate columns in the input data.
    date_offset : str, optional
        A string that represents an offset in the date, in the format
        'days:hours:minutes:seconds'. Defaults to None.
    seconds_shift : int, optional
        An integer that represents a time shift in seconds. Defaults to 0.
    timezone_identifier : str, optional
        What timezone the data is in. Defaults to 'UTC'.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays - epoch_time and data_array:
        - epoch_time : np.ndarray
            A 1-D numpy array of epoch times.
        - data_array : np.ndarray
            A 2-D numpy array of data values.

    Raises:
    -------
    ValueError:
        - If the data value is not in the correct format.
        - If no match for data value is found.
    """
    # flake8: noqa
    # pylint disable: too-many-arguments
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
                    'ON', 'on', 'On', 'oN', '1', 'True', 'true',
                    'TRUE', 'tRUE', 't', 'T', 'Yes', 'yes', 'YES',
                    'yES', 'y', 'Y'
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
    Formats and samples the data to get the time and data streams.

    Args:
    ----------
    data : list
        The list of strings containing the data.
    data_checks : dict
        A dictionary of data format checks to apply to the data.
    data_column : list
        The list of indices of the columns containing the data.
    time_column : Union[int, List[int]]
        The index or indices of the column(s) containing the time information.
    time_format : str
        The format of the time information, e.g. '%Y-%m-%d %H:%M:%S'.
    delimiter : str, default=','
        The delimiter used to separate columns in the data.
    date_offset : str, default=None
        A fixed date offset to add to the timestamp in front.
    seconds_shift : int, default=0
        A number of seconds to add to the timestamp.

    Returns:
    -------
    Tuple[np.array, np.array]
        A tuple containing two np.array objects: the first contains the
        epoch times, and the second contains the data.
    """

    # find str matching in header row and gets index
    if isinstance(data_column[0], str):
        data_header = data[header_row].split(delimiter)
        # Get data column indices
        data_column = [data_header.index(x)
                       for x in data_column]

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
    Formats data from a particle sizer.

    Parameters
    ----------
    data : List[str]
        The data to be formatted.
    data_checks : Dict[str, Any]
        Dictionary specifying the formatting requirements for the data.
    data_sizer_reader : Dict[str, str]
        Dictionary containing information about the sizer data format.
    time_column : int
        The index of the time column in the data.
    time_format : str
        The format of the time information.
    delimiter : str, default=','
        The delimiter used in the data.
    date_offset : str, default=None
        The date offset to add to the timestamp.
    seconds_shift : int, default=0
        The number of seconds to add to the timestamp.
    timezone_identifier : str, default='UTC'
        The timezone identifier for the data.

    Returns
    -------
    Tuple[np.ndarray, List(str) np.ndarray, np.ndarray]
        A tuple containing the epoch time, the Dp header, and the data arrays.
    """

    # Get Dp range and columns
    data_header = data[header_row].split(delimiter)
    # check if start and end keywords are in the header
    if data_sizer_reader["Dp_start_keyword"] not in data_header:
        # rise error with snip of data header
        raise ValueError(
            f"Cannot find '{data_sizer_reader['Dp_start_keyword']}' in header"
            + f" {data_header[:20]}..."
        )
    if data_sizer_reader["Dp_end_keyword"] not in data_header:
        # rise error with snip of data header
        raise ValueError(
            f"Cannot find '{data_sizer_reader['Dp_end_keyword']}' in header"
            + f" {data_header[:20]}..."
        )
    dp_range = [
        data_header.index(data_sizer_reader["Dp_start_keyword"]),
        data_header.index(data_sizer_reader["Dp_end_keyword"])
    ]
    dp_columns = list(range(dp_range[0], dp_range[1] + 1))  # +1 to include end
    header = [data_header[i] for i in dp_columns]
    # change from np.array

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

    Args:
    ----------
    data : list
        A list of strings representing the data.
    date_location : dict
        A dictionary specifying the method for extracting the date from the
        data.
        Supported methods include:
            - 'file_header_block': The date is located in the file header
                block, and its position is specified by the 'row',
                'delimiter', and 'index' keys.

    Returns:
    -------
    str
        The date extracted from the specified location in the data.

    Raises:
    ------
    ValueError
        If an unsupported or invalid method is specified in date_location.
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
    ----------
    path : str
        The path to the parent folder.
    subfolder : str
        The name of the subfolder containing the files.
    filename_regex : str
        A regular expression pattern for matching the filenames.
    min_size : int, optional
        The minimum file size in bytes (default is 10).

    Returns:
    -------
    Tuple[List[str], List[str], List[int]]
        A tuple containing three lists:
        - The filenames that match the pattern and size criteria
        - The full paths to the files
        - The file sizes in bytes
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


def save_lake(
        path: str,
        lake: Lake,
        sufix_name: Optional[str] = None):
    """
    Save lake object as a pickle file.

    Parameters
    ----------
    data_lake : DataLake
        DataLake object to be saved.
    path : str
        Path to save pickle file.
    sufix_name : str, optional
        Suffix to add to pickle file name. The default is None.
    """
    print('Saving lake...')
    # create output folder if it does not exist
    output_folder = os.path.join(path, 'output')
    os.makedirs(output_folder, exist_ok=True)

    # add suffix to file name if present
    file_name = f'lake_{sufix_name}.pk' if sufix_name is not None else 'lake.pk'
    # path to save pickle file
    file_path = os.path.join(output_folder, file_name)

    # save datalake
    with open(file_path, 'wb') as file:
        pickle.dump(lake, file)
    print('Lake saved')


def load_lake(path: str, sufix_name: Optional[str] = None) -> object:
    """
    Load datalake object from a pickle file.

    Parameters
    ----------
    path : str
        Path to load pickle file.

    Returns
    -------
    data_lake : DataLake
        Loaded DataLake object.
    """
    file_name = f'lake_{sufix_name}.pk' if sufix_name is not None else 'lake.pk'
    # path to load pickle file
    file_path = os.path.join(path, 'output', file_name)

    # load datalake
    with open(file_path, 'rb') as file:
        lake = pickle.load(file)

    return lake


# pylint: disable-all
def netcdf_get_epoch_time(
        file_path: str,
        settings: dict
) -> np.ndarray:
    """
    Given a netCDF file path and settings, returns an array of epoch times in
    seconds as a float.

    Currently only uses ARM 1.2 netCDF files (base_time + time_offset)

    Args:
    ----------
        file_path (str): The path to the netCDF file.
        settings (dict): A dictionary containing settings for the instrument.

    Returns:
    -------
        np.ndarray: An array of epoch times, in seconds as a float.
    """
    nc_file = nc.Dataset(file_path)  # type: ignore

    epoch_time = np.zeros(nc_file.dimensions['time'].size)

    for time_col in settings['time_column']:
        epoch_time += nc_file.variables.get(time_col)[:]
    epoch_time = np.array(epoch_time.astype(float))
    nc_file.close()

    return epoch_time


# pylint: disable-all
def netcdf_data_1d_load(
        file_path: str,
        settings: dict
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Given a netCDF file path and settings, returns a tuple containing the
    epoch time, header, and data as a numpy array. We do apply the mask to the
    data, and fill the masked values with nan.

    Args:
    ----------
        file_path (str): The path to the netCDF file.
        settings (dict): A dictionary containing settings for the instrument.

    Returns:
    -------
        Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
        header, and data as a numpy array.

    Errors:
    ------
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
    ----------
        file_path (str): The path to the netCDF file.
        settings (dict): A dictionary containing settings for the instrument.

    Returns:
    -------
        Tuple[np.ndarray, list, np.ndarray]: A tuple containing the epoch time,
        header, and data as a numpy array.

    Errors:
    ------
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
    ----------
        file_path (str): The path to the netCDF file.
        file_return (bool): If True, returns the netCDF file object.
            Defaults to False.

    Returns:
    -------
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
