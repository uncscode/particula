"""Callable to generate settings file from template."""
# pylint: disable=all


from typing import List, Optional
import json
import warnings
import os

from particula.data.loader import get_files_in_folder_with_size


def for_general_1d_load(
        relative_data_folder: str = 'instrument_data',
        filename_regex: str = '*.csv',
        file_min_size_bytes: int = 10,
        header_row: int = 0,
        data_checks: Optional[dict] = None,
        data_column: list = [3, 5],
        data_header: List[str] = ['data 1', 'data 3'],
        time_column: List[int] = [0, 1],
        time_format: str = '%Y-%m-%d %H:%M:%S.%f',
        delimiter: str = ',',
        time_shift_seconds: int = 0,
        timezone_identifier: str = 'UTC',
) -> dict:
    """
    Generate a settings dictionary for loading and checking 1D data from CSV
    files.

    Parameters:
    ------------
    - relative_data_folder (str): The folder path relative to the main script
        where data files are located. Default is 'instrument_data'.
    - filename_regex (str): Regular expression pattern to match filenames in
        the data folder. Default is '*.csv'.
    - file_min_size_bytes (int): Minimum size in bytes for files to be
        considered valid. Default is 10.
    - header_row (int): The index of the row containing column headers
        (0-indexed). Default is 0.
    - data_checks (Optional[dict]): A dictionary containing data quality
        checks such as character length, required character counts, rows to
        skip at the beginning or end. Defaults to basic checks if None.
    - data_column (list of int): List of indices for columns containing data
        points to be loaded. Default is [3, 5].
    - data_header (List[str]): List of strings representing the header names
        for data columns. Default is ['data 1', 'data 3'].
    - time_column (List[int]): List of indices for columns containing time
        information. Default is [0, 1].
    - time_format (str): String format for parsing time columns, using
        strftime conventions. Default is '%Y-%m-%d %H:%M:%S.%f'.
    - delimiter (str): Character used to separate values in the file.
        Default is ','.
    - time_shift_seconds (int): Number of seconds by which to shift time data
        (positive or negative). Default is 0.
    - timezone_identifier (str): Timezone identifier for time conversion.
        Default is 'UTC'.

    Returns:
    - dict: A dictionary with settings for data loading procedures including
        file paths, size requirements, header information, and data check
        parameters.
    """
    if data_checks is None:
        data_checks = {
            "characters": [10, 100],
            "char_counts": {",": 4, ":": 0},
            "skip_rows": 0,
            "skip_end": 0,
        }
    return {
        'relative_data_folder': relative_data_folder,
        'filename_regex': filename_regex,
        'MIN_SIZE_BYTES': file_min_size_bytes,
        'data_loading_function': 'general_1d_load',
        'header_row': header_row,
        'data_checks': data_checks,
        'data_column': data_column,
        'data_header': data_header,
        'time_column': time_column,
        'time_format': time_format,
        'delimiter': delimiter,
        'time_shift_seconds': time_shift_seconds,
        'timezone_identifier': timezone_identifier,
    }


def for_general_sizer_1d_2d_load(
        relative_data_folder: str = 'instrument_data',
        filename_regex: str = '*.csv',
        file_min_size_bytes: int = 10,
        header_row: int = 0,
        data_checks: Optional[dict] = None,
        data_1d_column: list = [3, 5],
        data_1d_header: List[str] = ['data 1', 'data 3'],
        data_2d_dp_start_keyword: str = "Date Time",
        data_2d_dp_end_keyword: str = "Total Conc",
        data_2d_convert_concentration_from: str = "dw/dlogdp",  # or dw
        time_column: List[int] = [0, 1],
        time_format: str = '%Y-%m-%d %H:%M:%S.%f',
        delimiter: str = ',',
        time_shift_seconds: int = 0,
        timezone_identifier: str = 'UTC',
) -> tuple:
    """
    Generate settings for the 1D general file loader and the 2D general sizer
        file loader.

    Parameters:
    - relative_data_folder (str): Path to the folder containing data files,
        relative to the script's location.
    - filename_regex (str): Regex pattern to match filenames for loading.
    - file_min_size_bytes (int): Minimum file size in bytes for a file to be
        considered valid for loading.
    - header_row (int): Row index for the header (0-based) in the data files.
    - data_checks (dict, optional): Specifications for data integrity checks
        to apply when loading data.
    - data_1d_column (list of int): Column indices for 1D data extraction.
    - data_1d_header (list of str): Header names corresponding to the
        `data_1d_column` indices.
    - data_2d_dp_start_keyword (str): Keyword indicating the start of 2D data
        points in a file.
    - data_2d_dp_end_keyword (str): Keyword indicating the end of 2D data
        points in a file.
    - data_2d_convert_concentration_from (str, optional): Unit to convert from
        if concentration scaling is needed for 2D data.
    - time_column (list of int): Column indices for time data extraction.
    - time_format (str): Format string for parsing time data.
    - delimiter (str): Delimiter character for splitting data in the file.
    - time_shift_seconds (int): Seconds to shift the time data by.
    - timezone_identifier (str): Timezone ID for time data interpretation.

    Returns:
    - tuple of (dict, dict): A tuple containing two dictionaries with settings
        for the 1D and 2D data loaders.

    The function defaults `data_checks` to basic validation criteria if not
        provided. It returns separate dictionaries for settings applicable to
        1D and 2D data loaders, which include file paths, size checks, and
        data parsing rules.
    """
    if data_checks is None:
        data_checks = {
            "characters": [10, 100],
            "char_counts": {",": 4, ":": 0},
            "skip_rows": 0,
            "skip_end": 0,
        }
    settings_1d = {
        'relative_data_folder': relative_data_folder,
        'filename_regex': filename_regex,
        'MIN_SIZE_BYTES': file_min_size_bytes,
        'data_loading_function': 'general_1d_load',
        'header_row': header_row,
        'data_checks': data_checks,
        'data_column': data_1d_column,
        'data_header': data_1d_header,
        'time_column': time_column,
        'time_format': time_format,
        'delimiter': delimiter,
        'time_shift_seconds': time_shift_seconds,
        'timezone_identifier': timezone_identifier,
    }
    settings_2d = {
        'relative_data_folder': relative_data_folder,
        'filename_regex': filename_regex,
        'MIN_SIZE_BYTES': file_min_size_bytes,
        'data_loading_function': 'general_2d_load',
        'header_row': header_row,
        'data_checks': data_checks,
        'data_sizer_reader': {
            "Dp_start_keyword": data_2d_dp_start_keyword,
            "Dp_end_keyword": data_2d_dp_end_keyword,
            "convert_scale_from": data_2d_convert_concentration_from,
        },
        'time_column': time_column,
        'time_format': time_format,
        'delimiter': delimiter,
        'time_shift_seconds': time_shift_seconds,
        'timezone_identifier': timezone_identifier,
    }
    return settings_1d, settings_2d


def load_settings_for_stream(
            path: str,
            subfolder: str,
            settings_sufix: str = ''
            ) -> dict:
    """auto loader for lake data settings. given a path, it will look for the
    json file in each folder containing the data and return the settings
    dictionary for the data in that folder.
    """

    settings_file_name = 'stream_settings' + settings_sufix + '.json'
    file_list, full_path, file_size_in_bytes = get_files_in_folder_with_size(
        path=path,
        subfolder=subfolder,
        filename_regex=settings_file_name,
        min_size=10)

    # if there is no settings file, raise an error
    if len(file_size_in_bytes) == 0:
        raise FileNotFoundError(
            f'No stream_settings file found in {path}/{subfolder}.')
    if len(file_size_in_bytes) > 1:
        raise warnings.warn(
            f'More than one stream_settings file found in {path}/{subfolder}.'
            'Using the first one found.')  # type: ignore

    return json.load(open(full_path[0], 'r'))


def save_settings_for_stream(
        settings: dict,
        path: str,
        subfolder: str,
        settings_sufix: str = ''
        ) -> None:
    """auto saver for lake data settings. given a path, it will look for the
    json file in each folder containing the data and save the settings
    dictionary for the data in that folder.
    """

    settings_file_name = 'stream_settings' + settings_sufix + '.json'

    save_path = os.path.join(path, subfolder, settings_file_name)
    # write the json, with 4 space indentation
    json.dump(settings, open(save_path, 'w'), indent=4)
