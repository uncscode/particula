"""Callable to generate settings file from template."""
# pylint: disable=all
# pytype: skip-file

from typing import List, Optional


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
    """Generate settings file for 1d general file."""
    if data_checks is None:
        data_checks = {
            "characters": [10, 100],
            "char_counts": {",": 4, ":": 0},
            "skip_rows": 0,
            "skip_end": 0,
        },
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
    """Generate settings file for 1d general file loader and
    2d general sizer file loader.

    Returns
    -------
    Tuple[dict, dict]
        The settings for the 1d loader and the 2d loader.
    """
    if data_checks is None:
        data_checks = {
            "characters": [10, 100],
            "char_counts": {",": 4, ":": 0},
            "skip_rows": 0,
            "skip_end": 0,
        },
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
